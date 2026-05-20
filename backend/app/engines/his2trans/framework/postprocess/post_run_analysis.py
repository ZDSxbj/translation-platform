#!/usr/bin/env python3
"""
Post-run analyzer for batch_test_staged.sh outputs.

What it does (best-effort, no network):
1) Safety: unsafe usage ratio across Rust outputs.
2) Idiomaticity: optional CodeBLEU-like *proxy* score (disabled by default; requires a reference).
3) Failure breakdown: aggregate translation failure counters from translation_stats.json.

Outputs a JSON report under the run results directory.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


RUST_KEYWORDS = {
    "as",
    "async",
    "await",
    "break",
    "const",
    "continue",
    "crate",
    "dyn",
    "else",
    "enum",
    "extern",
    "false",
    "fn",
    "for",
    "if",
    "impl",
    "in",
    "let",
    "loop",
    "match",
    "mod",
    "move",
    "mut",
    "pub",
    "ref",
    "return",
    "self",
    "Self",
    "static",
    "struct",
    "super",
    "trait",
    "true",
    "type",
    "unsafe",
    "use",
    "where",
    "while",
}

_MISSING_ITEM_E0425_RE = re.compile(
    r"error\[E0425\]:\s+cannot find function, tuple struct or tuple variant `([^`]+)` in this scope"
)
_MISSING_MACRO_RE = re.compile(r"cannot find macro `([^`]+)` in this scope")


def _scan_tu_context_closure(project_dir: Path) -> Dict[str, Any]:
    """
    Read stage1-generated TU context map to detect "input closure" gaps:
    - file not present in compile_commands (no entry)
    - preprocessing failed (missing headers/sysroot/toolchain mismatch)
    - preprocessed `.i` path recorded but file missing

    These are not translation errors; they indicate the build context available to the framework is incomplete.
    """
    ws = project_dir / "workspace"
    tu_map_path = ws / ".preprocessed" / "tu_context_map.json"
    if not tu_map_path.exists():
        return {"enabled": False, "reason": "tu_context_map_missing", "path": str(tu_map_path)}

    try:
        data = json.loads(_read_text(tu_map_path) or "{}")
    except Exception as e:
        return {"enabled": False, "reason": f"tu_context_map_parse_failed: {e}", "path": str(tu_map_path)}

    files = data.get("files") if isinstance(data, dict) else None
    if not isinstance(files, dict):
        return {"enabled": False, "reason": "tu_context_map_no_files", "path": str(tu_map_path)}

    total_files = len(files)
    excluded_by_cc = 0
    included_files = 0
    entry_present = 0
    entry_missing = 0
    pre_ok = 0
    pre_failed = 0
    pre_not_attempted = 0
    pre_file_missing = 0

    fail_samples: List[Dict[str, Any]] = []
    missing_samples: List[Dict[str, Any]] = []

    for safe_name, rec in files.items():
        if not isinstance(rec, dict):
            continue
        if rec.get("excluded_by_compile_commands"):
            excluded_by_cc += 1
            continue
        included_files += 1
        entry = rec.get("compile_commands_entry")
        has_entry = isinstance(entry, dict) and bool(entry.get("command") or entry.get("arguments"))
        if has_entry:
            entry_present += 1
        else:
            entry_missing += 1

        pre = rec.get("preprocessed_file")
        err = (rec.get("error") or "").strip()
        if pre:
            try:
                p = Path(str(pre))
                if p.exists():
                    pre_ok += 1
                else:
                    pre_file_missing += 1
                    if len(missing_samples) < 10:
                        missing_samples.append(
                            {
                                "file_group": str(safe_name),
                                "source_file_rel": rec.get("source_file_rel"),
                                "preprocessed_file": str(p),
                            }
                        )
            except Exception:
                pre_file_missing += 1
        else:
            if err:
                pre_failed += 1
                if len(fail_samples) < 10:
                    fail_samples.append(
                        {
                            "file_group": str(safe_name),
                            "source_file_rel": rec.get("source_file_rel"),
                            "error": err[:240],
                        }
                    )
            else:
                pre_not_attempted += 1

    preprocessing_enabled = bool(data.get("preprocessing_enabled")) if isinstance(data, dict) else False
    use_preprocessing = bool(data.get("use_preprocessing")) if isinstance(data, dict) else False

    # Decide if this project has "closure gaps" worth surfacing.
    has_gaps = (
        (not preprocessing_enabled and use_preprocessing)
        or entry_missing > 0
        or pre_failed > 0
        or pre_file_missing > 0
    )

    return {
        "enabled": True,
        "path": str(tu_map_path),
        "use_preprocessing": use_preprocessing,
        "preprocessing_enabled": preprocessing_enabled,
        "strategy": data.get("preprocessing_strategy"),
        "target_config": data.get("target_config"),
        "selected_profile": data.get("selected_profile"),
        "counts": {
            "file_groups_total": total_files,
            "file_groups_included": included_files,
            "excluded_by_compile_commands": excluded_by_cc,
            "compile_commands_entry_present": entry_present,
            "compile_commands_entry_missing": entry_missing,
            "preprocessed_ok": pre_ok,
            "preprocessed_failed": pre_failed,
            "preprocessed_not_attempted": pre_not_attempted,
            "preprocessed_file_missing": pre_file_missing,
        },
        "has_closure_gaps": has_gaps,
        "failure_samples": fail_samples,
        "missing_file_samples": missing_samples,
    }


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _iter_project_dirs(run_intermediate_dir: Path) -> List[Path]:
    project_dirs: List[Path] = []
    for p in sorted(run_intermediate_dir.iterdir()):
        if not p.is_dir():
            continue
        if p.name in ("cache", "logs"):
            continue
        if (p / "workspace").is_dir():
            project_dirs.append(p)
    return project_dirs


def _find_first(path: Path, pattern: str) -> Optional[Path]:
    matches = list(path.glob(pattern))
    return matches[0] if matches else None


def _find_translation_stats(project_dir: Path) -> Optional[Path]:
    ws = project_dir / "workspace"
    return _find_first(ws, "incremental_work/*/translate_by_*/translation_stats.json")


def _detect_rust_root(project_dir: Path, project_name: str, llm_name: Optional[str]) -> Optional[Path]:
    ws = project_dir / "workspace"
    if llm_name:
        p = ws / "final_projects" / project_name / f"translate_by_{llm_name}"
        if (p / "src").is_dir():
            return p
        p = ws / "incremental_work" / project_name / f"translate_by_{llm_name}"
        if (p / "src").is_dir():
            return p
    # best-effort: any translate_by_*
    p = _find_first(ws, f"final_projects/{project_name}/translate_by_*/src")
    if p is not None:
        return p.parent
    p = _find_first(ws, f"incremental_work/{project_name}/translate_by_*/src")
    if p is not None:
        return p.parent

    # skeleton-only fallback
    p = ws / "skeletons" / project_name
    if (p / "src").is_dir():
        return p
    return None


def _scan_repair_rounds(project_dir: Path, *, project_name: str, llm_name: Optional[str]) -> Dict[str, Any]:
    """
    Best-effort scan: compute per-project repair rounds from incremental repair history.

    We interpret "repair rounds" as the number of compile-guided iterations recorded as:
      attempt_<k>_error.txt

    Layout (per incremental_translate.py):
      intermediate/<proj>/workspace/repair_history/<proj>/translate_by_<llm>/<func>/attempt_k_*.{txt,rs}
    """
    ws = project_dir / "workspace"
    repair_base = ws / "repair_history" / project_name
    if not repair_base.is_dir():
        return {"enabled": False, "reason": "repair_history_missing", "repair_base": str(repair_base)}

    candidates: List[Path] = []
    if llm_name:
        p = repair_base / f"translate_by_{llm_name}"
        if p.is_dir():
            candidates.append(p)
    if not candidates:
        candidates = sorted([p for p in repair_base.glob("translate_by_*") if p.is_dir()])
    if not candidates:
        return {"enabled": False, "reason": "no_translate_by_dir", "repair_base": str(repair_base)}

    repair_root = candidates[0]

    attempts_hist: Counter[int] = Counter()
    total_attempts = 0
    funcs_in_repair_phase = 0
    funcs_with_attempts = 0
    repair_success = 0
    repair_failed = 0

    for func_dir in sorted(repair_root.iterdir()):
        if not func_dir.is_dir():
            continue
        if func_dir.name.startswith("_"):
            continue

        funcs_in_repair_phase += 1
        n = len(list(func_dir.glob("attempt_*_error.txt")))
        attempts_hist[n] += 1
        total_attempts += n
        if n > 0:
            funcs_with_attempts += 1
        if (func_dir / "repair_success.txt").is_file():
            repair_success += 1
        if (func_dir / "repair_failed.txt").is_file():
            repair_failed += 1

    avg_per_repair_func = (total_attempts / funcs_in_repair_phase) if funcs_in_repair_phase else 0.0
    avg_per_attempted_func = (total_attempts / funcs_with_attempts) if funcs_with_attempts else 0.0

    return {
        "enabled": True,
        "repair_root": str(repair_root),
        "functions_in_repair_phase": funcs_in_repair_phase,
        "functions_with_attempts": funcs_with_attempts,
        "repair_success": repair_success,
        "repair_failed": repair_failed,
        "repair_attempts_total": total_attempts,
        "avg_rounds_per_repair_phase_function": round(avg_per_repair_func, 6),
        "avg_rounds_per_attempted_function": round(avg_per_attempted_func, 6),
        "attempts_histogram": {str(k): int(v) for k, v in sorted(attempts_hist.items(), key=lambda x: x[0])},
    }


def _scan_timeouts(run_intermediate_dir: Path) -> Dict[str, Any]:
    """
    Best-effort scan for timeout / stall signals from staged logs.

    This is intentionally heuristic: it helps answer "卡在哪里/超时多吗" without
    requiring changes to the runner. Patterns can be extended over time.
    """
    logs_dir = run_intermediate_dir / "logs"
    if not logs_dir.is_dir():
        return {"enabled": False, "reason": "logs_dir_missing", "logs_dir": str(logs_dir)}

    # We mainly care about step7 (LLM translate/repair) and step5 (Jina GPU rerank),
    # but scanning all step*.log is cheap enough and keeps it robust.
    step_logs = sorted(logs_dir.glob("batch_staged_*/step*.log"))
    if not step_logs:
        return {"enabled": False, "reason": "no_step_logs", "logs_dir": str(logs_dir)}

    vllm_pat = re.compile(r"Request timed out", re.IGNORECASE)
    vllm_cn_pat = re.compile(r"vLLM\s*调用超时", re.IGNORECASE)
    gpu_wait_timeout_pat = re.compile(r"等待超时", re.IGNORECASE)

    vllm_by_project: Counter[str] = Counter()
    gpu_wait_by_project: Counter[str] = Counter()

    total_vllm_hits = 0
    total_gpu_wait_hits = 0

    for path in step_logs:
        # logs_dir/batch_staged_<proj>/stepX.log
        try:
            project_name = path.parent.name.replace("batch_staged_", "", 1)
        except Exception:
            project_name = path.parent.name

        try:
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if vllm_pat.search(line) or vllm_cn_pat.search(line):
                        total_vllm_hits += 1
                        vllm_by_project[project_name] += 1
                    if gpu_wait_timeout_pat.search(line):
                        total_gpu_wait_hits += 1
                        gpu_wait_by_project[project_name] += 1
        except Exception:
            continue

    return {
        "enabled": True,
        "logs_dir": str(logs_dir),
        "files_scanned": len(step_logs),
        "vllm": {
            "pattern": "Request timed out / vLLM 调用超时",
            "total_hits": total_vllm_hits,
            "projects": dict(vllm_by_project.most_common()),
        },
        "gpu_wait": {
            "pattern": "等待超时",
            "total_hits": total_gpu_wait_hits,
            "projects": dict(gpu_wait_by_project.most_common()),
        },
    }


def _scan_missing_symbol_issues(project_dir: Path, *, project_name: str, llm_name: Optional[str]) -> Dict[str, Any]:
    """
    Best-effort scan: find "missing symbol/macro declaration" signals from rustc errors.

    We mainly look for:
    - E0425 missing function / tuple-ctor (often caused by external C APIs/macros not being declared in Rust)
    - missing macro in scope
    """
    ws = project_dir / "workspace"
    if not ws.is_dir():
        return {"enabled": False, "reason": "workspace_missing", "workspace": str(ws)}

    repair_root = ws / "repair_history" / project_name
    if not repair_root.is_dir():
        return {"enabled": False, "reason": "repair_history_missing", "repair_root": str(repair_root)}

    candidates: List[Path] = []
    if llm_name:
        p = repair_root / f"translate_by_{llm_name}"
        if p.is_dir():
            candidates.append(p)
    if not candidates:
        candidates = sorted(repair_root.glob("translate_by_*"))
    if not candidates:
        return {"enabled": False, "reason": "no_translate_by_dir", "repair_root": str(repair_root)}

    scan_root = candidates[0]
    error_files = sorted(scan_root.glob("**/attempt_1_error.txt"))
    if not error_files:
        error_files = sorted(scan_root.glob("**/attempt_*_error.txt"))
    if not error_files:
        return {"enabled": False, "reason": "no_attempt_error_files", "scan_root": str(scan_root)}

    missing_items: Counter[str] = Counter()
    missing_macros: Counter[str] = Counter()
    for p in error_files:
        try:
            text = _read_text(p)
        except Exception:
            continue
        for m in _MISSING_ITEM_E0425_RE.finditer(text):
            sym = (m.group(1) or "").strip()
            if sym:
                missing_items[sym] += 1
        for m in _MISSING_MACRO_RE.finditer(text):
            sym = (m.group(1) or "").strip()
            if sym:
                missing_macros[sym] += 1

    return {
        "enabled": True,
        "scan_root": str(scan_root),
        "files_scanned": len(error_files),
        "unique_missing_item_symbols": len(missing_items),
        "total_missing_item_occurrences": int(sum(missing_items.values())),
        "missing_item_symbols_top": [
            {"name": name, "count": int(cnt)} for name, cnt in missing_items.most_common(20)
        ],
        "unique_missing_macro_symbols": len(missing_macros),
        "total_missing_macro_occurrences": int(sum(missing_macros.values())),
        "missing_macro_symbols_top": [
            {"name": name, "count": int(cnt)} for name, cnt in missing_macros.most_common(20)
        ],
    }


@dataclass(frozen=True)
class UnsafeMetrics:
    rs_files: int
    total_lines: int
    unsafe_keyword_lines: int
    unsafe_keyword_count: int
    unsafe_block_count: int
    unsafe_fn_count: int

    def to_dict(self) -> Dict[str, Any]:
        ratio_lines = (self.unsafe_keyword_lines / self.total_lines) if self.total_lines else 0.0
        return {
            "rs_files": self.rs_files,
            "total_lines": self.total_lines,
            "unsafe_keyword_lines": self.unsafe_keyword_lines,
            "unsafe_keyword_count": self.unsafe_keyword_count,
            "unsafe_block_count": self.unsafe_block_count,
            "unsafe_fn_count": self.unsafe_fn_count,
            "unsafe_keyword_line_ratio": round(ratio_lines, 6),
        }


def _compute_unsafe_metrics(rust_root: Path) -> UnsafeMetrics:
    src_dir = rust_root / "src"
    rs_files = sorted([p for p in src_dir.rglob("*.rs") if p.is_file()])
    total_lines = 0
    unsafe_keyword_lines = 0
    unsafe_keyword_count = 0
    unsafe_block_count = 0
    unsafe_fn_count = 0

    for f in rs_files:
        text = _read_text(f)
        lines = text.splitlines()
        total_lines += len(lines)
        unsafe_keyword_lines += sum(1 for ln in lines if re.search(r"\bunsafe\b", ln))
        unsafe_keyword_count += len(re.findall(r"\bunsafe\b", text))
        unsafe_block_count += len(re.findall(r"\bunsafe\s*\{", text))
        unsafe_fn_count += len(re.findall(r"\bunsafe\s+fn\b", text))

    return UnsafeMetrics(
        rs_files=len(rs_files),
        total_lines=total_lines,
        unsafe_keyword_lines=unsafe_keyword_lines,
        unsafe_keyword_count=unsafe_keyword_count,
        unsafe_block_count=unsafe_block_count,
        unsafe_fn_count=unsafe_fn_count,
    )


def _rust_tokenize(code: str) -> List[str]:
    # Identifiers, numbers, string/char literals (rough), operators/punct.
    # This is not a full lexer; it is only used for lightweight similarity metrics.
    token_re = re.compile(
        r"""
        (?:0x[0-9A-Fa-f_]+|\d[\d_]*\.\d[\d_]*|\d[\d_]*)
        |(?:'(?:[^'\\]|\\.)*')
        |(?:\"(?:[^\"\\]|\\.)*\")
        |(?:[A-Za-z_][A-Za-z0-9_]*)
        |(?:==|!=|<=|>=|->|=>|::|\.\.|&&|\|\||<<|>>|[{}()[\];,.:<>+\-*/%&|^!~=])
        """,
        re.VERBOSE,
    )
    return token_re.findall(code)


def _rust_tokenize_flat(code: str) -> List[str]:
    # Kept for backward compatibility with earlier drafts.
    return _rust_tokenize(code)


def _ngrams(tokens: Sequence[str], n: int) -> List[Tuple[str, ...]]:
    if n <= 0:
        return []
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(0, len(tokens) - n + 1)]


def _bleu_score(reference: Sequence[str], hypothesis: Sequence[str], max_n: int = 4) -> float:
    if not reference or not hypothesis:
        return 0.0

    precisions: List[float] = []
    for n in range(1, max_n + 1):
        ref_counts = Counter(_ngrams(reference, n))
        hyp_counts = Counter(_ngrams(hypothesis, n))
        if not hyp_counts:
            precisions.append(0.0)
            continue
        overlap = sum(min(count, ref_counts.get(ng, 0)) for ng, count in hyp_counts.items())
        total = sum(hyp_counts.values())
        # smoothing (add-1)
        precisions.append((overlap + 1.0) / (total + 1.0))

    # geometric mean
    log_p = sum(math.log(p) for p in precisions) / max_n
    bp = 1.0
    ref_len = len(reference)
    hyp_len = len(hypothesis)
    if hyp_len == 0:
        return 0.0
    if hyp_len < ref_len:
        bp = math.exp(1.0 - (ref_len / hyp_len))
    return float(bp * math.exp(log_p))


def _weighted_bleu(reference: Sequence[str], hypothesis: Sequence[str], max_n: int = 4) -> float:
    if not reference or not hypothesis:
        return 0.0

    precisions: List[float] = []
    ref_kw = set(t for t in reference if t in RUST_KEYWORDS)
    hyp_kw = set(t for t in hypothesis if t in RUST_KEYWORDS)
    active_kw = ref_kw | hyp_kw

    for n in range(1, max_n + 1):
        ref_counts = Counter(_ngrams(reference, n))
        hyp_counts = Counter(_ngrams(hypothesis, n))
        if not hyp_counts:
            precisions.append(0.0)
            continue

        def weight(ng: Tuple[str, ...]) -> float:
            return 1.0 + sum(1 for t in ng if t in active_kw)

        overlap_w = 0.0
        total_w = 0.0
        for ng, hyp_c in hyp_counts.items():
            w = weight(ng)
            total_w += hyp_c * w
            overlap_w += min(hyp_c, ref_counts.get(ng, 0)) * w
        # smoothing
        precisions.append((overlap_w + 1.0) / (total_w + 1.0))

    log_p = sum(math.log(p) for p in precisions) / max_n
    bp = 1.0
    ref_len = len(reference)
    hyp_len = len(hypothesis)
    if hyp_len == 0:
        return 0.0
    if hyp_len < ref_len:
        bp = math.exp(1.0 - (ref_len / hyp_len))
    return float(bp * math.exp(log_p))


def _token_type(tok: str) -> str:
    if tok in RUST_KEYWORDS:
        return "kw"
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", tok):
        return "id"
    if re.fullmatch(r"0x[0-9A-Fa-f_]+|\d[\d_]*\.\d[\d_]*|\d[\d_]*", tok):
        return "num"
    if tok.startswith('"') or tok.startswith("'"):
        return "lit"
    return "sym"


def _syntax_similarity(reference: Sequence[str], hypothesis: Sequence[str]) -> float:
    if not reference or not hypothesis:
        return 0.0
    ref_types = [_token_type(t) for t in reference]
    hyp_types = [_token_type(t) for t in hypothesis]
    ref_bi = set(_ngrams(ref_types, 2)) or set(ref_types)
    hyp_bi = set(_ngrams(hyp_types, 2)) or set(hyp_types)
    inter = len(ref_bi & hyp_bi)
    union = len(ref_bi | hyp_bi) or 1
    return inter / union


def _dataflow_similarity(reference: Sequence[str], hypothesis: Sequence[str]) -> float:
    ref_ids = {t for t in reference if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", t) and t not in RUST_KEYWORDS}
    hyp_ids = {t for t in hypothesis if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", t) and t not in RUST_KEYWORDS}
    if not ref_ids and not hyp_ids:
        return 1.0
    if not ref_ids or not hyp_ids:
        return 0.0
    inter = len(ref_ids & hyp_ids)
    precision = inter / len(hyp_ids)
    recall = inter / len(ref_ids)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def codebleu_proxy(reference_code: str, hypothesis_code: str) -> Dict[str, float]:
    ref_toks = _rust_tokenize_flat(reference_code)
    hyp_toks = _rust_tokenize_flat(hypothesis_code)
    ngram = _bleu_score(ref_toks, hyp_toks)
    wngram = _weighted_bleu(ref_toks, hyp_toks)
    syn = _syntax_similarity(ref_toks, hyp_toks)
    df = _dataflow_similarity(ref_toks, hyp_toks)
    score = (ngram + wngram + syn + df) / 4.0
    return {
        "ngram_bleu": float(ngram),
        "weighted_ngram_bleu": float(wngram),
        "syntax_similarity": float(syn),
        "dataflow_similarity": float(df),
        "codebleu_proxy": float(score),
    }


def _extract_c_function_name(c_code: str) -> Optional[str]:
    for ln in (c_code or "").splitlines():
        s = ln.strip()
        if not s:
            continue
        # Avoid matching function calls inside comments.
        if s.startswith("//"):
            continue
        m = re.search(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", s)
        if m:
            return m.group(1)
        # If first line doesn't contain it, don't over-scan to avoid picking up callees.
        break
    return None


def _extract_extern_c_fns_from_rs(text: str) -> Dict[str, str]:
    """
    Extract extern "C" function items (best-effort) from a Rust source file.
    Returns: name -> full function block text.
    """
    out: Dict[str, str] = {}
    pat = re.compile(r'(?m)^(?:pub\s+)?(?:unsafe\s+)?extern\s+"C"\s+fn\s+([A-Za-z_][A-Za-z0-9_]*)\b')
    for m in pat.finditer(text):
        name = m.group(1)
        start_line = text.rfind("\n", 0, m.start())
        start = 0 if start_line < 0 else start_line + 1
        brace = text.find("{", m.end())
        if brace < 0:
            continue
        depth = 0
        end = None
        for i in range(brace, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end is None:
            continue
        out[name] = text[start:end].strip()
    return out


def _index_project_extern_c_functions(rust_root: Path) -> Dict[str, str]:
    src = rust_root / "src"
    if not src.is_dir():
        return {}
    index: Dict[str, str] = {}
    for rs in sorted(src.rglob("*.rs")):
        try:
            text = _read_text(rs)
        except Exception:
            continue
        for name, block in _extract_extern_c_fns_from_rs(text).items():
            # If duplicates exist, keep the longer one.
            if name not in index or len(block) > len(index[name]):
                index[name] = block
    return index


def _is_stub_rust_function(fn_block: str) -> bool:
    if not fn_block:
        return True
    # Skeleton stubs.
    if re.search(r"\bunimplemented!\s*\(", fn_block):
        return True
    if re.search(r"\btodo!\s*\(", fn_block):
        return True
    return False


def _normalize_text_for_bm25(text: str) -> List[str]:
    """
    Must be compatible with build_knowledge_base.py / elastic_search.py tokenization.
    Falls back gracefully if nltk data is unavailable.
    """
    words = re.findall(r"[a-zA-Z0-9]+|[^\s\w]+", text)
    try:
        from nltk.corpus import stopwords
        from nltk.stem import PorterStemmer

        stop_words = set(stopwords.words("english"))
        words = [w for w in words if w not in stop_words]
        pattern = r"[A-Z][a-z]+|[a-z]+"
        new_words: List[str] = []
        for word in words:
            tmp = re.findall(pattern, word)
            tmp = [w.lower() for w in tmp]
            new_words.extend(tmp)
        if new_words:
            words = new_words
        stemmer = PorterStemmer()
        words = [stemmer.stem(w) for w in words]
        symbol_pattern = re.compile(r"^[^\w\s]+$")
        words = [w for w in words if not symbol_pattern.match(w)]
        return words
    except Exception:
        # No nltk data; use a simplified normalizer.
        return [w.lower() for w in words if re.search(r"[A-Za-z0-9]", w)]


def _load_bm25(kb_json: Path, bm25_pkl: Path) -> Tuple[List[Dict[str, Any]], Any]:
    kb = json.loads(_read_text(kb_json))
    with bm25_pkl.open("rb") as f:
        bm25 = pickle.load(f)
    return kb, bm25


def _bm25_top1_index(bm25: Any, tokenized_query: List[str]) -> Optional[int]:
    try:
        import numpy as np

        scores = bm25.get_scores(tokenized_query)
        if scores is None:
            return None
        return int(np.argmax(scores))
    except Exception:
        try:
            scores = bm25.get_scores(tokenized_query)
            if not scores:
                return None
            best_i = max(range(len(scores)), key=lambda i: scores[i])
            return int(best_i)
        except Exception:
            return None


def _summarize_mean(values: Sequence[float]) -> Dict[str, Any]:
    vals = [v for v in values if isinstance(v, (int, float)) and not math.isnan(v)]
    if not vals:
        return {"count": 0, "mean": None, "median": None, "p10": None, "p90": None}
    vals_sorted = sorted(vals)
    n = len(vals_sorted)

    def pct(p: float) -> float:
        if n == 1:
            return float(vals_sorted[0])
        idx = int(round((n - 1) * p))
        idx = max(0, min(n - 1, idx))
        return float(vals_sorted[idx])

    mean = sum(vals_sorted) / n
    median = pct(0.5)
    return {"count": n, "mean": mean, "median": median, "p10": pct(0.1), "p90": pct(0.9)}


def main() -> int:
    ap = argparse.ArgumentParser(description="Post-run analyzer (unsafe ratio / CodeBLEU proxy / failure breakdown)")
    ap.add_argument("--run-intermediate-dir", required=True, type=Path)
    ap.add_argument("--run-results-dir", required=True, type=Path)
    ap.add_argument("--llm-name", default=os.environ.get("LLM_NAME") or os.environ.get("VLLM_MODEL_NAME") or "")
    ap.add_argument("--output-json", default="", help="Override output json path (default: <run-results-dir>/post_run_analysis.json)")
    ap.add_argument("--codebleu-sample-per-project", type=int, default=20)
    ap.add_argument("--codebleu-max-projects", type=int, default=0, help="For quick runs: limit number of projects (0=all)")
    ap.add_argument("--threads", type=int, default=max(4, (os.cpu_count() or 8) // 2))
    ap.add_argument("--kb-json", type=Path, default=Path("workspace/rag/knowledge_base.json"))
    ap.add_argument("--bm25-pkl", type=Path, default=Path("workspace/rag/bm25_index.pkl"))
    # CodeBLEU（严格版本）需要“被认可的参考Rust”作为对照。当前实现只是一个 proxy（用RAG top1当参考），
    # 容易引入偏差，所以默认关闭；后续你提供权威参考集后，可以再启用或替换为真正的 CodeBLEU。
    ap.add_argument("--enable-codebleu-proxy", action="store_true", help="Enable CodeBLEU-like proxy scoring (not ground truth)")
    ap.add_argument("--codebleu-disable", action="store_true", help="(deprecated) Disable CodeBLEU proxy")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run_intermediate_dir: Path = args.run_intermediate_dir.resolve()
    run_results_dir: Path = args.run_results_dir.resolve()
    run_results_dir.mkdir(parents=True, exist_ok=True)

    llm_name = (args.llm_name or "").strip() or None

    project_dirs = _iter_project_dirs(run_intermediate_dir)
    if args.codebleu_max_projects and args.codebleu_max_projects > 0:
        project_dirs = project_dirs[: args.codebleu_max_projects]

    unsafe_by_project: Dict[str, Dict[str, Any]] = {}
    unsafe_overall_acc = Counter()

    failure_by_project: Dict[str, Dict[str, Any]] = {}
    failure_overall = Counter()

    repair_by_project: Dict[str, Dict[str, Any]] = {}
    repair_overall_acc: Counter[str] = Counter()
    repair_overall_hist: Counter[int] = Counter()

    # CodeBLEU proxy per project (default OFF)
    codebleu_enabled = (
        bool(args.enable_codebleu_proxy)
        and (not args.codebleu_disable)
        and args.kb_json.exists()
        and args.bm25_pkl.exists()
    )
    kb: List[Dict[str, Any]] = []
    bm25 = None
    if codebleu_enabled:
        try:
            kb, bm25 = _load_bm25(args.kb_json, args.bm25_pkl)
        except Exception as e:
            codebleu_enabled = False
            kb, bm25 = [], None
            print(f"[WARN] CodeBLEU proxy disabled: failed to load KB/BM25: {e}")

    codebleu_by_project: Dict[str, Dict[str, Any]] = {}
    overall_codebleu_vals: List[float] = []
    overall_codebleu_sampled = 0

    rng = random.Random(args.seed)

    def analyze_project(project_dir: Path) -> Dict[str, Any]:
        project_name = project_dir.name
        stats_file = _find_translation_stats(project_dir)
        rust_root = _detect_rust_root(project_dir, project_name, llm_name)

        project_result: Dict[str, Any] = {
            "project": project_name,
            "stats_file": str(stats_file) if stats_file else None,
            "rust_root": str(rust_root) if rust_root else None,
        }

        # Failure stats (function-level counters)
        if stats_file and stats_file.exists():
            try:
                stats = json.loads(_read_text(stats_file))
                total = int(stats.get("total", 0) or 0)
                translated = int(stats.get("translated", 0) or 0)
                compiled = int(stats.get("compiled", 0) or 0)
                repaired = int(stats.get("repaired", 0) or 0)
                failed = int(stats.get("failed", 0) or 0)
                skipped = int(stats.get("skipped", 0) or 0)
                inj = int(stats.get("injection_failed", 0) or 0)
                still_unimpl = int(stats.get("still_unimplemented", 0) or 0)
                direct = max(0, compiled - repaired)
                project_result["failure"] = {
                    "total": total,
                    "translated": translated,
                    "compiled_total": compiled,
                    "compiled_direct": direct,
                    "compiled_after_repair": repaired,
                    "failed_reverted": failed,
                    "skipped": skipped,
                    "injection_failed": inj,
                    "still_unimplemented": still_unimpl,
                }
            except Exception as e:
                project_result["failure"] = {"error": str(e)}

        # Avg repair rounds (compile-guided iterations)
        try:
            rr = _scan_repair_rounds(project_dir, project_name=project_name, llm_name=llm_name)
            # Attach per-denominator averages when we have translation stats.
            fm = project_result.get("failure") or {}
            if isinstance(rr, dict) and rr.get("enabled") and isinstance(fm, dict) and "error" not in fm:
                attempts = int(rr.get("repair_attempts_total", 0) or 0)
                total = int(fm.get("total", 0) or 0)
                translated = int(fm.get("translated", 0) or 0)
                rr["avg_rounds_per_total_function"] = round((attempts / total) if total else 0.0, 6)
                rr["avg_rounds_per_translated_function"] = round((attempts / translated) if translated else 0.0, 6)
            project_result["repair_rounds"] = rr
        except Exception as e:
            project_result["repair_rounds"] = {"enabled": False, "error": str(e)}

        # Missing symbol/macro declaration signals (E0425 / cannot find macro)
        try:
            project_result["missing_symbol_issues"] = _scan_missing_symbol_issues(
                project_dir,
                project_name=project_name,
                llm_name=llm_name,
            )
        except Exception as e:
            project_result["missing_symbol_issues"] = {"enabled": False, "error": str(e)}

        # TU context closure (compile_commands/preprocess `.i` availability)
        try:
            project_result["tu_context"] = _scan_tu_context_closure(project_dir)
        except Exception as e:
            project_result["tu_context"] = {"enabled": False, "error": str(e)}

        # Unsafe metrics (Rust text scan)
        if rust_root and (rust_root / "src").is_dir():
            try:
                m = _compute_unsafe_metrics(rust_root)
                project_result["unsafe"] = m.to_dict()
            except Exception as e:
                project_result["unsafe"] = {"error": str(e)}

        # CodeBLEU proxy (sample)
        if codebleu_enabled and bm25 is not None and kb and rust_root:
            try:
                # Build extern "C" fn index once.
                fn_index = _index_project_extern_c_functions(rust_root)
                functions_dir = project_dir / "workspace" / "extracted" / project_name / "functions"
                if not functions_dir.is_dir():
                    project_result["codebleu_proxy"] = {"enabled": False, "reason": "functions_dir_missing"}
                else:
                    candidates: List[Tuple[str, str]] = []  # (c_code, rust_fn)
                    for f in sorted(functions_dir.glob("*.txt")):
                        c_code = _read_text(f)
                        fn_name = _extract_c_function_name(c_code)
                        if not fn_name:
                            continue
                        rust_fn = fn_index.get(fn_name)
                        if not rust_fn or _is_stub_rust_function(rust_fn):
                            continue
                        candidates.append((c_code, rust_fn))

                    if not candidates:
                        project_result["codebleu_proxy"] = {"enabled": False, "reason": "no_non_stub_functions"}
                    else:
                        sample_n = min(max(1, args.codebleu_sample_per_project), len(candidates))
                        sampled = rng.sample(candidates, sample_n) if len(candidates) > sample_n else candidates
                        vals: List[float] = []
                        sub: List[Dict[str, float]] = []
                        for c_code, rust_fn in sampled:
                            q = _normalize_text_for_bm25(c_code)
                            top1 = _bm25_top1_index(bm25, q)
                            if top1 is None:
                                continue
                            ref_rust = kb[top1].get("rust_code", "")
                            if not isinstance(ref_rust, str) or not ref_rust.strip():
                                continue
                            s = codebleu_proxy(ref_rust, rust_fn)
                            vals.append(s["codebleu_proxy"])
                            sub.append(s)
                        project_result["codebleu_proxy"] = {
                            "enabled": True,
                            "sampled_functions": len(vals),
                            "sample_target": sample_n,
                            "summary": _summarize_mean(vals),
                            "components_mean": {
                                k: (sum(x[k] for x in sub) / len(sub)) if sub else None
                                for k in ("ngram_bleu", "weighted_ngram_bleu", "syntax_similarity", "dataflow_similarity")
                            },
                        }
            except Exception as e:
                project_result["codebleu_proxy"] = {"enabled": False, "error": str(e)}
        return project_result

    # Parallel per-project analysis (mostly IO-bound).
    results: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.threads))) as ex:
        futs = {ex.submit(analyze_project, p): p for p in project_dirs}
        for fut in as_completed(futs):
            results.append(fut.result())

    # Aggregate
    for r in results:
        proj = r["project"]

        um = r.get("unsafe")
        if isinstance(um, dict) and "error" not in um and um.get("rs_files"):
            unsafe_by_project[proj] = um
            unsafe_overall_acc.update(
                {
                    "rs_files": int(um.get("rs_files", 0) or 0),
                    "total_lines": int(um.get("total_lines", 0) or 0),
                    "unsafe_keyword_lines": int(um.get("unsafe_keyword_lines", 0) or 0),
                    "unsafe_keyword_count": int(um.get("unsafe_keyword_count", 0) or 0),
                    "unsafe_block_count": int(um.get("unsafe_block_count", 0) or 0),
                    "unsafe_fn_count": int(um.get("unsafe_fn_count", 0) or 0),
                }
            )

        fm = r.get("failure")
        if isinstance(fm, dict) and "error" not in fm and fm.get("total") is not None:
            failure_by_project[proj] = fm
            failure_overall.update({k: int(fm.get(k, 0) or 0) for k in fm.keys() if isinstance(fm.get(k), int)})

        cm = r.get("codebleu_proxy")
        if isinstance(cm, dict) and cm.get("enabled") is True:
            codebleu_by_project[proj] = cm
            summary = cm.get("summary") or {}
            if isinstance(summary, dict) and summary.get("mean") is not None:
                overall_codebleu_vals.append(float(summary["mean"]))
            overall_codebleu_sampled += int(cm.get("sampled_functions", 0) or 0)

        rm = r.get("repair_rounds")
        if isinstance(rm, dict) and rm.get("enabled") is True:
            repair_by_project[proj] = rm
            repair_overall_acc.update(
                {
                    "repair_attempts_total": int(rm.get("repair_attempts_total", 0) or 0),
                    "functions_in_repair_phase": int(rm.get("functions_in_repair_phase", 0) or 0),
                    "functions_with_attempts": int(rm.get("functions_with_attempts", 0) or 0),
                    "repair_success": int(rm.get("repair_success", 0) or 0),
                    "repair_failed": int(rm.get("repair_failed", 0) or 0),
                }
            )
            h = rm.get("attempts_histogram") or {}
            if isinstance(h, dict):
                for k, v in h.items():
                    try:
                        kk = int(k)
                        vv = int(v or 0)
                    except Exception:
                        continue
                    repair_overall_hist[kk] += vv

    unsafe_overall = {
        "projects_with_rust": len(unsafe_by_project),
        "rs_files": unsafe_overall_acc.get("rs_files", 0),
        "total_lines": unsafe_overall_acc.get("total_lines", 0),
        "unsafe_keyword_lines": unsafe_overall_acc.get("unsafe_keyword_lines", 0),
        "unsafe_keyword_count": unsafe_overall_acc.get("unsafe_keyword_count", 0),
        "unsafe_block_count": unsafe_overall_acc.get("unsafe_block_count", 0),
        "unsafe_fn_count": unsafe_overall_acc.get("unsafe_fn_count", 0),
    }
    unsafe_overall["unsafe_keyword_line_ratio"] = (
        (unsafe_overall["unsafe_keyword_lines"] / unsafe_overall["total_lines"]) if unsafe_overall["total_lines"] else 0.0
    )

    failure_overall_summary = dict(failure_overall)
    if failure_overall_summary:
        total = int(failure_overall_summary.get("total", 0) or 0)
        if total > 0:
            for k in (
                "compiled_total",
                "compiled_direct",
                "compiled_after_repair",
                "failed_reverted",
                "skipped",
                "injection_failed",
                "still_unimplemented",
            ):
                v = int(failure_overall_summary.get(k, 0) or 0)
                failure_overall_summary[f"{k}_rate"] = v / total

    # Repair rounds overall summary (AvgRepair)
    repair_overall_summary: Dict[str, Any] = {
        "projects_with_repair_history": len(repair_by_project),
        "repair_attempts_total": int(repair_overall_acc.get("repair_attempts_total", 0) or 0),
        "functions_in_repair_phase": int(repair_overall_acc.get("functions_in_repair_phase", 0) or 0),
        "functions_with_attempts": int(repair_overall_acc.get("functions_with_attempts", 0) or 0),
        "repair_success": int(repair_overall_acc.get("repair_success", 0) or 0),
        "repair_failed": int(repair_overall_acc.get("repair_failed", 0) or 0),
        "attempts_histogram": {str(k): int(v) for k, v in sorted(repair_overall_hist.items(), key=lambda x: x[0])},
    }
    total_attempts_all = int(repair_overall_summary.get("repair_attempts_total", 0) or 0)
    funcs_in_repair_phase_all = int(repair_overall_summary.get("functions_in_repair_phase", 0) or 0)
    funcs_with_attempts_all = int(repair_overall_summary.get("functions_with_attempts", 0) or 0)
    total_funcs_all = int(failure_overall_summary.get("total", 0) or 0) if failure_overall_summary else 0
    translated_all = int(failure_overall.get("translated", 0) or 0)
    repair_overall_summary["avg_rounds_per_repair_phase_function"] = round(
        (total_attempts_all / funcs_in_repair_phase_all) if funcs_in_repair_phase_all else 0.0, 6
    )
    repair_overall_summary["avg_rounds_per_attempted_function"] = round(
        (total_attempts_all / funcs_with_attempts_all) if funcs_with_attempts_all else 0.0, 6
    )
    repair_overall_summary["avg_rounds_per_total_function"] = round(
        (total_attempts_all / total_funcs_all) if total_funcs_all else 0.0, 6
    )
    repair_overall_summary["avg_rounds_per_translated_function"] = round(
        (total_attempts_all / translated_all) if translated_all else 0.0, 6
    )

    report = {
        "meta": {
            "generated_at": datetime.now().isoformat(),
            "llm_name": llm_name,
            "run_intermediate_dir": str(run_intermediate_dir),
            "run_results_dir": str(run_results_dir),
            "projects_total": len(project_dirs),
        },
        "timeouts": _scan_timeouts(run_intermediate_dir),
        "unsafe": {"overall": unsafe_overall, "by_project": unsafe_by_project},
        "failure_breakdown": {
            "projects_with_stats": len(failure_by_project),
            "overall": failure_overall_summary,
            "by_project": failure_by_project,
        },
        "repair_rounds": {
            "overall": repair_overall_summary,
            "by_project": repair_by_project,
        },
        "codebleu_proxy": {
            "enabled": codebleu_enabled,
            "reference": "BM25 top-1 Rust snippet from existing RAG knowledge_base.json (proxy; not ground truth)",
            "projects_with_score": len(codebleu_by_project),
            "overall": _summarize_mean(overall_codebleu_vals),
            "overall_sampled_functions": overall_codebleu_sampled,
            "by_project": codebleu_by_project,
        },
        "projects": results,
    }

    out_path = Path(args.output_json).expanduser() if args.output_json else (run_results_dir / "post_run_analysis.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] Post-run analysis written: {out_path}")
    print(
        f"  - unsafe_keyword_line_ratio={unsafe_overall['unsafe_keyword_line_ratio']:.6f} "
        f"({unsafe_overall['unsafe_keyword_lines']}/{unsafe_overall['total_lines']} lines)"
    )
    if failure_overall_summary and failure_overall_summary.get("total"):
        print(
            "  - function pass rate: "
            f"{failure_overall_summary.get('compiled_total_rate', 0.0)*100:.2f}%"
        )
    if translated_all:
        print(
            "  - avg repair rounds (per translated fn): "
            f"{repair_overall_summary.get('avg_rounds_per_translated_function', 0.0):.6f} "
            f"(attempts={total_attempts_all}/{translated_all})"
        )
    # Print projects that likely suffered from "external symbol/macro not declared in Rust"
    missing_projects: List[Tuple[str, Dict[str, Any]]] = []
    for r in results:
        ms = r.get("missing_symbol_issues") or {}
        if not isinstance(ms, dict) or not ms.get("enabled"):
            continue
        if int(ms.get("unique_missing_item_symbols", 0) or 0) > 0 or int(ms.get("unique_missing_macro_symbols", 0) or 0) > 0:
            missing_projects.append((r.get("project") or "unknown", ms))
    if missing_projects:
        print("[WARN] Missing symbol/macro declarations detected (may cause E0425 and force fallbacks):")
        for proj, ms in sorted(missing_projects, key=lambda x: x[0]):
            items = ms.get("missing_item_symbols_top") or []
            macros = ms.get("missing_macro_symbols_top") or []
            top_items = ", ".join([x.get("name") for x in items[:5] if isinstance(x, dict) and x.get("name")]) or "-"
            top_macros = ", ".join([x.get("name") for x in macros[:5] if isinstance(x, dict) and x.get("name")]) or "-"
            print(
                f"  - {proj}: missing_items={int(ms.get('unique_missing_item_symbols', 0) or 0)} (top: {top_items}); "
                f"missing_macros={int(ms.get('unique_missing_macro_symbols', 0) or 0)} (top: {top_macros})"
            )

    # Print projects with TU/input-closure gaps (compile_commands entry missing / preprocess failed / .i missing)
    tu_gap_projects: List[Tuple[str, Dict[str, Any]]] = []
    for r in results:
        tc = r.get("tu_context") or {}
        if not isinstance(tc, dict) or not tc.get("enabled"):
            continue
        if bool(tc.get("has_closure_gaps")):
            tu_gap_projects.append((r.get("project") or "unknown", tc))
    if tu_gap_projects:
        print("[WARN] TU/input-closure gaps detected (NOT translation bugs; build context incomplete):")
        for proj, tc in sorted(tu_gap_projects, key=lambda x: x[0]):
            counts = tc.get("counts") or {}
            print(
                f"  - {proj}: entry_missing={int(counts.get('compile_commands_entry_missing', 0) or 0)}, "
                f"pre_failed={int(counts.get('preprocessed_failed', 0) or 0)}, "
                f"pre_file_missing={int(counts.get('preprocessed_file_missing', 0) or 0)}; "
                f"map={tc.get('path')}"
            )
            samples = tc.get("failure_samples") or []
            if isinstance(samples, list) and samples:
                s0 = samples[0] if isinstance(samples[0], dict) else {}
                err0 = (s0.get("error") or "").strip()
                src0 = s0.get("source_file_rel") or s0.get("file_group") or "-"
                if err0:
                    print(f"    sample: {src0}: {err0[:160]}")
    if codebleu_enabled:
        overall = report["codebleu_proxy"]["overall"]
        print(f"  - codebleu_proxy(mean over projects)={overall.get('mean')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
