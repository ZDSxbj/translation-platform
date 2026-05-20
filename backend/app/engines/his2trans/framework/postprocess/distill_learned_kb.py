#!/usr/bin/env python3
"""
Distill a *new, independent* knowledge base from successful translation outputs.

This is intended for experiments:
- The learned KB is stored under a run-named folder (easy to delete).
- It does NOT modify the existing workspace/rag KB.
- Later runs can load both KBs (see elastic_search.py extra-KB support).

Outputs:
  <out-dir>/rag/knowledge_base.json
  <out-dir>/rag/bm25_index.pkl
  <out-dir>/distill_summary.json
  <out-dir>/symbols/types_generation_reports.jsonl   (optional, best-effort)
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from vllm_parallel_knowledge_extractor import (
    VllmSampling,
    convert_llm_record_to_extracted_knowledge,
    run_parallel_vllm_knowledge_extraction,
)


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


def _detect_rust_root(project_dir: Path, project_name: str, llm_name: Optional[str]) -> Optional[Path]:
    ws = project_dir / "workspace"
    if llm_name:
        p = ws / "final_projects" / project_name / f"translate_by_{llm_name}"
        if (p / "src").is_dir():
            return p
        p = ws / "incremental_work" / project_name / f"translate_by_{llm_name}"
        if (p / "src").is_dir():
            return p
    p = _find_first(ws, f"final_projects/{project_name}/translate_by_*/src")
    if p is not None:
        return p.parent
    p = _find_first(ws, f"incremental_work/{project_name}/translate_by_*/src")
    if p is not None:
        return p.parent
    return None


def _extract_c_function_name(c_code: str) -> Optional[str]:
    for ln in (c_code or "").splitlines():
        s = ln.strip()
        if not s:
            continue
        if s.startswith("//"):
            continue
        m = re.search(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", s)
        if m:
            return m.group(1)
        break
    return None


def _extract_extern_c_fns_from_rs(text: str) -> Dict[str, str]:
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
            if name not in index or len(block) > len(index[name]):
                index[name] = block
    return index


def _is_stub_rust_function(fn_block: str) -> bool:
    if not fn_block:
        return True
    if re.search(r"\bunimplemented!\s*\(", fn_block):
        return True
    if re.search(r"\btodo!\s*\(", fn_block):
        return True
    return False


def _normalize_text_for_bm25(text: str) -> List[str]:
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
        return [w.lower() for w in words if re.search(r"[A-Za-z0-9]", w)]


def _write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def main() -> int:
    # 获取 generation.py 的默认配置
    try:
        from generate.generation import (
            USE_VLLM, VLLM_BASE_URL, VLLM_API_KEY, VLLM_MODEL_NAME, VLLM_REQUEST_TIMEOUT,
            EXTERNAL_API_BASE_URL, EXTERNAL_API_KEY, EXTERNAL_API_MODEL, EXTERNAL_API_TIMEOUT
        )
        if USE_VLLM:
            default_url = VLLM_BASE_URL
            default_model = VLLM_MODEL_NAME
            default_api_key = VLLM_API_KEY
            default_timeout = VLLM_REQUEST_TIMEOUT
        else:
            default_url = EXTERNAL_API_BASE_URL
            default_model = EXTERNAL_API_MODEL
            default_api_key = EXTERNAL_API_KEY
            default_timeout = EXTERNAL_API_TIMEOUT
    except ImportError:
        default_url = os.environ.get("VLLM_BASE_URL") or os.environ.get("OPENAI_API_BASE") or "http://localhost:8000/v1"
        default_model = os.environ.get("VLLM_MODEL_NAME") or os.environ.get("LLM_NAME") or ""
        default_api_key = os.environ.get("VLLM_API_KEY") or "EMPTY"
        default_timeout = float(os.environ.get("VLLM_REQUEST_TIMEOUT", os.environ.get("VLLM_TIMEOUT", "180.0")))

    ap = argparse.ArgumentParser(description="Distill a learned (independent) KB from run outputs")
    ap.add_argument("--run-intermediate-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path, help="Output directory for the learned KB (e.g. translation_outputs/shared/learned_kb/Test)")
    ap.add_argument("--llm-name", default=os.environ.get("LLM_NAME") or os.environ.get("VLLM_MODEL_NAME") or default_model)
    ap.add_argument("--min-rust-lines", type=int, default=5, help="Reject very small function blocks as low-signal")
    ap.add_argument("--max-pairs", type=int, default=0, help="0=all; otherwise cap total pairs (sampled)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--extract-knowledge", action="store_true", help="Call vLLM to extract API_Mapping/Partial knowledge into extracted_knowledge")
    ap.add_argument("--vllm-base-url", default=os.environ.get("VLLM_BASE_URL") or os.environ.get("OPENAI_API_BASE") or default_url)
    ap.add_argument("--vllm-api-key", default=os.environ.get("VLLM_API_KEY") or default_api_key)
    ap.add_argument("--vllm-model", default=os.environ.get("VLLM_MODEL_NAME") or os.environ.get("LLM_NAME") or default_model)
    ap.add_argument("--vllm-num-workers", type=int, default=int(os.environ.get("VLLM_NUM_WORKERS", "48")))
    ap.add_argument("--vllm-timeout", type=float, default=float(os.environ.get("VLLM_REQUEST_TIMEOUT", os.environ.get("VLLM_TIMEOUT", str(default_timeout)))))
    ap.add_argument("--vllm-max-tokens", type=int, default=int(os.environ.get("C2R_KB_VLLM_MAX_TOKENS", "4096")))
    ap.add_argument("--vllm-resume", action="store_true", help="Resume from existing symbols/knowledge_extraction_vllm.jsonl")
    args = ap.parse_args()

    run_intermediate_dir = args.run_intermediate_dir.resolve()
    out_dir = args.out_dir.resolve()
    rag_dir = out_dir / "rag"
    symbols_dir = out_dir / "symbols"
    rag_dir.mkdir(parents=True, exist_ok=True)
    symbols_dir.mkdir(parents=True, exist_ok=True)

    llm_name = (args.llm_name or "").strip() or None
    rng = random.Random(args.seed)

    project_dirs = _iter_project_dirs(run_intermediate_dir)

    kb_entries: List[Dict[str, Any]] = []
    per_project_counts: Dict[str, Dict[str, int]] = {}
    skipped_reasons = Counter()

    # Optional: collect types_generation_report.json to a separate JSONL (useful for debugging/next-run hints).
    types_reports: List[Dict[str, Any]] = []
    copied_learned_data: List[str] = []

    for project_dir in project_dirs:
        project_name = project_dir.name
        rust_root = _detect_rust_root(project_dir, project_name, llm_name)
        functions_dir = project_dir / "workspace" / "extracted" / project_name / "functions"

        counts = {"pairs": 0, "c_functions": 0, "rust_extern_c_index": 0, "skipped": 0}

        # types_generation_report.json (best-effort)
        tgr = project_dir / "workspace" / "skeletons" / project_name / "types_generation_report.json"
        if tgr.exists():
            try:
                data = json.loads(_read_text(tgr))
                types_reports.append({"project": project_name, "path": str(tgr), "report": data})
            except Exception:
                pass

        if not functions_dir.is_dir():
            skipped_reasons["no_functions_dir"] += 1
            per_project_counts[project_name] = counts
            continue

        c_files = sorted(functions_dir.glob("*.txt"))
        counts["c_functions"] = len(c_files)

        if not rust_root or not (rust_root / "src").is_dir():
            skipped_reasons["no_rust_root"] += 1
            per_project_counts[project_name] = counts
            continue

        fn_index = _index_project_extern_c_functions(rust_root)
        counts["rust_extern_c_index"] = len(fn_index)

        for c_file in c_files:
            c_code = _read_text(c_file)
            fn_name = _extract_c_function_name(c_code)
            if not fn_name:
                counts["skipped"] += 1
                skipped_reasons["c_name_parse_failed"] += 1
                continue
            rust_fn = fn_index.get(fn_name)
            if not rust_fn:
                counts["skipped"] += 1
                skipped_reasons["rust_fn_missing"] += 1
                continue
            if _is_stub_rust_function(rust_fn):
                counts["skipped"] += 1
                skipped_reasons["rust_fn_stub"] += 1
                continue
            if len(rust_fn.splitlines()) < max(1, int(args.min_rust_lines)):
                counts["skipped"] += 1
                skipped_reasons["rust_fn_too_small"] += 1
                continue

            kb_entries.append(
                {
                    "c_code": c_code,
                    "rust_code": rust_fn,
                    "original_file": f"{project_name}::{c_file.stem}",
                    "c_file_count": 1,
                    # Keep schema compatible with the existing KB (V10): extracted_knowledge is a list of dicts.
                    # When --extract-knowledge is enabled, it will be populated with API_Mapping/Partial/Full entries.
                    "extracted_knowledge": [],
                    # Lightweight provenance (ignored by downstream RAG consumers; useful for auditing).
                    "learned_from": {
                        "project": project_name,
                        "function": fn_name,
                        "source_c_file": str(c_file),
                        "source_rust_root": str(rust_root),
                        "note": "Auto-distilled from successful non-stub translation outputs (proxy success).",
                    },
                }
            )
            counts["pairs"] += 1

        per_project_counts[project_name] = counts

    # Copy run-scoped learned_data cache (macros/types/constants learners may write here).
    try:
        run_cache_learned = run_intermediate_dir / "cache" / "learned_data"
        if run_cache_learned.is_dir():
            dest = symbols_dir / "learned_data"
            dest.mkdir(parents=True, exist_ok=True)
            for f in sorted(run_cache_learned.glob("*.json")):
                shutil.copy2(f, dest / f.name)
                copied_learned_data.append(f.name)
    except Exception:
        pass

    # Optional cap
    if args.max_pairs and args.max_pairs > 0 and len(kb_entries) > args.max_pairs:
        kb_entries = rng.sample(kb_entries, int(args.max_pairs))
        skipped_reasons["capped_by_max_pairs"] += 1

    # Optional: LLM knowledge extraction (vLLM parallel)
    llm_extract = {
        "enabled": bool(args.extract_knowledge),
        "pairs": 0,
        "ok": 0,
        "error": 0,
        "output_jsonl": None,
        "notes": "",
    }
    if args.extract_knowledge and kb_entries:
        try:
            vllm_model = (args.vllm_model or "").strip() or (args.llm_name or "").strip()
            if not vllm_model:
                raise ValueError("missing vllm model name (use --vllm-model or set VLLM_MODEL_NAME/LLM_NAME)")

            # Fast connectivity check before spawning a multiprocessing pool (avoids noisy worker tracebacks).
            try:
                from openai import OpenAI  # type: ignore

                OpenAI(
                    base_url=str(args.vllm_base_url),
                    api_key=str(args.vllm_api_key),
                    timeout=float(args.vllm_timeout),
                ).models.list()
            except Exception as e:
                raise RuntimeError(f"vLLM endpoint not reachable: {type(e).__name__}: {e}") from e

            out_jsonl = symbols_dir / "knowledge_extraction_vllm.jsonl"
            llm_extract["output_jsonl"] = str(out_jsonl)

            # Build tasks
            tasks: List[Dict[str, Any]] = []
            for e in kb_entries:
                tid = e.get("original_file") or ""
                meta = e.get("learned_from") or {}
                tasks.append(
                    {
                        "task_id": tid,
                        "original_file": tid,
                        "project": meta.get("project"),
                        "function": meta.get("function"),
                        "c_len": len(e.get("c_code") or ""),
                        "rust_len": len(e.get("rust_code") or ""),
                        "c_code": e.get("c_code") or "",
                        "rust_code": e.get("rust_code") or "",
                    }
                )

            sampling = VllmSampling(max_tokens=int(args.vllm_max_tokens))
            results = run_parallel_vllm_knowledge_extraction(
                tasks,
                output_jsonl=out_jsonl,
                base_url=str(args.vllm_base_url),
                api_key=str(args.vllm_api_key),
                model=vllm_model,
                num_workers=int(args.vllm_num_workers),
                timeout=float(args.vllm_timeout),
                sampling=sampling,
                resume=bool(args.vllm_resume),
            )

            # Attach extracted_knowledge into KB entries
            by_id = {e.get("original_file"): e for e in kb_entries if e.get("original_file")}
            for tid, rec in results.items():
                ent = by_id.get(tid)
                if not ent:
                    continue
                if rec.get("llm_ok") is True:
                    ent["extracted_knowledge"] = convert_llm_record_to_extracted_knowledge(
                        rec,
                        source="learned_kb_vllm",
                    )
                    llm_extract["ok"] += 1
                else:
                    llm_extract["error"] += 1
            llm_extract["pairs"] = len(tasks)
        except Exception as e:
            llm_extract["notes"] = f"{type(e).__name__}: {e}"

    # Build BM25 index (same tokenizer as build_knowledge_base.py/elastic_search.py).
    bm25_written = False
    bm25_error = None
    try:
        if not kb_entries:
            bm25_error = "no_pairs"
        else:
            from rank_bm25 import BM25Plus

            tokenized_corpus = [_normalize_text_for_bm25(e["c_code"]) for e in kb_entries]
            bm25 = BM25Plus(tokenized_corpus)
            with (rag_dir / "bm25_index.pkl").open("wb") as f:
                pickle.dump(bm25, f)
            bm25_written = True
    except Exception as e:
        bm25_error = str(e)

    (rag_dir / "knowledge_base.json").write_text(json.dumps(kb_entries, ensure_ascii=False, indent=2), encoding="utf-8")

    types_n = 0
    try:
        if types_reports:
            # Write as JSONL to avoid huge single JSON.
            types_n = _write_jsonl(symbols_dir / "types_generation_reports.jsonl", types_reports)
    except Exception:
        types_n = 0

    summary = {
        "meta": {
            "generated_at": datetime.now().isoformat(),
            "run_intermediate_dir": str(run_intermediate_dir),
            "llm_name": llm_name,
        },
        "output": {
            "out_dir": str(out_dir),
            "rag_dir": str(rag_dir),
            "symbols_dir": str(symbols_dir),
            "rm_command": f"rm -rf {out_dir}",
        },
        "rag_kb": {
            "pairs": len(kb_entries),
            "knowledge_base_json": str(rag_dir / "knowledge_base.json"),
            "bm25_index_pkl": str(rag_dir / "bm25_index.pkl"),
            "bm25_written": bm25_written,
            "bm25_error": bm25_error,
        },
        "llm_knowledge_extraction": llm_extract,
        "symbols": {
            "types_generation_reports_jsonl": str(symbols_dir / "types_generation_reports.jsonl"),
            "types_generation_reports_written": types_n,
            "copied_learned_data_files": copied_learned_data,
        },
        "per_project_counts": per_project_counts,
        "skipped_reasons": dict(skipped_reasons),
    }

    (out_dir / "distill_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Learned KB written: {(rag_dir / 'knowledge_base.json')}")
    if bm25_written:
        print(f"[OK] BM25 index written: {(rag_dir / 'bm25_index.pkl')}")
    else:
        print(f"[WARN] BM25 index NOT written: {bm25_error}")
    print(f"[OK] Summary: {(out_dir / 'distill_summary.json')}")
    print(f"[RM] {summary['output']['rm_command']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
