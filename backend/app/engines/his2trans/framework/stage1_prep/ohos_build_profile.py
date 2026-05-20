#!/usr/bin/env python3
"""
OpenHarmony build profile (compile_commands) resolver.

Goal:
- Use `out_dir` as the physical primary key (a "profile").
- Use `product@vendor` as a logical label for humans.

This module loads an OpenHarmony `compile_commands_all/summary.tsv` registry,
selects the best profile for a project (AUTO), caches the decision per project,
and materializes the chosen compile_commands.json by extracting the `.zst`
archive to a stable path under `.cache/`.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


_C_EXTS = (".c", ".cc", ".cpp", ".cxx", ".C")


@dataclass(frozen=True)
class OhosProfile:
    """A physical profile keyed by out_dir."""

    out_dir: str  # e.g. "out/rk3568" or "out/neptune100/neptune_iotlink_demo"
    board: str  # e.g. "rk3568"
    label: str  # logical label: product@vendor (canonical for this out_dir)
    # Many products can share the same out_dir (e.g. multiple rk3568 products).
    # Each build can produce a different compile_commands.json subset.
    # We keep all OK archives so we can merge them into a higher-coverage compile_db.
    archive_paths: Tuple[Path, ...]  # all .zst archives for this out_dir
    archive_path: Path  # canonical representative archive (largest)
    products: Tuple[str, ...]  # all product@vendor entries that map to this out_dir


@dataclass(frozen=True)
class SelectedProfile:
    profile: OhosProfile
    compile_commands_json: Path  # extracted compile_commands.json (stable path)
    ohos_root: Path
    selection_reason: str  # "cache" | "pin" | "auto"
    score: Optional[Dict] = None  # optional diagnostics


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sanitize_out_dir(out_dir: str) -> str:
    s = out_dir.strip().replace("\\", "/")
    s = s.strip("/")
    return s.replace("/", "__")


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _get_cache_root() -> Path:
    # Keep all profile-related state under repo `.cache/` (stable, writable).
    root = _default_repo_root() / ".cache" / "ohos_profiles"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _get_extract_root() -> Path:
    env = os.environ.get("C2R_OHOS_CC_EXTRACT_ROOT", "").strip()
    if env:
        p = Path(env).expanduser().resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p
    p = _get_cache_root() / "compile_commands"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _get_project_cache_root() -> Path:
    env = os.environ.get("C2R_PROFILE_CACHE_DIR", "").strip()
    if env:
        p = Path(env).expanduser().resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p
    p = _get_cache_root() / "projects"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _iter_project_sources(project_root: Path, limit: int = 0) -> List[Path]:
    """Collect project sources (deterministic order)."""
    project_root = Path(project_root)
    files: List[Path] = []
    for dirpath, _, filenames in os.walk(project_root):
        d = Path(dirpath)
        for fn in filenames:
            if fn.endswith(_C_EXTS):
                files.append(d / fn)
    files.sort(key=lambda p: str(p))
    if limit and limit > 0:
        return files[:limit]
    return files


def _read_original_rel(project_root: Path) -> str:
    try:
        p = Path(project_root) / "original_path.txt"
        if not p.exists():
            return ""
        return p.read_text(encoding="utf-8", errors="ignore").strip()
    except Exception:
        return ""


def build_project_key(project_name: str, project_root: Path, ohos_root: Optional[Path]) -> str:
    """
    Stable-ish project fingerprint across workspaces.

    Preference order:
    1) original_path.txt content (SelfContained modules)
    2) project_name + absolute resolved project_root
    """
    original_rel = _read_original_rel(project_root)
    if original_rel:
        base = f"name={project_name}\noriginal_rel={original_rel}\n"
        if ohos_root:
            base += f"ohos_root={str(Path(ohos_root).resolve())}\n"
        return _sha256_text(base)[:16]
    return _sha256_text(f"name={project_name}\nroot={str(Path(project_root).resolve())}\n")[:16]


def create_ohos_path_mapper(project_root: Path, ohos_root: Optional[Path]):
    """
    Map a file path under `project_root` back into the OpenHarmony source tree
    using `original_path.txt` (if present).
    """
    project_root = Path(project_root).resolve()
    if not ohos_root:
        return lambda p: Path(p)
    ohos_root = Path(ohos_root).resolve()
    original_rel = _read_original_rel(project_root)
    if not original_rel:
        return lambda p: Path(p)

    ohos_project_root = (ohos_root / original_rel).resolve()
    if not ohos_project_root.exists():
        return lambda p: Path(p)

    def _map(p: Path) -> Path:
        try:
            p_resolved = Path(p).resolve()
            rel = p_resolved.relative_to(project_root)
            candidate = (ohos_project_root / rel).resolve()
            return candidate if candidate.exists() else Path(p)
        except Exception:
            return Path(p)

    return _map


def _normalize_registry_path(registry: Path) -> Path:
    registry = Path(registry).expanduser()
    if registry.is_dir():
        candidate = registry / "summary.tsv"
        return candidate
    return registry


def detect_default_registry() -> Optional[Path]:
    """
    Best-effort default registry discovery for this repo layout.
    """
    repo = _default_repo_root()
    default = (
        repo
        / "SelfContained"
        / "ohos_full"
        / "OpenHarmony-v5.0.1-Release"
        / "OpenHarmony"
        / "compile_commands_all"
        / "summary.tsv"
    )
    return default if default.exists() else None


def get_registry_path_from_env() -> Optional[Path]:
    env = os.environ.get("OHOS_CC_REGISTRY", "").strip()
    if env:
        p = _normalize_registry_path(Path(env))
        return p if p.exists() else None
    env = os.environ.get("OHOS_COMPILE_COMMANDS_REGISTRY", "").strip()
    if env:
        p = _normalize_registry_path(Path(env))
        return p if p.exists() else None
    return detect_default_registry()


def infer_ohos_root_from_registry(registry_path: Path) -> Optional[Path]:
    """
    registry_path: .../<ohos_root>/compile_commands_all/summary.tsv
    """
    try:
        rp = Path(registry_path).resolve()
        if rp.name != "summary.tsv":
            return None
        parent = rp.parent
        if parent.name != "compile_commands_all":
            return None
        return parent.parent
    except Exception:
        return None


def registry_fingerprint(registry_path: Path) -> str:
    """
    Fingerprint the registry so project cache can be invalidated when it changes.
    We use (size, mtime, sha256(prefix)) to avoid hashing huge files.
    """
    p = Path(registry_path)
    st = p.stat()
    # summary.tsv is small; hashing full file is ok and deterministic.
    digest = _sha256_file(p)
    return f"sha256={digest}|size={st.st_size}|mtime={int(st.st_mtime)}"


def load_profiles_from_summary(registry_path: Path) -> Dict[str, OhosProfile]:
    """
    Load registry and collapse products into out_dir-keyed profiles.
    For each out_dir, pick a canonical archive by largest archive size.
    """
    registry_path = Path(registry_path)
    if not registry_path.exists():
        raise FileNotFoundError(str(registry_path))

    rows: List[Dict[str, str]] = []
    with open(registry_path, "r", encoding="utf-8", errors="ignore") as f:
        header = None
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if header is None:
                header = parts
                continue
            if len(parts) < len(header):
                # tolerate truncated
                parts += [""] * (len(header) - len(parts))
            row = dict(zip(header, parts))
            rows.append(row)

    grouped: Dict[str, List[Dict[str, str]]] = {}
    for r in rows:
        status = (r.get("status") or "").strip().upper()
        out_dir = (r.get("out_dir") or "").strip()
        archive = (r.get("archive_path") or "").strip()
        if not out_dir:
            continue
        # Accept any successful row that produced an archive, even if the runner labeled it as SKIP.
        # In our `gen_compile_commands_all_products.sh` workflow, SKIP typically means "already built /
        # reused existing compile_commands", but the archive is still valid and should be used.
        if status in ("FAIL", "ERROR"):
            continue
        try:
            rc = int(str(r.get("rc") or "0").strip() or "0")
        except Exception:
            rc = 0
        if rc != 0:
            continue
        if not archive:
            continue
        grouped.setdefault(out_dir, []).append(r)

    profiles: Dict[str, OhosProfile] = {}
    for out_dir, items in grouped.items():
        # Pick canonical archive by largest compressed size.
        best = None
        best_size = -1
        products: List[str] = []
        archive_paths: List[Path] = []
        board = ""
        for r in items:
            prod = (r.get("product") or "").strip()
            if prod:
                products.append(prod)
            board = board or (r.get("board") or "").strip()
            ap = Path(r.get("archive_path") or "")
            if str(ap):
                ap = ap.expanduser()
                if ap not in archive_paths:
                    archive_paths.append(ap)
            try:
                size = ap.stat().st_size
            except Exception:
                size = -1
            if size > best_size:
                best_size = size
                best = r

        if not best:
            continue

        label = (best.get("product") or "").strip() or out_dir
        archive_path = Path(best.get("archive_path") or "").expanduser()
        if not archive_paths:
            archive_paths = [archive_path]
        # Deterministic ordering for caching/meta.
        archive_paths = sorted(archive_paths, key=lambda p: str(p))
        profiles[out_dir] = OhosProfile(
            out_dir=out_dir,
            board=board or "",
            label=label,
            archive_paths=tuple(archive_paths),
            archive_path=archive_path,
            products=tuple(sorted(set(products))),
        )

    return profiles


def ensure_compile_commands_json(profile: OhosProfile) -> Path:
    """
    Extract profile archive to a stable path:
      <extract_root>/<sanitize(out_dir)>/compile_commands.json
    """
    extract_root = _get_extract_root()
    out_dir_key = _sanitize_out_dir(profile.out_dir)
    target_dir = extract_root / out_dir_key
    target_dir.mkdir(parents=True, exist_ok=True)
    out_json = target_dir / "compile_commands.json"
    meta_json = target_dir / "profile.json"
    merge_strategy = "best_per_file_v1"

    def _latest_archive_mtime() -> float:
        latest = 0.0
        for ap in (profile.archive_paths or (profile.archive_path,)):
            try:
                latest = max(latest, ap.stat().st_mtime)
            except Exception:
                continue
        return latest

    # Fast path: existing and newer than all archives AND meta matches current archive set.
    try:
        if out_json.exists() and out_json.stat().st_size > 10:
            meta_ok = False
            try:
                if meta_json.exists():
                    meta = json.loads(meta_json.read_text(encoding="utf-8", errors="ignore"))
                    want = sorted(str(p) for p in (profile.archive_paths or (profile.archive_path,)))
                    got = meta.get("archive_paths")
                    if isinstance(got, list):
                        meta_ok = sorted(str(p) for p in got) == want
                        # If this out_dir is merged, also require a compatible merge strategy.
                        if meta_ok and len(want) > 1:
                            meta_ok = str(meta.get("merge_strategy") or "") == merge_strategy
                    else:
                        # legacy meta only stored single archive_path
                        got_one = str(meta.get("archive_path") or "")
                        if len(want) == 1 and got_one == want[0]:
                            meta_ok = True
            except Exception:
                meta_ok = False
            if meta_ok and out_json.stat().st_mtime >= _latest_archive_mtime():
                return out_json
    except Exception:
        pass

    # Extract with zstd (preferred, matches how archives were created).
    for ap in (profile.archive_paths or (profile.archive_path,)):
        if not ap.exists():
            raise FileNotFoundError(str(ap))

    zstd = shutil.which("zstd")
    if not zstd:
        raise RuntimeError("zstd not found in PATH; cannot extract compile_commands archive")

    tmp = out_json.with_suffix(f".json.tmp.{os.getpid()}")
    try:
        merged_count: Optional[int] = None
        # Single archive: keep old fast path.
        if len(profile.archive_paths) <= 1:
            with open(tmp, "wb") as f:
                proc = subprocess.run(
                    [zstd, "-d", "-c", str(profile.archive_path)],
                    stdout=f,
                    stderr=subprocess.PIPE,
                    check=False,
                )
            if proc.returncode != 0:
                err = proc.stderr.decode("utf-8", errors="ignore")[:4000]
                raise RuntimeError(f"zstd extract failed (rc={proc.returncode}): {err}")
            os.replace(tmp, out_json)
        else:
            # Multi-archive merge: union compile_commands entries across products that share the same out_dir.
            import json as _json

            best_by_file: Dict[str, dict] = {}
            best_score_by_file: Dict[str, Tuple[int, int, int]] = {}

            def _entry_text(e: dict) -> str:
                cmd = e.get("command")
                if cmd:
                    return str(cmd)
                args = e.get("arguments")
                if isinstance(args, list) and args:
                    return " ".join(str(a) for a in args)
                return ""

            def _entry_file_key(e: dict) -> str:
                f = str(e.get("file") or "")
                if not f:
                    return ""
                d = str(e.get("directory") or "")
                p = Path(f)
                if not p.is_absolute() and d:
                    p = Path(d) / p
                try:
                    return str(p.resolve(strict=False))
                except Exception:
                    return str(p)

            def _score_entry(e: dict) -> Tuple[int, int, int]:
                """
                Prefer target OHOS toolchain entries over host entries.
                Keep this consistent with CompileCommandsParser's internal heuristic.
                """
                directory = str(e.get("directory") or "").replace("\\", "/")
                text_norm = _entry_text(e).replace("\\", "/")
                has_target = ("--target=" in text_norm) or (" -target " in f" {text_norm} ")
                is_ohos_target = has_target and ("ohos" in text_norm)
                has_sysroot = (
                    ("--sysroot=" in text_norm)
                    or (" --sysroot " in f" {text_norm} ")
                    or (" -isysroot " in f" {text_norm} ")
                )
                host_hint = (
                    "/clang_x64/" in directory
                    or "/clang_x64/" in text_norm
                    or "x86_64-linux-gnu" in text_norm
                    or "--target=x86_64" in text_norm
                )
                return (1 if is_ohos_target else 0, 1 if has_sysroot else 0, 0 if host_hint else 1)

            for ap in profile.archive_paths:
                # Decompress into a temp file to avoid holding multiple archives in memory at once.
                with tempfile.NamedTemporaryFile(prefix="c2r_cc_", suffix=".json", delete=False) as tf:
                    tf_path = Path(tf.name)
                    proc = subprocess.run(
                        [zstd, "-d", "-c", str(ap)],
                        stdout=tf,
                        stderr=subprocess.PIPE,
                        check=False,
                    )
                if proc.returncode != 0:
                    err = proc.stderr.decode("utf-8", errors="ignore")[:4000]
                    try:
                        tf_path.unlink(missing_ok=True)  # py>=3.8
                    except Exception:
                        pass
                    raise RuntimeError(f"zstd extract failed (rc={proc.returncode}): {err}")

                try:
                    with open(tf_path, "r", encoding="utf-8", errors="ignore") as f:
                        arr = _json.load(f)
                    if not isinstance(arr, list):
                        continue
                    for entry in arr:
                        if not isinstance(entry, dict):
                            continue
                        fk = _entry_file_key(entry)
                        if not fk:
                            continue
                        sc = _score_entry(entry)
                        prev = best_score_by_file.get(fk)
                        if prev is None or sc > prev:
                            best_score_by_file[fk] = sc
                            best_by_file[fk] = entry
                finally:
                    try:
                        tf_path.unlink(missing_ok=True)  # py>=3.8
                    except Exception:
                        pass

            merged = list(best_by_file.values())
            merged.sort(key=lambda e: (_entry_file_key(e), str(e.get("directory") or "")))
            with open(tmp, "w", encoding="utf-8") as f:
                _json.dump(merged, f, ensure_ascii=False)
            merged_count = len(merged)
            os.replace(tmp, out_json)

        meta = {
            "out_dir": profile.out_dir,
            "board": profile.board,
            "label": profile.label,
            "products": list(profile.products),
            "archive_path": str(profile.archive_path),
            "archive_paths": [str(p) for p in (profile.archive_paths or (profile.archive_path,))],
            "merged": bool(len(profile.archive_paths) > 1),
            "merged_entry_count": merged_count,
            "merge_strategy": merge_strategy if len(profile.archive_paths) > 1 else "single",
            "extracted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        meta_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return out_json
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


def _project_cache_path(project_key: str) -> Path:
    return _get_project_cache_root() / f"{project_key}.json"


def load_project_profile_cache(project_key: str) -> Optional[Dict]:
    p = _project_cache_path(project_key)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return None


def save_project_profile_cache(project_key: str, data: Dict) -> None:
    p = _project_cache_path(project_key)
    tmp = p.with_suffix(f".json.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, p)


def _coerce_out_dir(pin: str) -> str:
    s = (pin or "").strip()
    if not s:
        return ""
    s = s.replace("\\", "/").strip()
    if s.startswith("out/"):
        return s
    # Accept bare board like "rk3568" -> "out/rk3568"
    if "/" not in s:
        return f"out/{s}"
    return s


def select_profile_for_project(
    *,
    project_name: str,
    project_root: Path,
    require_registry: bool = False,
) -> Optional[SelectedProfile]:
    """
    Resolve (out_dir profile -> compile_commands.json) for the current project.

    Env knobs:
    - OHOS_CC_REGISTRY / OHOS_COMPILE_COMMANDS_REGISTRY: registry path (summary.tsv or compile_commands_all dir)
    - OHOS_ROOT: optional override for ohos_root (used for path mapping)
    - C2R_PROFILE_PIN_OUT_DIR: pin a specific out_dir (e.g. out/rk3568)
    - C2R_PROFILE_RESCAN=1: ignore cache and re-probe
    - C2R_PROFILE_SAMPLE_N: sample file count for probing (default 30)
    - C2R_PROFILE_TOP_K: candidate out_dirs to probe (default 16). Set 0 to mean "no limit" (probe all).
    - C2R_PROFILE_FULL_SCAN=1: scan all profiles in registry for coverage (slower, but most accurate).
      If C2R_TRUTH_MODE=1, full-scan is enabled by default (can be disabled by setting C2R_PROFILE_FULL_SCAN=0).
    - C2R_PROFILE_PROBE_TIMEOUT: clang -E timeout seconds for probing (default 25)
    - C2R_PROFILE_PROBE_JOBS: parallel probe workers (default: min(top_k, cpu/2))
    """
    registry_path = get_registry_path_from_env()
    if not registry_path:
        if require_registry:
            raise FileNotFoundError("OpenHarmony compile_commands registry not found")
        return None

    ohos_root = None
    env_root = os.environ.get("OHOS_ROOT", "").strip()
    if env_root:
        p = Path(env_root).expanduser()
        if p.exists():
            ohos_root = p.resolve()
    if not ohos_root:
        ohos_root = infer_ohos_root_from_registry(registry_path)

    reg_fp = registry_fingerprint(registry_path)
    profiles = load_profiles_from_summary(registry_path)
    if not profiles:
        if require_registry:
            raise RuntimeError(f"no OK profiles found in registry: {registry_path}")
        return None

    # Hints from original_path.txt (SelfContained modules): help choosing a reasonable profile even when
    # the project itself is not compiled by any known product (coverage_hits==0 everywhere).
    original_rel_low = _read_original_rel(project_root).lower()

    def _hint_score_for_profile(p: OhosProfile) -> int:
        """Higher is better; only used as a tie-breaker / 0-coverage fallback."""
        s = 0
        board = (p.board or "").lower()
        out_dir = (p.out_dir or "").lower()
        label = (p.label or "").lower()
        if board and board in original_rel_low:
            s += 80
        if out_dir and out_dir in original_rel_low:
            s += 80
        if label and label in original_rel_low:
            s += 120
        # Match product names (e.g. "dayu210" in device/board/hihope/dayu210/...)
        for prod in (p.products or ()):
            prod_low = str(prod).lower()
            if prod_low and prod_low in original_rel_low:
                s += 140
            prod_name = prod_low.split("@", 1)[0]
            if prod_name and prod_name in original_rel_low:
                s += 120
        return s

    hint_score_map: Dict[str, int] = {od: _hint_score_for_profile(p) for od, p in profiles.items()}
    hinted_out_dirs = {od for od, sc in hint_score_map.items() if sc > 0}

    project_key = build_project_key(project_name, project_root, ohos_root)
    cache = load_project_profile_cache(project_key)

    force = os.environ.get("C2R_PROFILE_RESCAN", "").strip() == "1"
    pin = _coerce_out_dir(os.environ.get("C2R_PROFILE_PIN_OUT_DIR", "").strip())

    if pin:
        prof = profiles.get(pin)
        if not prof:
            # allow pin by suffix match (e.g. "rk3568" -> out/rk3568 already handled)
            raise RuntimeError(f"pinned out_dir not found in registry: {pin}")
        cc = ensure_compile_commands_json(prof)
        return SelectedProfile(
            profile=prof,
            compile_commands_json=cc,
            ohos_root=ohos_root or Path(),
            selection_reason="pin",
            score={"pinned_out_dir": pin, "registry": str(registry_path), "registry_fp": reg_fp},
        )

    if cache and not force:
        if cache.get("registry_fingerprint") == reg_fp and cache.get("chosen_out_dir") in profiles:
            cache_invalid = False
            # Guard against a bad historical cache where chosen_out_dir had zero coverage hits.
            # This happened when probe scoring favored large archives even if `attempted==0`.
            try:
                score = cache.get("score") or {}
                sel_cfg = score.get("selection_config") or {}
                probe_scores = score.get("probe_scores") or {}
                chosen = str(cache.get("chosen_out_dir") or "")
                chosen_hit = (probe_scores.get(chosen) or {}).get("coverage_hits")
                truth_mode = os.environ.get("C2R_TRUTH_MODE", "0").strip().lower() in ("1", "true", "yes", "on")
                full_scan_env = os.environ.get("C2R_PROFILE_FULL_SCAN", "").strip().lower()
                if full_scan_env in ("1", "true", "yes", "on"):
                    full_scan = True
                elif full_scan_env in ("0", "false", "no", "off"):
                    full_scan = False
                else:
                    full_scan = truth_mode
                # If we now request full-scan (or truth-mode default), but the cached selection was made
                # without full-scan config, re-probe to avoid sticking to a limited candidate set.
                if full_scan and not bool(sel_cfg.get("full_scan")):
                    cache_invalid = True
                # If we can infer better hints now, and the cached choice has no coverage, invalidate.
                if hinted_out_dirs and chosen not in hinted_out_dirs and int(chosen_hit or 0) == 0:
                    cache_invalid = True
                if chosen_hit == 0 and any(
                    int((ps or {}).get("coverage_hits", 0)) > 0 for ps in probe_scores.values()
                ):
                    cache_invalid = True
            except Exception:
                cache_invalid = False

            if not cache_invalid:
                prof = profiles[str(cache["chosen_out_dir"])]
                cc = ensure_compile_commands_json(prof)
                return SelectedProfile(
                    profile=prof,
                    compile_commands_json=cc,
                    ohos_root=Path(cache.get("ohos_root") or (ohos_root or "")),
                    selection_reason="cache",
                    score=cache.get("score"),
                )

    # AUTO selection
    truth_mode = os.environ.get("C2R_TRUTH_MODE", "0").strip().lower() in ("1", "true", "yes", "on")
    sample_n_raw = int(os.environ.get("C2R_PROFILE_SAMPLE_N", "30"))
    top_k_raw = int(os.environ.get("C2R_PROFILE_TOP_K", "16"))
    timeout_s = int(os.environ.get("C2R_PROFILE_PROBE_TIMEOUT", "25"))

    full_scan_env = os.environ.get("C2R_PROFILE_FULL_SCAN", "").strip().lower()
    if full_scan_env in ("1", "true", "yes", "on"):
        full_scan = True
    elif full_scan_env in ("0", "false", "no", "off"):
        full_scan = False
    else:
        full_scan = truth_mode
    # Also treat top_k<=0 as "no limit", which implies we should not pre-trim candidates by heuristics.
    if top_k_raw <= 0:
        full_scan = True
    probe_top_k = None if top_k_raw <= 0 else max(1, top_k_raw)

    use_all_sources_env = os.environ.get("C2R_PROFILE_USE_ALL_SOURCES", "").strip().lower()
    if use_all_sources_env in ("1", "true", "yes", "on"):
        use_all_sources = True
    elif use_all_sources_env in ("0", "false", "no", "off"):
        use_all_sources = False
    else:
        # In truth-mode (or full-scan), prioritize correctness over speed: probe all sources.
        use_all_sources = truth_mode or full_scan or (sample_n_raw <= 0)

    sources = _iter_project_sources(project_root, limit=0)
    if not sources:
        # No sources: cannot probe; pick the biggest profile as default.
        prof = max(profiles.values(), key=lambda p: p.archive_path.stat().st_size)
        cc = ensure_compile_commands_json(prof)
        selected = SelectedProfile(
            profile=prof,
            compile_commands_json=cc,
            ohos_root=ohos_root or Path(),
            selection_reason="auto",
            score={"reason": "no_sources_fallback"},
        )
        _write_project_cache(project_key, selected, registry_path, reg_fp, project_root, ohos_root, {})
        return selected

    if use_all_sources:
        sample = sources
    else:
        sample_n = max(1, sample_n_raw)
        sample = sources[:sample_n] if len(sources) > sample_n else sources

    mapper = create_ohos_path_mapper(project_root, ohos_root)
    mapped_sample = [mapper(p) for p in sample]

    # Candidate ranking by coverage (file has any entry).
    from compile_commands_parser import CompileCommandsParser  # local import to avoid cycles at module import

    def _archive_size(p: OhosProfile) -> int:
        try:
            return p.archive_path.stat().st_size
        except Exception:
            return 0

    # Candidate set:
    # - Full-scan: evaluate every profile in registry (slowest, but most accurate).
    # - Heuristic: evaluate a bounded subset (fast path).
    if full_scan:
        candidates_out_dirs = set(profiles.keys())
        small_out_dirs: set[str] = set()
    else:
        # Preselect candidates without parsing every compile_db:
        # - Always include the largest profiles
        # - Also include any profile whose board/out_dir is hinted by original_path.txt
        # NOTE: hinted_out_dirs is computed above via hint_score_map (board/out_dir/product@vendor matching).

        # Largest-N fallback (N grows with top_k, but capped)
        top_k_for_preselect = probe_top_k or 16
        largest_n = max(6, top_k_for_preselect * 2)
        all_by_size = sorted(profiles.values(), key=_archive_size, reverse=True)
        candidates_out_dirs = {p.out_dir for p in all_by_size[:largest_n]}
        candidates_out_dirs |= hinted_out_dirs

        # Always include "small" profiles (cheap to parse, but can be essential for liteos/mini targets)
        small_max_mb = float(os.environ.get("C2R_PROFILE_SMALL_MAX_MB", "0.25"))
        small_out_dirs = {
            p.out_dir for p in profiles.values() if (_archive_size(p) / (1024 * 1024)) <= small_max_mb
        }
        candidates_out_dirs |= small_out_dirs

        # If still too many, keep only top by archive size.
        max_candidates = max(12, top_k_for_preselect * 3)
        if len(candidates_out_dirs) > max_candidates:
            must_keep = set(hinted_out_dirs) | set(small_out_dirs)
            keep = {od for od in must_keep if od in profiles}
            others = [profiles[od] for od in candidates_out_dirs if od in profiles and od not in keep]

            allow_others = max(0, max_candidates - len(keep))
            reduced_others = sorted(others, key=_archive_size, reverse=True)[:allow_others]
            candidates_out_dirs = keep | {p.out_dir for p in reduced_others}

    # coverage: (out_dir, strict_hit_count, any_hit_count, archive_size)
    coverage: List[Tuple[str, int, int, int]] = []
    parser_errors: Dict[str, str] = {}
    strict_reasons = {"filename_index_exact_path", "filename_index_suffix_path", "full_scan_exact_str"}

    def _compute_coverage_for_out_dir(out_dir: str) -> Optional[Tuple[str, int, int, int]]:
        prof = profiles.get(out_dir)
        if not prof:
            return None
        try:
            cc = ensure_compile_commands_json(prof)
            parser = CompileCommandsParser(cc, ohos_root=ohos_root)
        except Exception as e:
            parser_errors[out_dir] = str(e)
            return None

        strict_hit = 0
        any_hit = 0
        for f in mapped_sample:
            try:
                entry, info = parser.get_entry_for_file_with_reason(Path(f))
                if not entry:
                    continue
                any_hit += 1
                if str(info.get("reason") or "") in strict_reasons:
                    strict_hit += 1
            except Exception:
                continue
        try:
            size = prof.archive_path.stat().st_size
        except Exception:
            size = 0

        # Avoid retaining large compile_db structures across many profiles in full-scan mode.
        try:
            del parser
        except Exception:
            pass
        return (out_dir, strict_hit, any_hit, size)

    for out_dir in sorted(candidates_out_dirs):
        row = _compute_coverage_for_out_dir(out_dir)
        if row:
            coverage.append(row)

    if not coverage:
        if require_registry:
            raise RuntimeError("all profiles failed to load")
        return None

    max_strict_hit = max((t[1] for t in coverage), default=0)
    max_any_hit = max((t[2] for t in coverage), default=0)

    # If any profile has non-zero *strict* coverage, ignore zero-strict profiles by default.
    # (They cannot provide a verifiable TU context for this project.)
    if max_strict_hit > 0:
        coverage = [t for t in coverage if t[1] > 0]
    elif (not truth_mode) and (max_any_hit > 0):
        # Non-truth-mode only: fall back to any-hit (weaker match; may indicate filename collisions).
        coverage = [t for t in coverage if t[2] > 0]

    # Prefer higher strict coverage, then any coverage; use hint_score as a tie-breaker.
    # In truth-mode with 0 strict hits, treat any-hit as untrusted (avoid picking by filename collisions).
    if truth_mode and max_strict_hit == 0:
        coverage.sort(key=lambda x: (hint_score_map.get(x[0], 0), x[3]), reverse=True)
    else:
        coverage.sort(key=lambda x: (x[1], x[2], hint_score_map.get(x[0], 0), x[3]), reverse=True)

    # Candidate set to probe:
    # - Full-scan/truth-mode: only probe the best-coverage tier (max strict, else max any).
    # - Heuristic: probe top-k ranked by coverage.
    if full_scan or truth_mode:
        if max_strict_hit > 0:
            candidates = [t for t in coverage if t[1] == max_strict_hit]
        else:
            candidates = coverage[:1]
    else:
        limit = probe_top_k or len(coverage)
        candidates = coverage[: max(1, min(limit, len(coverage)))]

    # Probe candidates with light clang -E checks (parallel).
    probe_scores: Dict[str, Dict] = {}
    best_out_dir = candidates[0][0]
    best_score = None

    with tempfile.TemporaryDirectory(prefix="c2r_ohos_profile_probe_") as tmpdir:
        probe_dir = Path(tmpdir)
        # Use a bounded thread pool to avoid spawning too many clang processes at once.
        try:
            default_jobs = max(1, (os.cpu_count() or 4) // 2)
        except Exception:
            default_jobs = 2
        try:
            jobs = int(os.environ.get("C2R_PROFILE_PROBE_JOBS", "").strip() or "0")
        except Exception:
            jobs = 0
        if jobs <= 0:
            jobs = min(len(candidates), default_jobs)
        jobs = max(1, min(jobs, len(candidates)))

        def _probe_one(out_dir: str, strict_hit: int, any_hit: int, size: int) -> Tuple[str, Dict, Tuple]:
            try:
                prof = profiles[out_dir]
                cc = ensure_compile_commands_json(prof)
                parser = CompileCommandsParser(cc, ohos_root=ohos_root)
            except Exception as e:
                hint_score = hint_score_map.get(out_dir, 0)
                score = (0, strict_hit, any_hit, hint_score, 0, 0, 0, size)
                stats = {
                    "coverage_hits": strict_hit,
                    "coverage_hits_any": any_hit,
                    "archive_size": size,
                    "attempted": 0,
                    "success": 0,
                    "failures": 0,
                    "hint_score": hint_score,
                    "total_functions": 0,
                    "total_macros": 0,
                    "error": str(e)[:300],
                    "score_tuple": list(score),
                }
                return out_dir, stats, score
            success = 0
            attempted = 0
            failures = 0
            total_functions = 0
            total_macros = 0

            # Keep per-profile outputs separate to avoid collisions across profiles.
            out_dir_key = _sanitize_out_dir(out_dir)
            out_probe_dir = probe_dir / out_dir_key
            out_probe_dir.mkdir(parents=True, exist_ok=True)

            # Only probe files that have a reliable entry in this profile.
            for f in mapped_sample:
                attempted += 1
                try:
                    entry, info = parser.get_entry_for_file_with_reason(Path(f))
                    if not entry:
                        attempted -= 1
                        continue
                    # In probe, reject the weakest filename-only candidates by default.
                    # They tend to be collisions (same basename in different dirs).
                    reason = str(info.get("reason") or "")
                    if reason not in strict_reasons:
                        attempted -= 1
                        continue
                    ctx = parser.preprocess_with_context(
                        Path(f),
                        entry,
                        output_dir=out_probe_dir,
                        timeout_sec=timeout_s,
                    )
                    if ctx and not getattr(ctx, "error", None):
                        success += 1
                        total_functions += int(getattr(ctx, "function_count", 0))
                        total_macros += int(getattr(ctx, "macro_count", 0))
                    else:
                        failures += 1
                except Exception:
                    failures += 1

            # score: prioritize preprocess success, then coverage hits, then hint_score, then reduce failures,
            # then prefer richer TU contexts, then archive size.
            hint_score = hint_score_map.get(out_dir, 0)
            score = (success, strict_hit, any_hit, hint_score, -failures, total_functions, total_macros, size)
            stats = {
                "coverage_hits": strict_hit,
                "coverage_hits_any": any_hit,
                "archive_size": size,
                "attempted": attempted,
                "success": success,
                "failures": failures,
                "hint_score": hint_score,
                "total_functions": total_functions,
                "total_macros": total_macros,
                "score_tuple": list(score),
            }
            return out_dir, stats, score

        with ThreadPoolExecutor(max_workers=jobs) as ex:
            futs = {
                ex.submit(_probe_one, out_dir, strict_hit, any_hit, size): (out_dir, strict_hit, any_hit, size)
                for out_dir, strict_hit, any_hit, size in candidates
            }
            for fut in as_completed(futs):
                try:
                    out_dir, stats, score = fut.result()
                except Exception as e:
                    out_dir, strict_hit, any_hit, size = futs.get(fut) or ("", 0, 0, 0)
                    err = str(e)[:300]
                    probe_scores[out_dir] = {
                        "coverage_hits": strict_hit,
                        "coverage_hits_any": any_hit,
                        "archive_size": size,
                        "attempted": 0,
                        "success": 0,
                        "failures": 0,
                        "hint_score": hint_score_map.get(out_dir, 0),
                        "total_functions": 0,
                        "total_macros": 0,
                        "error": err,
                        "score_tuple": [
                            0,
                            strict_hit,
                            any_hit,
                            hint_score_map.get(out_dir, 0),
                            0,
                            0,
                            0,
                            size,
                        ],
                    }
                    continue
                probe_scores[out_dir] = stats
                if best_score is None or score > best_score:
                    best_score = score
                    best_out_dir = out_dir

    prof = profiles[best_out_dir]
    cc = ensure_compile_commands_json(prof)
    selected = SelectedProfile(
        profile=prof,
        compile_commands_json=cc,
        ohos_root=ohos_root or Path(),
        selection_reason="auto",
        score={
            "selection_config": {
                "truth_mode": truth_mode,
                "full_scan": full_scan,
                "use_all_sources": use_all_sources,
                "sample_n": sample_n_raw,
                "top_k": top_k_raw,
            },
            "coverage_summary": {
                "total_files": len(mapped_sample),
                "max_strict_hit": max_strict_hit,
                "max_any_hit": max_any_hit,
            },
            "registry": str(registry_path),
            "registry_fingerprint": reg_fp,
            "project_key": project_key,
            "candidates": [
                {"out_dir": od, "coverage_hits": sh, "coverage_hits_any": ah, "archive_size": sz}
                for od, sh, ah, sz in candidates
            ],
            "probe_scores": probe_scores,
            "parser_errors": parser_errors,
        },
    )
    _write_project_cache(project_key, selected, registry_path, reg_fp, project_root, ohos_root, probe_scores)
    return selected


def _probe_best_context(
    parser,
    source_file: Path,
    entries: List[Dict],
    *,
    output_dir: Path,
    timeout_s: int,
) -> Optional[Dict]:
    """
    A light-weight probe that tries a few entries and returns best stats
    without generating large artifacts.
    """
    best = None
    best_score = None
    # Limit probing per file to avoid explosions when filename-collision returns many entries.
    max_entries = 6
    for entry in entries[:max_entries]:
        try:
            # We only need success/fail; keep output in a temp dir.
            ctx = parser.preprocess_with_context(
                source_file,
                entry,
                output_dir=output_dir,
                timeout_sec=timeout_s,
            )
            ok = (ctx is not None) and (not getattr(ctx, "error", None))
            # score by (ok, functions, macros)
            score = (
                1 if ok else 0,
                int(getattr(ctx, "function_count", 0)),
                int(getattr(ctx, "macro_count", 0)),
            )
            if best_score is None or score > best_score:
                best_score = score
                best = {
                    "ok": ok,
                    "functions": int(getattr(ctx, "function_count", 0)),
                    "macros": int(getattr(ctx, "macro_count", 0)),
                    "error": getattr(ctx, "error", None),
                }
        except Exception as e:
            if best is None:
                best = {"ok": False, "functions": 0, "macros": 0, "error": str(e)[:300]}
    return best


def _write_project_cache(
    project_key: str,
    selected: SelectedProfile,
    registry_path: Path,
    reg_fp: str,
    project_root: Path,
    ohos_root: Optional[Path],
    probe_scores: Dict,
) -> None:
    data = {
        "project_key": project_key,
        "project_root_hint": str(Path(project_root).resolve()),
        "original_path_rel": _read_original_rel(project_root),
        "ohos_root": str((ohos_root or Path()).resolve()) if ohos_root else "",
        "registry": str(Path(registry_path).resolve()),
        "registry_fingerprint": reg_fp,
        "chosen_out_dir": selected.profile.out_dir,
        "chosen_label": selected.profile.label,
        "chosen_board": selected.profile.board,
        "compile_commands_json": str(selected.compile_commands_json),
        "selection_reason": selected.selection_reason,
        "score": selected.score or {},
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    save_project_profile_cache(project_key, data)
