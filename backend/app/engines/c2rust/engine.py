"""C2Rust Translation Engine — mechanical C → Rust transpilation.

C2Rust (c2rust transpile) is a deterministic, mechanical transpiler that
produces raw unsafe Rust from C99 source code. It makes no attempt at
semantic optimization, safety guarantees, or idiomatic Rust output.

This engine provides a baseline for comparison against His2Trans.
"""

import os
import re
import shutil
import subprocess
import time
from typing import List, Set, Callable

from app.engines.base_engine import BaseEngine


class C2RustEngine(BaseEngine):
    """C2Rust mechanical C-to-Rust transpiler engine.

    Stages (2 total):
      1. stage1_transpile  — run `c2rust transpile` on every .c file
      2. stage2_postprocess — strip unstable features, downgrade extern types,
                               add pub qualifiers, compute quality metrics
    """

    # ------------------------------------------------------------------
    # BaseEngine interface
    # ------------------------------------------------------------------

    def get_display_name(self) -> str:
        return "C2Rust"

    def get_description(self) -> str:
        return (
            "C2Rust mechanical C → Rust transpiler. "
            "Produces raw unsafe Rust with no semantic optimization — "
            "a deterministic baseline for comparison with His2Trans."
        )

    def get_stages(self) -> list[dict]:
        return [
            {
                "id": "stage1_transpile",
                "name": "Stage 1: C2Rust Transpile",
                "description": "Run c2rust transpile on all .c source files",
            },
            {
                "id": "stage2_postprocess",
                "name": "Stage 2: Post-process & Metrics",
                "description": (
                    "Strip unstable #![feature] gates, downgrade extern types "
                    "to opaque structs, make top-level functions public, "
                    "compute quality metrics (unsafe count, raw ptr count, etc.)"
                ),
            },
        ]

    def run_stage(
        self,
        stage_id: str,
        source_path: str,
        output_path: str,
        config: dict,
        log_callback: Callable,
    ) -> dict:
        source_path = os.path.abspath(source_path)
        output_path = os.path.abspath(output_path)
        os.makedirs(output_path, exist_ok=True)

        # Resolve c2rust binary — it may be under ~/.cargo/bin which is
        # not always on the Flask process's default PATH.
        c2rust_bin = shutil.which("c2rust")
        if not c2rust_bin:
            cargo_bin = os.path.expanduser("~/.cargo/bin")
            candidate = os.path.join(cargo_bin, "c2rust")
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                c2rust_bin = candidate
                # Also add to PATH so subprocess inherits it
                os.environ["PATH"] = f"{cargo_bin}:{os.environ.get('PATH', '')}"
        if not c2rust_bin:
            raise RuntimeError(
                "c2rust not found on PATH. "
                "Install with: cargo install --locked c2rust"
            )

        # Ensure LLVM libraries are resolvable at runtime (c2rust links
        # against LLVM dynamically — it needs the shared libs on the
        # library path).
        llvm_lib = self._detect_llvm_lib_dir()
        if llvm_lib:
            existing = os.environ.get("DYLD_LIBRARY_PATH", "")
            os.environ["DYLD_LIBRARY_PATH"] = (
                f"{llvm_lib}:{existing}" if existing else llvm_lib
            )
            existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
            os.environ["LD_LIBRARY_PATH"] = (
                f"{llvm_lib}:{existing_ld}" if existing_ld else llvm_lib
            )

        if stage_id == "stage1_transpile":
            return self._run_transpile(
                c2rust_bin, source_path, output_path, config, log_callback
            )
        elif stage_id == "stage2_postprocess":
            return self._run_postprocess(
                output_path, config, log_callback
            )
        else:
            raise ValueError(f"Unknown stage: {stage_id}")

    # ------------------------------------------------------------------
    # Stage 1: Transpile
    # ------------------------------------------------------------------

    def _run_transpile(
        self,
        c2rust_bin: str,
        source_path: str,
        output_path: str,
        config: dict,
        log: Callable,
    ) -> dict:
        transpiled_dir = os.path.join(output_path, "transpiled")
        os.makedirs(transpiled_dir, exist_ok=True)

        # Find all .c files
        c_files = self._find_c_files(source_path)
        if not c_files:
            raise RuntimeError(f"No .c files found in source: {source_path}")

        log(f"Found {len(c_files)} C source file(s)", "info")

        # Collect include flags from compile_commands.json.  Resolve
        # the OHOS root against the bundled default if none was provided
        # (the C2Rust config panel hides the field, so it may be empty).
        ohos_root = config.get("ohos_root", "")
        if not ohos_root:
            # Fall back to the in-tree ohos_root_min bundled with the backend
            default_ohos = os.path.join(
                os.path.dirname(__file__), "..", "..", "..",
                "data", "ohos", "ohos_root_min",
            )
            if os.path.isdir(default_ohos):
                ohos_root = os.path.abspath(default_ohos)
        include_flags = self._gather_include_flags(source_path, ohos_root=ohos_root)
        if include_flags:
            log(f"Include flags ({len(include_flags)}): {include_flags[:5]}{'...' if len(include_flags) > 5 else ''}", "info")

        timeout = int(config.get("c2rust_timeout", 180))
        total = len(c_files)
        succeeded = 0
        failed = 0
        failed_files: List[str] = []

        for idx, c_file in enumerate(c_files, 1):
            rel = os.path.relpath(c_file, source_path)
            out_subdir = os.path.join(transpiled_dir, os.path.dirname(rel))
            os.makedirs(out_subdir, exist_ok=True)

            log(f"[{idx}/{total}] Transpiling: {rel}", "info")

            # Preprocess with clang first so c2rust sees fully-resolved types.
            # Raw .c files with unresolved includes cause c2rust to panic with
            # "TagTypeUnknown" — mirroring His2Trans's fallback in
            # incremental_translate.py:6190 which feeds preprocessed .i files.
            source_file = c_file
            try:
                source_file = self._preprocess_c_file(
                    c_file, out_subdir, include_flags, timeout, log
                )
            except Exception as e:
                log(f"  ⚠ Preprocessing failed for {rel}: {e}, trying raw .c", "warn")

            # c2rust transpile <input> -o <output_dir> -- <clang_flags>
            cmd = [
                c2rust_bin, "transpile",
                source_file,
                "-o", out_subdir,
                "--",
                "-Wno-error",
                "-Wno-macro-redefined",
                "-Wno-builtin-macro-redefined",
                "-Wno-ignored-attributes",
                "-Wno-unknown-attributes",
                "-Wno-unused-command-line-argument",
            ] + include_flags

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                if result.returncode == 0:
                    succeeded += 1
                    # Also copy the generated .rs to transpiled/ for flat viewing
                    log(f"  ✓ {rel} — OK", "info")
                else:
                    failed += 1
                    failed_files.append(rel)
                    stderr_tail = (result.stderr or "").strip().splitlines()[-5:]
                    log(f"  ✗ {rel} — c2rust failed (rc={result.returncode})", "error")
                    for line in stderr_tail:
                        log(f"    {line}", "error")
            except subprocess.TimeoutExpired:
                failed += 1
                failed_files.append(rel)
                log(f"  ✗ {rel} — timed out after {timeout}s", "error")

        # Generate minimal Cargo.toml so the output can be browsed as a project
        self._write_minimal_cargo_toml(transpiled_dir)

        # Save a raw (pre-postprocess) copy in workspace/transpiled_raw/
        # so Stage 1's "Show Files" shows the original c2rust output —
        # Stage 2 will modify transpiled/ in-place, making the two stages'
        # file views diverge.
        raw_dir = os.path.join(output_path, "workspace", "transpiled_raw")
        if os.path.exists(raw_dir):
            shutil.rmtree(raw_dir)
        try:
            shutil.copytree(transpiled_dir, raw_dir, symlinks=True)
        except Exception:
            pass  # best-effort

        # Symlink transpiled/ into workspace/ so the frontend workspace
        # file browser (StagePanel) can serve the (eventually postprocessed)
        # sources — Stage 2's view.
        ws_transpiled = os.path.join(output_path, "workspace", "transpiled")
        os.makedirs(os.path.dirname(ws_transpiled), exist_ok=True)
        if not os.path.exists(ws_transpiled):
            try:
                os.symlink(transpiled_dir, ws_transpiled)
            except OSError:
                pass  # best-effort; file browsing falls back gracefully

        summary = (
            f"Stage 1 complete: {succeeded}/{total} files transpiled"
            + (f", {failed} failed" if failed else "")
        )
        log(summary, "info")

        return {
            "summary": summary,
            "details": {
                "total_files": total,
                "transpiled": succeeded,
                "failed": failed,
                "failed_files": failed_files,
                "transpiled_dir": transpiled_dir,
            },
        }

    # ------------------------------------------------------------------
    # Stage 2: Post-process & Metrics
    # ------------------------------------------------------------------

    def _run_postprocess(
        self,
        output_path: str,
        config: dict,
        log: Callable,
    ) -> dict:
        transpiled_dir = os.path.join(output_path, "transpiled")
        if not os.path.isdir(transpiled_dir):
            raise RuntimeError(
                "Transpiled directory not found. Run Stage 1 first."
            )

        rs_files = self._find_rs_files(transpiled_dir)
        if not rs_files:
            raise RuntimeError("No .rs files found in transpiled directory")

        log(f"Post-processing {len(rs_files)} Rust file(s)", "info")

        # Accumulate metrics across all files
        metrics = {
            "total_files": 0,
            "total_lines": 0,
            "unsafe_blocks": 0,
            "unsafe_functions": 0,
            "unsafe_body_lines": 0,
            "extern_c_functions": 0,
            "raw_ptr_types": 0,
            "feature_gates_removed": 0,
            "extern_types_downgraded": 0,
        }

        for idx, rs_file in enumerate(rs_files, 1):
            rel = os.path.relpath(rs_file, transpiled_dir)
            log(f"[{idx}/{len(rs_files)}] Processing: {rel}", "info")

            try:
                with open(rs_file, "r", encoding="utf-8", errors="replace") as f:
                    original = f.read()
            except Exception:
                log(f"  ✗ Cannot read {rel}", "error")
                continue

            postprocessed = _postprocess_c2rust_output_for_stable(original)

            with open(rs_file, "w", encoding="utf-8") as f:
                f.write(postprocessed)

            # Compute per-file metrics
            file_metrics = _compute_file_metrics(original, postprocessed)
            for key in metrics:
                metrics[key] += file_metrics.get(key, 0)
            metrics["total_files"] += 1

        # Write report.json
        report_path = os.path.join(output_path, "report.json")
        import json
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        # Run cargo check to get real compile stats
        compile_passed, compile_failed = self._run_cargo_check(
            transpiled_dir, log
        )
        metrics["compile_passed"] = compile_passed
        metrics["compile_failed"] = compile_failed
        compile_total = compile_passed + compile_failed
        compile_rate = f"{compile_passed}/{compile_total}" if compile_total > 0 else "N/A"

        summary = (
            f"Stage 2 complete: {metrics['total_files']} files post-processed. "
            f"{metrics['unsafe_blocks']} unsafe blocks, "
            f"{metrics['extern_c_functions']} extern C fns, "
            f"{metrics['raw_ptr_types']} raw ptrs. "
            f"Compile: {compile_rate} passed."
        )
        log(summary, "info")

        return {
            "summary": summary,
            "details": metrics,
        }

    # ------------------------------------------------------------------
    # Stage 2 helper: cargo check
    # ------------------------------------------------------------------

    @staticmethod
    def _run_cargo_check(transpiled_dir: str, log) -> tuple:
        """Run ``cargo check`` on the transpiled output and return
        (passed_count, failed_count) based on the compiler exit code.

        A zero exit code means all files compile.  A non-zero exit code
        counts all files as failed (c2rust output is typically not
        compilable as a whole crate — individual files may have errors).
        """
        # Ensure a minimal Cargo project exists
        cargo_toml = os.path.join(transpiled_dir, "Cargo.toml")
        src_dir = os.path.join(transpiled_dir, "src")
        os.makedirs(src_dir, exist_ok=True)

        # Collect all .rs files recursively for file count
        rs_files_all = []
        for root, dirs, files in os.walk(transpiled_dir):
            # Skip target/ and hidden dirs
            dirs[:] = [d for d in dirs if d != "target" and not d.startswith(".")]
            for fn in files:
                if fn.endswith(".rs"):
                    rs_files_all.append(os.path.relpath(os.path.join(root, fn), transpiled_dir))
        total_files = len(rs_files_all) or 1

        # Ensure Cargo.toml exists (generated in Stage 1, but may have been removed)
        if not os.path.exists(cargo_toml):
            C2RustEngine._write_minimal_cargo_toml(transpiled_dir)

        log(f"Running cargo check on {total_files} transpiled .rs file(s)...", "info")
        try:
            result = subprocess.run(
                ["cargo", "check"],
                cwd=transpiled_dir,
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode == 0:
                log(f"  ✓ cargo check passed ({total_files} file(s))", "info")
                return (total_files, 0)
            else:
                stderr_tail = (result.stderr or "").strip().splitlines()[-5:]
                log(f"  ✗ cargo check failed (exit {result.returncode})", "warn")
                for line in stderr_tail:
                    log(f"    {line[:150]}", "warn")
                return (0, total_files)
        except subprocess.TimeoutExpired:
            log("  ✗ cargo check timed out", "warn")
            return (0, total_files)
        except FileNotFoundError:
            log("  ✗ cargo not found on PATH — skipping compile check", "warn")
            return (0, 0)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_clang() -> str:
        """Find a usable clang binary (prefer the llvm@18 one on macOS)."""
        for candidate in ("clang-18", "clang"):
            path = shutil.which(candidate)
            if path:
                return path
        # Homebrew llvm@18 may not be on PATH
        brew_clang = "/opt/homebrew/opt/llvm@18/bin/clang"
        if os.path.isfile(brew_clang) and os.access(brew_clang, os.X_OK):
            return brew_clang
        raise RuntimeError(
            "clang not found. Install with: brew install llvm@18"
        )

    @staticmethod
    def _preprocess_c_file(
        c_file: str,
        work_dir: str,
        include_flags: list,
        timeout: int,
        log,
    ) -> str:
        """Preprocess a .c file with clang -E so c2rust sees resolved types.

        Returns the path to a preprocessed .c file (written beside the
        original, using the same stem).  This mirrors the His2Trans
        pattern in ``incremental_translate.py`` where ``.i`` (clang -E
        output) is fed to c2rust instead of raw ``.c``.
        """
        clang = C2RustEngine._detect_clang()
        stem = os.path.splitext(os.path.basename(c_file))[0]
        preprocessed = os.path.join(work_dir, f"{stem}_preprocessed.c")

        cmd = [
            clang, "-E",
            c_file,
            "-o", preprocessed,
            "-Wno-error",
            "-Wno-macro-redefined",
            "-Wno-builtin-macro-redefined",
            "-Wno-ignored-attributes",
        ] + include_flags

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=max(10, timeout),
        )
        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            raise RuntimeError(
                f"clang -E failed (rc={result.returncode}): {stderr[:200]}"
            )
        if not os.path.isfile(preprocessed) or os.path.getsize(preprocessed) == 0:
            raise RuntimeError("clang -E produced empty output")

        return preprocessed

    @staticmethod
    def _detect_llvm_lib_dir() -> str | None:
        """Auto-detect the LLVM shared-library directory needed by c2rust.

        c2rust is linked against LLVM dynamically.  On macOS / Linux the
        installed ``llvm-config`` binary can tell us where the libraries
        live.  Falls back to common Homebrew / system paths.
        """
        # 1. Try llvm-config (most reliable)
        for candidate in ("llvm-config-18", "llvm-config", "llvm-config-17"):
            path = shutil.which(candidate)
            if path:
                try:
                    result = subprocess.run(
                        [path, "--libdir"],
                        capture_output=True, text=True, timeout=5,
                    )
                    if result.returncode == 0 and result.stdout.strip():
                        return result.stdout.strip()
                except Exception:
                    pass

        # 2. Common Homebrew paths (macOS Apple Silicon)
        for brew_llvm in (
            "/opt/homebrew/opt/llvm@18/lib",
            "/opt/homebrew/opt/llvm@17/lib",
            "/opt/homebrew/opt/llvm/lib",
            "/usr/local/opt/llvm@18/lib",
            "/usr/local/opt/llvm@17/lib",
            "/usr/local/opt/llvm/lib",
        ):
            if os.path.isdir(brew_llvm):
                return brew_llvm

        # 3. Linux system paths
        for sys_path in ("/usr/lib/llvm-18/lib", "/usr/lib/llvm-17/lib"):
            if os.path.isdir(sys_path):
                return sys_path

        return None

    @staticmethod
    def _find_c_files(root: str) -> List[str]:
        """Recursively find all .c files, skipping hidden and test dirs."""
        skip_dirs = {
            ".git", "__pycache__", "node_modules", "target",
            "test", "tests", "unittest", "unittests", ".hidden",
        }
        results = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in skip_dirs and not d.startswith(".")]
            for f in filenames:
                if f.endswith(".c"):
                    results.append(os.path.join(dirpath, f))
        return sorted(results)

    @staticmethod
    def _find_rs_files(root: str) -> List[str]:
        """Recursively find all .rs files."""
        results = []
        for dirpath, _, filenames in os.walk(root):
            for f in filenames:
                if f.endswith(".rs"):
                    results.append(os.path.join(dirpath, f))
        return sorted(results)

    @staticmethod
    def _gather_include_flags(source_path: str, ohos_root: str = "") -> List[str]:
        """Extract -I flags from compile_commands.json, resolving relative paths.

        Paths are resolved against *source_path* first; any path that still
        doesn't exist is re-resolved against *ohos_root* (if provided) so
        that ``../../ohos_root_min/...`` entries from the original build
        machine are mapped to the local OHOS SDK.
        """
        cc_path = os.path.join(source_path, "compile_commands.json")
        if not os.path.isfile(cc_path):
            return []

        try:
            import json
            with open(cc_path, "r", encoding="utf-8") as f:
                entries = json.load(f)
        except Exception:
            return []

        include_dirs: set = set()
        for entry in entries[:50]:
            cmd = entry.get("command", "")
            if not cmd and "arguments" in entry:
                cmd = " ".join(entry["arguments"])
            # Some entries have a "directory" key — prepend it for resolution
            build_dir = entry.get("directory", source_path)
            for token in cmd.split():
                if token.startswith("-I") and len(token) > 2:
                    d = token[2:]
                    # Try absolute first, then relative to build_dir, then relative to source
                    if os.path.isdir(d):
                        include_dirs.add(d)
                        continue
                    resolved = os.path.normpath(os.path.join(build_dir, d))
                    if os.path.isdir(resolved):
                        include_dirs.add(resolved)
                        continue
                    resolved = os.path.normpath(os.path.join(source_path, d))
                    if os.path.isdir(resolved):
                        include_dirs.add(resolved)
                        continue
                    # Try resolving ohos_root_min relative paths against ohos_root
                    if ohos_root and "ohos_root_min" in d:
                        # strip leading ../../... up to ohos_root_min
                        idx = d.find("ohos_root_min")
                        if idx >= 0:
                            tail = d[idx + len("ohos_root_min"):]
                            resolved = os.path.normpath(
                                os.path.join(ohos_root, tail.lstrip("/\\"))
                            )
                            if os.path.isdir(resolved):
                                include_dirs.add(resolved)
                                continue

        return [f"-I{d}" for d in sorted(include_dirs)]

    @staticmethod
    def _write_minimal_cargo_toml(output_dir: str):
        """Generate Cargo.toml + module files for the transpiled output.

        Handles arbitrarily nested ``.rs`` files by creating intermediate
        ``mod.rs`` files where needed.  Rust ``mod`` declarations do not
        support ``::`` paths, so we cannot write ``pub mod a::b;`` —
        instead we emit ``pub mod a;`` and ensure ``a/mod.rs`` (or
        ``a.rs``) exists with ``pub mod b;``.
        """
        src_dir = os.path.join(output_dir, "src")
        os.makedirs(src_dir, exist_ok=True)

        # Recursively collect .rs files (skip lib.rs / main.rs / target/)
        # Store as (directory_chain, file_stem) pairs.
        entries = []
        for root, dirs, files in os.walk(output_dir):
            dirs[:] = [d for d in dirs if d != "target" and not d.startswith(".")]
            for fn in files:
                if fn.endswith(".rs") and fn not in ("lib.rs", "main.rs"):
                    rel_dir = os.path.relpath(root, src_dir)
                    stem = fn[:-3]
                    chain = rel_dir.split(os.sep) if rel_dir != "." else []
                    entries.append((chain, stem))

        # Ensure intermediate mod.rs files exist and parent modules know
        # about child directories.
        all_dirs = set()
        for chain, _stem in entries:
            for depth in range(len(chain) + 1):
                all_dirs.add(tuple(chain[:depth]))

        for dchain in sorted(all_dirs, key=lambda c: len(c)):
            if not dchain:
                continue  # skip root — handled by lib.rs
            # Directory at src/a/b/ — ensure parent declares "pub mod b;"
            parent_chain = dchain[:-1]
            child_name = dchain[-1]
            if parent_chain:
                parent_mod = os.path.join(src_dir, *parent_chain, "mod.rs")
            else:
                parent_mod = os.path.join(src_dir, "lib.rs")
            os.makedirs(os.path.dirname(parent_mod), exist_ok=True)
            marker = f"pub mod {child_name};"
            existing = ""
            if os.path.exists(parent_mod):
                existing = open(parent_mod, encoding="utf-8", errors="ignore").read()
            if marker not in existing:
                with open(parent_mod, "a", encoding="utf-8") as f:
                    f.write(marker + "\n")

        # Ensure every subdirectory has a mod.rs
        for dchain in all_dirs:
            if not dchain:
                continue
            mod_rs = os.path.join(src_dir, *dchain, "mod.rs")
            os.makedirs(os.path.dirname(mod_rs), exist_ok=True)
            if not os.path.exists(mod_rs):
                with open(mod_rs, "w", encoding="utf-8") as f:
                    f.write("// Auto-generated module declarations.\n")

        # Write leaf pub mod declarations
        for chain, stem in entries:
            if chain:
                parent_mod = os.path.join(src_dir, *chain, "mod.rs")
            else:
                parent_mod = os.path.join(src_dir, "lib.rs")
            marker = f"pub mod {stem};"
            existing = open(parent_mod, encoding="utf-8", errors="ignore").read() if os.path.exists(parent_mod) else ""
            if marker not in existing:
                with open(parent_mod, "a", encoding="utf-8") as f:
                    f.write(marker + "\n")

        cargo_path = os.path.join(output_dir, "Cargo.toml")
        if not os.path.exists(cargo_path):
            with open(cargo_path, "w", encoding="utf-8") as f:
                f.write("""[package]
name = "c2rust-output"
version = "0.1.0"
edition = "2021"

[dependencies]
""")


# ======================================================================
# Standalone post-process (extracted from His2Trans incremental_translate.py)
# ======================================================================

def _postprocess_c2rust_output_for_stable(rs_text: str) -> str:
    """Make c2rust output compile on stable Rust toolchains.

    c2rust commonly emits:
      - ``#![feature(extern_types)]`` + ``extern "C" { pub type Foo; }``
      - ``#[no_mangle]`` on functions (can collide with linked C objects)
      - ``extern type`` declarations (unstable on stable Rust)

    We:
      - remove all ``#![feature(...)]`` lines
      - remove ``#[no_mangle]`` attributes
      - rewrite ``extern type`` declarations into opaque zero-sized structs
      - promote top-level fns to ``pub``
      - replace ``compile_error!`` with ``panic!``
    """
    if not rs_text:
        return ""

    lines_in = rs_text.splitlines()
    stripped: List[str] = []
    saw_allow_deref_nullptr = False
    for ln in lines_in:
        s = ln.strip()
        if re.match(r"^#!\s*\[\s*feature\s*\(.*\)\s*\]\s*$", s):
            continue
        if re.match(r"^#\s*\[\s*no_mangle\s*\]\s*$", s):
            continue
        if re.match(r"^#!\s*\[\s*allow\s*\(\s*deref_nullptr\s*\)\s*\]\s*$", s):
            saw_allow_deref_nullptr = True
        stripped.append(ln)

    if not saw_allow_deref_nullptr:
        stripped.insert(0, "#![allow(deref_nullptr)]")

    text = "\n".join(stripped) + ("\n" if rs_text.endswith("\n") else "")

    # Collect already-defined types to avoid duplicate definitions
    defined: Set[str] = set()
    try:
        defined.update(re.findall(r"\bpub\s+(?:struct|enum|union)\s+([A-Za-z_]\w*)\b", text))
        defined.update(re.findall(r"\bpub\s+type\s+([A-Za-z_]\w*)\s*=", text))
    except Exception:
        defined = set()

    extern_type_names: List[str] = []
    out_lines: List[str] = []
    i = 0
    in_extern = False
    extern_buf: List[str] = []

    def _flush_extern_block(buf: List[str]) -> None:
        nonlocal out_lines
        if not buf:
            return
        body_lines = [
            b for b in buf
            if b.strip() not in ('extern "C" {', 'unsafe extern "C" {', "}")
        ]
        has_real = any(
            b.strip() and not b.strip().startswith("//")
            for b in body_lines
        )
        if has_real:
            out_lines.extend(buf)

    while i < len(stripped):
        line = stripped[i]
        s = line.strip()
        if not in_extern and s in {'extern "C" {', 'unsafe extern "C" {'}:
            in_extern = True
            extern_buf = [line]
            i += 1
            continue
        if in_extern:
            if s == "}":
                extern_buf.append(line)
                in_extern = False
                _flush_extern_block(extern_buf)
                extern_buf = []
                i += 1
                continue
            m = re.match(r"^pub\s+type\s+([A-Za-z_]\w*)\s*;\s*$", s)
            if m:
                extern_type_names.append(m.group(1))
                i += 1
                continue
            extern_buf.append(line)
            i += 1
            continue
        out_lines.append(line)
        i += 1

    # Insert opaque structs for extern types (dedup + avoid redefinition)
    uniq: List[str] = []
    seen: Set[str] = set()
    for n in extern_type_names:
        if not n or n in seen or n in defined:
            continue
        seen.add(n)
        uniq.append(n)

    if not uniq:
        out_text = "\n".join(out_lines) + ("\n" if rs_text.endswith("\n") else "")
        try:
            out_text = re.sub(
                r'(?m)^(\s*)unsafe\s+extern\s+"C"\s+fn\s+',
                r'\1pub unsafe extern "C" fn ', out_text,
            )
            out_text = re.sub(
                r'(?m)^(\s*)extern\s+"C"\s+fn\s+',
                r'\1pub extern "C" fn ', out_text,
            )
            out_text = re.sub(
                r'(?m)^(\s*)unsafe\s+fn\s+([A-Za-z_])',
                r'\1pub unsafe fn \2', out_text,
            )
            out_text = re.sub(
                r'(?m)^(\s*)fn\s+([A-Za-z_])',
                r'\1pub fn \2', out_text,
            )
            out_text = re.sub(r'pub\s+pub\s+', 'pub ', out_text)
        except Exception:
            pass
        out_text = out_text.replace("compile_error!(", "panic!(")
        return out_text

    opaque_defs: List[str] = []
    opaque_defs.append("// === C2R_C2RUST_EXTERN_TYPES_BEGIN ===")
    opaque_defs.append(
        "// Auto-generated: downgraded c2rust `extern type` "
        "to stable-safe opaque structs."
    )
    for n in uniq:
        opaque_defs.append("#[repr(C)]")
        opaque_defs.append("#[derive(Copy, Clone)]")
        opaque_defs.append(f"pub struct {n} {{")
        opaque_defs.append("    _unused: [u8; 0],")
        opaque_defs.append("}")
        opaque_defs.append("")
    opaque_defs.append("// === C2R_C2RUST_EXTERN_TYPES_END ===")
    opaque_lines = opaque_defs + [""]

    # Insert after leading inner attributes
    insert_at = 0
    i = 0
    while i < len(out_lines) and out_lines[i].lstrip().startswith("#!["):
        bracket_balance = 0
        j = i
        while j < len(out_lines):
            ln = out_lines[j] or ""
            bracket_balance += ln.count("[")
            bracket_balance -= ln.count("]")
            j += 1
            if bracket_balance <= 0:
                break
        i = j
    insert_at = i
    out_lines = out_lines[:insert_at] + opaque_lines + out_lines[insert_at:]
    out_text = "\n".join(out_lines) + ("\n" if rs_text.endswith("\n") else "")

    try:
        out_text = re.sub(
            r'(?m)^(\s*)unsafe\s+extern\s+"C"\s+fn\s+',
            r'\1pub unsafe extern "C" fn ', out_text,
        )
        out_text = re.sub(
            r'(?m)^(\s*)extern\s+"C"\s+fn\s+',
            r'\1pub extern "C" fn ', out_text,
        )
        out_text = re.sub(
            r'(?m)^(\s*)unsafe\s+fn\s+([A-Za-z_])',
            r'\1pub unsafe fn \2', out_text,
        )
        out_text = re.sub(
            r'(?m)^(\s*)fn\s+([A-Za-z_])',
            r'\1pub fn \2', out_text,
        )
        out_text = re.sub(r'pub\s+pub\s+', 'pub ', out_text)
    except Exception:
        pass
    out_text = out_text.replace("compile_error!(", "panic!(")
    return out_text


# ======================================================================
# Quality metrics
# ======================================================================

def _compute_file_metrics(original: str, postprocessed: str) -> dict:
    """Compute quality metrics for a single .rs file."""

    def _count(pattern: str, text: str) -> int:
        return len(re.findall(pattern, text))

    unsafe_fn_re = re.compile(r'\bunsafe\s+(?:extern\s+"C"\s+)?fn\b')

    # Count lines inside unsafe fn bodies
    unsafe_body_lines = 0
    for m in re.finditer(
        r'\bunsafe\s+(?:extern\s+"C"\s+)?fn\s+\w+[^{]*\{', postprocessed
    ):
        pos = m.end() - 1
        depth = 1
        i = pos + 1
        while i < len(postprocessed) and depth > 0:
            if postprocessed[i] == '{':
                depth += 1
            elif postprocessed[i] == '}':
                depth -= 1
            i += 1
        unsafe_body_lines += postprocessed[m.start():i].count('\n') + 1

    return {
        "total_lines": len(postprocessed.splitlines()),
        "unsafe_blocks": (
            _count(r"\bunsafe\s*\{", postprocessed)
            + len(unsafe_fn_re.findall(postprocessed))
        ),
        "unsafe_functions": len(unsafe_fn_re.findall(postprocessed)),
        "unsafe_body_lines": unsafe_body_lines,
        "extern_c_functions": _count(r'extern\s+"C"\s+fn', postprocessed),
        "raw_ptr_types": _count(r"\*const\b|\*mut\b", postprocessed),
        "feature_gates_removed": _count(
            r"(?m)^#!\s*\[\s*feature\s*\(.*\)\s*\]", original
        ),
        "extern_types_downgraded": _count(
            r"// === C2R_C2RUST_EXTERN_TYPES_BEGIN ===", postprocessed
        ),
    }
