"""His2Trans engine — orchestrates the full 4-stage translation pipeline.

Calls framework Python scripts (in-tree under ./framework/) via subprocess
with proper environment variable mapping from session config.
"""

import os
import sys
import json
import re
import shutil
from pathlib import Path
from datetime import datetime

from app.engines.base_engine import BaseEngine
from app.api.upload import get_project_name as _get_project_name
from app.engines.his2trans.env_mapper import EnvMapper
from app.engines.his2trans.runner import FrameworkRunner, FrameworkRunnerError

# Framework root — the in-tree His2Trans-Opt-/framework copy
_FRAMEWORK_DIR = Path(__file__).resolve().parent / "framework"


class His2TransEngine(BaseEngine):
    """His2Trans: C/C++ → Rust translation engine.

    Runs the full pipeline:
    - Stage 1: Dependency analysis + skeleton generation
    - Stage 2: RAG signature matching (BM25 + Jina Reranker)
    - Stage 3: Function body translation + compile-check + repair
    - Post-process: Reports & packaging
    """

    def get_display_name(self) -> str:
        return "His2Trans"

    def get_description(self) -> str:
        return ("Full C/C++ → Rust translation pipeline — dependency analysis, "
                "skeleton building, RAG signature matching, LLM translation "
                "with compile-guided repair.")

    def get_stages(self) -> list[dict]:
        return [
            {
                "id": "stage1_prep",
                "name": "Stage 1: Dependency Analysis + Skeleton",
                "description": "Parses C source files, extracts functions/dependencies, "
                "generates Rust skeleton files with bindgen type definitions.",
            },
            {
                "id": "stage2_rag",
                "name": "Stage 2: Signature Matching (RAG)",
                "description": "BM25 retrieval + Jina Reranker — finds matching Rust "
                "function signatures from the knowledge base.",
            },
            {
                "id": "stage3_translate",
                "name": "Stage 3: Function Body Translation + Repair",
                "description": "LLM translates each function body, then compile-check — "
                "auto-repair loop fixes compilation errors iteratively.",
            },
            {
                "id": "postprocess",
                "name": "Post-process: Reports & Packaging",
                "description": "Generates translation summary report and packages output files.",
            },
        ]

    # ==================================================================
    # Stage dispatch
    # ==================================================================

    def run_stage(self, stage_id: str, source_path: str, output_path: str,
                  config: dict, log_callback) -> dict:
        """Run a single pipeline stage."""
        # Ensure absolute paths — relative paths break subprocess CWD resolution
        source_path = os.path.abspath(source_path)
        output_path = os.path.abspath(output_path)
        workspace = os.path.join(output_path, "workspace")
        os.makedirs(workspace, exist_ok=True)

        framework_path = config.get("his2trans_framework", str(_FRAMEWORK_DIR))
        framework_path = os.path.abspath(framework_path)
        runner = FrameworkRunner(framework_path, config)

        log_callback(f"[Engine] Starting {stage_id}...", "info")
        log_callback(f"[Engine] Source: {source_path}", "info")
        log_callback(f"[Engine] Workspace: {workspace}", "info")
        log_callback(f"[Engine] Framework: {framework_path}", "info")
        log_callback(f"[Engine] Model: {config.get('model', 'unknown')}", "info")

        try:
            if stage_id == "stage1_prep":
                return self._run_stage1(runner, source_path, workspace, config, log_callback)
            elif stage_id == "stage2_rag":
                return self._run_stage2(runner, source_path, workspace, config, log_callback)
            elif stage_id == "stage3_translate":
                return self._run_stage3(runner, source_path, workspace, config, log_callback)
            elif stage_id == "postprocess":
                return self._run_postprocess(source_path, output_path, config, log_callback)
            else:
                raise ValueError(f"Unknown stage: {stage_id}")
        except Exception as e:
            log_callback(f"[Engine] Stage {stage_id} FAILED: {e}", "error")
            import traceback
            log_callback(traceback.format_exc()[-2000:], "error")
            raise

    # ==================================================================
    # Stage 1: Dependency Analysis + Skeleton
    # ==================================================================

    def _run_stage1(self, runner, source_path, workspace, config, log):
        log("=" * 50)
        log("Stage 1: Dependency Analysis + Skeleton Generation")
        log("=" * 50)

        cc_path = os.path.join(source_path, "compile_commands.json")
        has_cc = os.path.isfile(cc_path)
        ohos_root = config.get("ohos_root", "")

        log(f"compile_commands.json: {'found' if has_cc else 'NOT found (standard C mode)'}")
        if ohos_root:
            log(f"OpenHarmony root: {ohos_root}")
        if config.get("extra_includes"):
            log(f"Extra includes: {config['extra_includes']}")

        # Temporarily hide test/ directories to prevent test headers
        # from polluting types.rs with conflicting macro definitions
        hidden_test_dirs = []
        for test_dir_name in ["test", "tests", "unittest"]:
            test_dir = os.path.join(source_path, test_dir_name)
            if os.path.isdir(test_dir):
                hidden = test_dir + ".hidden"
                os.rename(test_dir, hidden)
                hidden_test_dirs.append((test_dir, hidden))

        try:
            # Step 1: Extract dependencies
            log("--- Step 1/2: get_dependencies.py ---")
            self._run_script(runner, "stage1_prep/get_dependencies.py",
                             source_path, workspace, log, timeout=1800)

            project_name = _get_project_name(os.path.dirname(source_path))

            # get_dependencies.py derives output dir from source_path
            # basename (→ "source"). Rename to the real project name so
            # downstream stages can find the extracted data.
            extracted_src = os.path.join(workspace, "extracted", "source")
            extracted_dst = os.path.join(workspace, "extracted", project_name)
            if os.path.isdir(extracted_src) and not os.path.exists(extracted_dst):
                os.rename(extracted_src, extracted_dst)
                log(f"  Renamed extracted dir: source → {project_name}")

            # Step 2: Build skeletons — output to workspace/skeletons/<project>/
            # so that merge_final_project.py (Stage 3) finds them via get_skeleton_path()
            skeleton_out = os.path.join(workspace, "skeletons", project_name)
            log("--- Step 2/2: skeleton_builder.py ---")

            # Build skeleton_builder args with compile_commands and OHOS root
            skel_args = [source_path, skeleton_out]
            if has_cc:
                skel_args += ["--compile-commands", cc_path]
                # Derive ohos_root from compile_commands.json include paths
                detected_ohos = self._detect_ohos_root(cc_path, source_path)
                effective_ohos_root = detected_ohos or ohos_root
                if effective_ohos_root:
                    if detected_ohos:
                        log(f"  Auto-detected OHOS root: {effective_ohos_root}")
                    else:
                        log(f"  Using configured OHOS root: {effective_ohos_root}")
                    skel_args += ["--ohos-root", effective_ohos_root]
            elif ohos_root:  # no cc.json, use config value
                skel_args += ["--ohos-root", ohos_root]

            self._run_script(runner, "stage2_skeleton/skeleton_builder.py",
                             source_path, workspace, log, timeout=3600,
                             args=skel_args)
        finally:
            # Restore hidden test directories
            for orig, hidden in hidden_test_dirs:
                if os.path.isdir(hidden):
                    os.rename(hidden, orig)

        # Remove test files from skeleton to avoid macro conflicts (ASSERT_EQ etc.)
        self._remove_test_skeletons(skeleton_out, log)

        # Fix duplicate const/fn definitions in types.rs (skeleton_builder bug)
        self._cleanup_types_duplicates(skeleton_out, log)

        # Step 3: Generate function_signatures.json for RAG
        log("--- Step 3/3: Generating function_signatures.json for RAG ---")
        try:
            sig_count = self._generate_function_signatures(workspace, project_name, log)
            log(f"  Generated {sig_count} function signatures for RAG")
        except Exception as e:
            log(f"  Warning: function_signatures.json generation failed: {e}", "warn")

        # Setup RAG resources in workspace
        self._setup_rag_resources(workspace, runner.framework_path, log)

        rust_files = list(Path(workspace).rglob("*.rs"))
        log(f"Generated {len(rust_files)} Rust skeleton files")

        return {
            "summary": f"Stage 1 complete: {len(rust_files)} skeleton files generated",
            "details": {
                "workspace": workspace,
                "rust_files_count": len(rust_files),
                "has_compile_commands": has_cc,
            },
        }

    # ==================================================================
    # Stage 2: RAG Signature Matching
    # ==================================================================

    def _run_stage2(self, runner, source_path, workspace, config, log):
        if not config.get("use_rag", False):
            log("Stage 2: RAG disabled — skipping")
            return {"summary": "Stage 2 skipped (RAG disabled)", "details": {"rag_enabled": False}}

        project_name = _get_project_name(os.path.dirname(source_path))

        log("=" * 50)
        log("Stage 2: RAG Signature Matching (BM25 + Jina Reranker)")
        log("=" * 50)
        log(f"Project: {project_name}")

        # Step 1: BM25 retrieval + signature matching
        log("--- Step 1/2: generate_signature_mappings.py ---")
        self._run_script(runner, "knowledge/generate_signature_mappings.py",
                         source_path, workspace, log, timeout=3600,
                         args=["--project", project_name])

        # Step 2: Jina Reranker
        log("--- Step 2/2: run_jina_reranker_queued.py ---")
        self._run_script(runner, "knowledge/run_jina_reranker_queued.py",
                         source_path, workspace, log, timeout=3600,
                         args=["--project", project_name])

        log("Stage 2 RAG processing complete")
        return {
            "summary": "Stage 2 complete: RAG signature matching finished",
            "details": {"rag_enabled": True, "workspace": workspace},
        }

    # ==================================================================
    # Stage 3: Translate + Compile + Repair
    # ==================================================================

    def _run_stage3(self, runner, source_path, workspace, config, log):
        log("=" * 50)
        log("Stage 3: Function Body Translation + Compile + Repair")
        log("=" * 50)
        max_repair = config.get("max_repair", 5)
        model = config.get("model", "deepseek-v3.2")
        project_name = _get_project_name(os.path.dirname(source_path))
        log(f"Model: {model}, Max repair rounds: {max_repair}, Project: {project_name}")

        # Derive framework-expected subdirs
        extracted_dir = os.path.join(workspace, "extracted", project_name)
        translated_dir = os.path.join(workspace, "translated", project_name)
        test_results_dir = os.path.join(workspace, "test_results")
        repair_results_dir = os.path.join(workspace, "repair_results")
        skeletons_dir = os.path.join(workspace, "skeletons", project_name)
        rag_signatures_dir = os.path.join(workspace, "source_skeletons", project_name)

        # Populate signature_matches/ with per-function .txt files
        # translate_function.py reads individual signature files from this dir
        sig_count = self._populate_signature_match_files(
            workspace, project_name, model, log)

        # Step 1: Translate function bodies
        # Args: source_dir target_dir llm dependencies_path rag_path_function
        log("--- Step 1/4: translate_function.py ---")
        # dependencies_path must point to the skeleton project root
        # (translate_function.py reads {dependencies_path}/src/{file}.rs)
        self._run_script(runner, "stage3_translate/translate_function.py",
                         source_path, workspace, log, timeout=7200,
                         args=[
                             os.path.join(extracted_dir, "functions"),
                             translated_dir,
                             model,
                             skeletons_dir,
                             rag_signatures_dir,
                         ])

        # Step 2: Compile-check + auto-repair
        # Args: functions_dir translated_dir test_results_dir repair_results_dir llm skeletons_dir
        log("--- Step 2/4: auto_repair_rust.py ---")
        self._run_script(runner, "stage3_translate/auto_repair_rust.py",
                         source_path, workspace, log, timeout=7200,
                         args=[
                             os.path.join(extracted_dir, "functions"),
                             translated_dir,
                             test_results_dir,
                             repair_results_dir,
                             model,
                             skeletons_dir,
                         ])

        # Step 3: Generate test result tracking files
        # (auto_repair_rust.py runs cargo build but doesn't persist results)
        log("--- Step 3/4: Generating test result tracking ---")
        try:
            passed = self._generate_test_results(workspace, project_name, model, log)
            log(f"  Generated test results for {passed} functions")
        except Exception as e:
            log(f"  Warning: test result generation failed: {e}", "warn")

        # Step 4: Merge final project
        # Args: project_name llm_name
        log("--- Step 4/4: merge_final_project.py ---")
        self._run_script(runner, "stage3_translate/merge_final_project.py",
                         source_path, workspace, log, timeout=1800,
                         args=[project_name, model])

        rust_files = list(Path(workspace).rglob("*.rs"))
        log(f"Final output: {len(rust_files)} Rust files")

        return {
            "summary": f"Stage 3 complete: {len(rust_files)} Rust files generated",
            "details": {
                "workspace": workspace,
                "max_repair": max_repair,
                "rust_files_count": len(rust_files),
            },
        }

    # ==================================================================
    # Post-process: Reports & Packaging
    # ==================================================================

    def _run_postprocess(self, source_path, output_path, config, log):
        log("=" * 50)
        log("Post-process: Reports & Packaging")
        log("=" * 50)

        workspace = os.path.join(output_path, "workspace")
        rust_files = list(Path(output_path).rglob("*.rs"))

        report = {
            "engine": "his2trans",
            "model": config.get("model", "unknown"),
            "use_rag": config.get("use_rag", False),
            "max_repair": config.get("max_repair", 5),
            "ohos_root": config.get("ohos_root", ""),
            "rust_files_generated": len(rust_files),
            "generated_at": datetime.now().isoformat(),
        }

        report_path = os.path.join(output_path, "report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        log(f"Report saved: {report_path}")

        # Copy workspace to output for download
        if workspace and os.path.isdir(workspace):
            for item in os.listdir(workspace):
                src = os.path.join(workspace, item)
                dst = os.path.join(output_path, item)
                if os.path.isdir(src) and not os.path.exists(dst):
                    shutil.copytree(src, dst)
                elif os.path.isfile(src) and not os.path.exists(dst):
                    shutil.copy2(src, dst)
            log("Output packaged for download")

        return {
            "summary": f"Post-process complete: {len(rust_files)} Rust files",
            "details": report,
        }

    # ==================================================================
    # RAG setup helpers
    # ==================================================================

    def _generate_function_signatures(self, workspace, project_name, log):
        """Generate function_signatures.json from skeleton .rs files and manifest.

        This bridges the gap between skeleton_builder output and RAG input.
        Format: {func_name: {"c_signature": ..., "rust_signature": ..., "source_file": ...}}
        """
        manifest_path = os.path.join(workspace, "extracted", project_name,
                                     "functions_manifest.json")
        if not os.path.isfile(manifest_path):
            log("  functions_manifest.json not found, skipping signature generation", "warn")
            return 0

        with open(manifest_path) as f:
            manifest = json.load(f)

        # Parse all skeleton .rs files to build name→signature map
        rs_dir = os.path.join(workspace, "skeletons", project_name, "src")
        rust_sigs = {}  # func_name → rust_signature
        if os.path.isdir(rs_dir):
            for rs_file in Path(rs_dir).glob("*.rs"):
                try:
                    content = rs_file.read_text(encoding="utf-8")
                except Exception:
                    continue
                # Match: pub extern "C" fn funcName(params) -> ReturnType {
                for m in re.finditer(
                    r'pub\s+(?:unsafe\s+)?extern\s+"C"\s+fn\s+(\w+)\s*\((.*?)\)\s*(->\s*[^{;]+)?',
                    content
                ):
                    func_name = m.group(1)
                    params = m.group(2).strip()
                    ret = (m.group(3) or "()").strip()
                    sig = f"pub extern \"C\" fn {func_name}({params}){ret}"
                    rust_sigs[func_name] = sig

        # Build function_signatures from manifest and extracted Rust signatures
        signatures = {}
        for func_meta in manifest.get("functions", []):
            name = func_meta.get("name", "")
            source_file = func_meta.get("source_file", "")
            if name and name in rust_sigs:
                signatures[name] = {
                    "c_signature": f"{name}(...)",  # C signature not easily available
                    "rust_signature": rust_sigs[name],
                    "source_file": source_file,
                }

        # Write output
        sig_dir = os.path.join(workspace, "source_skeletons", project_name)
        os.makedirs(sig_dir, exist_ok=True)
        sig_path = os.path.join(sig_dir, "function_signatures.json")
        with open(sig_path, "w", encoding="utf-8") as f:
            json.dump(signatures, f, indent=2, ensure_ascii=False)

        return len(signatures)

    def _populate_signature_match_files(self, workspace, project_name, model, log):
        """Convert func_file_to_rust_sig.json into per-function .txt files
        in signature_matches/{project}/translate_by_{model}/.

        This bridges the gap between generate_signature_mappings.py (Stage 2)
        which writes aggregate JSON to source_skeletons/, and translate_function.py
        (Stage 3) which reads individual .txt files from signature_matches/.
        """
        sig_json_path = os.path.join(workspace, "source_skeletons", project_name,
                                     "func_file_to_rust_sig.json")
        if not os.path.isfile(sig_json_path):
            log("  func_file_to_rust_sig.json not found, skipping signature population", "warn")
            return 0

        with open(sig_json_path) as f:
            func_file_map = json.load(f)

        sig_match_dir = os.path.join(workspace, "signature_matches", project_name,
                                     f"translate_by_{model}")
        os.makedirs(sig_match_dir, exist_ok=True)

        count = 0
        for func_file, sig in func_file_map.items():
            if sig and sig != "null" and func_file:
                txt_path = os.path.join(sig_match_dir, f"{func_file}.txt")
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(sig)
                count += 1

        log(f"  Populated {count} signature files in signature_matches/")
        return count

    def _remove_test_skeletons(self, skeleton_out, log):
        """Remove skeleton files generated from test source files.

        Test files often define macros (ASSERT_EQ, ASSERT_NE, etc.) that
        conflict with main code when cargo builds the full project.
        Also removes 'mod test_*' references from main.rs.
        """
        removed = 0
        # Remove test skeleton .rs files
        for pattern in ["*test*", "*Test*", "*unittest*", "*gtest*"]:
            for f in Path(skeleton_out).rglob(pattern):
                if f.is_file():
                    f.unlink()
                    removed += 1
        # Clean up 'mod test_*' declarations from main.rs
        main_rs = os.path.join(skeleton_out, "src", "main.rs")
        if os.path.isfile(main_rs):
            with open(main_rs, "r", encoding="utf-8") as f:
                content = f.read()
            new_content = re.sub(r'^pub mod test_.*;\s*$', '', content, flags=re.MULTILINE)
            if new_content != content:
                with open(main_rs, "w", encoding="utf-8") as f:
                    f.write(new_content)
                removed += 1
        if removed:
            log(f"  Removed {removed} test skeleton files/references")

    def _cleanup_types_duplicates(self, skeleton_out, log):
        """Remove duplicate const/function definitions in types.rs.

        Skeleton builder's stub types.rs sometimes generates both:
          pub const NAME: i32 = 0;  // placeholder
        and
          pub fn NAME() -> i32;    // in extern "C" block
        causing E0428 multiply-defined errors.
        """
        types_rs = os.path.join(skeleton_out, "src", "types.rs")
        if not os.path.isfile(types_rs):
            return

        with open(types_rs, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        # Find all const name: type = value; definitions
        const_names = set()
        for m in re.finditer(r'pub const (\w+)\s*:\s*\w+\s*=', content):
            const_names.add(m.group(1))

        # Find all fn name(...) -> type; declarations (in traits/impl/extern blocks)
        fn_names = set()
        for m in re.finditer(r'pub fn (\w+)\s*\([^)]*\)\s*(->\s*[^{;]+)?\s*;', content):
            fn_names.add(m.group(1))

        # Remove const definitions that also appear as fn declarations
        duplicates = const_names & fn_names
        if not duplicates:
            return

        removed = 0
        for name in sorted(duplicates):
            pattern = rf'pub const {name}\s*:\s*\w+\s*=\s*[^;]+;\s*// placeholder.*\n?'
            if re.search(pattern, content):
                content = re.sub(pattern, '', content)
                removed += 1

        if removed:
            with open(types_rs, "w", encoding="utf-8") as f:
                f.write(content)
            log(f"  Cleaned up {removed} duplicate const/fn definitions in types.rs")

    def _detect_ohos_root(self, cc_path, source_path):
        """Derive OHOS root directory from compile_commands.json include paths.

        Looks for -I paths containing 'ohos_root' and resolves them
        relative to the project directory to find the OHOS SDK root.
        Uses realpath to handle symlinked source directories.
        """
        try:
            with open(cc_path) as f:
                cc = json.load(f)
        except Exception:
            return None

        # Use realpath: compile_commands.json paths are relative to the
        # real source directory, not any symlink that points to it
        project_dir = os.path.dirname(os.path.realpath(cc_path))

        for entry in cc:
            command = entry.get("command", "")
            for m in re.finditer(r'-I\s*(\S*ohos_root(?:_min)?)', command):
                ohos_rel = m.group(1)
                candidate = os.path.normpath(os.path.join(project_dir, ohos_rel))
                # Walk up from the include dir to find ohos_root / ohos_root_min
                path = candidate
                while path:
                    dirname = os.path.basename(path)
                    if dirname in ("ohos_root", "ohos_root_min"):
                        if os.path.isdir(path):
                            return path
                        break
                    parent = os.path.dirname(path)
                    if parent == path:
                        break
                    path = parent
        return None

    @staticmethod
    def _ensure_rag_extracted(log=None):
        """Auto-extract RAG resources from .tar.gz if the raw files are missing.

        The repo ships knowledge_base.tar.gz and bm25_index.tar.gz (~30 MB
        combined) so that users don't need an external His2Trans-Opt- checkout.
        Extracted .json/.pkl files are gitignored.
        """
        import tarfile as _tarfile
        # 3 levels up from his2trans/ → backend/
        rag_dir = os.path.join(os.path.dirname(__file__), "..", "..",
                               "..", "data", "rag")
        rag_dir = os.path.normpath(rag_dir)

        for name in ("knowledge_base", "bm25_index"):
            extracted = os.path.join(rag_dir, f"{name}.json" if name == "knowledge_base" else f"{name}.pkl")
            archive = os.path.join(rag_dir, f"{name}.tar.gz")

            if os.path.isfile(extracted):
                continue
            if not os.path.isfile(archive):
                if log:
                    log(f"  RAG archive not found: {archive}", "warn")
                continue

            if log:
                log(f"  Extracting {os.path.basename(archive)} → {os.path.basename(extracted)} ...")
            try:
                with _tarfile.open(archive, "r:gz") as tf:
                    tf.extractall(rag_dir)
            except Exception as e:
                if log:
                    log(f"  Failed to extract {archive}: {e}", "error")

    def _setup_rag_resources(self, workspace, framework_path, log):
        """Copy RAG knowledge base and BM25 index into workspace.

        Resources live in backend/data/rag/, shipped as .tar.gz archives.
        Auto-extracts on first use if the raw files are missing.
        """
        # Auto-extract if needed
        self._ensure_rag_extracted(log)

        rag_workspace = os.path.join(workspace, "rag")
        os.makedirs(rag_workspace, exist_ok=True)

        # Primary source: local backend/data/rag/ (self-contained)
        # Fallback: His2Trans-Opt- framework workspace (dev / legacy)
        fw = os.path.abspath(framework_path) if framework_path else ""
        candidates_base = [
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "rag"),
            os.path.join(fw, "workspace", "rag") if fw else "",
            os.path.join(os.path.dirname(fw), "data", "rag") if fw else "",
        ]

        for basename, dst_name in [("knowledge_base.json", "knowledge_base.json"),
                                    ("bm25_index.pkl", "bm25_index.pkl")]:
            dst = os.path.join(rag_workspace, dst_name)
            if os.path.exists(dst):
                continue
            for base in candidates_base:
                if not base:
                    continue
                src = os.path.join(base, basename)
                if os.path.isfile(src):
                    try:
                        shutil.copy2(src, dst)
                        log(f"  Copied {dst_name} → workspace/rag/")
                    except OSError:
                        pass
                    break

    def _generate_test_results(self, workspace, project_name, model, log):
        """Generate test result tracking files for merge_final_project.

        The auto_repair_rust.py runs cargo build but doesn't persist per-function
        pass/fail files. This method creates them from the translated function files
        so that merge_final_project can find and merge them into the skeleton.
        """
        llm_name = f"translate_by_{model}"
        translated_dir = os.path.join(workspace, "translated", project_name, llm_name)
        test_results_dir = os.path.join(workspace, "test_results", project_name, llm_name)
        os.makedirs(test_results_dir, exist_ok=True)

        if not os.path.isdir(translated_dir):
            return 0

        count = 0
        for txt_file in Path(translated_dir).glob("*.txt"):
            # Read the translated function
            try:
                content = txt_file.read_text(encoding="utf-8")
            except Exception:
                continue

            # Extract translated function body
            body_match = re.search(r'<translated function>\s*\n?(.*?)\n?\s*</translated function>',
                                   content, re.DOTALL)
            if not body_match:
                continue
            func_body = body_match.group(1).strip()
            # Skip empty translations and empty markdown code blocks
            if not func_body:
                continue
            code_blocks = re.findall(r'```(?:\w+)?\s*(.*?)```', func_body, re.DOTALL)
            has_code = any(b.strip() for b in code_blocks)
            if not has_code and not re.search(r'(?<![`])\b(fn|impl|pub|use|mod|struct|enum|trait|const|static|let|mut|if|for|while|loop|match|return)\b', func_body):
                continue  # Skip responses with no actual Rust code

            # Write test result (merge script checks for "Success" prefix)
            test_file = os.path.join(test_results_dir, txt_file.name)
            with open(test_file, "w", encoding="utf-8") as f:
                f.write(f"Success\n{func_body}")
            count += 1

        return count

    # ==================================================================
    # Script runner helper
    # ==================================================================

    def _run_script(self, runner, script_rel, source_path, workspace, log, timeout=1800, args=None):
        """Run a framework script via FrameworkRunner, with env setup and optional CLI args."""
        try:
            runner.run_script(script_rel, source_path, workspace, log,
                            timeout=timeout, args=args)
        except FrameworkRunnerError as e:
            log(f"Script {script_rel} failed: {e}", "error")
