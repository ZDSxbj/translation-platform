"""His2Trans Engine Integration Tests.

Tests the full translation pipeline with:
1. An OHOS project (shared__541f4e547bdb) — uses compile_commands.json + RAG
2. A standard C project — no compile_commands.json, no RAG

Parameters optimized for best results:
- RAG enabled (for OHOS)
- JINA_MIN_MEMORY_GB=4.0 (avoids GPU wait loop)
- Max repair rounds: 8
- Model: deepseek-v3.2
- Temperature: 0.0

Usage:
    pytest tests/test_his2trans_engine.py -v          # All tests
    pytest tests/test_his2trans_engine.py -v -k ohos  # OHOS-only
    pytest tests/test_his2trans_engine.py -v -k standard  # Standard C only
    pytest tests/test_his2trans_engine.py -v -k env   # Env mapping tests
"""

import os
import sys
import json
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.engines.his2trans.engine import His2TransEngine
from app.engines.his2trans.runner import FrameworkRunner, FrameworkRunnerError
from app.engines.his2trans.env_mapper import EnvMapper


# ------------------------------------------------------------------
# Engine basics (always run, no framework needed)
# ------------------------------------------------------------------

class TestEngineBasics:
    """Tests that work without the His2Trans framework installed."""

    def test_engine_stages(self, engine):
        """Engine should return 4 stages with correct IDs."""
        stages = engine.get_stages()
        stage_ids = [s["id"] for s in stages]
        assert stage_ids == ["stage1_prep", "stage2_rag", "stage3_translate", "postprocess"]

    def test_get_display_name(self, engine):
        assert engine.get_display_name() == "His2Trans"

    def test_get_description(self, engine):
        desc = engine.get_description()
        assert "C/C++" in desc
        assert "Rust" in desc


# ------------------------------------------------------------------
# Framework availability tests
# ------------------------------------------------------------------

@pytest.mark.framework
class TestFrameworkAvailability:
    """Tests that verify the in-tree framework is intact."""

    def test_framework_dir_exists(self):
        """The in-tree framework/ directory must exist."""
        from app.engines.his2trans.engine import _FRAMEWORK_DIR
        assert _FRAMEWORK_DIR.is_dir(), f"Framework not found at {_FRAMEWORK_DIR}"

    def test_key_scripts_exist(self):
        """Verify key framework scripts are present."""
        from app.engines.his2trans.engine import _FRAMEWORK_DIR
        required = [
            "stage1_prep/get_dependencies.py",
            "stage2_skeleton/skeleton_builder.py",
            "stage3_translate/translate_function.py",
            "stage3_translate/auto_repair_rust.py",
            "stage3_translate/merge_final_project.py",
            "knowledge/generate_signature_mappings.py",
            "knowledge/run_jina_reranker_queued.py",
        ]
        missing = []
        for script in required:
            if not (_FRAMEWORK_DIR / script).is_file():
                missing.append(script)
        assert len(missing) == 0, f"Missing framework scripts: {missing}"

    def test_runner_available(self, framework_path, base_config):
        """FrameworkRunner should detect the in-tree framework."""
        runner = FrameworkRunner(framework_path, base_config)
        assert runner.check_framework_available() is True

    def test_runner_script_exists(self, framework_path, base_config):
        """FrameworkRunner should find key scripts."""
        runner = FrameworkRunner(framework_path, base_config)
        assert runner.check_script_exists("stage1_prep/get_dependencies.py") is True


# ------------------------------------------------------------------
# Env Mapping tests
# ------------------------------------------------------------------

class TestEnvMapping:
    """Verify the full env var chain: config → EnvMapper → subprocess env."""

    def test_all_critical_env_vars_set(self, base_config):
        """Every critical framework env var should be present in the built env."""
        env = EnvMapper.build_env(base_config, source_path="/test/src", workspace="/test/ws")

        critical = [
            "USE_VLLM",            # Must be 'false'
            "JINA_MIN_MEMORY_GB",  # Must be '4.0'
            "EXTERNAL_API_KEY",
            "EXTERNAL_API_BASE_URL",
            "EXTERNAL_API_MODEL",
            "C2R_WORKSPACE_ROOT",
            "PROJECT_PATH",
            "WORKSPACE_PATH",
        ]
        for key in critical:
            assert key in env, f"Critical env var '{key}' missing"
            assert env[key], f"Critical env var '{key}' is empty"

    def test_use_vllm_disabled(self, base_config):
        """USE_VLLM must be 'false' to force external API mode."""
        env = EnvMapper.build_env(base_config)
        assert env["USE_VLLM"] == "false"

    def test_jina_gpu_threshold_lowered(self, base_config):
        """JINA_MIN_MEMORY_GB must be lowered from default 8.0 to avoid GPU wait-loop."""
        env = EnvMapper.build_env(base_config)
        assert env["JINA_MIN_MEMORY_GB"] == "4.0"

    def test_framework_paths_added(self, framework_path):
        """add_framework_paths should set PYTHONPATH with framework subdirs."""
        env = os.environ.copy()
        EnvMapper.add_framework_paths(env, framework_path)
        assert "PYTHONPATH" in env
        assert "stage1_prep" in env["PYTHONPATH"]
        assert "stage2_skeleton" in env["PYTHONPATH"]
        assert "stage3_translate" in env["PYTHONPATH"]

    def test_resource_paths_added(self, framework_path):
        """add_resource_paths should set NLTK_DATA and knowledge base paths."""
        env = os.environ.copy()
        EnvMapper.add_resource_paths(env, framework_path)
        # Should at least set NLTK_DATA if available
        if "NLTK_DATA" in env:
            assert os.path.isdir(env["NLTK_DATA"])


# ------------------------------------------------------------------
# OHOS Project Tests
# ------------------------------------------------------------------

@pytest.mark.ohos
class TestOhosProject:
    """OHOS project translation tests.

    Requires: His2Trans framework + OHOS test project + compile_commands.json
    """

    def test_ohos_project_exists(self, ohos_project_path):
        """Verify OHOS test project directory and compile_commands.json exist."""
        assert os.path.isdir(ohos_project_path)
        cc_path = os.path.join(ohos_project_path, "compile_commands.json")
        assert os.path.isfile(cc_path), f"compile_commands.json not found in {ohos_project_path}"

    def test_ohos_stage1_skeleton_generation(self, engine, ohos_project_path,
                                              workspace, ohos_config, log_collector):
        """Stage 1 on OHOS project should generate skeleton .rs files."""
        result = engine.run_stage(
            "stage1_prep", ohos_project_path, workspace,
            ohos_config, log_collector,
        )

        assert result is not None
        assert "summary" in result
        print(f"Stage 1 result: {result['summary']}")

        # Check for generated files
        ws = os.path.join(workspace, "workspace")
        rust_files = list(Path(ws).rglob("*.rs")) if os.path.isdir(ws) else []
        print(f"Generated {len(rust_files)} .rs files in workspace")

    def test_ohos_stage2_rag(self, engine, ohos_project_path, workspace,
                              ohos_config, log_collector):
        """Stage 2 on OHOS project with RAG should run successfully."""
        # First run Stage 1
        engine.run_stage(
            "stage1_prep", ohos_project_path, workspace,
            ohos_config, log_collector,
        )

        # Then Stage 2
        result = engine.run_stage(
            "stage2_rag", ohos_project_path, workspace,
            ohos_config, log_collector,
        )

        assert result is not None
        assert "summary" in result

    @pytest.mark.slow
    def test_ohos_full_pipeline(self, engine, ohos_project_path, workspace,
                                 ohos_config, log_collector):
        """Full pipeline on OHOS project should complete all 4 stages."""
        results = {}
        for stage_id in ["stage1_prep", "stage2_rag", "stage3_translate", "postprocess"]:
            result = engine.run_stage(
                stage_id, ohos_project_path, workspace,
                ohos_config, log_collector,
            )
            results[stage_id] = result
            assert result is not None, f"Stage {stage_id} returned None"
            assert "summary" in result, f"Stage {stage_id} missing 'summary' key"

        # Check output
        report_path = os.path.join(workspace, "report.json")
        if os.path.isfile(report_path):
            with open(report_path) as f:
                report = json.load(f)
            assert "rust_files_generated" in report
            print(f"Full pipeline complete: {report['rust_files_generated']} Rust files")


# ------------------------------------------------------------------
# Standard C Project Tests
# ------------------------------------------------------------------

@pytest.mark.standard_c
class TestStandardCProject:
    """Standard C project translation tests.

    Requires: His2Trans framework + standard C test project
    """

    def test_standard_c_project_exists(self, standard_c_project_path):
        """Verify standard C test project exists with source files."""
        assert os.path.isdir(standard_c_project_path)
        c_files = list(Path(standard_c_project_path).rglob("*.c"))
        assert len(c_files) > 0, f"No .c files found in {standard_c_project_path}"
        print(f"Standard C project: {len(c_files)} .c files")

    def test_standard_c_no_compile_commands(self, standard_c_project_path):
        """Standard C project should NOT have compile_commands.json."""
        cc_path = os.path.join(standard_c_project_path, "compile_commands.json")
        assert not os.path.isfile(cc_path), \
            "Standard C test project should not have compile_commands.json"

    def test_standard_c_stage1(self, engine, standard_c_project_path,
                                workspace, standard_c_config, log_collector):
        """Stage 1 on standard C project should work without compile_commands.json."""
        result = engine.run_stage(
            "stage1_prep", standard_c_project_path, workspace,
            standard_c_config, log_collector,
        )

        assert result is not None
        assert "summary" in result
        print(f"Stage 1 result: {result['summary']}")

        # Check for generated files
        ws = os.path.join(workspace, "workspace")
        rust_files = list(Path(ws).rglob("*.rs")) if os.path.isdir(ws) else []
        print(f"Generated {len(rust_files)} .rs files in workspace")

    def test_standard_c_stage2_skipped(self, engine, standard_c_project_path,
                                        workspace, standard_c_config, log_collector):
        """Stage 2 should be skipped for standard C project (RAG disabled)."""
        result = engine.run_stage(
            "stage2_rag", standard_c_project_path, workspace,
            standard_c_config, log_collector,
        )

        assert "skip" in result["summary"].lower() or "RAG disabled" in result["summary"]

    @pytest.mark.slow
    def test_standard_c_full_pipeline(self, engine, standard_c_project_path,
                                       workspace, standard_c_config, log_collector):
        """Full pipeline on standard C project should complete all 4 stages."""
        results = {}
        for stage_id in ["stage1_prep", "stage2_rag", "stage3_translate", "postprocess"]:
            result = engine.run_stage(
                stage_id, standard_c_project_path, workspace,
                standard_c_config, log_collector,
            )
            results[stage_id] = result
            assert result is not None, f"Stage {stage_id} returned None"
            assert "summary" in result, f"Stage {stage_id} missing 'summary' key"

        report_path = os.path.join(workspace, "report.json")
        if os.path.isfile(report_path):
            with open(report_path) as f:
                report = json.load(f)
            print(f"Full pipeline complete: {report}")
