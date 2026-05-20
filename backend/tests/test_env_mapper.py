"""Unit tests for EnvMapper — the platform→framework env var bridge."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.engines.his2trans.env_mapper import EnvMapper


class TestEnvMapper:
    """Verify critical environment variable mappings."""

    def test_use_vllm_disabled(self):
        """USE_VLLM must be 'false' to force external API mode."""
        env = EnvMapper.build_env({})
        assert env["USE_VLLM"] == "false", \
            f"USE_VLLM should be 'false', got '{env.get('USE_VLLM')}'. Framework defaults to vLLM local mode!"

    def test_jina_memory_threshold_lowered(self):
        """JINA_MIN_MEMORY_GB must be 4.0 to avoid GPU wait-loop."""
        env = EnvMapper.build_env({})
        assert env["JINA_MIN_MEMORY_GB"] == "4.0", \
            f"JINA_MIN_MEMORY_GB should be '4.0', got '{env.get('JINA_MIN_MEMORY_GB')}'"

    def test_llm_key_mapping(self):
        """Platform's api_key must map to framework's EXTERNAL_API_KEY."""
        env = EnvMapper.build_env({"api_key": "sk-test-key"})
        assert env["EXTERNAL_API_KEY"] == "sk-test-key"
        # API_KEY is also present (from env or test override)
        assert "API_KEY" in env
        assert len(env["API_KEY"]) > 0

    def test_llm_base_url_mapping(self):
        """Platform's api_base_url must map to EXTERNAL_API_BASE_URL."""
        env = EnvMapper.build_env({"api_base_url": "https://api.test.com/v1"})
        assert env["EXTERNAL_API_BASE_URL"] == "https://api.test.com/v1"

    def test_llm_model_mapping(self):
        """Platform's model must map to EXTERNAL_API_MODEL."""
        env = EnvMapper.build_env({"model": "deepseek-v3.2"})
        assert env["EXTERNAL_API_MODEL"] == "deepseek-v3.2"

    def test_ohos_root_passthrough(self):
        """ohos_root must set both OHOS_ROOT and OPENHARMONY_SOURCE_ROOT."""
        env = EnvMapper.build_env({"ohos_root": "/test/ohos/root"})
        assert env["OHOS_ROOT"] == "/test/ohos/root"
        assert env["OPENHARMONY_SOURCE_ROOT"] == "/test/ohos/root"

    def test_extra_includes_colon_joined(self):
        """extra_includes list must be colon-joined in env var."""
        env = EnvMapper.build_env({"extra_includes": ["/inc/a", "/inc/b"]})
        assert env["EXTRA_INCLUDE_DIRS"] == "/inc/a:/inc/b"

    def test_max_repair_rounds(self):
        """max_repair config must set MAX_REPAIR_ROUNDS env var."""
        env = EnvMapper.build_env({"max_repair": 8})
        assert env["MAX_REPAIR_ROUNDS"] == "8"

    def test_workspace_paths(self):
        """PROJECT_PATH and C2R_WORKSPACE_ROOT must be set."""
        env = EnvMapper.build_env({}, source_path="/src", workspace="/ws")
        assert env["PROJECT_PATH"] == "/src"
        assert env["C2R_WORKSPACE_ROOT"] == "/ws"
        assert env["WORKSPACE_PATH"] == "/ws"

    def test_retry_config(self):
        """VLLM_MAX_RETRIES must be set."""
        env = EnvMapper.build_env({"vllm_max_retries": "5"})
        assert env["VLLM_MAX_RETRIES"] == "5"

    def test_empty_config_no_crash(self):
        """Empty config should not cause KeyError or crash."""
        env = EnvMapper.build_env({})
        assert len(env) > 0

    def test_none_values_handled(self):
        """None values should not appear in output."""
        env = EnvMapper.build_env({})
        for k, v in env.items():
            assert v is not None, f"Env var '{k}' has None value"
