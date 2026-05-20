"""Platform config → His2Trans framework environment variable mapping.

The framework's generate/generation.py reads LLM config from environment
variables with DIFFERENT names than what the platform passes. This module
bridges that gap.

Key bugs fixed:
- USE_VLLM defaults to 'true' → force 'false' (external API mode)
- API_KEY → EXTERNAL_API_KEY, etc.
- JINA_MIN_MEMORY_GB defaults to 8.0 GB → lower to 4.0 to avoid GPU wait-loop
"""

import os
from pathlib import Path


class EnvMapper:
    """Maps platform config keys to framework environment variables."""

    # Platform config key → Framework env var name (LLM config)
    LLM_CONFIG_MAP = {
        "api_key": "EXTERNAL_API_KEY",
        "api_base_url": "EXTERNAL_API_BASE_URL",
        "model": "EXTERNAL_API_MODEL",
    }

    # Always-set overrides (fix framework defaults that are wrong for our use case)
    FIXED_OVERRIDES = {
        "USE_VLLM": "false",           # Force external API — framework defaults to local vLLM!
        "JINA_MIN_MEMORY_GB": "4.0",   # Lower from 8.0 GB to avoid GPU wait-loop
        "JINA_MIN_GPU_MEMORY_GB": "4.0",
        "JINA_MIN_FREE_MEMORY_GB": "2.0",
        "JINA_RERANKER_BATCH_SIZE": "8",
        "USE_LLM_SIGNATURES": "true",  # Enable LLM-based signature matching
    }

    # Platform config keys that map to env vars of the SAME name
    DIRECT_PASSTHROUGH = {
        "ohos_root": ("OHOS_ROOT", "OPENHARMONY_SOURCE_ROOT", "OHOS_SOURCE_ROOT"),
    }

    @classmethod
    def build_env(cls, config: dict, source_path: str = "",
                  workspace: str = "", extra_env: dict = None) -> dict:
        """Build the complete environment dict for framework subprocess calls.

        This is the single source of truth for config→env mapping.
        All framework scripts MUST be launched with this environment.
        """
        env = os.environ.copy()

        # 1. Essential paths
        # Resolve project name: prefer original_path.txt, fall back to env var,
        # then to source-path basename.
        from app.api.upload import get_project_name as _get_project_name
        project_dir = os.path.dirname(source_path) if source_path else ""
        project_name = _get_project_name(project_dir) if project_dir else ""
        if not project_name or project_name == os.path.basename(project_dir.rstrip("/")):
            # get_project_name fell back to UUID — try env var or basename
            project_name = os.environ.get("PROJECT_NAME",
                          os.path.basename(source_path.rstrip("/")))
        env["PROJECT_PATH"] = source_path or ""
        env["PROJECT_ROOT"] = source_path or ""
        env["PROJECT_NAME"] = project_name
        env["WORKSPACE_PATH"] = workspace or ""
        env["C2R_WORKSPACE_ROOT"] = workspace or ""

        # 2. Fixed overrides (must come first — these fix framework defaults)
        for key, value in cls.FIXED_OVERRIDES.items():
            env[key] = value

        # 3. LLM config mapping (API_KEY → EXTERNAL_API_KEY, etc.)
        for config_key, env_var in cls.LLM_CONFIG_MAP.items():
            value = config.get(config_key, "")
            if value:
                env[env_var] = str(value)

        # 4. Also pass original API_* keys for scripts that read both naming conventions
        for config_key in ("api_key", "api_base_url", "model"):
            value = config.get(config_key, "")
            if value:
                env_key = config_key.upper()
                if env_key not in env or not env[env_key]:
                    env[env_key] = str(value)

        # 5. LLM extended params — both EXTERNAL_API_* and VLLM_* prefixes
        env["EXTERNAL_API_MAX_TOKENS"] = str(config.get("api_max_tokens", "8192"))
        env["EXTERNAL_API_TEMPERATURE"] = str(config.get("api_temperature", "0.0"))
        env["EXTERNAL_API_TIMEOUT"] = str(config.get("api_timeout", "600"))
        env["EXTERNAL_API_MAX_RETRIES"] = str(config.get("vllm_max_retries", "3"))
        env["EXTERNAL_API_TOP_P"] = str(config.get("api_top_p", "0.95"))

        env["VLLM_MAX_TOKENS"] = str(config.get("api_max_tokens", "8192"))
        env["VLLM_TEMPERATURE"] = str(config.get("api_temperature", "0.0"))
        env["VLLM_TOP_P"] = str(config.get("api_top_p", "0.95"))
        env["VLLM_REQUEST_TIMEOUT"] = str(config.get("api_timeout", "600"))
        env["VLLM_MAX_RETRIES"] = str(config.get("vllm_max_retries", "3"))
        env["VLLM_RETRY_BACKOFF_SEC"] = str(config.get("vllm_retry_backoff_sec", "2"))
        env["VLLM_RETRY_BACKOFF_MAX_SEC"] = str(config.get("vllm_retry_backoff_max_sec", "30"))

        # 6. Direct passthrough config keys
        for config_key, env_vars in cls.DIRECT_PASSTHROUGH.items():
            value = config.get(config_key, "")
            if value:
                for var_name in env_vars:
                    env[var_name] = str(value)

        # 7. Extra include directories (colon-joined list)
        extra_includes = config.get("extra_includes", [])
        if extra_includes:
            env["EXTRA_INCLUDE_DIRS"] = ":".join(extra_includes)

        # 8. Repair rounds for Stage 3
        env["MAX_REPAIR_ROUNDS"] = str(config.get("max_repair", "5"))

        # 9. Parallelism / resource controls
        env["VLLM_CONCURRENT_LIMIT"] = str(config.get("vllm_concurrent_limit", "4"))
        env["MAX_PARALLEL_WORKERS"] = str(config.get("max_parallel_workers", "4"))
        env["TRANSLATE_MAX_WORKERS"] = str(config.get("translate_max_workers", "2"))
        env["JINA_WORKERS"] = str(config.get("jina_workers", "1"))
        env["JINA_MAX_BATCH_SIZE"] = str(config.get("jina_max_batch_size", "16"))
        env["JINA_MAX_SLOTS_PER_GPU"] = str(config.get("jina_max_slots_per_gpu", "1"))
        env["JINA_CHECK_INTERVAL"] = str(config.get("jina_check_interval", "15"))
        env["GPU_BATCH_SIZE_AUTO"] = str(config.get("gpu_batch_size_auto", "true"))
        env["GPU_MIN_FREE_MEMORY_GB"] = str(config.get("gpu_min_free_memory_gb", "1.0"))

        # 10. Feature toggles
        env["USE_BINDGEN"] = str(config.get("use_bindgen", "true"))
        env["USE_LIBCLANG"] = str(config.get("use_libclang", "false"))
        env["USE_LAYERED_SKELETON"] = str(config.get("use_layered_skeleton", "true"))
        env["USE_SELF_HEALING"] = str(config.get("use_self_healing", "true"))
        env["SKELETON_RULE_FIX_ROUNDS"] = str(config.get("skeleton_rule_fix_rounds", "2"))

        # 11. Merge extra env (stage-specific overrides)
        if extra_env:
            env.update(extra_env)

        return env

    @classmethod
    def add_framework_paths(cls, env: dict, framework_dir: str):
        """Add PYTHONPATH entries for framework module imports."""
        framework_path = os.path.abspath(framework_dir)
        subdirs = [
            framework_path,
            os.path.join(framework_path, "stage1_prep"),
            os.path.join(framework_path, "stage2_skeleton"),
            os.path.join(framework_path, "stage3_translate"),
            os.path.join(framework_path, "knowledge"),
            os.path.join(framework_path, "shared"),
            os.path.join(framework_path, "config"),
            os.path.join(framework_path, "generate"),
        ]
        pythonpath = os.pathsep.join(subdirs)
        existing = env.get("PYTHONPATH", "")
        if existing:
            pythonpath = pythonpath + os.pathsep + existing
        env["PYTHONPATH"] = pythonpath

    # Path to the platform's own data directory, relative to this file
    # env_mapper.py is at backend/app/engines/his2trans/env_mapper.py
    # 3 levels up from his2trans/ → backend/
    _PLATFORM_DATA_DIR = os.path.normpath(os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "data"))

    @classmethod
    def add_resource_paths(cls, env: dict, framework_dir: str):
        """Add NLTK data, knowledge base, and OHOS fallback paths.

        Primary source is the platform's own backend/data/ directory.
        Framework directory paths serve as fallback for dev environments.
        """
        fw = os.path.abspath(framework_dir)

        # NLTK data: platform's own copy first, then framework fallback
        nltk_candidates = [
            env.get("HIS2TRANS_NLTK_DATA", ""),
            os.path.join(cls._PLATFORM_DATA_DIR, "nltk_data"),
            os.path.join(fw, "data", "nltk_data"),
        ]
        for p in nltk_candidates:
            if p and os.path.isdir(p):
                env["NLTK_DATA"] = p
                break

        # Knowledge base (for RAG) — platform's own data/rag/ first
        kb_candidates = [
            os.path.join(cls._PLATFORM_DATA_DIR, "rag", "knowledge_base.json"),
            os.path.join(fw, "workspace", "rag", "knowledge_base.json"),
            os.path.join(os.path.dirname(fw), "data", "rag", "knowledge_base.json"),
        ]
        for p in kb_candidates:
            if os.path.isfile(p):
                env["KNOWLEDGE_BASE_PATH"] = p
                break

        # BM25 index (for RAG) — platform's own data/rag/ first
        bm25_candidates = [
            os.path.join(cls._PLATFORM_DATA_DIR, "rag", "bm25_index.pkl"),
            os.path.join(fw, "workspace", "rag", "bm25_index.pkl"),
            os.path.join(os.path.dirname(fw), "data", "rag", "bm25_index.pkl"),
        ]
        for p in bm25_candidates:
            if os.path.isfile(p):
                env["BM25_INDEX_PATH"] = p
                break

        # OHOS root fallback — auto-extract if .tar.gz present
        if not env.get("OHOS_ROOT"):
            ohos_dir = os.path.join(cls._PLATFORM_DATA_DIR, "ohos")
            ohos_extracted = os.path.join(ohos_dir, "ohos_root_min")
            ohos_archive = os.path.join(ohos_dir, "ohos_root_min.tar.gz")
            if not os.path.isdir(ohos_extracted) and os.path.isfile(ohos_archive):
                _extract_ohos_archive(ohos_archive, ohos_dir)
            ohos_candidates = [
                ohos_extracted,
                os.path.join(fw, "..", "..", "..", "data", "ohos", "ohos_root_min"),
                os.path.join(fw, "data", "ohos", "ohos_root_min"),
            ]
            for p in ohos_candidates:
                if os.path.isdir(p):
                    env["OHOS_ROOT"] = p
                    env["OPENHARMONY_SOURCE_ROOT"] = p
                    env["OHOS_SOURCE_ROOT"] = p
                    break

        # HuggingFace cache (Jina reranker downloads models here)
        hf_cache = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
        if not env.get("HF_HOME"):
            env["HF_HOME"] = hf_cache
        if not env.get("TRANSFORMERS_CACHE"):
            env["TRANSFORMERS_CACHE"] = hf_cache
        if not env.get("HF_HUB_CACHE"):
            env["HF_HUB_CACHE"] = os.path.join(hf_cache, "hub")
        env["HF_HUB_DISABLE_EXPERIMENTAL_WARNING"] = "1"


def _extract_ohos_archive(archive_path: str, dest_dir: str) -> None:
    """Extract ohos_root_min.tar.gz on first use."""
    import tarfile
    import sys
    print(f"[EnvMapper] Extracting OHOS SDK archive ({os.path.basename(archive_path)})...",
          file=sys.stderr)
    os.makedirs(dest_dir, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(dest_dir)
    print("[EnvMapper] OHOS SDK extraction complete.", file=sys.stderr)
