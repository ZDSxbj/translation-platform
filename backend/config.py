import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")
    UPLOAD_FOLDER = os.environ.get("UPLOAD_FOLDER", str(BASE_DIR / "uploads"))
    OUTPUT_FOLDER = os.environ.get("OUTPUT_FOLDER", str(BASE_DIR / "outputs"))
    MAX_CONTENT_LENGTH = int(os.environ.get("MAX_CONTENT_LENGTH", 500 * 1024 * 1024))

    # LLM API
    API_KEY = os.environ.get("API_KEY", "")
    API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.apiyi.com/v1")
    API_MODEL = os.environ.get("API_MODEL", "deepseek-v3.2")
    API_MAX_TOKENS = int(os.environ.get("API_MAX_TOKENS", "8192"))
    API_TEMPERATURE = float(os.environ.get("API_TEMPERATURE", "0.0"))
    API_TIMEOUT = float(os.environ.get("API_TIMEOUT", "600.0"))

    # His2Trans framework path
    HIS2TRANS_FRAMEWORK = os.environ.get(
        "HIS2TRANS_FRAMEWORK",
        str(BASE_DIR.parent.parent / "His2Trans-Opt-" / "framework"),
    )
    HIS2TRANS_DATA = os.environ.get(
        "HIS2TRANS_DATA",
        str(BASE_DIR.parent.parent / "His2Trans-Opt-" / "data"),
    )

    # His2Trans RAG resources (migrated to backend/data/)
    HIS2TRANS_RAG_KB = os.environ.get(
        "HIS2TRANS_RAG_KB",
        str(BASE_DIR / "data" / "rag" / "knowledge_base.json"),
    )
    HIS2TRANS_RAG_INDEX = os.environ.get(
        "HIS2TRANS_RAG_INDEX",
        str(BASE_DIR / "data" / "rag" / "bm25_index.pkl"),
    )
    HIS2TRANS_OHOS_ROOT = os.environ.get(
        "HIS2TRANS_OHOS_ROOT",
        str(BASE_DIR / "data" / "ohos" / "ohos_root_min"),
    )
    HIS2TRANS_NLTK_DATA = os.environ.get(
        "HIS2TRANS_NLTK_DATA",
        str(BASE_DIR / "data" / "nltk_data"),
    )
    HIS2TRANS_PROMPTS = os.environ.get(
        "HIS2TRANS_PROMPTS",
        str(BASE_DIR / "data" / "prompts"),
    )

    # Jina Reranker GPU memory threshold (lower to avoid wait-loop)
    JINA_MIN_MEMORY_GB = os.environ.get("JINA_MIN_MEMORY_GB", "4.0")
    JINA_RERANKER_BATCH_SIZE = os.environ.get("JINA_RERANKER_BATCH_SIZE", "8")

    # LLM retry config (passed to framework's generation.py)
    VLLM_MAX_RETRIES = os.environ.get("VLLM_MAX_RETRIES", "3")
    VLLM_REQUEST_TIMEOUT = os.environ.get("VLLM_REQUEST_TIMEOUT", "600")

    # Session TTL (seconds)
    SESSION_TTL = int(os.environ.get("SESSION_TTL", "86400"))

    # CORS
    CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*")


class DevelopmentConfig(Config):
    DEBUG = True


class ProductionConfig(Config):
    DEBUG = False


config = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "default": DevelopmentConfig,
}
