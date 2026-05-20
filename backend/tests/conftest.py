"""Shared fixtures for His2Trans engine tests."""

import os
import sys
import pytest
import tempfile
import shutil
from pathlib import Path

# Ensure backend is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.engines.his2trans.engine import His2TransEngine


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "framework: tests requiring His2Trans framework")
    config.addinivalue_line("markers", "ohos: tests requiring OHOS project")
    config.addinivalue_line("markers", "standard_c: tests requiring standard C project")
    config.addinivalue_line("markers", "slow: slow integration tests")


@pytest.fixture
def engine():
    """Create a His2TransEngine instance."""
    return His2TransEngine()


@pytest.fixture
def test_data_dir():
    """Path to test data directory."""
    return Path(__file__).resolve().parent / "test_data"


@pytest.fixture
def ohos_project_path(test_data_dir):
    """Path to an OHOS test project (shared__541f4e547bdb)."""
    path = test_data_dir / "ohos_project"
    assert path.is_dir(), f"OHOS project not found at {path}"
    return str(path)


@pytest.fixture
def standard_c_project_path(test_data_dir):
    """Path to a standard C test project."""
    path = test_data_dir / "standard_c_project"
    assert path.is_dir(), f"Standard C project not found at {path}"
    return str(path)


@pytest.fixture
def workspace():
    """Temporary workspace directory."""
    tmp = tempfile.mkdtemp(prefix="his2trans_test_")
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def framework_path():
    """Path to His2Trans framework.

    Checks in order:
    1. HIS2TRANS_FRAMEWORK env var
    2. In-tree framework/ under the engine module
    3. External His2Trans-Opt-/framework
    """
    candidates = [
        os.environ.get("HIS2TRANS_FRAMEWORK", ""),
        str(Path(__file__).resolve().parent.parent / "app" / "engines" / "his2trans" / "framework"),
        str(Path(__file__).resolve().parent.parent.parent.parent / "His2Trans-Opt-" / "framework"),
    ]
    for path in candidates:
        if path and os.path.isdir(path):
            return path
    pytest.skip("His2Trans framework not found — set HIS2TRANS_FRAMEWORK env var")


@pytest.fixture
def base_config(framework_path):
    """Base translation config with optimized parameters."""
    return {
        "engine": "his2trans",
        "model": "deepseek-v3.2",
        "use_rag": False,
        "max_repair": 3,
        "api_key": os.environ.get("API_KEY", ""),
        "api_base_url": os.environ.get("API_BASE_URL", "https://api.apiyi.com/v1"),
        "api_temperature": 0.0,
        "his2trans_framework": framework_path,
        "ohos_root": "",
        "extra_includes": [],
    }


@pytest.fixture
def ohos_config(base_config):
    """Config for OHOS project translation."""
    config = dict(base_config)
    config.update({
        "use_rag": True,
        "max_repair": 8,
        "ohos_root": os.environ.get(
            "HIS2TRANS_OHOS_ROOT",
            str(Path(__file__).resolve().parent.parent / "data" / "ohos" / "ohos_root_min"),
        ),
    })
    return config


@pytest.fixture
def standard_c_config(base_config):
    """Config for standard C project translation."""
    config = dict(base_config)
    config.update({
        "use_rag": False,
        "max_repair": 5,
    })
    return config


@pytest.fixture
def log_collector():
    """Collect log messages for assertions."""
    class LogCollector:
        def __init__(self):
            self.messages = []

        def __call__(self, msg, level="info"):
            self.messages.append({"msg": msg, "level": level})

        def contains(self, text):
            return any(text in m["msg"] for m in self.messages)

        def errors(self):
            return [m for m in self.messages if m["level"] in ("error", "warn")]

    return LogCollector()
