"""Unit tests for FrameworkRunner.

These tests do NOT require the His2Trans framework — they test
the runner's error handling, script discovery, and path resolution.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.engines.his2trans.runner import FrameworkRunner, FrameworkRunnerError, _is_progress_line


class TestFrameworkRunnerBasics:
    """Test FrameworkRunner without needing the actual framework."""

    def test_check_framework_available_exists(self, tmp_path):
        """Should return True for an existing directory."""
        runner = FrameworkRunner(str(tmp_path), {})
        assert runner.check_framework_available() is True

    def test_check_framework_available_missing(self):
        """Should return False for a non-existent directory."""
        runner = FrameworkRunner("/nonexistent/path/to/framework", {})
        assert runner.check_framework_available() is False

    def test_check_script_exists(self, tmp_path):
        """Should return True for an existing script file."""
        script = tmp_path / "exists.py"
        script.write_text("# test")
        runner = FrameworkRunner(str(tmp_path), {})
        assert runner.check_script_exists("exists.py") is True

    def test_check_script_missing(self, tmp_path):
        """Should return False for a non-existent script file."""
        runner = FrameworkRunner(str(tmp_path), {})
        assert runner.check_script_exists("does_not_exist.py") is False

    def test_run_script_not_found_raises(self, tmp_path, base_config):
        """Should raise FrameworkRunnerError when script not found."""
        runner = FrameworkRunner(str(tmp_path), base_config)

        # We need a log callback
        def noop_log(msg, level="info"):
            pass

        with __import__('pytest').raises(FrameworkRunnerError) as exc_info:
            runner.run_script("nonexistent.py", "/src", "/ws", noop_log)
        assert "not found" in str(exc_info.value)


class TestProgressLineDetection:
    """Test _is_progress_line heuristics."""

    def test_error_line_detected(self):
        assert _is_progress_line("ERROR: Something went wrong")

    def test_warning_line_detected(self):
        assert _is_progress_line("WARNING: Low memory")

    def test_progress_with_bracket(self):
        assert _is_progress_line("[5/20] Translating functions...")

    def test_verbose_output_filtered(self):
        """Plain verbose output without keywords should not be flagged."""
        assert not _is_progress_line("  verbose debug line with nothing special")
        assert not _is_progress_line("")

    def test_generated_line_detected(self):
        assert _is_progress_line("Generated 15 Rust skeleton files.")
        assert _is_progress_line("Found 12 C functions.")
