"""Framework script runner — launches His2Trans framework Python scripts as subprocesses.

Provides:
- FrameworkRunner: subprocess execution with proper env mapping
- RunResult: structured result from a script execution
- FrameworkRunnerError: raised on script failure, not-found, or timeout
- Progress streaming via threading (Popen + line-by-line stdout)
"""

import os
import sys
import subprocess
import threading
from dataclasses import dataclass, field
from typing import Callable

from app.engines.his2trans.env_mapper import EnvMapper


class FrameworkRunnerError(Exception):
    """Raised when a framework script fails, is not found, or times out."""
    def __init__(self, message: str, script: str = "", returncode: int = None):
        super().__init__(message)
        self.script = script
        self.returncode = returncode


@dataclass
class RunResult:
    """Result from running a framework script."""
    script: str
    returncode: int = 0
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    error: str = ""


class FrameworkRunner:
    """Runs His2Trans framework scripts as subprocesses.

    Handles env var mapping, working directory setup, timeout,
    and line-by-line stdout streaming for progress reporting.
    """

    def __init__(self, framework_path: str, config: dict):
        self.framework_path = os.path.abspath(framework_path)
        self.config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_framework_available(self) -> bool:
        """Verify the framework path exists and is a directory."""
        return os.path.isdir(self.framework_path)

    def check_script_exists(self, script_rel: str) -> bool:
        """Check if a specific framework script exists."""
        script_path = os.path.join(self.framework_path, script_rel)
        return os.path.isfile(script_path)

    def run_script(self, script_rel: str, source_path: str, workspace: str,
                   log_callback: Callable, timeout: int = 1800,
                   extra_env: dict = None, args: list = None) -> RunResult:
        """Run a framework Python script and return structured result.

        Args:
            script_rel: Relative path to script within the framework.
            source_path: Absolute path to project source directory.
            workspace: Absolute path to workspace directory.
            log_callback: Callable(msg: str, level: str) for progress logging.
            timeout: Max execution time in seconds (default 30 min).
            extra_env: Additional env vars for this specific script.
            args: Optional list of command-line arguments to pass to the script.

        Returns:
            RunResult with returncode, stdout, stderr, and timing info.

        Raises:
            FrameworkRunnerError: If script not found.
        """
        script_path = os.path.join(self.framework_path, script_rel)
        if not os.path.isfile(script_path):
            raise FrameworkRunnerError(
                f"Framework script not found: {script_path}",
                script=script_rel,
            )

        # Build environment
        env = EnvMapper.build_env(
            self.config,
            source_path=source_path,
            workspace=workspace,
            extra_env=extra_env,
        )

        # Add framework Python paths and resource paths
        EnvMapper.add_framework_paths(env, self.framework_path)
        EnvMapper.add_resource_paths(env, self.framework_path)

        # Working directory: script's own directory (framework scripts use relative imports)
        cwd = os.path.dirname(script_path)

        # Build command with optional CLI args
        cmd = [sys.executable, script_path]
        if args:
            cmd.extend([str(a) for a in args])

        log_callback(f"[Runner] Running: {script_rel}", "info")

        try:
            result = self._run_with_streaming(
                cmd, env, cwd, timeout, log_callback
            )
        except subprocess.TimeoutExpired:
            raise FrameworkRunnerError(
                f"Script timed out after {timeout}s: {script_rel}",
                script=script_rel,
            )

        if result.returncode != 0:
            log_callback(
                f"[Runner] Script {script_rel} exited with code {result.returncode}",
                "warn",
            )

        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_with_streaming(self, cmd: list, env: dict, cwd: str,
                            timeout: int, log_callback: Callable) -> RunResult:
        """Run a script with real-time stdout streaming via Popen + thread."""
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd,
        )

        # Thread to read stdout line-by-line for progress reporting
        stdout_lines = []
        stderr_lines = []

        def read_stream(stream, collector, cb_level="info"):
            for line in iter(stream.readline, ""):
                stripped = line.rstrip("\n")
                if stripped:
                    collector.append(stripped)
                    # Rate-limit: only callback lines with key patterns
                    if _is_progress_line(stripped):
                        log_callback(stripped[:500], cb_level)

        stdout_thread = threading.Thread(
            target=read_stream,
            args=(process.stdout, stdout_lines, "info"),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=read_stream,
            args=(process.stderr, stderr_lines, "warn"),
            daemon=True,
        )

        stdout_thread.start()
        stderr_thread.start()

        # Wait with timeout
        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            raise

        stdout_thread.join(timeout=5)
        stderr_thread.join(timeout=5)

        result = RunResult(
            script=os.path.basename(cmd[0]) if cmd else "",
            returncode=returncode,
            stdout="\n".join(stdout_lines),
            stderr="\n".join(stderr_lines),
        )

        # Log remaining non-progress stderr lines
        for line in stderr_lines[-20:]:
            if not _is_progress_line(line):
                log_callback(f"STDERR: {line[:500]}", "warn")

        return result


def _is_progress_line(line: str) -> bool:
    """Check if a stdout/stderr line looks like progress worth logging.

    Avoids spamming the log with every line from verbose scripts.
    """
    keywords = [
        "ERROR", "WARNING", "error", "warning",
        "Processing", "Translating", "Compiling", "Repair",
        "✓", "✗", "完成", "错误", "成功",
        "progress", "Progress", "done", "Done",
        "Generated", "generated", "Found", "found",
        "[",  # progress bars and counters like [1/10]
    ]
    return any(kw in line for kw in keywords)
