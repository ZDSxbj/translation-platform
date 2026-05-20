"""Compile commands analysis and path resolution service.

Detects project type (OHOS vs Standard C), analyzes include paths,
identifies broken paths, and provides path relativization.
"""

import os
import re
import json
from pathlib import Path
from collections import defaultdict

# Standard C library headers — if ALL includes are from this set,
# the project is "standard_c" (Type B).
STD_C_HEADERS = {
    # C standard library
    "assert.h", "complex.h", "ctype.h", "errno.h", "fenv.h", "float.h",
    "inttypes.h", "iso646.h", "limits.h", "locale.h", "math.h", "setjmp.h",
    "signal.h", "stdalign.h", "stdarg.h", "stdatomic.h", "stdbool.h",
    "stddef.h", "stdint.h", "stdio.h", "stdlib.h", "stdnoreturn.h",
    "string.h", "tgmath.h", "threads.h", "time.h", "uchar.h", "wchar.h",
    "wctype.h",
    # POSIX / common system
    "unistd.h", "fcntl.h", "sys/types.h", "sys/stat.h", "sys/time.h",
    "sys/socket.h", "sys/mman.h", "dirent.h", "pthread.h", "dlfcn.h",
    "poll.h", "sched.h", "semaphore.h", "termios.h", "utime.h",
    # C++
    "algorithm", "vector", "string", "map", "set", "unordered_map",
    "unordered_set", "iostream", "fstream", "sstream", "memory",
    "functional", "thread", "mutex", "condition_variable", "atomic",
    "chrono", "regex", "random", "numeric", "iterator", "type_traits",
    "utility", "tuple", "array", "deque", "list", "forward_list",
    "stack", "queue", "bitset",
}

# Known SDK/framework prefixes that indicate an external / non-standard-C dependency
SDK_PREFIXES = [
    "ohos/", "hdf_", "hilog/", "drivers/hdf",
    "bounds_checking_function",
]


class PathService:
    """Analyzes compile_commands.json for project classification and path health."""

    # Standard C headers cache (set of lowercase header names)
    _stdc_norm = {h.lower() for h in STD_C_HEADERS}

    def __init__(self, project_source_dir: str):
        self.source_dir = os.path.abspath(project_source_dir)
        self.cc_path = os.path.join(self.source_dir, "compile_commands.json")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self) -> dict:
        """Run full project analysis.

        Returns a dict with keys:
        - project_type: "ohos" | "standard_c" | "unknown"
        - has_compile_commands: bool
        - compile_commands_info: dict (if present)
        - source_files: list of relative paths
        - detected_dependencies: dict
        - recommendation: dict
        """
        result = {
            "project_type": "unknown",
            "has_compile_commands": False,
            "compile_commands_info": None,
            "source_files": [],
            "detected_dependencies": {
                "standard_c_only": None,
                "external_sdks": [],
                "missing_headers_estimate": 0,
            },
            "recommendation": {
                "needs_compile_commands": False,
                "needs_openharmony_root": False,
                "can_auto_compile": None,
                "path_fixup_needed": False,
            },
        }

        # Gather source files
        result["source_files"] = self._gather_source_files()

        if not os.path.isfile(self.cc_path):
            # No compile_commands.json — try to classify from includes
            result["has_compile_commands"] = False
            ext_deps = self._check_external_dependencies_from_sources()
            if ext_deps["has_external"]:
                result["project_type"] = "ohos"
                result["recommendation"]["needs_compile_commands"] = True
                result["recommendation"]["needs_openharmony_root"] = True
                result["recommendation"]["can_auto_compile"] = False
                result["recommendation"]["path_fixup_needed"] = True
                result["detected_dependencies"]["external_sdks"] = ext_deps["sdks"]
                result["detected_dependencies"]["standard_c_only"] = False
            else:
                result["project_type"] = "standard_c"
                result["recommendation"]["can_auto_compile"] = True
                result["recommendation"]["needs_compile_commands"] = False
                result["detected_dependencies"]["standard_c_only"] = True
            return result

        result["has_compile_commands"] = True

        try:
            with open(self.cc_path, "r", encoding="utf-8") as f:
                ccdb = json.load(f)
        except (json.JSONDecodeError, OSError):
            result["recommendation"]["needs_compile_commands"] = True
            return result

        if not isinstance(ccdb, list) or len(ccdb) == 0:
            return result

        # Analyze include paths
        includes_info = self._analyze_includes(ccdb)
        # Check for broken / unresolvable paths
        broken_info = self._detect_broken_paths(ccdb)
        # Classify project
        proj_type, ext_sdks = self._classify_project(includes_info)

        result["project_type"] = proj_type
        result["compile_commands_info"] = {
            "entry_count": len(ccdb),
            "source_files_in_db": list(includes_info["source_files"])[:50],
            "include_dirs": list(includes_info["include_dirs"])[:50],
            "external_includes": list(includes_info["external_includes"])[:30],
            "has_absolute_paths": broken_info["has_absolute"],
            "broken_paths": broken_info["broken"],
            "path_prefix": includes_info.get("common_prefix", ""),
        }
        result["detected_dependencies"] = {
            "standard_c_only": not includes_info["has_external"],
            "external_sdks": ext_sdks,
            "missing_headers_estimate": broken_info["missing_count"],
        }
        result["recommendation"] = {
            "needs_compile_commands": proj_type == "ohos" or broken_info["has_absolute"],
            "needs_openharmony_root": proj_type == "ohos",
            "can_auto_compile": proj_type == "standard_c" and not broken_info["has_absolute"],
            "path_fixup_needed": broken_info["has_absolute"],
        }

        return result

    def relativize_paths(self) -> dict:
        """Rewrite absolute paths in compile_commands.json to be relative to project root.

        Returns a summary dict with counts of changed entries.
        """
        if not os.path.isfile(self.cc_path):
            return {"success": False, "message": "No compile_commands.json found"}

        try:
            with open(self.cc_path, "r", encoding="utf-8") as f:
                ccdb = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            return {"success": False, "message": f"Failed to read: {e}"}

        changed = 0
        for entry in ccdb:
            if not isinstance(entry, dict):
                continue

            base_dir = entry.get("directory", ".")
            if not os.path.isabs(base_dir):
                base_dir = os.path.normpath(os.path.join(self.source_dir, base_dir))

            # Fix directory field
            if os.path.isabs(entry.get("directory", "")):
                try:
                    rel = os.path.relpath(entry["directory"], self.source_dir)
                    entry["directory"] = rel
                    changed += 1
                except ValueError:
                    pass

            # Fix file field
            file_path = entry.get("file", "")
            if os.path.isabs(file_path):
                try:
                    rel = os.path.relpath(file_path, self.source_dir)
                    entry["file"] = rel
                    changed += 1
                except ValueError:
                    pass

            # Fix -I paths in command/arguments
            command = entry.get("command", "")
            if command:
                new_command = self._relativize_command_paths(command, base_dir)
                if new_command != command:
                    entry["command"] = new_command
                    changed += 1

            arguments = entry.get("arguments", [])
            if arguments:
                new_args = []
                for arg in arguments:
                    new_args.append(self._relativize_arg(arg, base_dir))
                if new_args != arguments:
                    entry["arguments"] = new_args
                    changed += 1

        # Write back
        try:
            with open(self.cc_path, "w", encoding="utf-8") as f:
                json.dump(ccdb, f, indent=2)
        except OSError as e:
            return {"success": False, "message": f"Failed to write: {e}"}

        return {
            "success": True,
            "message": f"Relativized {changed} entries",
            "changed_entries": changed,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _gather_source_files(self) -> list:
        """List all C/C++ source files in the project directory."""
        source_exts = {".c", ".h", ".cpp", ".hpp", ".cc", ".cxx", ".hxx"}
        files = []
        for root, dirs, _ in os.walk(self.source_dir):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
            for name in sorted(os.listdir(root)):
                full = os.path.join(root, name)
                if os.path.isfile(full):
                    ext = os.path.splitext(name)[1].lower()
                    if ext in source_exts:
                        files.append(os.path.relpath(full, self.source_dir))
        return files

    def _analyze_includes(self, ccdb: list) -> dict:
        """Extract include directories and classify includes."""
        include_dirs = set()
        external_includes = set()
        source_files = set()
        has_external = False
        common_prefix = None

        include_re = re.compile(r'-I\s*(\S+)')

        for entry in ccdb:
            if not isinstance(entry, dict):
                continue
            sf = entry.get("file", "")
            if sf:
                source_files.add(sf)

            command = entry.get("command", entry.get("arguments", ""))
            if isinstance(command, list):
                command = " ".join(command)

            for m in include_re.finditer(command):
                inc_path = m.group(1).strip('"')
                include_dirs.add(inc_path)
                if not self._is_standard_c_path(inc_path):
                    external_includes.add(inc_path)
                    has_external = True

        # Compute common prefix
        if include_dirs:
            common_prefix = os.path.commonpath(list(include_dirs)) if len(include_dirs) > 1 else ""

        return {
            "include_dirs": include_dirs,
            "external_includes": external_includes,
            "source_files": source_files,
            "has_external": has_external,
            "common_prefix": common_prefix,
        }

    def _detect_broken_paths(self, ccdb: list) -> dict:
        """Find include/file paths that don't resolve on this machine."""
        broken = []
        has_absolute = False
        missing_count = 0

        include_re = re.compile(r'-I\s*(\S+)')

        for entry in ccdb[:20]:  # Sample first 20 entries
            if not isinstance(entry, dict):
                continue
            base_dir = entry.get("directory", ".")
            if not os.path.isabs(base_dir):
                base_dir = os.path.join(self.source_dir, base_dir)
            base_dir = os.path.normpath(base_dir)

            command = entry.get("command", entry.get("arguments", ""))
            if isinstance(command, list):
                command = " ".join(command)

            for m in include_re.finditer(command):
                inc = m.group(1).strip('"')
                if os.path.isabs(inc):
                    has_absolute = True
                    if not os.path.isdir(inc):
                        broken.append(inc)
                        missing_count += 1
                else:
                    resolved = os.path.normpath(os.path.join(base_dir, inc))
                    if not os.path.isdir(resolved):
                        broken.append(inc)
                        missing_count += 1

        return {
            "has_absolute": has_absolute,
            "broken": list(set(broken))[:30],
            "missing_count": missing_count,
        }

    def _classify_project(self, includes_info: dict) -> tuple:
        """Classify as 'ohos' or 'standard_c' based on external includes."""
        ext_includes = includes_info.get("external_includes", set())
        if not ext_includes:
            return ("standard_c", [])

        sdks = set()
        for inc in ext_includes:
            inc_lower = inc.lower()
            for prefix in SDK_PREFIXES:
                if prefix in inc_lower:
                    sdks.add(prefix.rstrip("/").replace("_", " ").title())
                    break

        if sdks:
            return ("ohos", sorted(sdks))
        else:
            # Has external includes but not matching known SDK patterns
            # Could be a custom library — still classify as "ohos" since it needs
            # external dependency resolution
            return ("ohos", sorted(sdks) if sdks else ["Unknown External"])

    def _check_external_dependencies_from_sources(self) -> dict:
        """Scan source files' #include lines when no compile_commands.json exists."""
        include_re = re.compile(r'^\s*#\s*include\s*[<"]([^>"]+)[>"]', re.MULTILINE)
        external = []
        has_external = False

        for root, dirs, files in os.walk(self.source_dir):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for name in files:
                ext = os.path.splitext(name)[1].lower()
                if ext not in (".c", ".h", ".cpp", ".hpp", ".cc", ".cxx"):
                    continue
                full = os.path.join(root, name)
                try:
                    with open(full, "r", encoding="utf-8", errors="replace") as f:
                        content = f.read()
                except OSError:
                    continue
                for m in include_re.finditer(content):
                    header = m.group(1)
                    norm = header.lower()
                    if norm not in self._stdc_norm:
                        external.append(header)
                        has_external = True

        sdks = set()
        for h in external:
            for prefix in SDK_PREFIXES:
                if prefix in h.lower():
                    sdks.add(prefix.rstrip("/").replace("_", " ").title())

        return {
            "has_external": has_external,
            "external_headers": list(set(external))[:30],
            "sdks": sorted(sdks),
        }

    def _is_standard_c_path(self, inc_path: str) -> bool:
        """Check if an include path looks like it points to standard C/POSIX headers."""
        # If it contains a subdirectory not in standard locations, it's external
        path_lower = inc_path.lower()
        for prefix in SDK_PREFIXES:
            if prefix in path_lower:
                return False
        # Relative paths that are just the project's own source tree are fine
        if inc_path in (".", "include", "src", ".."):
            return True
        if inc_path.startswith("-"):
            return True
        # System paths
        if inc_path.startswith("/usr/include"):
            return True
        if inc_path.startswith("/usr/local/include"):
            return True
        return False

    def _relativize_command_paths(self, command: str, base_dir: str) -> str:
        """Rewrite absolute -I paths in a command string."""
        include_re = re.compile(r'(-I\s*)(\S+)')
        result = []
        last_end = 0
        for m in include_re.finditer(command):
            result.append(command[last_end:m.start()])
            prefix = m.group(1)
            path = m.group(2).strip('"')
            if os.path.isabs(path):
                try:
                    rel = os.path.relpath(path, self.source_dir)
                    result.append(f'{prefix}{rel}')
                except ValueError:
                    result.append(m.group(0))
            else:
                result.append(m.group(0))
            last_end = m.end()
        result.append(command[last_end:])
        return "".join(result)

    def _relativize_arg(self, arg: str, base_dir: str) -> str:
        """Relativize a single argument (handles -Ipath pattern)."""
        if arg.startswith("-I") and len(arg) > 2:
            path = arg[2:]
            if os.path.isabs(path):
                try:
                    rel = os.path.relpath(path, self.source_dir)
                    return f"-I{rel}"
                except ValueError:
                    pass
        elif arg == "-I":
            return arg
        return arg
