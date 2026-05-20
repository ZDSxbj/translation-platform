#!/usr/bin/env python3
"""
Generate C accessor shims to make Rust skeletons compile when struct layouts are unknown.

Scheme-B idea:
- Keep Rust types potentially opaque (or incomplete).
- Delegate *field address calculation* to C (which sees the real struct layout).
- Rust uses the returned `void*` and casts to `*mut _` at the call site.

Outputs (per Rust project dir):
- native/c2r_accessors.c
- native/include/c2r_accessors.h
- src/compat.rs (inject extern decls between markers)
- c2r_manifest.json (record generated shims)
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple


_HEADER_EXTS = (".h", ".hpp", ".hh", ".hxx", ".H")
_SOURCE_EXTS = (".c", ".cpp", ".cc", ".cxx", ".C")


def _sanitize_ident(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return "_"
    # Replace non identifier chars with underscore.
    s = re.sub(r"[^0-9A-Za-z_]", "_", s)
    # Avoid leading digit.
    if re.match(r"^[0-9]", s):
        s = "_" + s
    return s


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _ensure_file(path: Path, content: str) -> None:
    if path.exists():
        return
    _write_text(path, content)


def _insert_unique_between_markers(
    path: Path,
    begin_marker: str,
    end_marker: str,
    new_lines: Sequence[str],
    existing_predicate=None,
) -> bool:
    """
    Insert `new_lines` between markers. If a line already exists (by predicate or exact match),
    it will not be inserted again.
    Returns True if file changed.
    """
    text = _read_text(path) if path.exists() else ""
    if begin_marker not in text or end_marker not in text:
        raise ValueError(f"Missing markers in {path}: {begin_marker} / {end_marker}")

    before, rest = text.split(begin_marker, 1)
    middle, after = rest.split(end_marker, 1)
    middle_lines = middle.splitlines()

    existing_set = set(l.rstrip("\n") for l in middle_lines)

    to_add: List[str] = []
    for l in new_lines:
        line = l.rstrip("\n")
        if not line:
            continue
        if existing_predicate:
            if any(existing_predicate(line, ex) for ex in existing_set):
                continue
        else:
            if line in existing_set:
                continue
        to_add.append(line)

    if not to_add:
        return False

    # Keep a trailing blank line for readability.
    middle_out = "\n".join([ln for ln in middle_lines if ln.strip() != "" or True]).rstrip("\n")
    if middle_out and not middle_out.endswith("\n"):
        middle_out += "\n"
    middle_out += "\n".join(to_add) + "\n"

    new_text = before + begin_marker + middle_out + end_marker + after
    _write_text(path, new_text)
    return True


@dataclass
class FieldShim:
    c_type: str
    field: str
    function: str
    header: str


class CAccessorShimManager:
    """
    Manage per-project C accessor shims + Rust declarations.
    """

    def __init__(self, rust_project_root: Path, c_search_dirs: Sequence[Path]):
        self.rust_project_root = Path(rust_project_root)
        self.c_search_dirs = [Path(p) for p in (c_search_dirs or [])]

        self.native_dir = self.rust_project_root / "native"
        self.native_include_dir = self.native_dir / "include"
        self.shims_h = self.native_include_dir / "c2r_accessors.h"
        self.shims_c = self.native_dir / "c2r_accessors.c"
        self.compat_rs = self.rust_project_root / "src" / "compat.rs"
        self.manifest_path = self.rust_project_root / "c2r_manifest.json"

        self._known_funcs: Set[str] = set()

    def _load_known_funcs(self) -> None:
        if self._known_funcs:
            return
        if self.shims_h.exists():
            text = _read_text(self.shims_h)
            for m in re.finditer(r"\b(c2r_field_ptr_[A-Za-z0-9_]+)\s*\(", text):
                self._known_funcs.add(m.group(1))

    def _ensure_scaffold(self) -> None:
        self.native_include_dir.mkdir(parents=True, exist_ok=True)

        _ensure_file(
            self.shims_h,
            "\n".join(
                [
                    "/* Auto-generated C accessor shims (field address helpers). */",
                    "#ifndef C2R_ACCESSORS_H",
                    "#define C2R_ACCESSORS_H",
                    "",
                    "#ifdef __cplusplus",
                    'extern "C" {',
                    "#endif",
                    "",
                    "// === C2R_FIELD_PTR_DECLS_BEGIN ===",
                    "// (auto-appended declarations will be inserted here)",
                    "// === C2R_FIELD_PTR_DECLS_END ===",
                    "",
                    "#ifdef __cplusplus",
                    "}",
                    "#endif",
                    "",
                    "#endif  /* C2R_ACCESSORS_H */",
                    "",
                ]
            ),
        )

        _ensure_file(
            self.shims_c,
            "\n".join(
                [
                    "/* Auto-generated C accessor shims (field address helpers). */",
                    "#include <stddef.h>",
                    "#include <stdint.h>",
                    '#include "c2r_accessors.h"',
                    "",
                    "// === C2R_INCLUDE_BEGIN ===",
                    "// (auto-appended includes will be inserted here)",
                    "// === C2R_INCLUDE_END ===",
                    "",
                    "// === C2R_FIELD_PTR_DEFS_BEGIN ===",
                    "// (auto-appended definitions will be inserted here)",
                    "// === C2R_FIELD_PTR_DEFS_END ===",
                    "",
                ]
            ),
        )

        if not self.compat_rs.exists():
            raise FileNotFoundError(
                f"compat.rs not found: {self.compat_rs}. "
                "Make sure skeleton_builder generates src/compat.rs."
            )

    def _load_manifest(self) -> Dict:
        if self.manifest_path.exists():
            try:
                return json.loads(_read_text(self.manifest_path))
            except Exception:
                pass
        return {
            "version": 1,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "field_access_shims": [],
        }

    def _save_manifest(self, manifest: Dict) -> None:
        manifest["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        _write_text(self.manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2))

    def _find_header_for_type(self, type_name: str) -> Optional[Path]:
        """
        Best-effort: search header files in `c_search_dirs` that likely contain the type.
        We prefer headers to avoid `#include .c` (which risks duplicate symbols).
        Falls back to .c source files if header not found.
        """
        type_name = (type_name or "").strip()
        if not type_name:
            return None

        # Collect candidates: headers first, then source files as fallback
        header_candidates: List[Path] = []
        source_candidates: List[Path] = []

        for root in self.c_search_dirs:
            if not root.exists():
                continue
            for p in root.rglob("*"):
                if not p.is_file():
                    continue
                if p.name.endswith(_HEADER_EXTS):
                    header_candidates.append(p)
                elif p.name.endswith(_SOURCE_EXTS):
                    source_candidates.append(p)

        header_candidates.sort(key=lambda p: str(p))
        source_candidates.sort(key=lambda p: str(p))

        # Patterns for struct/typedef detection:
        # 1. "struct TypeName" (forward decl or definition)
        struct_pat = re.compile(rf"\bstruct\s+{re.escape(type_name)}\b")
        # 2. "typedef ... TypeName ..." (simple typedef, TypeName as alias)
        typedef_simple_pat = re.compile(rf"\btypedef\s+\w+\s+{re.escape(type_name)}\s*;")
        # 3. "typedef struct { ... } TypeName;" (anonymous struct typedef)
        typedef_end_pat = re.compile(r"\btypedef\b.*?\}\s*" + re.escape(type_name) + r"\s*;", re.DOTALL)

        def check_patterns(txt: str) -> bool:
            return bool(struct_pat.search(txt) or typedef_simple_pat.search(txt) or typedef_end_pat.search(txt))

        # First try headers
        for p in header_candidates:
            try:
                txt = _read_text(p)
            except Exception:
                continue
            if check_patterns(txt):
                return p

        # Fallback to source files (for static/local struct definitions)
        for p in source_candidates:
            try:
                txt = _read_text(p)
            except Exception:
                continue
            if check_patterns(txt):
                return p

        return None

    def ensure_field_ptr(self, c_type: str, field: str, header_hint: Optional[str] = None) -> Optional[FieldShim]:
        """
        Ensure a `void* c2r_field_ptr_<type>__<field>(void*)` shim exists.
        Returns the shim descriptor if created or already present; None if type header not found.
        """
        self._ensure_scaffold()
        self._load_known_funcs()

        c_type = (c_type or "").strip()
        field = (field or "").strip()
        if not c_type or not field:
            return None

        func = f"c2r_field_ptr_{_sanitize_ident(c_type)}__{_sanitize_ident(field)}"
        if func in self._known_funcs:
            return FieldShim(c_type=c_type, field=field, function=func, header=header_hint or "")

        header_path: Optional[Path] = None
        if header_hint:
            try:
                hp = Path(header_hint)
                if hp.exists() and hp.is_file() and (hp.name.endswith(_HEADER_EXTS) or hp.name.endswith(_SOURCE_EXTS)):
                    header_path = hp
            except Exception:
                header_path = None
        if header_path is None:
            header_path = self._find_header_for_type(c_type)
        if header_path is None:
            return None

        header_abs = str(header_path.resolve())

        # 1) Inject include into C file (dedup)
        _insert_unique_between_markers(
            self.shims_c,
            "// === C2R_INCLUDE_BEGIN ===",
            "// === C2R_INCLUDE_END ===",
            [f'#include "{header_abs}"'],
        )

        # 2) Add C declaration (header)
        decl = f"void* {func}(void* base);"
        _insert_unique_between_markers(
            self.shims_h,
            "// === C2R_FIELD_PTR_DECLS_BEGIN ===",
            "// === C2R_FIELD_PTR_DECLS_END ===",
            [decl],
        )

        # 3) Add C definition
        # We cast as `c_type*` (works for typedef). If the C type is a struct tag without typedef,
        # this will fail; users can fix by adding a typedef or adjusting header discovery.
        definition = "\n".join(
            [
                f"void* {func}(void* base) {{",
                f"    return (void*)(&(({c_type}*)base)->{field});",
                "}",
            ]
        )
        _insert_unique_between_markers(
            self.shims_c,
            "// === C2R_FIELD_PTR_DEFS_BEGIN ===",
            "// === C2R_FIELD_PTR_DEFS_END ===",
            [definition],
            existing_predicate=lambda new, ex: ex.startswith(f"void* {func}("),
        )

        # 4) Inject Rust extern decl into compat.rs
        rust_decl = "\n".join(
            [
                "#[allow(improper_ctypes)]",
                'extern "C" {',
                f"    pub fn {func}(base: *mut ::core::ffi::c_void) -> *mut ::core::ffi::c_void;",
                "}",
            ]
        )
        _insert_unique_between_markers(
            self.compat_rs,
            "// === C2R_ACCESSOR_SHIMS_BEGIN ===",
            "// === C2R_ACCESSOR_SHIMS_END ===",
            [rust_decl],
            existing_predicate=lambda new, ex: f"pub fn {func}(" in ex,
        )

        # 5) Record manifest
        manifest = self._load_manifest()
        manifest.setdefault("field_access_shims", [])
        manifest["field_access_shims"].append(
            {
                "c_type": c_type,
                "field": field,
                "function": func,
                "header": header_abs,
            }
        )
        self._save_manifest(manifest)

        self._known_funcs.add(func)
        return FieldShim(c_type=c_type, field=field, function=func, header=header_abs)

