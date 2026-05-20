#!/usr/bin/env python3
"""
On-demand type recovery for incremental translation.

Goals:
- Keep the Rust project compiling (cargo check) without sacrificing function translation quality.
- Prefer deterministic fixes first (placeholders, C struct parsing), then LLM inference as a last resort.

This module is intentionally conservative:
- It only edits `src/types.rs` (and optionally `src/globals.rs`) inside the incremental work dir.
- Any generated items are clearly marked with `C2R_...` comments for later review.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)

_RUST_KEYWORDS = {
    # 2018/2021 keywords + reserved words commonly used by bindgen sanitizers.
    "as",
    "break",
    "const",
    "continue",
    "crate",
    "else",
    "enum",
    "extern",
    "false",
    "fn",
    "for",
    "if",
    "impl",
    "in",
    "let",
    "loop",
    "match",
    "mod",
    "move",
    "mut",
    "pub",
    "ref",
    "return",
    "self",
    "Self",
    "static",
    "struct",
    "super",
    "trait",
    "true",
    "type",
    "unsafe",
    "use",
    "where",
    "while",
    # 2018+ edition keywords
    "async",
    "await",
    "dyn",
    # reserved for future use (still blocked as identifiers in many contexts)
    "try",
    "yield",
}


def _sanitize_rust_field_name(name: str, used: Set[str]) -> str:
    """
    Sanitize a C struct field name into a valid Rust field identifier.

    Important: some Rust keywords cannot be used even as raw identifiers (e.g., `super`).
    To keep things simple and robust, we always use the bindgen-like strategy:
    - replace non-identifier chars with `_`
    - if keyword, append `_`
    - de-dup by appending a numeric suffix
    """
    raw = (name or "").strip()
    s = re.sub(r"[^A-Za-z0-9_]", "_", raw)
    if not s:
        s = "_c2r_field"
    if s[0].isdigit():
        s = "_" + s
    if s in _RUST_KEYWORDS:
        s = s + "_"
    base = s
    i = 1
    while s in used:
        s = f"{base}_{i}"
        i += 1
    used.add(s)
    return s

TYPE_MAPPER_AVAILABLE = False
TypeMapper = None
try:
    from type_mapper import TypeMapper as _TypeMapper

    TypeMapper = _TypeMapper
    TYPE_MAPPER_AVAILABLE = True
except Exception:
    TYPE_MAPPER_AVAILABLE = False

GENERATION_AVAILABLE = False
generation = None
try:
    from generate.generation import generation as _generation

    generation = _generation
    GENERATION_AVAILABLE = True
except Exception:
    GENERATION_AVAILABLE = False


def _sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def _strip_rust_path(type_expr: str) -> str:
    """Reduce a Rust type expression to a likely bare type name."""
    if not type_expr:
        return ""
    s = type_expr.strip()
    # Strip common wrappers/pointers
    s = re.sub(r"^[\s\(\)]*", "", s)
    s = re.sub(r"^(?:\*const|\*mut)\s+", "", s)
    s = re.sub(r"^&(?:mut\s+)?", "", s)
    s = s.strip("() \t\n\r")
    # Drop generic wrappers like Option<...>, Box<...>, etc (best-effort)
    generic_match = re.match(r"^[A-Za-z0-9_:]+<(.+)>$", s)
    if generic_match:
        s = generic_match.group(1).strip()
        s = re.sub(r"^(?:\*const|\*mut)\s+", "", s)
        s = s.strip("() \t\n\r")
    # Keep last path segment
    if "::" in s:
        s = s.split("::")[-1]
    # Remove trailing punctuation
    s = s.strip("` ,;")
    # Keep only identifier-ish
    m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)$", s)
    return m.group(1) if m else s


def _normalize_rust_type_for_types_rs(rust_type: str) -> str:
    """types.rs is `crate::types` module; avoid `crate::types::` self-paths."""
    if not rust_type:
        return "usize"
    t = rust_type.strip()
    t = t.replace("crate::types::", "")
    t = t.replace("crate::globals::", "")
    return t


def parse_missing_symbols(error_msg: str) -> Tuple[Set[str], Set[str]]:
    """
    Extract missing (types, values) from rustc output.

    Returns:
        (missing_types, missing_values)
    """
    missing_types: Set[str] = set()
    missing_values: Set[str] = set()
    if not error_msg:
        return missing_types, missing_values

    # Types
    type_patterns = [
        r"cannot find type `(\w+)`",
        r"use of undeclared type `(\w+)`",
        r"failed to resolve: use of undeclared type `(\w+)`",
    ]
    for pat in type_patterns:
        for m in re.finditer(pat, error_msg):
            missing_types.add(m.group(1))

    # Values/constants
    value_patterns = [
        r"cannot find value `(\w+)`",
        r"cannot find function `(\w+)`",
        r"cannot find static `(\w+)`",
        r"failed to resolve: could not find `(\w+)` in `types`",
        r"not found in this scope.*`(\w+)`",
    ]
    for pat in value_patterns:
        for m in re.finditer(pat, error_msg):
            missing_values.add(m.group(1))

    # Filter obvious Rust keywords
    blacklist = {
        "self",
        "Self",
        "super",
        "crate",
        "std",
        "core",
        "alloc",
        "Option",
        "Result",
        "Box",
        "Vec",
        "String",
        "Ok",
        "Err",
        "Some",
        "None",
        "true",
        "false",
    }
    missing_types = {t for t in missing_types if t not in blacklist}
    missing_values = {v for v in missing_values if v not in blacklist}
    return missing_types, missing_values


def parse_no_field_errors(error_msg: str) -> List[Tuple[str, str]]:
    """
    Extract (field_name, type_expr) from E0609-like messages.
    """
    results: List[Tuple[str, str]] = []
    if not error_msg:
        return results
    # Examples:
    #   no field `data` on type `*mut Foo`
    #   no field 'data' on type '*mut Foo'
    patterns = [
        r"no field `(\w+)` on type `([^`]+)`",
        r"no field '(\w+)' on type '([^']+)'",
    ]
    for pat in patterns:
        for m in re.finditer(pat, error_msg):
            results.append((m.group(1), m.group(2)))
    return results


def _looks_like_const(name: str) -> bool:
    return bool(re.match(r"^[A-Z][A-Z0-9_]*$", name))


def _looks_like_global(name: str) -> bool:
    return name.startswith("g_") or name.startswith("G_")


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", errors="ignore")


def _has_definition(types_rs: str, name: str) -> bool:
    if not name:
        return True
    # Direct item definitions.
    if re.search(rf"\b(pub\s+)?(struct|enum|type|union)\s+{re.escape(name)}\b", types_rs):
        return True
    # Field names (very loose heuristic; kept for backwards compatibility).
    if re.search(rf"\b{re.escape(name)}\b\s*:\s*", types_rs):
        return True
    # Also treat `pub use path::{A,B,C};` as "already defined" to avoid generating duplicate
    # placeholders when TU-pinned bindgen types are re-exported from `types.rs`.
    try:
        # `pub use foo::bar::{A, B as C};`
        for grp in re.findall(r"(?m)^\s*pub\s+use\s+[^;]*\{([^}]*)\}\s*;\s*$", types_rs):
            for part in (p.strip() for p in (grp or "").split(",")):
                if not part:
                    continue
                head = part.split(" as ", 1)[0].strip()
                if head == name:
                    return True
        # `pub use foo::bar::Baz;`
        for single in re.findall(r"(?m)^\s*pub\s+use\s+[^;]*::\s*([A-Za-z_][A-Za-z0-9_]*)\s*;\s*$", types_rs):
            if single == name:
                return True
    except Exception:
        pass
    return False


def _append_marked_block(path: Path, block_name: str, lines: Sequence[str]) -> bool:
    """
    Append a marked block to a Rust source file, avoiding duplication.
    """
    if not lines:
        return False
    content = _read_text(path)
    marker_begin = f"// === C2R_{block_name}_BEGIN ==="
    marker_end = f"// === C2R_{block_name}_END ==="
    if marker_begin in content and marker_end in content:
        # Append inside existing block if new lines are not present
        begin_idx = content.find(marker_begin)
        end_idx = content.find(marker_end)
        if begin_idx == -1 or end_idx == -1 or end_idx <= begin_idx:
            return False
        existing_block = content[begin_idx:end_idx]
        # IMPORTANT:
        # Do NOT de-duplicate per-line here.
        # For AUTO_MISSING_TYPES, different placeholders share common lines like `#[repr(C)]`.
        # Line-level de-dup would drop attributes for later types, causing missing derives (e.g. LosMux not Copy/Debug).
        non_empty = [ln for ln in lines if str(ln).strip()]
        if non_empty and all(str(ln) in existing_block for ln in non_empty):
            return False
        new_content = content[:end_idx] + "\n" + "\n".join(lines) + "\n" + content[end_idx:]
        _write_text(path, new_content)
        return True

    addition = "\n" + marker_begin + "\n" + "\n".join(lines) + "\n" + marker_end + "\n"
    _write_text(path, content.rstrip() + addition)
    return True


def _find_opaque_struct_span(types_rs: str, type_name: str) -> Optional[Tuple[int, int, str]]:
    """
    Find the span (start, end, block_text) of a `pub struct TypeName { ... }` definition.
    """
    if not type_name:
        return None
    # Match struct block including optional attributes above it.
    # Use a non-greedy match for the body.
    pattern = rf"(?s)(?:^\s*#\[[^\]]+\]\s*\n)*^\s*pub\s+struct\s+{re.escape(type_name)}\s*\{{.*?\}}\s*"
    m = re.search(pattern, types_rs, re.MULTILINE)
    if not m:
        return None
    return m.start(), m.end(), m.group(0)


def _is_trivial_opaque_struct(block_text: str) -> bool:
    """
    Heuristic: treat a struct as "opaque placeholder" if it only contains _opaque/_unused fields.
    """
    if not block_text:
        return False
    body_match = re.search(r"(?s)\{(.*)\}", block_text)
    if not body_match:
        return False
    body = body_match.group(1)
    # If there is any `pub <name>:` other than _opaque/_unused/_private, treat as non-trivial.
    for m in re.finditer(r"\b(pub\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*:", body):
        name = m.group(2)
        if name not in {"_opaque", "_unused", "_private", "_c2r_private", "_reserved"}:
            return False
    return True


def _extract_existing_struct_fields(block_text: str) -> Dict[str, str]:
    """
    Extract field_name -> type from a Rust struct block (best effort).
    """
    fields: Dict[str, str] = {}
    if not block_text:
        return fields
    body_match = re.search(r"(?s)\{(.*)\}", block_text)
    if not body_match:
        return fields
    body = body_match.group(1)
    for m in re.finditer(r"^\s*(pub\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*:\s*([^,]+),", body, re.MULTILINE):
        name = m.group(2)
        ty = m.group(3).strip()
        if name and ty:
            fields[name] = ty
    return fields


def _build_struct_block(type_name: str, fields: Dict[str, str], *, comments: Optional[List[str]] = None) -> str:
    """
    Build a conservative `#[repr(C)] pub struct` block for types.rs.
    """
    field_lines = []
    used_names: Set[str] = set()
    for fname, fty in sorted(fields.items()):
        safe_name = _sanitize_rust_field_name(fname, used_names)
        if safe_name != fname:
            field_lines.append(f"    // C2R_FIELD_RENAME: {fname} -> {safe_name}")
        field_lines.append(f"    pub {safe_name}: {fty},")
    if not field_lines:
        field_lines.append("    pub _c2r_private: [u8; 0],")
    else:
        field_lines.append("    pub _c2r_private: [u8; 0],")
    comment_lines: List[str] = []
    if comments:
        for ln in comments:
            if not ln or not str(ln).strip():
                continue
            s = str(ln).rstrip()
            comment_lines.append(s if s.lstrip().startswith("//") else f"// {s}")
    if not comment_lines:
        comment_lines = [f"// C2R_STRUCT_RECOVERY: inferred/parsed fields for `{type_name}`"]
    header = "\n".join(comment_lines) + "\n"
    return (
        header
        + "#[repr(C)]\n"
        + f"pub struct {type_name} {{\n"
        + "\n".join(field_lines)
        + "\n}\n"
    )


def _replace_struct_block(types_rs: str, type_name: str, new_block: str) -> Tuple[str, bool]:
    span = _find_opaque_struct_span(types_rs, type_name)
    if not span:
        return types_rs, False
    start, end, _old = span
    return types_rs[:start] + new_block + types_rs[end:], True


def _rg_list_files(root: Path, pattern: str, max_files: int = 40) -> List[Path]:
    """
    Use ripgrep to find candidate files quickly. Falls back to empty list.
    """
    if not root or not root.exists():
        return []
    if not shutil.which("rg"):
        return []
    try:
        cmd = ["rg", "-l", "-S", pattern, str(root)]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        if proc.returncode not in (0, 1):
            return []
        files = [Path(p) for p in proc.stdout.splitlines() if p.strip()]
        return [p for p in files if p.exists()][:max_files]
    except Exception:
        return []


def _strip_c_comments(text: str) -> str:
    # Remove /* ... */ and // ...
    text = re.sub(r"(?s)/\*.*?\*/", " ", text)
    text = re.sub(r"//.*?$", " ", text, flags=re.MULTILINE)
    return text


def _find_struct_body_in_text(text: str, type_name: str) -> Optional[str]:
    """
    Return the inside of `{ ... }` for a matching struct definition.
    Handles:
      - struct TypeName { ... };
      - typedef struct TypeName { ... } TypeName;
      - typedef struct { ... } TypeName;
    """
    if not text or not type_name:
        return None
    hay = text

    # 1) struct TypeName { ... }
    for m in re.finditer(rf"\bstruct\s+{re.escape(type_name)}\b\s*\{{", hay):
        brace_pos = m.end() - 1
        body = _extract_brace_block(hay, brace_pos)
        if body is not None:
            return body

    # 2) typedef struct TypeName { ... } TypeName;
    for m in re.finditer(rf"\btypedef\s+struct\s+{re.escape(type_name)}\b\s*\{{", hay):
        brace_pos = m.end() - 1
        body = _extract_brace_block(hay, brace_pos)
        if body is not None:
            return body

    # 3) typedef struct { ... } TypeName;
    for m in re.finditer(r"\btypedef\s+struct\s*\{", hay):
        brace_pos = m.end() - 1
        body = _extract_brace_block(hay, brace_pos)
        if body is None:
            continue
        tail = hay[brace_pos + len(body) + 2 : brace_pos + len(body) + 260]  # rough window
        if re.search(rf"\b{re.escape(type_name)}\b\s*;", tail):
            return body
    return None


def _find_struct_body_in_text_with_line(text: str, type_name: str) -> Optional[Tuple[str, int]]:
    """
    Like _find_struct_body_in_text, but also returns the (1-based) line number where
    the matching definition starts (best effort).
    """
    if not text or not type_name:
        return None
    hay = text

    def _line_of(idx: int) -> int:
        try:
            return hay.count("\n", 0, idx) + 1
        except Exception:
            return 1

    # 1) struct TypeName { ... }
    for m in re.finditer(rf"\bstruct\s+{re.escape(type_name)}\b\s*\{{", hay):
        brace_pos = m.end() - 1
        body = _extract_brace_block(hay, brace_pos)
        if body is not None:
            return body, _line_of(m.start())

    # 2) typedef struct TypeName { ... } TypeName;
    for m in re.finditer(rf"\btypedef\s+struct\s+{re.escape(type_name)}\b\s*\{{", hay):
        brace_pos = m.end() - 1
        body = _extract_brace_block(hay, brace_pos)
        if body is not None:
            return body, _line_of(m.start())

    # 3) typedef struct { ... } TypeName;
    for m in re.finditer(r"\btypedef\s+struct\s*\{", hay):
        brace_pos = m.end() - 1
        body = _extract_brace_block(hay, brace_pos)
        if body is None:
            continue
        tail = hay[brace_pos + len(body) + 2 : brace_pos + len(body) + 260]
        if re.search(rf"\b{re.escape(type_name)}\b\s*;", tail):
            return body, _line_of(m.start())
    return None


def _extract_brace_block(text: str, brace_open_index: int) -> Optional[str]:
    if brace_open_index < 0 or brace_open_index >= len(text) or text[brace_open_index] != "{":
        return None
    depth = 0
    i = brace_open_index
    start = brace_open_index + 1
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                return text[start:end]
        i += 1
    return None


def _infer_field_type_from_struct_body(body: str, field_name: str) -> Optional[str]:
    """
    Best-effort extraction of a C field type for `field_name` from a struct body.
    Returns a C type string (not Rust).
    """
    if not body or not field_name:
        return None
    cleaned = _strip_c_comments(body)
    # Drop preprocessor lines
    lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    # Try to find a line containing the field name.
    for ln in lines:
        # Skip likely function pointers in struct (too hard)
        if "(" in ln and ")" in ln and field_name in ln:
            continue
        # Accept bitfields: `u32 flags:1;`
        m = re.match(
            rf"^(?P<type>.+?)\b{re.escape(field_name)}\b\s*(?:\[[^\]]*\])?\s*(?::\s*\d+)?\s*;",
            ln,
        )
        if not m:
            continue
        c_type = m.group("type").strip()
        if not c_type:
            continue
        # Normalize whitespace around '*'
        c_type = re.sub(r"\s*\*\s*", "*", c_type)
        return c_type
    return None


def _map_c_type_to_rust(c_type: str) -> str:
    if not c_type:
        return "usize"
    # Common OpenHarmony / LiteOS aliases (avoid creating bogus opaque structs for these)
    basic_aliases = {
        # libc-like core aliases
        "c_void": "core::ffi::c_void",
        "c_char": "core::ffi::c_char",
        "c_schar": "i8",
        "c_uchar": "u8",
        "c_short": "i16",
        "c_ushort": "u16",
        "c_int": "i32",
        "c_uint": "u32",
        "c_long": "i64",
        "c_ulong": "u64",
        "c_longlong": "i64",
        "c_ulonglong": "u64",

        "UINT8": "u8",
        "UINT16": "u16",
        "UINT32": "u32",
        "UINT64": "u64",
        "INT8": "i8",
        "INT16": "i16",
        "INT32": "i32",
        "INT64": "i64",
        "CHAR": "std::ffi::c_char",
        "UCHAR": "u8",
        "USHORT": "u16",
        "UINT": "u32",
        "ULONG": "u64",
        "LONG": "i64",
        "VOID": "std::ffi::c_void",

        "size_t": "usize",
        "ssize_t": "isize",
        "ptrdiff_t": "isize",
        "intptr_t": "isize",
        "uintptr_t": "usize",
    }
    c_norm = c_type.strip()
    if c_norm in basic_aliases:
        return basic_aliases[c_norm]
    if TYPE_MAPPER_AVAILABLE and TypeMapper is not None:
        rust = TypeMapper.map_c_type(c_type, False, False)
        return _normalize_rust_type_for_types_rs(rust)
    # Fallback mapping (very conservative)
    if "*" in c_type:
        return "*mut std::ffi::c_void"
    c = c_type.strip()
    if c in {"int", "INT32", "int32_t"}:
        return "i32"
    if c in {"unsigned", "unsigned int", "UINT32", "uint32_t"}:
        return "u32"
    if c in {"size_t"}:
        return "usize"
    return "usize"


def _sanitize_inferred_rust_type(rust_type: str, *, types_rs_snapshot: str = "") -> str:
    """
    Constrain inferred Rust types to a small, compile-friendly FFI subset.

    This is primarily for Tier-2 (LLM) inference: we don't want complex Rust types
    to leak into `types.rs`.
    """
    if not rust_type:
        return "usize"

    t = _normalize_rust_type_for_types_rs(rust_type)
    t = re.sub(r"\s+", " ", t).strip()

    if t in {"u8", "u16", "u32", "u64", "i8", "i16", "i32", "i64", "usize", "isize"}:
        return t

    if re.match(r"^(core|std)::ffi::c_[A-Za-z0-9_]+$", t):
        return t

    # raw pointers
    m = re.match(r"^(\*const|\*mut)\s+(.+)$", t)
    if m:
        base = m.group(2).strip()
        if base in {"u8", "u16", "u32", "u64", "i8", "i16", "i32", "i64", "usize", "isize"}:
            return f"{m.group(1)} {base}"
        if re.match(r"^(core|std)::ffi::c_[A-Za-z0-9_]+$", base):
            return f"{m.group(1)} {base}"
        if base and re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", base) and types_rs_snapshot:
            if re.search(rf"\b(pub\s+)?(struct|enum|type|union)\s+{re.escape(base)}\b", types_rs_snapshot):
                return f"{m.group(1)} {base}"
        # Fallback pointer for unknowns
        return "*mut core::ffi::c_void"

    # arrays
    if re.match(r"^\[(u8|i8|u16|i16|u32|i32|u64|i64); *\d+\]$", t):
        return t

    # known identifiers (already defined in types.rs)
    if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", t) and types_rs_snapshot:
        if re.search(rf"\b(pub\s+)?(struct|enum|type|union)\s+{re.escape(t)}\b", types_rs_snapshot):
            return t

    return "usize"


@dataclass
class StructFieldCacheEntry:
    fields: Dict[str, str] = field(default_factory=dict)  # field -> rust_type
    source: str = ""  # "c" | "llm"
    c_snippet_hash: str = ""


class TypeRecoveryManager:
    def __init__(
        self,
        project_name: str,
        project_root: Path,
        work_dir: Path,
        context_cache_dir: Path,
        llm_name: str = "qwen3_coder",
        enable_llm: bool = True,
    ):
        self.project_name = project_name
        self.project_root = Path(project_root) if project_root else Path(".")
        self.work_dir = Path(work_dir) if work_dir else Path(".")
        self.context_cache_dir = Path(context_cache_dir) if context_cache_dir else Path(".")
        self.llm_name = llm_name
        self.enable_llm = enable_llm and GENERATION_AVAILABLE

        self.types_rs_path = self.work_dir / "src" / "types.rs"
        self.globals_rs_path = self.work_dir / "src" / "globals.rs"
        self.cache_path = self.context_cache_dir / f"type_recovery_cache_{self.project_name}.json"
        self.cache: Dict[str, StructFieldCacheEntry] = {}
        self._load_cache()

        self.report_path = self.work_dir / "types_recovery_report.json"
        self.report: Dict = self._load_report()

        self.ohos_root: Optional[Path] = self._detect_ohos_root()
        self.original_relpath: Optional[str] = self._read_original_relpath()

        self._compile_commands_path: Optional[Path] = None
        self._cc_parser = None
        self._use_compile_commands = os.environ.get("C2R_TYPE_RECOVERY_USE_COMPILE_COMMANDS", "true").lower() in ("true", "1", "yes")

        # IMPORTANT: missing const placeholders are risky:
        # - They can mask real build-context issues (macro/header not in closure)
        # - They can pollute the global namespace and trigger Rust constant-pattern parsing surprises
        #   (e.g., `let DEVICE_NUM = 3;` becomes a refutable const pattern if `pub const DEVICE_NUM: i32 = 0;` exists).
        #
        # Default: do NOT emit `pub const FOO: i32 = 0;` placeholders; let compile errors drive repair.
        self._emit_missing_consts = os.environ.get("C2R_TYPE_RECOVERY_EMIT_MISSING_CONSTS", "false").lower() in (
            "true",
            "1",
            "yes",
        )

    def _detect_ohos_root(self) -> Optional[Path]:
        # Prefer explicit OHOS_ROOT (best signal; needed when compile_commands.json is extracted from registry archives).
        try:
            env_root = (os.environ.get("OHOS_ROOT") or "").strip()
            if env_root:
                p = Path(env_root).expanduser().resolve()
                if p.exists():
                    return p
        except Exception:
            pass

        # Fall back to inferring OHOS root from the compile_commands_all registry path, if configured.
        try:
            from ohos_build_profile import get_registry_path_from_env, infer_ohos_root_from_registry

            reg = get_registry_path_from_env()
            if reg:
                inferred = infer_ohos_root_from_registry(reg)
                if inferred and Path(inferred).exists():
                    return Path(inferred).resolve()
        except Exception:
            pass

        # Derive from compile_commands.json (only reliable when it lives under <ohos_root>/out/...).
        try:
            from workspace_config import get_compile_commands_path

            cc = get_compile_commands_path()
            if cc and cc.exists():
                # If this compile_commands.json is extracted from compile_commands_all archives, do not
                # "infer" OHOS root from the extracted location; it will be wrong.
                try:
                    if (cc.parent / "profile.json").exists():
                        return None
                except Exception:
                    pass
                return cc.parent.parent.parent.resolve()
        except Exception:
            return None
        return None

    def _read_original_relpath(self) -> Optional[str]:
        try:
            p = self.project_root / "original_path.txt"
            if p.exists():
                rel = p.read_text(encoding="utf-8", errors="ignore").strip()
                return rel or None
        except Exception:
            return None
        return None

    def _utc_now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _load_report(self) -> Dict:
        if not self.report_path.exists():
            return {
                "version": 1,
                "project": self.project_name,
                "created_at": self._utc_now(),
                "updated_at": self._utc_now(),
                "missing_types": {},  # name -> {kind, mapped_to}
                "missing_consts": [],  # [{name}]
                "missing_globals": [],  # [{name}]
                "structs": {},  # type -> details
            }
        try:
            data = json.loads(self.report_path.read_text(encoding="utf-8", errors="ignore") or "{}")
            if not isinstance(data, dict):
                raise ValueError("report not dict")
            data.setdefault("version", 1)
            data.setdefault("project", self.project_name)
            data.setdefault("created_at", self._utc_now())
            data.setdefault("updated_at", self._utc_now())
            data.setdefault("missing_types", {})
            data.setdefault("missing_consts", [])
            data.setdefault("missing_globals", [])
            data.setdefault("structs", {})
            return data
        except Exception:
            return {
                "version": 1,
                "project": self.project_name,
                "created_at": self._utc_now(),
                "updated_at": self._utc_now(),
                "missing_types": {},
                "missing_consts": [],
                "missing_globals": [],
                "structs": {},
            }

    def _save_report(self) -> None:
        try:
            self.report["updated_at"] = self._utc_now()
            self.report_path.write_text(json.dumps(self.report, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            logger.debug(f"write types_recovery_report failed: {e}")

    def _ensure_cc_parser(self) -> None:
        if not self._use_compile_commands or self._cc_parser is not None:
            return
        try:
            from workspace_config import get_compile_commands_path
            from compile_commands_parser import CompileCommandsParser

            cc_path = get_compile_commands_path()
            if not cc_path or not cc_path.exists():
                return
            self._compile_commands_path = cc_path
            self._cc_parser = CompileCommandsParser(cc_path, self.ohos_root)  # may load cache
        except Exception as e:
            logger.debug(f"init CompileCommandsParser failed: {e}")
            self._cc_parser = None

    def _map_source_to_ohos_path(self, source_file: Optional[Path]) -> Optional[Path]:
        if not source_file or not self.ohos_root or not self.original_relpath:
            return None
        try:
            src = Path(source_file).resolve()
            base = self.project_root.resolve()
            rel = src.relative_to(base)
            mapped = (self.ohos_root / self.original_relpath / rel).resolve()
            return mapped if mapped.exists() else None
        except Exception:
            return None

    def _get_include_dirs_for_source(self, source_file: Optional[Path]) -> List[Path]:
        self._ensure_cc_parser()
        if not self._cc_parser or not source_file:
            return []
        try:
            mapped = self._map_source_to_ohos_path(source_file) or Path(source_file)
            dirs = self._cc_parser.get_includes_for_file(mapped, normalize_paths=True) or []
            # Keep only existing dirs
            uniq: List[Path] = []
            seen = set()
            for d in dirs:
                try:
                    p = Path(d).resolve()
                except Exception:
                    continue
                if str(p) in seen or not p.exists():
                    continue
                seen.add(str(p))
                uniq.append(p)
            return uniq
        except Exception:
            return []

    def _record_missing_type(self, name: str, *, kind: str, mapped_to: str = "") -> None:
        try:
            mt = self.report.setdefault("missing_types", {})
            prev = mt.get(name) or {}
            if isinstance(prev, dict) and prev.get("kind") == "alias" and kind != "alias":
                return
            mt[name] = {"kind": kind, "mapped_to": mapped_to, "last_seen": self._utc_now()}
        except Exception:
            return

    def _record_missing_const(self, name: str) -> None:
        try:
            arr = self.report.setdefault("missing_consts", [])
            if any(isinstance(x, dict) and x.get("name") == name for x in arr):
                return
            arr.append({"name": name, "last_seen": self._utc_now()})
        except Exception:
            return

    def _record_missing_global(self, name: str) -> None:
        try:
            arr = self.report.setdefault("missing_globals", [])
            if any(isinstance(x, dict) and x.get("name") == name for x in arr):
                return
            arr.append({"name": name, "last_seen": self._utc_now()})
        except Exception:
            return

    def _record_struct_recovery(
        self,
        type_name: str,
        *,
        fields: Dict[str, str],
        sources: Dict[str, str],
        evidence: Optional[Dict] = None,
        complete: bool = False,
        confidence: float = 0.0,
        inferred: bool = False,
    ) -> None:
        try:
            structs = self.report.setdefault("structs", {})
            cur = structs.get(type_name) or {}
            if not isinstance(cur, dict):
                cur = {}
            cur_fields = cur.get("fields") if isinstance(cur.get("fields"), dict) else {}
            cur_sources = cur.get("sources") if isinstance(cur.get("sources"), dict) else {}
            for k, v in (fields or {}).items():
                cur_fields[str(k)] = str(v)
            for k, v in (sources or {}).items():
                cur_sources[str(k)] = str(v)
            cur["fields"] = cur_fields
            cur["sources"] = cur_sources
            if evidence:
                cur["evidence"] = evidence
            cur["complete"] = bool(complete)
            cur["confidence"] = float(confidence)
            cur["inferred"] = bool(inferred)
            cur["last_updated"] = self._utc_now()
            structs[type_name] = cur
        except Exception:
            return

    def _load_cache(self) -> None:
        if not self.cache_path.exists():
            return
        try:
            data = json.loads(self.cache_path.read_text(encoding="utf-8", errors="ignore"))
            structs = data.get("structs") or {}
            for type_name, entry in structs.items():
                if not isinstance(entry, dict):
                    continue
                fields = entry.get("fields") or {}
                if not isinstance(fields, dict):
                    continue
                self.cache[type_name] = StructFieldCacheEntry(
                    fields={str(k): str(v) for k, v in fields.items()},
                    source=str(entry.get("source") or ""),
                    c_snippet_hash=str(entry.get("c_snippet_hash") or ""),
                )
        except Exception as e:
            logger.debug(f"type recovery cache load failed: {e}")

    def _save_cache(self) -> None:
        try:
            payload = {
                "version": 1,
                "project": self.project_name,
                "structs": {
                    k: {"fields": v.fields, "source": v.source, "c_snippet_hash": v.c_snippet_hash}
                    for k, v in sorted(self.cache.items())
                },
            }
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            logger.debug(f"type recovery cache save failed: {e}")

    def apply_from_compile_error(
        self,
        error_msg: str,
        *,
        source_file: Optional[Path] = None,
        c_code: str = "",
    ) -> bool:
        """
        Apply on-demand fixes based on rustc error output.

        Returns:
            Whether any file was modified.
        """
        modified = False
        if not error_msg:
            return False
        if not self.types_rs_path.exists():
            return False

        types_rs = _read_text(self.types_rs_path)

        # (2) Missing types / values
        missing_types, missing_values = parse_missing_symbols(error_msg)

        # Add missing type placeholders
        type_lines: List[str] = []
        for t in sorted(missing_types):
            if not t or _has_definition(types_rs, t):
                continue
            placeholder = self._placeholder_for_type(t)
            type_lines.extend(placeholder)
            # record (best effort)
            mapped_to = ""
            kind = "opaque"
            m = re.search(rf"\bpub\s+type\s+{re.escape(t)}\s*=\s*([^;]+);", "\n".join(placeholder))
            if m:
                kind = "alias"
                mapped_to = m.group(1).strip()
            self._record_missing_type(t, kind=kind, mapped_to=mapped_to)
        if type_lines:
            modified |= _append_marked_block(self.types_rs_path, "AUTO_MISSING_TYPES", type_lines)
            if modified:
                types_rs = _read_text(self.types_rs_path)

        # Add missing constants / globals placeholders (heuristic)
        const_lines: List[str] = []
        globals_lines: List[str] = []
        for v in sorted(missing_values):
            if not v:
                continue
            if _looks_like_global(v):
                if self.globals_rs_path.exists():
                    g_content = _read_text(self.globals_rs_path)
                    if re.search(rf"\bstatic\s+mut\s+{re.escape(v)}\b", g_content):
                        continue
                    self._record_missing_global(v)
                    globals_lines.append(f"/// C2R_AUTO_GLOBAL: placeholder for `{v}`")
                    globals_lines.append(f"pub static mut {v}: i32 = 0;")
                    globals_lines.append("")
                continue
            if not _looks_like_const(v):
                continue
            if re.search(rf"\bpub\s+const\s+{re.escape(v)}\b", types_rs):
                continue
            self._record_missing_const(v)
            if self._emit_missing_consts:
                const_lines.append(f"/// C2R_AUTO_CONST: placeholder for `{v}` (value unknown)")
                const_lines.append(f"pub const {v}: i32 = 0;")
                const_lines.append("")

        if const_lines:
            modified |= _append_marked_block(self.types_rs_path, "AUTO_MISSING_CONSTS", const_lines)
            if modified:
                types_rs = _read_text(self.types_rs_path)

        if globals_lines and self.globals_rs_path.exists():
            modified |= _append_marked_block(self.globals_rs_path, "AUTO_MISSING_GLOBALS", globals_lines)

        # (3) Struct field recovery for E0609
        no_fields = parse_no_field_errors(error_msg)
        fields_by_type: Dict[str, Set[str]] = {}
        for field_name, type_expr in no_fields:
            type_name = _strip_rust_path(type_expr)
            if not type_name or not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", type_name):
                continue
            if type_name in {"c_void", "c_char"}:
                continue
            fields_by_type.setdefault(type_name, set()).add(field_name)

        for type_name, fields in fields_by_type.items():
            did = self._ensure_struct_fields(
                types_rs,
                type_name,
                sorted(fields),
                source_file=source_file,
                c_code=c_code,
            )
            if did:
                modified = True
                types_rs = _read_text(self.types_rs_path)

        if modified:
            self._save_cache()
            self._save_report()
        return modified

    def _placeholder_for_type(self, type_name: str) -> List[str]:
        # Heuristic placeholders: keep them small and compile-friendly.
        lines: List[str] = []
        lines.append(f"/// C2R_AUTO_TYPE: placeholder for external type `{type_name}`")
        # Basic aliases: prefer correct primitives over opaque placeholders
        alias = _map_c_type_to_rust(type_name)
        if alias and alias != "usize" and (re.match(r"^(u|i)\d+$", alias) or alias.startswith(("std::ffi::", "core::ffi::"))):
            lines.append(f"pub type {type_name} = {alias};")
            lines.append("")
            return lines
        if type_name.endswith("_t") or type_name.endswith("_T"):
            lines.append(f"pub type {type_name} = usize;")
        else:
            lines.append("#[repr(C)]")
            lines.append("#[derive(Debug, Copy, Clone)]")
            lines.append(f"pub struct {type_name} {{ _opaque: [u8; 0] }}")
        lines.append("")
        return lines

    def _search_roots(self, source_file: Optional[Path]) -> List[Path]:
        roots: List[Path] = []
        # Closest source dir first (often includes local headers)
        if source_file and source_file.exists():
            roots.append(source_file.parent)

        # Compile-commands guided include dirs (more precise than a global grep)
        try:
            for inc in (self._get_include_dirs_for_source(source_file) or [])[:40]:
                if inc.exists():
                    roots.append(inc)
        except Exception:
            pass

        for sub in ("include", "inc", "headers", "src"):
            p = self.project_root / sub
            if p.exists():
                roots.append(p)
        if self.project_root.exists():
            roots.append(self.project_root)
        # If we know the original OHOS relpath, search that subtree in the full OHOS tree.
        if self.ohos_root and self.original_relpath:
            candidate = self.ohos_root / self.original_relpath
            if candidate.exists():
                roots.append(candidate)
        return roots

    def _ensure_struct_fields(
        self,
        types_rs_snapshot: str,
        type_name: str,
        field_names: List[str],
        *,
        source_file: Optional[Path],
        c_code: str,
    ) -> bool:
        # Check existing struct definition in types.rs
        span = _find_opaque_struct_span(types_rs_snapshot, type_name)
        if not span:
            return False
        _start, _end, block = span

        existing_fields = _extract_existing_struct_fields(block)
        missing = [f for f in field_names if f and f not in existing_fields]
        if not missing:
            return False

        # Collect/merge fields for this struct (cache + existing)
        entry = self.cache.get(type_name) or StructFieldCacheEntry()
        merged: Dict[str, str] = dict(existing_fields)
        # Drop placeholder-only fields
        merged.pop("_opaque", None)
        merged.pop("_unused", None)
        merged.pop("_private", None)
        merged.pop("_c2r_private", None)
        merged.update(entry.fields)

        # Determine rust types for missing fields:
        # 1) Try parse from C struct definition once
        from_c: Dict[str, str] = {}
        c_evidence: Optional[Dict] = None
        field_sources: Dict[str, str] = {}
        try:
            from_c, c_evidence = self._try_get_field_types_from_c(type_name, missing, source_file=source_file)
        except Exception:
            from_c, c_evidence = {}, None
        if from_c:
            entry.source = entry.source or "c"
            for k in from_c.keys():
                field_sources[k] = "c"

        # 2) For any still unknown, ask LLM once (last resort)
        still_missing = [f for f in missing if f not in from_c]
        from_llm: Dict[str, str] = {}
        llm_meta: Dict = {"source": "llm", "confidence": 0.0, "evidence": {}}
        if still_missing:
            from_llm, llm_meta = self._try_infer_field_types_with_llm(
                type_name, still_missing, source_file=source_file, c_code=c_code
            )
            if from_llm:
                entry.source = entry.source or "llm"
                for k in from_llm.keys():
                    field_sources[k] = "llm"

        for f in missing:
            rust_ty = from_c.get(f) or from_llm.get(f) or merged.get(f) or "usize"
            if field_sources.get(f) == "llm":
                rust_ty = _sanitize_inferred_rust_type(rust_ty, types_rs_snapshot=types_rs_snapshot)
            merged[f] = _normalize_rust_type_for_types_rs(rust_ty)

        entry.fields = {k: v for k, v in merged.items() if k and v}
        if c_code:
            entry.c_snippet_hash = _sha1_text(c_code[:2000])
        self.cache[type_name] = entry

        # Only rewrite structs that are placeholders or previously generated by this recovery logic.
        # Avoid touching bindgen-generated real layouts.
        safe_to_rewrite = _is_trivial_opaque_struct(block) or ("_c2r_private" in block) or ("C2R_STRUCT_RECOVERY" in block)
        if not safe_to_rewrite:
            return False

        inferred = any(v == "llm" for v in field_sources.values())
        complete = bool(from_c) and len(from_c) == len(missing)
        if complete:
            confidence = 0.95
        else:
            # combine evidence: partial C + LLM meta
            frac = (len(from_c) / len(missing)) if missing else 0.0
            confidence = max(float(llm_meta.get("confidence") or 0.0), 0.45 + 0.35 * frac)
            confidence = max(0.10, min(0.90, confidence))

        evidence = c_evidence or (llm_meta.get("evidence") if isinstance(llm_meta, dict) else None) or {}

        comments: List[str] = [f"C2R_STRUCT_RECOVERY: `{type_name}`"]
        if c_evidence and isinstance(c_evidence, dict) and c_evidence.get("file"):
            comments.append(f"C2R_FROM_C: {c_evidence.get('file')}:{c_evidence.get('line', '')}")
        if inferred:
            comments.append(f"C2R_INFERRED: confidence={confidence:.2f}")
        new_block = _build_struct_block(type_name, entry.fields, comments=comments)
        new_types_rs, changed = _replace_struct_block(types_rs_snapshot, type_name, new_block)
        if not changed:
            return False
        _write_text(self.types_rs_path, new_types_rs)

        # Report only the fields we had to add for this round
        added_fields = {k: entry.fields.get(k, "usize") for k in missing if k}
        added_sources = {k: field_sources.get(k, "unknown") for k in missing if k}
        self._record_struct_recovery(
            type_name,
            fields=added_fields,
            sources=added_sources,
            evidence=evidence if evidence else None,
            complete=complete,
            confidence=confidence,
            inferred=inferred,
        )
        return True

    def _try_get_field_types_from_c(
        self, type_name: str, field_names: List[str], *, source_file: Optional[Path]
    ) -> Tuple[Dict[str, str], Optional[Dict]]:
        roots = self._search_roots(source_file)
        if not roots:
            return {}, None

        # First use rg to narrow candidates (fast). Only scan a few files.
        candidate_files: List[Path] = []
        # Prefer definition-like matches to avoid false positives from mere usage.
        def_patterns = [
            rf"\bstruct\s+{re.escape(type_name)}\b\s*\{{",
            rf"\btypedef\s+struct\s+{re.escape(type_name)}\b\s*\{{",
        ]
        broad_patterns = [
            rf"\b{re.escape(type_name)}\b",
        ]

        for root in roots:
            for pat in def_patterns:
                candidate_files.extend(_rg_list_files(root, pat, max_files=30))
        if not candidate_files:
            for root in roots:
                for pat in broad_patterns:
                    candidate_files.extend(_rg_list_files(root, pat, max_files=30))

        # Prefer header-like files
        if candidate_files:
            def _prio(p: Path) -> Tuple[int, int]:
                ext = p.suffix.lower()
                is_header = 0 if ext in {".h", ".hpp", ".hh", ".hxx"} else 1
                return (is_header, len(str(p)))

            candidate_files = sorted({p.resolve() for p in candidate_files if p.exists()}, key=_prio)

        # Fallback: if rg unavailable, scan a small set of headers in include/
        if not candidate_files:
            for root in roots:
                try:
                    for p in list(root.rglob("*.h"))[:30]:
                        candidate_files.append(p)
                except Exception:
                    continue
                if candidate_files:
                    break

        for path in candidate_files[:40]:
            text = _read_text(path)
            if not text:
                continue
            found = _find_struct_body_in_text_with_line(text, type_name)
            if not found:
                continue
            body, line_no = found
            out: Dict[str, str] = {}
            for field_name in field_names:
                c_type = _infer_field_type_from_struct_body(body, field_name)
                if not c_type:
                    continue
                rust = _map_c_type_to_rust(c_type)
                if rust:
                    out[field_name] = rust
            if out:
                return out, {"file": str(path), "line": int(line_no)}
        return {}, None

    def _try_infer_field_types_with_llm(
        self,
        type_name: str,
        field_names: List[str],
        *,
        source_file: Optional[Path],
        c_code: str,
    ) -> Tuple[Dict[str, str], Dict]:
        meta: Dict = {"source": "llm", "confidence": 0.0, "evidence": {}}
        if not self.enable_llm or not GENERATION_AVAILABLE or generation is None:
            meta["source"] = "disabled"
            return {}, meta
        # Cache hits
        entry = self.cache.get(type_name)
        if entry:
            cached = {f: entry.fields.get(f) for f in field_names if f in entry.fields}
            if len(cached) == len(field_names):
                meta["source"] = "cache"
                meta["confidence"] = 0.0
                return cached, meta

        # Provide a small amount of C context (function code + nearby struct snippet if available).
        c_context = (c_code or "").strip()
        c_context = c_context[:2500]

        struct_snippet = ""
        roots = self._search_roots(source_file)
        for root in roots:
            for path in _rg_list_files(root, rf"\b{re.escape(type_name)}\b", max_files=10):
                text = _read_text(path)
                if not text:
                    continue
                body = _find_struct_body_in_text(text, type_name)
                if body:
                    snippet = body.strip()
                    struct_snippet = snippet[:2000]
                    break
            if struct_snippet:
                break
        struct_snippet_found = bool(struct_snippet)

        system_prompt = (
            "You are a Rust FFI engineer. "
            "Infer the most likely Rust field types for C struct fields, "
            "only to make the project compile (cargo check). "
            "Return minimal FFI-safe types only: integers, raw pointers, arrays, or known identifiers; "
            "if unsure, use `usize`."
        )
        wanted = ", ".join(field_names)
        user_prompt = f"""We need to fix Rust compilation errors: missing fields [{wanted}] on type `{type_name}`.

Context:
- This is inside a `types.rs` module (so do NOT prefix types with `crate::types::`).
- We only need a compile-friendly type. Runtime correctness is not required in this fallback.

Known C usage (function snippet, may be partial):
```c
{c_context or '/* unavailable */'}
```

Known/likely struct snippet (may be partial):
```c
{struct_snippet or '/* not found */'}
```

Output ONLY a JSON object like:
{{\"type_name\": \"{type_name}\", \"fields\": {{\"field_a\": \"usize\"}}}}
"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            resp = generation(messages)
            if isinstance(resp, dict):
                resp = resp.get("content", "") or ""
            obj = _extract_first_json_obj(resp or "")
            if not obj:
                return {}, meta
            if str(obj.get("type_name", "")) and str(obj.get("type_name", "")).strip() != type_name:
                return {}, meta
            fields_obj = obj.get("fields") or {}
            if not isinstance(fields_obj, dict):
                return {}, meta
            types_rs_snapshot = _read_text(self.types_rs_path)
            out: Dict[str, str] = {}
            for f in field_names:
                v = fields_obj.get(f)
                if not v:
                    continue
                out[f] = _sanitize_inferred_rust_type(str(v).strip(), types_rs_snapshot=types_rs_snapshot)
            if not out:
                return {}, meta

            # Confidence heuristic (bounded): prefer evidence (struct snippet + c snippet)
            conf = 0.30
            if c_context:
                conf += 0.20
            if struct_snippet_found:
                conf += 0.30
            fallback_count = sum(1 for v in out.values() if v in {"usize", "*mut core::ffi::c_void"})
            conf -= 0.05 * fallback_count
            conf = max(0.10, min(0.90, conf))
            meta["confidence"] = conf
            meta["evidence"] = {
                "struct_snippet_found": struct_snippet_found,
                "c_snippet_hash": _sha1_text(c_context) if c_context else "",
                "source_file": str(source_file) if source_file else "",
            }

            # Store to cache
            entry = self.cache.get(type_name) or StructFieldCacheEntry()
            entry.fields.update(out)
            entry.source = entry.source or "llm"
            entry.c_snippet_hash = _sha1_text(c_context) if c_context else entry.c_snippet_hash
            self.cache[type_name] = entry
            return out, meta
        except Exception as e:
            logger.debug(f"LLM field type inference failed for {type_name} ({wanted}): {e}")
            return {}, meta


def _extract_first_json_obj(text: str) -> Optional[Dict]:
    if not text:
        return None
    # Fast path: try direct parse if looks like json
    s = text.strip()
    # Strip code fences
    s = re.sub(r"^```(?:json)?", "", s).strip()
    s = re.sub(r"```$", "", s).strip()
    # Extract first {...}
    m = re.search(r"\{.*\}", s, re.DOTALL)
    if not m:
        return None
    blob = m.group(0)
    # Remove trailing commas (common)
    blob = re.sub(r",\s*([}\]])", r"\1", blob)
    try:
        obj = json.loads(blob)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None
