#!/usr/bin/env python3
"""
Context slicer for reducing prompt size (types.rs).

Goal:
- Avoid putting the entire bindgen-generated `types.rs` into every LLM prompt.
- Build a symbol registry (name -> code block + deps) from `types.rs`.
- For each function/file, extract a small set of relevant symbols and expand a dependency closure.
- Render only that slice to be injected into the prompt.

Design notes:
- Prefer tree-sitter-rust when available for robust item extraction.
- Dependencies are computed via "known-symbol token intersection" (deterministic, fast).
- This module does NOT modify any generated files; it only slices context text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

try:
    from tree_sitter import Language, Parser, QueryCursor  # type: ignore

    try:
        import tree_sitter_rust as tsrust  # type: ignore

        _RUST_LANGUAGE = Language(tsrust.language(), "rust")
        _RUST_PARSER: Optional[Parser] = Parser()
        _RUST_PARSER.set_language(_RUST_LANGUAGE)
    except Exception:
        _RUST_LANGUAGE = None  # type: ignore
        _RUST_PARSER = None
except Exception:
    _RUST_LANGUAGE = None  # type: ignore
    _RUST_PARSER = None


_IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")


def _query_captures_compat(query, root_node):
    """Return a stable List[(node, capture_name)] across tree-sitter versions."""
    if hasattr(query, "captures"):
        captures = query.captures(root_node)
    else:
        cursor = QueryCursor(query)
        captures = cursor.captures(root_node)

    if isinstance(captures, dict):
        out = []
        for cap_name, nodes in captures.items():
            for node in nodes:
                out.append((node, cap_name))
        return out
    return captures or []


def _normalize_symbol(s: str) -> str:
    """Best-effort: normalize a candidate symbol token to a Rust identifier."""
    if not s:
        return ""
    s = s.strip()
    s = s.strip("`")
    # Rust paths: crate::types::Foo / types::Foo
    if "::" in s:
        s = s.split("::")[-1]
    # C tags embedded in display names: "struct Foo" / "enum Bar"
    s = s.replace("struct ", "").replace("enum ", "").replace("union ", "").strip()
    # Pointers/refs in Rust types: "*mut Foo" -> Foo
    s = s.replace("*const", "").replace("*mut", "").replace("&mut", "").replace("&", "").strip()
    # Remove trailing punctuation
    s = s.strip(" ,;()[]{}")
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", s):
        return ""
    return s


def extract_identifiers(text: str) -> Set[str]:
    """Extract identifier tokens (deterministic regex)."""
    if not text:
        return set()
    return set(_IDENT_RE.findall(text))


def extract_types_from_rust_signature(sig: str) -> Set[str]:
    """Extract `types::X` / `crate::types::X` symbols from a Rust signature string."""
    if not sig:
        return set()
    out: Set[str] = set()
    for m in re.finditer(r"(?:crate::)?types::([A-Za-z_][A-Za-z0-9_]*)", sig):
        out.add(m.group(1))
    return out


def extract_missing_symbols_from_rustc_output(rustc_output: str) -> Set[str]:
    """
    Parse rustc output and extract missing identifiers likely belonging to `crate::types`.

    This is used to "expand the slice" for repair prompts without falling back to full types.rs.
    """
    if not rustc_output:
        return set()
    out: Set[str] = set()

    # Most common:
    # - cannot find type `X` in module `crate::types`
    # - cannot find value `X` in module `crate::types`
    for m in re.finditer(
        r"cannot find (?:type|value|constant) `([^`]+)` in module `crate::types`",
        rustc_output,
    ):
        n = _normalize_symbol(m.group(1))
        if n:
            out.add(n)

    # Sometimes the module is omitted; keep it conservative.
    for m in re.finditer(r"cannot find type `([^`]+)` in this scope", rustc_output):
        n = _normalize_symbol(m.group(1))
        if n:
            out.add(n)

    for m in re.finditer(r"cannot find value `([^`]+)` in this scope", rustc_output):
        n = _normalize_symbol(m.group(1))
        if n:
            out.add(n)

    # E0433 (common with missing types):
    #   failed to resolve: use of undeclared type `X`
    for m in re.finditer(r"use of undeclared type `([^`]+)`", rustc_output):
        n = _normalize_symbol(m.group(1))
        if n:
            out.add(n)

    # Another common variant:
    #   failed to resolve: could not find `X` in `types`
    for m in re.finditer(r"could not find `([^`]+)` in `(?:crate::)?types`", rustc_output):
        n = _normalize_symbol(m.group(1))
        if n:
            out.add(n)

    return out


@dataclass(frozen=True)
class SymbolEntry:
    name: str
    kind: str  # struct|enum|union|type|const
    code: str
    start_byte: int
    deps: Tuple[str, ...] = ()


class TypesRsRegistry:
    """
    Build a symbol registry from bindgen `types.rs`.

    - Collect top-level definitions: struct/enum/union/type/const.
    - Compute dependencies as a closure over known symbol identifiers.
    - Slice the registry by a set of seed symbols.
    """

    def __init__(self, entries: Dict[str, SymbolEntry], *, full_text: str):
        self._entries = dict(entries)
        self._full_text = full_text
        self._name_set = set(self._entries.keys())

    @property
    def names(self) -> Set[str]:
        return set(self._name_set)

    @property
    def full_text(self) -> str:
        return self._full_text

    @classmethod
    def from_types_rs(cls, types_rs_path: Path) -> "TypesRsRegistry":
        text = types_rs_path.read_text(encoding="utf-8", errors="ignore")
        entries = _build_entries_from_rust(text)
        entries = _fill_deps(entries)
        return cls(entries, full_text=text)

    def filter_existing(self, candidates: Iterable[str]) -> Set[str]:
        out: Set[str] = set()
        for c in candidates:
            n = _normalize_symbol(str(c))
            if n and n in self._name_set:
                out.add(n)
        return out

    def slice_symbols(self, seed_symbols: Iterable[str], *, extra: Iterable[str] = ()) -> List[SymbolEntry]:
        seeds = set(self.filter_existing(seed_symbols))
        seeds.update(self.filter_existing(extra))
        if not seeds:
            return []

        visited: Set[str] = set()
        queue: List[str] = sorted(seeds)
        while queue:
            cur = queue.pop(0)
            if cur in visited:
                continue
            visited.add(cur)
            ent = self._entries.get(cur)
            if not ent:
                continue
            for d in ent.deps:
                if d not in visited:
                    queue.append(d)

        # Deterministic order: keep file order by start_byte.
        out_entries = [self._entries[n] for n in visited if n in self._entries]
        out_entries.sort(key=lambda e: e.start_byte)
        return out_entries

    def render_slice(self, seed_symbols: Iterable[str], *, extra: Iterable[str] = (), header: bool = True) -> str:
        entries = self.slice_symbols(seed_symbols, extra=extra)
        if not entries:
            return ""
        blocks: List[str] = []
        if header:
            blocks.append("// === types.rs slice (bindgen) ===")
        for e in entries:
            blocks.append(e.code.rstrip())
        return "\n\n".join(blocks).strip() + "\n"


def _build_entries_from_rust(text: str) -> Dict[str, SymbolEntry]:
    """
    Build entries from Rust source.
    Prefer tree-sitter; fallback to regex (best-effort).
    """
    if _RUST_PARSER is None or _RUST_LANGUAGE is None:
        return _build_entries_from_rust_regex(text)

    try:
        tree = _RUST_PARSER.parse(text.encode("utf-8"))
        query = _RUST_LANGUAGE.query(
            """
            (struct_item name: (type_identifier) @name) @item
            (enum_item name: (type_identifier) @name) @item
            (union_item name: (type_identifier) @name) @item
            (type_item name: (type_identifier) @name) @item
            (const_item name: (identifier) @name) @item
            """
        )
        captures = _query_captures_compat(query, tree.root_node)

        # We want stable "item -> name", so we record the last seen name per item node.
        item_to_name: Dict[int, str] = {}
        item_nodes: Dict[int, object] = {}
        for node, cap in captures:
            if cap == "name":
                name = text[node.start_byte : node.end_byte]
                # Walk up to the top-level item node.
                it = node.parent
                while it is not None and it.type not in ("struct_item", "enum_item", "union_item", "type_item", "const_item"):
                    it = it.parent
                if it is None:
                    continue
                item_to_name[id(it)] = name
                item_nodes[id(it)] = it

        out: Dict[str, SymbolEntry] = {}
        for it_id, name in item_to_name.items():
            it = item_nodes.get(it_id)
            if it is None:
                continue
            n = _normalize_symbol(name)
            if not n or n in out:
                continue
            kind_map = {
                "struct_item": "struct",
                "enum_item": "enum",
                "union_item": "union",
                "type_item": "type",
                "const_item": "const",
            }
            kind = kind_map.get(getattr(it, "type", ""), "item")
            code = text[it.start_byte : it.end_byte].strip()
            out[n] = SymbolEntry(name=n, kind=kind, code=code, start_byte=int(it.start_byte), deps=())
        return out
    except Exception:
        return _build_entries_from_rust_regex(text)


def _build_entries_from_rust_regex(text: str) -> Dict[str, SymbolEntry]:
    """
    Fallback entry builder.
    This is intentionally conservative and may miss some items; it is only used
    when tree-sitter-rust is unavailable.
    """
    out: Dict[str, SymbolEntry] = {}

    # type aliases + consts (line-based)
    for m in re.finditer(r"(?m)^\\s*pub\\s+type\\s+([A-Za-z_][A-Za-z0-9_]*)\\s*=.*?;\\s*$", text):
        name = m.group(1)
        if name not in out:
            out[name] = SymbolEntry(name=name, kind="type", code=m.group(0).strip(), start_byte=m.start(), deps=())
    for m in re.finditer(r"(?m)^\\s*pub\\s+const\\s+([A-Za-z_][A-Za-z0-9_]*)\\b.*?;\\s*$", text):
        name = m.group(1)
        if name not in out:
            out[name] = SymbolEntry(name=name, kind="const", code=m.group(0).strip(), start_byte=m.start(), deps=())

    # structs/enums/unions: best-effort "pub struct Name { ... }" block.
    # This is not a full Rust parser; it can be confused by nested braces in impls/macros.
    block_pat = re.compile(r"(?s)\\bpub\\s+(struct|enum|union)\\s+([A-Za-z_][A-Za-z0-9_]*)\\b.*?\\n\\}", re.MULTILINE)
    for m in block_pat.finditer(text):
        kind = m.group(1)
        name = m.group(2)
        if name not in out:
            out[name] = SymbolEntry(name=name, kind=kind, code=m.group(0).strip(), start_byte=m.start(), deps=())

    return out


def _fill_deps(entries: Dict[str, SymbolEntry]) -> Dict[str, SymbolEntry]:
    names = set(entries.keys())
    out: Dict[str, SymbolEntry] = {}
    for name, ent in entries.items():
        tokens = set(_IDENT_RE.findall(ent.code))
        deps = sorted((tokens & names) - {name})
        out[name] = SymbolEntry(
            name=ent.name,
            kind=ent.kind,
            code=ent.code,
            start_byte=ent.start_byte,
            deps=tuple(deps),
        )
    return out
