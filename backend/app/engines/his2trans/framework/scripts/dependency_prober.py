#!/usr/bin/env python3
"""全局依赖声明探针。

扫描指定 C/C++ 项目，针对符号名称提取最相关的全局声明/定义，
便于在增量翻译阶段注入额外上下文。
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from tree_sitter import Language, Parser
import tree_sitter_cpp as tscpp


logger = logging.getLogger(__name__)

try:
    CPP_LANGUAGE = Language(tscpp.language(), "cpp")
except TypeError:
    CPP_LANGUAGE = Language(tscpp.language())
CPP_PARSER = Parser()
try:
    CPP_PARSER.set_language(CPP_LANGUAGE)
except Exception:
    CPP_PARSER = Parser(CPP_LANGUAGE)

CODE_EXTS = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".inl"}
GLOBAL_WRAPPERS = {
    "translation_unit",
    "preproc_if",
    "preproc_ifdef",
    "preproc_elif",
    "preproc_else",
    "preproc_if_section",
}


def _node_text(code: bytes, node) -> str:
    return code[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")


def _truncate(text: str, limit: int = 1000) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "\n/* ...truncated... */"


def _is_global_node(node) -> bool:
    current = node
    while current is not None:
        parent = current.parent
        if parent is None:
            return False
        if parent.type == "translation_unit":
            return True
        if parent.type in GLOBAL_WRAPPERS:
            current = parent
            continue
        # 允许 typedef/声明这类中间节点
        if parent.type in {"declaration", "type_definition"}:
            current = parent
            continue
        return False


@dataclass
class DeclarationEntry:
    symbol: str
    code: str
    source_path: str
    kind: str


class DependencyProber:
    def __init__(self, project_root: Path, max_entries_per_symbol: int = 2, max_files_per_symbol: int = 6):
        self.project_root = Path(project_root)
        self.max_entries_per_symbol = max_entries_per_symbol
        self.max_files_per_symbol = max_files_per_symbol
        self._file_index_cache: Dict[Path, Dict[str, List[DeclarationEntry]]] = {}

    def probe(self, symbols: Iterable[str]) -> Dict[str, List[Dict[str, str]]]:
        results: Dict[str, List[Dict[str, str]]] = {}
        seen: set = set()
        clean_symbols: List[str] = []
        for sym in symbols:
            normalized = (sym or "").strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            clean_symbols.append(normalized)
        for symbol in clean_symbols:
            entries = self._collect_symbol(symbol)
            if entries:
                results[symbol] = [entry.__dict__ for entry in entries]
        return results

    def _collect_symbol(self, symbol: str) -> List[DeclarationEntry]:
        candidates = self._candidate_files(symbol)
        collected: List[DeclarationEntry] = []
        for path in candidates:
            index = self._index_file(path)
            if symbol not in index:
                continue
            for entry in index[symbol]:
                if not entry.code.strip():
                    continue
                collected.append(entry)
                if len(collected) >= self.max_entries_per_symbol:
                    return collected
        # fallback: 直接截取附近文本
        for path in candidates:
            fallback = self._fallback_snippet(path, symbol)
            if fallback:
                collected.append(
                    DeclarationEntry(
                        symbol=symbol,
                        code=fallback,
                        source_path=str(path),
                        kind="text_snippet"
                    )
                )
                break
        return collected

    def _candidate_files(self, symbol: str) -> List[Path]:
        if not self.project_root.exists():
            return []
        pattern = rf"\\b{re.escape(symbol)}\\b"
        try:
            cmd = [
                "rg",
                "--with-filename",
                "--no-heading",
                "--line-number",
                "--color",
                "never",
                pattern,
                "."
            ]
            proc = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=False
            )
            lines = proc.stdout.splitlines()
        except FileNotFoundError:
            lines = []
        files: List[Path] = []
        seen: set = set()
        for line in lines:
            if not line:
                continue
            path_part = line.split(":", 1)[0]
            candidate = (self.project_root / path_part).resolve()
            if candidate in seen:
                continue
            if candidate.suffix.lower() not in CODE_EXTS:
                continue
            seen.add(candidate)
            files.append(candidate)
            if len(files) >= self.max_files_per_symbol:
                break
        if files:
            return files
        # 兜底：遍历有限数量的源文件
        iter_count = 0
        for candidate in self.project_root.rglob("*"):
            if not candidate.is_file() or candidate.suffix.lower() not in CODE_EXTS:
                continue
            files.append(candidate)
            iter_count += 1
            if iter_count >= self.max_files_per_symbol:
                break
        return files

    def _index_file(self, path: Path) -> Dict[str, List[DeclarationEntry]]:
        if path in self._file_index_cache:
            return self._file_index_cache[path]
        try:
            code = path.read_text(encoding="utf-8", errors="ignore")
        except Exception as exc:
            logger.debug("读取文件失败 %s: %s", path, exc)
            self._file_index_cache[path] = {}
            return {}
        code_bytes = code.encode("utf-8")
        try:
            tree = CPP_PARSER.parse(code_bytes)
        except Exception as exc:
            logger.debug("tree-sitter 解析失败 %s: %s", path, exc)
            self._file_index_cache[path] = {}
            return {}
        index: Dict[str, List[DeclarationEntry]] = {}
        for symbol, kind, node in _gather_declarations(tree.root_node, code_bytes):
            snippet = _truncate(_node_text(code_bytes, node))
            entry = DeclarationEntry(
                symbol=symbol,
                code=snippet,
                source_path=str(path),
                kind=kind
            )
            index.setdefault(symbol, []).append(entry)
        self._file_index_cache[path] = index
        return index

    def _fallback_snippet(self, path: Path, symbol: str) -> Optional[str]:
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            return None
        matches = [i for i, line in enumerate(lines) if symbol in line]
        if not matches:
            return None
        idx = matches[0]
        start = max(0, idx - 3)
        end = min(len(lines), idx + 4)
        snippet = "\n".join(lines[start:end])
        return _truncate(snippet, limit=600)


_FUNCTION_DEF_QUERY = CPP_LANGUAGE.query(
    """
    (function_definition
      declarator: (function_declarator
        declarator: (identifier) @symbol
      )
    ) @decl
    """
)

_FUNCTION_DECL_QUERY = CPP_LANGUAGE.query(
    """
    (declaration
      declarator: (function_declarator
        declarator: (identifier) @symbol
      )
    ) @decl
    """
)

_STRUCT_QUERY = CPP_LANGUAGE.query(
    """
    (struct_specifier
      name: (type_identifier) @symbol
    ) @decl
    """
)

_CLASS_QUERY = CPP_LANGUAGE.query(
    """
    (class_specifier
      name: (type_identifier) @symbol
    ) @decl
    """
)

_ENUM_QUERY = CPP_LANGUAGE.query(
    """
    (enum_specifier
      name: (type_identifier) @symbol
    ) @decl
    """
)

_TYPEDEF_QUERY = CPP_LANGUAGE.query(
    """
    (type_definition
      declarator: (type_identifier) @symbol
    ) @decl
    """
)

_VAR_DECL_QUERY = CPP_LANGUAGE.query(
    """
    (declaration
      declarator: (identifier) @symbol
    ) @decl
    """
)

_VAR_INIT_DECL_QUERY = CPP_LANGUAGE.query(
    """
    (declaration
      declarator: (init_declarator
        declarator: (identifier) @symbol
      )
    ) @decl
    """
)

_MACRO_QUERY = CPP_LANGUAGE.query(
    """
    (preproc_def
      name: (identifier) @symbol
    ) @decl
    """
)

_MACRO_FUNC_QUERY = CPP_LANGUAGE.query(
    """
    (preproc_function_def
      name: (identifier) @symbol
    ) @decl
    """
)


def _gather_declarations(root, code_bytes: bytes) -> List[Tuple[str, str, object]]:
    mapping = []
    queries = [
        ("function_definition", _FUNCTION_DEF_QUERY),
        ("function_declaration", _FUNCTION_DECL_QUERY),
        ("struct", _STRUCT_QUERY),
        ("class", _CLASS_QUERY),
        ("enum", _ENUM_QUERY),
        ("typedef", _TYPEDEF_QUERY),
        ("var_decl", _VAR_DECL_QUERY),
        ("var_decl", _VAR_INIT_DECL_QUERY),
        ("macro", _MACRO_QUERY),
        ("macro", _MACRO_FUNC_QUERY),
    ]
    for kind, query in queries:
        for _, captures in query.matches(root):
            node = captures.get("decl")
            symbol_node = captures.get("symbol")
            if not node or not symbol_node:
                continue
            if not _is_global_node(node):
                continue
            symbol = _node_text(code_bytes, symbol_node).strip()
            if not symbol:
                continue
            mapping.append((symbol, kind, node))
    return mapping


_PROBER_CACHE: Dict[Path, DependencyProber] = {}


def collect_global_declarations(project_root: Path, symbols: Sequence[str]) -> Dict[str, List[Dict[str, str]]]:
    project_root = Path(project_root).resolve()
    if not symbols or not project_root.exists():
        return {}
    prober = _PROBER_CACHE.get(project_root)
    if prober is None:
        prober = DependencyProber(project_root)
        _PROBER_CACHE[project_root] = prober
    return prober.probe(symbols)


def main():
    parser = argparse.ArgumentParser(description="Collect global declarations for unresolved symbols.")
    parser.add_argument("project_root", help="项目根目录")
    parser.add_argument("symbol", nargs="+", help="待解析的符号名称")
    parser.add_argument("--output", help="可选的 JSON 输出路径")
    args = parser.parse_args()

    result = collect_global_declarations(Path(args.project_root), args.symbol)
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    main()
