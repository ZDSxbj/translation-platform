#!/usr/bin/env python3
"""
基于 tree-sitter 的语义切片提取工具。

给定 C/C++ 源文件与函数名，收集该函数在同文件内依赖的其它函数定义，
并输出结构化数据，供增量翻译阶段注入上下文。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Set

from tree_sitter import Language, Parser
import tree_sitter_cpp as tscpp


try:
    CPP_LANGUAGE = Language(tscpp.language(), "cpp")
except TypeError:
    CPP_LANGUAGE = Language(tscpp.language())
CPP_PARSER = Parser()
try:
    CPP_PARSER.set_language(CPP_LANGUAGE)
except Exception:
    CPP_PARSER = Parser(CPP_LANGUAGE)


def _node_text(code: bytes, node) -> str:
    return code[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")


def _iter_nodes(node):
    stack = [node]
    while stack:
        current = stack.pop()
        yield current
        stack.extend(reversed(current.children))


def _extract_function_name(node, code: bytes) -> Optional[str]:
    declarator = node.child_by_field_name("declarator")
    if declarator is None:
        return None
    identifier = declarator.child_by_field_name("declarator")
    if identifier is None:
        return None
    return _node_text(code, identifier)


def _find_function_node(root, code: bytes, func_name: str):
    for node in _iter_nodes(root):
        if node.type != "function_definition":
            continue
        name = _extract_function_name(node, code)
        if name == func_name:
            return node
    return None


def _collect_function_definitions(root, code: bytes) -> Dict[str, str]:
    defs = {}
    for node in _iter_nodes(root):
        if node.type != "function_definition":
            continue
        name = _extract_function_name(node, code)
        if not name:
            continue
        snippet = _node_text(code, node).strip()
        defs[name] = snippet
    return defs


def _sanitize_symbol(text: str) -> str:
    """
    清洗符号名称，去掉命名空间、指针访问等噪声。
    """
    if not text:
        return ""
    for sep in ("->", ".", "::"):
        if sep in text:
            text = text.split(sep)[-1]
    text = text.split("(")[0]
    return text.strip()


def _collect_referenced_symbols(func_node, code: bytes) -> Set[str]:
    referenced: Set[str] = set()
    body = func_node.child_by_field_name("body")
    if body is None:
        body = func_node

    for node in _iter_nodes(body):
        if node.type == "call_expression":
            target = node.child_by_field_name("function")
            if target is not None:
                referenced.add(_sanitize_symbol(_node_text(code, target)))
        elif node.type == "identifier":
            referenced.add(_sanitize_symbol(_node_text(code, node)))
        elif node.type == "field_expression":
            field = node.child_by_field_name("field")
            if field is not None:
                referenced.add(_sanitize_symbol(_node_text(code, field)))

    referenced = {name for name in referenced if name and not name.isnumeric()}
    return referenced


def extract_semantic_slice(source_path: Path, function_name: str) -> Dict:
    """
    提取目标函数在同文件内依赖的定义。

    返回:
        {
            "func_name": str,
            "source_path": str,
            "infile_definitions": [{"name": str, "code": str}, ...],
            "unresolved_symbols": [str, ...]
        }
    """
    source_path = Path(source_path)
    code = source_path.read_text(encoding="utf-8", errors="ignore")
    code_bytes = code.encode("utf-8")
    tree = CPP_PARSER.parse(code_bytes)
    root = tree.root_node

    function_defs = _collect_function_definitions(root, code_bytes)
    target_node = _find_function_node(root, code_bytes, function_name)

    if target_node is None:
        return {
            "func_name": function_name,
            "source_path": str(source_path),
            "infile_definitions": [],
            "unresolved_symbols": []
        }

    referenced = _collect_referenced_symbols(target_node, code_bytes)
    referenced.discard(function_name)

    infile_defs: List[Dict[str, str]] = []
    unresolved: List[str] = []

    for symbol in sorted(referenced):
        if symbol in function_defs:
            snippet = function_defs[symbol]
            if len(snippet) > 2000:
                snippet = snippet[:2000] + "\n/* ...truncated... */"
            infile_defs.append({
                "name": symbol,
                "code": snippet
            })
        else:
            unresolved.append(symbol)

    return {
        "func_name": function_name,
        "source_path": str(source_path),
        "infile_definitions": infile_defs,
        "unresolved_symbols": unresolved
    }


def main():
    parser = argparse.ArgumentParser(description="Extract semantic slice for a function.")
    parser.add_argument("--source", required=True, help="C/C++ source file path")
    parser.add_argument("--function", required=True, help="Function name")
    parser.add_argument("--output", help="Optional output JSON file")
    args = parser.parse_args()

    result = extract_semantic_slice(Path(args.source), args.function)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
