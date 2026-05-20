#!/usr/bin/env python3
"""依赖用法示例索引构建工具。

扫描 C/C++ 工程中的函数调用，按照符号名聚合调用片段，
并缓存到 JSON 文件，供增量翻译阶段引入 "Usage Examples" 上下文。
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


logger = logging.getLogger(__name__)

CODE_GLOBS = ["*.c", "*.cc", "*.cpp", "*.cxx", "*.h", "*.hh", "*.hpp"]
CODE_EXTS = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp"}
MAX_SNIPPET_LEN = 500


def _load_cache(cache_file: Path) -> Dict[str, List[Dict]]:
    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.debug("读取用法缓存失败 %s: %s", cache_file, exc)
    return {}


def _save_cache(cache_file: Path, data: Dict[str, List[Dict]]) -> None:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_file_lines(path: Path) -> List[str]:
    try:
        return path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception as exc:
        logger.debug("读取文件失败 %s: %s", path, exc)
        return []


def _build_snippet(lines: List[str], line_no: int, radius: int = 1) -> str:
    if not lines:
        return ""
    idx = max(0, line_no - 1)
    start = max(0, idx - radius)
    end = min(len(lines), idx + radius + 1)
    snippet = "\n".join(lines[start:end]).strip()
    snippet = snippet.replace("\t", "    ")
    if len(snippet) > MAX_SNIPPET_LEN:
        snippet = snippet[:MAX_SNIPPET_LEN] + "\n/* ...truncated... */"
    return snippet


def _estimate_arg_count(snippet: str, symbol: str) -> Optional[int]:
    pattern = re.compile(rf"{re.escape(symbol)}\s*\((.*?)\)", re.S)
    match = pattern.search(snippet)
    if not match:
        return None
    inside = match.group(1).strip()
    if not inside:
        return 0
    # 忽略括号嵌套的简单估计
    items = [seg.strip() for seg in inside.split(',')]
    items = [seg for seg in items if seg]
    return len(items)


def _collect_examples(project_root: Path, symbol: str, limit: int = 3) -> List[Dict[str, str]]:
    pattern = rf"{re.escape(symbol)}\s*\("
    project_root = project_root.resolve()
    if not project_root.exists():
        return []
    cmd: List[str] = [
        "rg",
        "-n",
        "-H",
        "--no-heading",
        "--color",
        "never",
        "-m",
        str(limit * 4),
    ]
    for glob in CODE_GLOBS:
        cmd.extend(["--glob", glob])
    cmd.extend([pattern, str(project_root)])
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )
    except FileNotFoundError:
        logger.warning("未找到 rg 命令，无法构建用法索引")
        return []
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    examples: List[Dict[str, str]] = []
    file_cache: Dict[Path, List[str]] = {}
    seen_snippets: set = set()
    for line in lines:
        try:
            file_part, line_part, _ = line.split(":", 2)
        except ValueError:
            continue
        file_path = Path(file_part)
        if not file_path.is_absolute():
            file_path = (project_root / file_path).resolve()
        if file_path.suffix.lower() not in CODE_EXTS:
            continue
        try:
            line_no = int(line_part)
        except ValueError:
            continue
        if file_path not in file_cache:
            file_cache[file_path] = _read_file_lines(file_path)
        snippet = _build_snippet(file_cache[file_path], line_no)
        if not snippet:
            continue
        dedup_key = (file_path, line_no, snippet)
        if dedup_key in seen_snippets:
            continue
        seen_snippets.add(dedup_key)
        arg_count = _estimate_arg_count(snippet, symbol)
        examples.append({
            "symbol": symbol,
            "file": str(file_path),
            "line": line_no,
            "snippet": snippet,
            "arg_count": arg_count
        })
        if len(examples) >= limit:
            break
    return examples


def ensure_usage_examples(
    project_root: Path,
    cache_file: Path,
    symbols: Sequence[str],
    max_examples_per_symbol: int = 2
) -> Dict[str, List[Dict[str, str]]]:
    project_root = Path(project_root)
    cache_file = Path(cache_file)
    cache = _load_cache(cache_file)
    updated = False
    ordered_symbols: List[str] = []
    seen = set()
    for sym in symbols:
        normalized = (sym or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered_symbols.append(normalized)
    targets: List[str] = []
    for normalized in ordered_symbols:
        current = cache.get(normalized, [])
        if len(current) >= max_examples_per_symbol:
            continue
        targets.append(normalized)
    for symbol in targets:
        snippets = _collect_examples(project_root, symbol, limit=max_examples_per_symbol)
        if snippets:
            cache.setdefault(symbol, [])
            cache[symbol].extend(snippets)
            deduped: List[Dict] = []
            seen_keys = set()
            for item in cache[symbol]:
                key = (item.get("file"), item.get("line"), item.get("snippet"))
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                deduped.append(item)
                if len(deduped) >= max_examples_per_symbol:
                    break
            cache[symbol] = deduped
            updated = True
    if updated:
        _save_cache(cache_file, cache)
    return {sym: cache.get(sym, []) for sym in ordered_symbols}


def main():
    parser = argparse.ArgumentParser(description="构建/更新依赖用法示例索引")
    parser.add_argument("project_root", help="C/C++ 工程根目录")
    parser.add_argument("cache_file", help="缓存文件路径 (JSON)")
    parser.add_argument("symbol", nargs="+", help="要索引的符号名称")
    parser.add_argument("--max-examples", type=int, default=2, help="每个符号保留的示例数量")
    args = parser.parse_args()

    project_root = Path(args.project_root)
    cache_file = Path(args.cache_file)
    symbols = args.symbol
    result = ensure_usage_examples(project_root, cache_file, symbols, max_examples_per_symbol=args.max_examples)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    main()
