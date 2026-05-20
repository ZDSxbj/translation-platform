import re
import subprocess
from typing import Tuple, List, Optional
from tree_sitter import Language, Parser, QueryCursor
try:
    import tree_sitter_rust as tsrust  # type: ignore
    _HAS_TREE_SITTER_RUST = True
except Exception:
    tsrust = None  # type: ignore
    _HAS_TREE_SITTER_RUST = False


def _query_captures(query, root_node):
    """Compat wrapper: tree_sitter >=0.25 removed Query.captures()."""
    cursor = QueryCursor(query)
    captures = cursor.captures(root_node)
    if isinstance(captures, dict):
        out = []
        for cap_name, nodes in captures.items():
            for node in nodes:
                out.append((node, cap_name))
        return out
    return captures or []


import json
from pathlib import Path
import os
import sys

try:
    from project_config import get_config
    # 检查是否有项目名
    try:
        PROJECT_NAME = get_config().get("project_name", "")
    except:
        PROJECT_NAME = ""
except ImportError:
    PROJECT_NAME = ""

def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _iter_rust_code_chars(code: str, start: int = 0):
    """
    Yield (idx, ch) for characters that are *not* inside strings/comments.
    This is NOT a full Rust lexer, but is robust enough for:
    - matching braces even when strings contain '{' / '}'
    - finding delimiters like '{' / ';' safely
    """
    i = max(0, start)
    n = len(code)
    mode = "normal"
    block_depth = 0
    raw_hashes = 0

    while i < n:
        ch = code[i]

        if mode == "normal":
            if code.startswith("//", i):
                mode = "line_comment"
                i += 2
                continue
            if code.startswith("/*", i):
                mode = "block_comment"
                block_depth = 1
                i += 2
                continue

            # raw strings: r#"..."# / br#"..."# / rb#"..."#
            if code.startswith("br", i) or code.startswith("rb", i):
                j = i + 2
                hashes = 0
                while j < n and code[j] == "#":
                    hashes += 1
                    j += 1
                if j < n and code[j] == '"':
                    mode = "raw_string"
                    raw_hashes = hashes
                    i = j + 1
                    continue

            if ch == "r":
                j = i + 1
                hashes = 0
                while j < n and code[j] == "#":
                    hashes += 1
                    j += 1
                if j < n and code[j] == '"':
                    mode = "raw_string"
                    raw_hashes = hashes
                    i = j + 1
                    continue

            # byte string/char: b"..." / b'...'
            if ch == "b" and i + 1 < n and code[i + 1] in "\"'":
                mode = "string" if code[i + 1] == '"' else "char"
                i += 2
                continue

            if ch == '"':
                mode = "string"
                i += 1
                continue
            if ch == "'":
                mode = "char"
                i += 1
                continue

            yield i, ch
            i += 1
            continue

        if mode == "line_comment":
            if ch == "\n":
                mode = "normal"
            i += 1
            continue

        if mode == "block_comment":
            if code.startswith("/*", i):
                block_depth += 1
                i += 2
                continue
            if code.startswith("*/", i):
                block_depth -= 1
                i += 2
                if block_depth <= 0:
                    mode = "normal"
                continue
            i += 1
            continue

        if mode == "string":
            if ch == "\\":
                i += 2
                continue
            if ch == '"':
                mode = "normal"
                i += 1
                continue
            i += 1
            continue

        if mode == "char":
            if ch == "\\":
                i += 2
                continue
            if ch == "'":
                mode = "normal"
                i += 1
                continue
            i += 1
            continue

        if mode == "raw_string":
            if ch == '"':
                if raw_hashes == 0:
                    mode = "normal"
                    i += 1
                    continue
                tail = '"' + ("#" * raw_hashes)
                if code.startswith(tail, i):
                    i += len(tail)
                    mode = "normal"
                    continue
            i += 1
            continue


def _rust_find_matching_brace(code: str, open_brace_idx: int) -> Optional[int]:
    """Find the matching '}' for code[open_brace_idx] == '{', ignoring strings/comments."""
    if open_brace_idx < 0 or open_brace_idx >= len(code) or code[open_brace_idx] != "{":
        return None
    depth = 1
    for idx, ch in _iter_rust_code_chars(code, open_brace_idx + 1):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return idx
    return None


def _rust_find_next_delim(code: str, start: int, delims: set[str]) -> Optional[tuple[int, str]]:
    """Find next delimiter char in `delims`, ignoring strings/comments."""
    for idx, ch in _iter_rust_code_chars(code, start):
        if ch in delims:
            return idx, ch
    return None


def _rust_find_extern_c_block_ranges(code: str) -> List[tuple[int, int]]:
    """Return ranges [start, end) for `extern \"C\" { ... }` blocks (best-effort)."""
    ranges: List[tuple[int, int]] = []
    pat = re.compile(r'extern\s+"C"\s*\{')
    for m in pat.finditer(code):
        brace_idx = code.find("{", m.start(), m.end() + 2)
        if brace_idx == -1:
            continue
        close_idx = _rust_find_matching_brace(code, brace_idx)
        if close_idx is None:
            continue
        ranges.append((m.start(), close_idx + 1))
    return ranges


def extract_function_block(code: str, signature: str, func_name: str) -> Optional[str]:
    """
    Extract a function item block from code, matching by signature first then name.
    Returns the entire function item text (starting at line start), or None.
    """
    if not code:
        return None

    target_sig_norm = _norm_ws(signature)
    extern_ranges = _rust_find_extern_c_block_ranges(code)

    def _in_extern(pos: int) -> bool:
        for a, b in extern_ranges:
            if a <= pos < b:
                return True
        return False

    candidates: List[tuple[int, int, str]] = []
    for m in re.finditer(r"\bfn\s+" + re.escape(func_name) + r"\b", code):
        if _in_extern(m.start()):
            continue
        line_start = code.rfind("\n", 0, m.start()) + 1
        delim = _rust_find_next_delim(code, m.end(), {"{", ";"})
        if not delim:
            continue
        delim_idx, delim_ch = delim
        if delim_ch == ";":
            end = delim_idx + 1
        else:
            close = _rust_find_matching_brace(code, delim_idx)
            if close is None:
                continue
            end = close + 1
        header_norm = _norm_ws(code[line_start:delim_idx])
        candidates.append((line_start, end, header_norm))

    if not candidates:
        return None

    best_start, best_end, _ = candidates[0]
    best_score = -1
    for start, end, header_norm in candidates:
        score = 0
        if target_sig_norm:
            if header_norm == target_sig_norm:
                score = 10_000_000
            elif target_sig_norm in header_norm:
                score = 1_000_000 + len(target_sig_norm)
            elif header_norm in target_sig_norm:
                score = 500_000 + len(header_norm)
        score += min(len(header_norm), 10_000)
        if score > best_score:
            best_score = score
            best_start, best_end = start, end

    return code[best_start:best_end]


def run(translated_dir, test_output_dir, llm_name, project_name, timeout=700):
    """Wrapper run() expected by auto_repair_rust.py.

    Locates the translated Rust project, runs cargo build, and returns results.
    """
    target_project = os.path.join(translated_dir, llm_name, project_name)
    if not os.path.isdir(target_project):
        # Fallback: try the translated_dir directly
        target_project = translated_dir
    # Run cargo build in the target project directory
    return run_tests(target_project, ["cargo", "build"], timeout=timeout)


def run_tests(target_project, test_cmd, timeout=700):
    """
    运行编译测试命令
    
    使用 RUSTFLAGS 抑制无害警告，专注于真正的编译错误：
    - unused_imports: LLM 倾向于添加防御性 use 语句
    - dead_code: 增量翻译中间状态
    - unused_variables: 占位参数
    - unused_mut: 防御性可变声明
    
    Args:
        target_project: 项目目录
        test_cmd: 测试命令（如 ["cargo", "build"]）
        timeout: 超时时间（秒）
        
    Returns:
        (output, error, success): 标准输出、标准错误、是否成功
    """
    # 设置 RUSTFLAGS 抑制无害警告
    env = os.environ.copy()
    env["RUSTFLAGS"] = "-A unused_imports -A dead_code -A unused_variables -A unused_mut"
    
    # 并行批量运行时，限制 cargo 并行度避免过度抢占 CPU/内存
    try:
        from cargo_utils import with_cargo_jobs
        test_cmd = with_cargo_jobs(list(test_cmd))
    except Exception:
        pass

    result = subprocess.run(
        test_cmd,
        cwd=target_project, 
        timeout=timeout, 
        capture_output=True, 
        text=True,
        env=env
    )
    
    output = result.stdout
    error = result.stderr
    # 成功判断：优先使用返回码，其次匹配 stdout 中的 'Finished'
    key = (result.returncode == 0) or ("Finished" in output)
    
    if key:
        return output, error, True
    else:
        return output, error, False


def find_and_replace_function_signature(skeleton_code: str, signature: str, new_function_body: str) -> tuple:
    """
    在骨架代码中查找函数签名并替换为新的函数体。
    支持多种签名格式（有/无缩进，有/无分号，extern "C" 声明等）。

    重要：不会替换 extern "C" 块内的函数声明（这些是FFI声明，不应该有函数体）

    关键改进（2025-12-07）：
    1. 验证 new_function_body 是否包含完整的函数定义
    2. 如果 new_function_body 只是代码片段（没有 fn 关键字），拒绝替换

    关键改进（2025-12-23）：
    1. 在验证前修复常见的 C 语法混入问题
    2. 更宽松的匹配策略确保注入成功

    Returns:
        (replaced_code, success): 替换后的代码和是否成功
    """
    fn_match = re.search(r'fn\s+(\w+)', signature)
    if not fn_match:
        return skeleton_code, False

    func_name = fn_match.group(1)

    # 预处理：修复 LLM 常见的 C 语法混入问题
    def _fix_c_syntax(code: str) -> str:
        if not code:
            return code
        # 修复 (void)var; -> let _ = var;
        code = re.sub(r'\(void\)\s*(\w+)\s*;', r'let _ = \1;', code)
        # 修复 (void)(expr);
        code = re.sub(r'\(void\)\s*\(([^)]+)\)\s*;', r'let _ = \1;', code)
        # 修复 NULL -> std::ptr::null_mut()
        code = re.sub(r'\bNULL\b', 'std::ptr::null_mut()', code)
        return code

    new_function_body = _fix_c_syntax(new_function_body)

    # 防御：拒绝"只有片段/错名"的返回，避免破坏文件结构
    new_body_stripped = (new_function_body or "").strip()
    if not re.search(r'\bfn\s+' + re.escape(func_name) + r'\b', new_body_stripped):
        print(f"[DEBUG] 拒绝替换: new_function_body 未包含目标函数名 fn {func_name}")
        print(f"[DEBUG] 代码开头: {new_body_stripped[:120]}...")
        return skeleton_code, False

    # 结构校验：优先 tree-sitter（若可用），否则使用 lexer 检查花括号匹配（忽略字符串/注释）
    def _validate_snippet(snippet: str) -> bool:
        if _HAS_TREE_SITTER_RUST and parser is not None:
            try:
                wrapped = "mod __c2r_tmp {\n" + snippet + "\n}\n"
                tree = parser.parse(bytes(wrapped, "utf-8"))

                def _has_error(node):
                    if node.type == "ERROR" or getattr(node, "is_missing", False):
                        return True
                    for ch in getattr(node, "children", []) or []:
                        if _has_error(ch):
                            return True
                    return False

                return not _has_error(tree.root_node)
            except Exception:
                pass

        depth = 0
        for _, ch in _iter_rust_code_chars(snippet, 0):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth < 0:
                    return False
        return depth == 0

    if not _validate_snippet(new_body_stripped):
        print(f"[DEBUG] 拒绝替换: new_function_body 结构校验失败 (fn {func_name})")
        print(f"[DEBUG] 代码开头: {new_body_stripped[:150]}...")
        return skeleton_code, False

    def _ts_replace_function_item(code: str) -> tuple[str, bool]:
        """
        使用 tree-sitter 精确替换 function_item（避免简单 brace_count 在字符串/注释/raw string 中误计数）
        """
        if not (_HAS_TREE_SITTER_RUST and parser is not None):
            return code, False
        try:
            code_bytes = code.encode("utf-8")
            tree = parser.parse(code_bytes)

            target_sig_norm = _norm_ws(signature)
            candidates = []

            def walk(node, inside_extern: bool = False):
                nonlocal candidates
                now_inside = inside_extern or (node.type == "extern_block")

                if node.type == "function_item" and not now_inside:
                    name_node = node.child_by_field_name("name")
                    if name_node is not None:
                        name = code_bytes[name_node.start_byte:name_node.end_byte].decode(
                            "utf-8", errors="ignore"
                        )
                        if name == func_name:
                            body_node = node.child_by_field_name("body")
                            if body_node is not None:
                                header_bytes = code_bytes[node.start_byte:body_node.start_byte]
                            else:
                                header_bytes = code_bytes[node.start_byte:node.end_byte]
                            header_norm = _norm_ws(
                                header_bytes.decode("utf-8", errors="ignore")
                            )
                            candidates.append((node, header_norm))

                for ch in getattr(node, "children", []) or []:
                    walk(ch, now_inside)

            walk(tree.root_node, False)
            if not candidates:
                return code, False

            best_node = None
            best_score = -1
            for node, header_norm in candidates:
                score = 0
                if target_sig_norm:
                    if header_norm == target_sig_norm:
                        score = 10_000_000
                    elif target_sig_norm in header_norm:
                        score = 1_000_000 + len(target_sig_norm)
                    elif header_norm in target_sig_norm:
                        score = 500_000 + len(header_norm)
                # 兜底：至少按“更长的 header”倾向选择更具体的候选
                score += min(len(header_norm), 10_000)

                if score > best_score:
                    best_score = score
                    best_node = node

            if best_node is None:
                best_node = candidates[0][0]

            new_bytes = (
                code_bytes[:best_node.start_byte]
                + new_function_body.encode("utf-8")
                + code_bytes[best_node.end_byte:]
            )
            replaced = new_bytes.decode("utf-8", errors="ignore")
            print(f"[DEBUG] tree-sitter: 成功替换定义 {func_name}")
            return replaced, True
        except Exception as e:
            print(f"[DEBUG] tree-sitter: 替换失败 {func_name}: {e}")
            return code, False

    # 1) 首选：tree-sitter 精准替换（若可用）
    new_code, ok = _ts_replace_function_item(skeleton_code)
    if ok:
        return new_code, True

    # 2) 兜底：lexer 精确替换（可替换定义/声明；忽略字符串/注释中的花括号）
    target_sig_norm = _norm_ws(signature)
    extern_ranges = _rust_find_extern_c_block_ranges(skeleton_code)

    def _in_extern(pos: int) -> bool:
        for a, b in extern_ranges:
            if a <= pos < b:
                return True
        return False

    candidates: List[tuple[int, int, str]] = []
    for m in re.finditer(r"\bfn\s+" + re.escape(func_name) + r"\b", skeleton_code):
        if _in_extern(m.start()):
            continue
        line_start = skeleton_code.rfind("\n", 0, m.start()) + 1
        delim = _rust_find_next_delim(skeleton_code, m.end(), {"{", ";"})
        if not delim:
            continue
        delim_idx, delim_ch = delim
        if delim_ch == ";":
            end = delim_idx + 1
        else:
            close = _rust_find_matching_brace(skeleton_code, delim_idx)
            if close is None:
                continue
            end = close + 1
        header_norm = _norm_ws(skeleton_code[line_start:delim_idx])
        candidates.append((line_start, end, header_norm))

    if not candidates:
        print(f"[DEBUG] 所有策略失败: {func_name} (no candidates)")
        return skeleton_code, False

    best_start, best_end, _ = candidates[0]
    best_score = -1
    for start, end, header_norm in candidates:
        score = 0
        if target_sig_norm:
            if header_norm == target_sig_norm:
                score = 10_000_000
            elif target_sig_norm in header_norm:
                score = 1_000_000 + len(target_sig_norm)
            elif header_norm in target_sig_norm:
                score = 500_000 + len(header_norm)
        score += min(len(header_norm), 10_000)
        if score > best_score:
            best_score = score
            best_start, best_end = start, end

    # Preserve indentation of the original line.
    indent = re.match(r'^\s*', skeleton_code[best_start:]).group(0)

    def _indent_lines(snippet: str, prefix: str) -> str:
        if not prefix:
            return snippet
        if snippet.startswith(prefix):
            return snippet
        lines = snippet.splitlines(True)
        return "".join((prefix + ln if ln.strip() else ln) for ln in lines)

    replacement = (new_function_body.rstrip() + "\n")
    replacement = _indent_lines(replacement, indent)

    replaced = skeleton_code[:best_start] + replacement + skeleton_code[best_end:]
    print(f"[DEBUG] lexer: 成功替换 {func_name}")
    return replaced, True


if _HAS_TREE_SITTER_RUST:
    # tree-sitter >= 0.22 requires Language(..., name). Keep a fallback for older bindings.
    try:
        RS_LANGUAGE = Language(tsrust.language(), "rust")
    except TypeError:
        RS_LANGUAGE = Language(tsrust.language())
    parser = Parser()
    try:
        parser.set_language(RS_LANGUAGE)
    except Exception:
        # Backward compatibility for older tree-sitter bindings.
        parser = Parser(RS_LANGUAGE)
    query_function_defin_text = """
    (
        (function_item) @function.defin
    )
    """

    query_import_text = """
    (
        (use_declaration) @use.name
    )
    """
    query_function_defin = RS_LANGUAGE.query(query_function_defin_text)
    query_import = RS_LANGUAGE.query(query_import_text)
else:
    RS_LANGUAGE = None
    parser = None
    query_function_defin = None
    query_import = None


def traverse_target_node(node, source_code, target_node, target_statements):
    if len(target_statements) != 0:
        return
    
    if node.type == target_node:
        # call_functions.append(node)
        target_statements.append(node.start_byte)
        return
    
    for child in node.children:
        traverse_target_node(child, source_code, target_node, target_statements)

def get_frist_function_position_start(source_code):
    """
    获取代码中第一个函数定义的起始位置
    
    Args:
        source_code: 源代码字符串

    Returns:
        第一个函数的起始字节位置，如果没有找到函数则返回代码末尾位置
    """
    # tree-sitter 可用时使用 AST；否则回退到正则（足够用于“插到第一个 fn 之前”）。
    if parser is None:
        m = re.search(r'(?m)^[ \t]*(?:pub(?:\([^)]*\))?\s+)?(?:unsafe\s+)?(?:extern\s+"C"\s+)?fn\s+\w+\b', source_code)
        return m.start() if m else len(source_code)

    byte_source_code = bytes(source_code, "utf-8")
    tree = parser.parse(byte_source_code)
    translated_code_function_signature = []
    traverse_target_node(tree.root_node, byte_source_code, "function_item", translated_code_function_signature)

    if not translated_code_function_signature:
        return len(source_code)

    return translated_code_function_signature[0]


def add_import_to_translated_result(content, imports: list):
    """
    将导入语句添加到翻译结果中
    
    Args:
        content: 源代码（可以是字符串或行列表）
        imports: 要添加的导入语句列表
    
    Returns:
        添加导入后的代码（与输入类型相同）
    """
    # 处理输入类型
    input_was_string = isinstance(content, str)
    if input_was_string:
        content = content.splitlines(keepends=True)
        # 如果原字符串没有换行符，添加它们
        if content and not content[-1].endswith('\n'):
            content[-1] += '\n'
    
    # 查找现有 use 语句的位置
    index = next((i for i, line in enumerate(content) if line.strip().startswith("use ")), None)
    
    # 如果没有找到 use 语句，在文件开头添加
    if index is None:
        index = 0
    
    # 过滤掉已存在的导入
    existing_imports = set()
    for line in content:
        stripped = line.strip()
        if stripped.startswith("use "):
            existing_imports.add(stripped)
    
    new_imports = []
    for imp in imports:
        imp_stripped = imp.strip()
        if imp_stripped and imp_stripped not in existing_imports:
            new_imports.append(imp_stripped)
    
    # 插入新导入
    for imp in reversed(new_imports):
        content.insert(index, imp + "\n")
    
    # 返回与输入相同的类型
    if input_was_string:
        return "".join(content)
    return content


def read_translated_function(code: str) -> tuple:
    """
    解析翻译后的 Rust 代码，提取函数签名、函数体和导入语句

    Args:
        code: 翻译后的 Rust 代码字符串

    Returns:
        (签名列表, 函数体列表, 导入列表)
    """
    # Extract only the <translated function> section to avoid
    # parsing prompt template content (which contains Rust signatures)
    tf_match = re.search(r'<translated function>\s*\n?(.*?)\n?\s*</translated function>',
                         code, re.DOTALL)
    if tf_match:
        code = tf_match.group(1).strip()
        if not code:
            return [], [], []

    # tree-sitter 路径（更精确）
    if parser is not None and query_function_defin is not None and query_import is not None:
        byte_code = bytes(code, "utf-8")
        tree = parser.parse(byte_code)

        function_captures = _query_captures(query_function_defin, tree.root_node)
        functions = []
        signatures = []

        for node, _ in function_captures:
            func_text = code[node.start_byte:node.end_byte]
            functions.append(func_text)
            brace_idx = func_text.find('{')
            if brace_idx != -1:
                signatures.append(func_text[:brace_idx].strip())

        import_captures = _query_captures(query_import, tree.root_node)
        imports = []
        for node, _ in import_captures:
            imports.append(code[node.start_byte:node.end_byte])

        return signatures, functions, imports

    # fallback：正则 + lexer（不依赖 tree_sitter_rust）
    functions: List[str] = []
    signatures: List[str] = []
    imports: List[str] = []

    for line in code.splitlines():
        s = line.strip()
        if (s.startswith("use ") or s.startswith("pub use ")) and s.endswith(";"):
            imports.append(s)

    fn_pat = re.compile(
        r'(?m)^[ \t]*(?:pub(?:\([^)]*\))?\s+)?(?:unsafe\s+)?(?:extern\s+"C"\s+)?fn\s+(\w+)\b'
    )
    for m in fn_pat.finditer(code):
        start = code.rfind("\n", 0, m.start()) + 1
        delim = _rust_find_next_delim(code, m.end(), {"{", ";"})
        if not delim:
            continue
        delim_idx, delim_ch = delim
        if delim_ch != "{":
            continue
        close = _rust_find_matching_brace(code, delim_idx)
        if close is None:
            continue
        func_text = code[start:close + 1]
        functions.append(func_text)
        sig = code[start:delim_idx].strip()
        if sig:
            signatures.append(sig)

    return signatures, functions, imports


def _merge_translated_into_skeleton(skeleton_code: str, translated_code: str, func_signature: str) -> str:
    """
    将翻译后的函数代码合并到骨架文件中
    
    Args:
        skeleton_code: 骨架代码
        translated_code: 翻译后的代码
        func_signature: 函数签名（用于定位）
        
    Returns:
        合并后的代码
    """
    # 解析翻译代码
    try:
        _, functions, imports = read_translated_function(translated_code)
        if functions:
            translated_function = functions[0]
        else:
            translated_function = translated_code
    except Exception:
        translated_function = translated_code
        imports = []
    
    # 使用签名匹配进行替换
    result_code, success = find_and_replace_function_signature(
        skeleton_code, func_signature, translated_function
                )
    
    if success:
        # 添加导入
        if imports:
            lines = result_code.splitlines(keepends=True)
            lines = add_import_to_translated_result(lines, imports)
            result_code = "".join(lines)
        return result_code
    else:
        # 如果签名匹配失败，在文件末尾添加
        frist_function_position = get_frist_function_position_start(skeleton_code)
        result = list(skeleton_code)
        result.insert(frist_function_position, "\n" + translated_function + "\n")
        result_code = "".join(result)
        
        if imports:
            lines = result_code.splitlines(keepends=True)
            lines = add_import_to_translated_result(lines, imports)
            result_code = "".join(lines)
        
        return result_code


def read_test_results(test_dir: str) -> dict:
    """
    读取测试结果目录中的测试结果
    
    Args:
        test_dir: 测试结果目录
        
    Returns:
        测试结果字典
    """
    results = {
        "passed": 0,
        "failed": 0,
        "errors": [],
        "details": {}
    }
    
    test_path = Path(test_dir)
    if not test_path.exists():
        results["errors"].append(f"测试目录不存在: {test_dir}")
        return results
    
    # 查找所有测试文件
    test_files = list(test_path.glob("*.txt"))
    
    for test_file in test_files:
        try:
            with open(test_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read().strip()
            
            # 判断测试结果
            if content.lower().startswith("success"):
                results["passed"] += 1
                results["details"][test_file.name] = "passed"
            else:
                results["failed"] += 1
                results["details"][test_file.name] = "failed"
                
        except Exception as e:
            results["errors"].append(f"读取测试文件失败 {test_file}: {e}")
            results["details"][test_file.name] = "error"
    
    return results


def extract_translated_import(code: str) -> list:
    """
    从代码中提取导入语句
    
    Args:
        code: Rust 代码
        
    Returns:
        导入语句列表
    """
    imports = []
    for line in code.splitlines():
        stripped = line.strip()
        if stripped.startswith("use "):
            imports.append(stripped)
    return imports


def find_function_in_code(code: str, func_name: str) -> tuple:
    """
    在代码中查找函数定义
    
    Args:
        code: 源代码
        func_name: 函数名
        
    Returns:
        (start_pos, end_pos) 或 (None, None)
    """
    if parser is None or query_function_defin is None:
        block = extract_function_block(code, f"fn {func_name}", func_name)
        if not block:
            return None, None
        start = code.find(block)
        if start == -1:
            return None, None
        return start, start + len(block)

    byte_code = bytes(code, "utf-8")
    tree = parser.parse(byte_code)

    function_captures = _query_captures(query_function_defin, tree.root_node)

    for node, _ in function_captures:
        func_text = code[node.start_byte:node.end_byte]
        fn_match = re.search(r'fn\s+(\w+)', func_text)
        if fn_match and fn_match.group(1) == func_name:
            return node.start_byte, node.end_byte

    return None, None


def remove_function_from_code(code: str, func_name: str) -> str:
    """
    从代码中移除指定的函数
    
    Args:
        code: 源代码
        func_name: 要移除的函数名
        
    Returns:
        移除函数后的代码
    """
    start, end = find_function_in_code(code, func_name)
    if start is not None and end is not None:
        return code[:start] + code[end:]
    return code


if __name__ == "__main__":
    # 简单的命令行接口用于测试
    import sys
    if len(sys.argv) > 1:
        test_dir = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else None
        results = run_tests(test_dir, output_dir)
        print(json.dumps(results, indent=2))
