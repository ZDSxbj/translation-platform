from tree_sitter import Language, Parser, Query, QueryCursor
try:
    import tree_sitter_rust as tsrust  # type: ignore
    _HAS_TREE_SITTER_RUST = True
except Exception:
    tsrust = None  # type: ignore
    _HAS_TREE_SITTER_RUST = False
import tree_sitter_cpp as tscpp

from generate.generation import generation
from auto_test_rust import run_tests
import sys
import os
import re
import shutil
from get_dependencies import get_dependencies
from project_config import PROJECT_NAME, PROJECT_ROOT, get_all_source_files_with_headers
from workspace_config import (
    get_skeleton_path,
    get_source_skeleton_path,
    get_repair_results_path,
    get_functions_path,
    get_dependencies_path,
    get_dependencies_not_in_file_path,
    get_extracted_path,
    ensure_dirs,
    get_compile_commands_path,
)
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import time
import logging
import json
from typing import Any, Dict, List

# 导入日志配置并初始化
from log_config import setup_logging, LogPrinter, ensure_logging_setup
ensure_logging_setup()

# 创建 logger 实例
logger = logging.getLogger(__name__)
log = LogPrinter(__name__)

# 导入新的分层骨架构建器
from skeleton_builder import (
    SkeletonBuilder,
    create_signature_translation_prompt,
    parse_llm_signature_response
)

# 导入调用图构建器
try:
    from call_graph import CallGraphBuilder, LLMContextProvider
    CALL_GRAPH_AVAILABLE = True
except ImportError:
    CALL_GRAPH_AVAILABLE = False
    CallGraphBuilder = None
    LLMContextProvider = None

# tree-sitter language init (compat with newer bindings)
try:
    CPP_LANGUAGE = Language(tscpp.language(), "cpp")
except TypeError:
    CPP_LANGUAGE = Language(tscpp.language())
parser = Parser()
try:
    parser.set_language(CPP_LANGUAGE)
except Exception:
    parser = Parser(CPP_LANGUAGE)

def traverse(node, source_code, depth=0):
    # Get the node text
    node_text = source_code[node.start_byte:node.end_byte].decode('utf-8')
    print(" " * depth + f"{node.type}: {node_text}")
    for child in node.children:
        # print(" " * depth + f"{child.type}: {source_code[child.start_byte:child.end_byte].decode('utf-8')}")
        traverse(child, source_code, depth + 1)

def read_file(path):
    with open(path, 'r', encoding='utf-8', errors='ignore') as input_file:
        content = input_file.read() 

    return content

def get_source_code(target_file_path):

    source_code = read_file(target_file_path)

    # 转化成bytes！！否则在出现中文注释时，根据偏移获得对应内容会出错
    source_code = bytes(source_code, "utf-8")

    return source_code

def traverse_target_node(node, source_code, target_node, target_statements):
    """
    迭代版遍历，避免递归过深导致 RecursionError。
    收集所有类型为 target_node 的语法节点对应源码片段。
    """
    stack = [node]
    while stack:
        cur = stack.pop()
        if cur.type == target_node:
            snippet = source_code[cur.start_byte:cur.end_byte].decode('utf-8', errors='ignore').strip()
            target_statements.append(snippet)
            # 不 return，继续收集可能的后续匹配
        # 压栈子节点
        # 为了接近原先的先序遍历，这里逆序压栈
        for child in reversed(cur.children):
            stack.append(child)

def get_include(source_code):
    source_code = bytes(source_code, "utf-8")
    tree = parser.parse(source_code)
    target_statements = []
    traverse_target_node(tree.root_node, source_code, "preproc_include", target_statements)
    return target_statements

def get_function(source_code):
    import re
    source_code_bytes = bytes(source_code, "utf-8")
    tree = parser.parse(source_code_bytes)
    target_statements = []
    traverse_target_node(tree.root_node, source_code_bytes, "function_definition", target_statements)

    function_name_to_statements = {}
    for function_statements in target_statements:
        function_name = []
        bytes_source_code = bytes(function_statements, "utf-8")
        tree = parser.parse(bytes_source_code)
        traverse_target_node(tree.root_node, bytes_source_code, "function_declarator", function_name)
        
        # 检查是否成功提取到函数名
        # 先检查列表是否为空，避免 IndexError
        has_valid_function_name = False
        if function_name and len(function_name) > 0:
            if function_name[0] and function_name[0].strip():
                has_valid_function_name = True
        
        if not has_valid_function_name:
            # 如果无法提取函数名，尝试从函数定义中直接提取
            # 使用更智能的正则表达式匹配
            try:
                # 移除注释和预处理指令
                clean_code = re.sub(r'//.*?$|/\*.*?\*/', '', function_statements, flags=re.MULTILINE | re.DOTALL)
                clean_code = re.sub(r'^\s*#.*?$', '', clean_code, flags=re.MULTILINE)
                
                # 尝试多种函数定义模式
                patterns = [
                    # 析构函数: ~ClassName()
                    (r'~\s*(\w+)\s*\([^)]*\)\s*{', 'destructor'),
                    # 运算符重载: operator+(参数)
                    (r'operator\s*([^\s(]+)\s*\([^)]*\)\s*{', 'operator'),
                    # 类成员函数: ClassName::functionName(参数)
                    (r'(\w+(?:::\w+)*)::(\w+)\s*\([^)]*\)\s*{', 'member'),
                    # 普通函数: functionName(参数) {
                    (r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)\s*(const\s*)?{', 'function'),
                ]
                
                # 需要排除的关键字和类型
                excluded_keywords = {
                    'if', 'for', 'while', 'switch', 'catch', 'try', 'else',
                    'return', 'void', 'int', 'bool', 'char', 'float', 
                    'double', 'long', 'short', 'unsigned', 'signed',
                    'const', 'static', 'extern', 'inline', 'virtual',
                    'class', 'struct', 'enum', 'union', 'typedef',
                    'namespace', 'using', 'template', 'typename',
                    'public', 'private', 'protected', 'friend',
                    'new', 'delete', 'this', 'nullptr', 'NULL',
                    'true', 'false', 'nullptr', 'auto', 'decltype'
                }
                
                func_name = None
                for pattern, pattern_type in patterns:
                    match = re.search(pattern, clean_code)
                    if match:
                        if pattern_type == 'destructor':
                            func_name = f"~{match.group(1)}"
                        elif pattern_type == 'operator':
                            func_name = f"operator{match.group(1)}"
                        elif pattern_type == 'member':
                            func_name = f"{match.group(1)}::{match.group(2)}"
                        elif pattern_type == 'function':
                            candidate = match.group(1)
                            if candidate not in excluded_keywords:
                                func_name = candidate
                        
                        if func_name and func_name not in excluded_keywords:
                            break
                        func_name = None
                
                if func_name:
                    # 如果函数名已存在，添加后缀
                    original_func_name = func_name
                    counter = 1
                    while func_name in function_name_to_statements:
                        func_name = f"{original_func_name}_{counter}"
                        counter += 1
                    function_name_to_statements[func_name] = function_statements
                    continue
            except Exception as e:
                # 如果正则提取也失败，继续到匿名函数处理
                pass
            
            # 如果仍然无法提取，使用索引作为函数名（带文件哈希以避免冲突）
            index = len(function_name_to_statements) + 1
            func_hash = hash(function_statements[:100]) % 10000
            function_name_to_statements[f"anonymous_function_{index}_{func_hash}"] = function_statements
            continue
        
        # 正常情况：使用提取到的函数名
        # function_name[0] 已经是字符串（由 traverse_target_node 转换）
        func_name = function_name[0].strip()
        
        # 处理函数名可能包含参数的情况
        func_name = func_name.split('(')[0].strip()
        
        # 清理函数名（移除可能的模板参数）
        if '<' in func_name:
            func_name = func_name.split('<')[0].strip()
        
        # 如果函数名已存在，添加后缀
        original_func_name = func_name
        counter = 1
        while func_name in function_name_to_statements:
            func_name = f"{original_func_name}_{counter}"
            counter += 1
        
        function_name_to_statements[func_name] = function_statements
    
    return function_name_to_statements

def get_variable(source_code):
    """提取变量声明（包括静态变量、全局变量）"""
    source_code_bytes = bytes(source_code, "utf-8")
    tree = parser.parse(source_code_bytes)
    target_statements = []
    traverse_target_node(tree.root_node, source_code_bytes, "declaration", target_statements)
    
    return target_statements


def get_static_variables(source_code):
    """
    提取源代码中的静态变量和全局变量声明
    
    返回: List[Dict] - 变量信息列表，每个元素包含 {name, type, declaration, is_pointer, is_array}
    """
    variables = []
    seen_vars = set()
    
    # 1. 匹配 static 变量声明
    # static Type var; 或 static Type var = value; 或 static Type *var;
    static_patterns = [
        # static Type *var = value;
        r'static\s+(?:const\s+)?([\w:]+)\s*\*\s*(\w+)\s*(?:=\s*([^;]+))?\s*;',
        # static Type var = value;
        r'static\s+(?:const\s+)?([\w:]+)\s+(\w+)\s*(?:=\s*([^;]+))?\s*;',
        # static Type var[SIZE];
        r'static\s+(?:const\s+)?([\w:]+)\s+(\w+)\s*\[([^\]]*)\]\s*(?:=\s*\{[^}]*\})?\s*;',
        # static Type *var[SIZE];
        r'static\s+(?:const\s+)?([\w:]+)\s*\*\s*(\w+)\s*\[([^\]]*)\]\s*;',
    ]
    
    for pattern in static_patterns:
        for match in re.finditer(pattern, source_code, re.MULTILINE):
            var_type = match.group(1).strip() if match.group(1) else "unknown"
            var_name = match.group(2).strip() if match.group(2) else None
            
            if not var_name or var_name in seen_vars:
                continue
            seen_vars.add(var_name)
            
            is_pointer = '*' in pattern or '*' in match.group(0)
            is_array = '[' in match.group(0)
            
            variables.append({
                'name': var_name,
                'type': var_type,
                'declaration': match.group(0).strip(),
                'is_pointer': is_pointer,
                'is_array': is_array
            })
    
    # 2. 匹配全局变量（g_xxx 或 G_XXX 开头）
    global_patterns = [
        # Type* g_varName = value;
        r'^[\t ]*([\w:]+)\s*\*\s*(g_\w+|G_\w+)\s*(?:=\s*([^;]+))?\s*;',
        # Type g_varName = value;
        r'^[\t ]*([\w:]+)\s+(g_\w+|G_\w+)\s*(?:=\s*([^;]+))?\s*;',
    ]
    
    for pattern in global_patterns:
        for match in re.finditer(pattern, source_code, re.MULTILINE):
            var_type = match.group(1).strip() if match.group(1) else "unknown"
            var_name = match.group(2).strip() if match.group(2) else None
            
            if not var_name or var_name in seen_vars:
                continue
            seen_vars.add(var_name)
            
            is_pointer = '*' in match.group(0)
            
            variables.append({
                'name': var_name,
                'type': var_type,
                'declaration': match.group(0).strip(),
                'is_pointer': is_pointer,
                'is_array': False
            })
    
    # 3. 扫描函数体中使用但可能未声明的全局变量（g_xxx 或 G_XXX）
    # 这是一个启发式方法，帮助识别遗漏的全局变量
    global_var_usage_pattern = r'\b(g_\w+|G_[A-Z_]+)\b'
    for match in re.finditer(global_var_usage_pattern, source_code):
        var_name = match.group(1)
        if var_name not in seen_vars:
            seen_vars.add(var_name)
            variables.append({
                'name': var_name,
                'type': 'unknown',  # 类型未知，让 LLM 推断
                'declaration': f'// Used but not declared: {var_name}',
                'is_pointer': True,  # 假设是指针（更安全）
                'is_array': False
            })
    
    return variables

def split_include_and_content(source_code):
    include_statements = get_include(source_code)
    content_statements = str(source_code)
    for include_statement in include_statements:

        # print(include_statement, " ", content_statements.find(include_statement))
        content_statements = content_statements.replace(include_statement, "")

    # print(content_statements)
    return include_statements, content_statements

def extract_import(translated_code):
    # translated_code 可能是 bytes 或 str
    if isinstance(translated_code, bytes):
        code_bytes = translated_code
        code_text = translated_code.decode("utf-8", errors="ignore")
    else:
        code_text = str(translated_code)
        code_bytes = code_text.encode("utf-8", errors="ignore")

    # tree-sitter-rust 不可用时，退化为简单的行级提取
    if not _HAS_TREE_SITTER_RUST:
        imports = []
        for line in code_text.splitlines():
            s = line.strip()
            if (s.startswith("use ") or s.startswith("pub use ")) and s.endswith(";"):
                imports.append(s)
        return imports

    try:
        RS_LANGUAGE = Language(tsrust.language(), "rust")
    except TypeError:
        RS_LANGUAGE = Language(tsrust.language())
    rust_parser = Parser()
    try:
        rust_parser.set_language(RS_LANGUAGE)
    except Exception:
        rust_parser = Parser(RS_LANGUAGE)

    query_import_text = """
    (
        (use_declaration) @use.name
    )
    """

    query_import = RS_LANGUAGE.query(query_import_text)

    def _query_captures_compat(query, root_node):
        """兼容不同 tree-sitter 版本的 captures 接口。
        返回 List[(node, capture_name)]。
        """
        if hasattr(query, "captures"):
            captures = query.captures(root_node)
        else:
            cursor = QueryCursor(query)
            captures = cursor.captures(root_node)

        # 有些实现返回 dict: {capture_name: [nodes]}
        if isinstance(captures, dict):
            out = []
            for cap_name, nodes in captures.items():
                for node in nodes:
                    out.append((node, cap_name))
            return out
        return captures or []
    
    node = rust_parser.parse(code_bytes).root_node

    # get import
    import_codes = []
    
    import_captures = _query_captures_compat(query_import, node)

    for import_capture in import_captures:
        import_node, _ = import_capture
        import_code = code_bytes[import_node.start_byte:import_node.end_byte].decode("utf-8", errors="ignore").strip()
        import_codes.append(f"{import_code}")
    # print(import_codes)
    return import_codes

def add_after_comment(include_statements, source_code):
    bytes_source_code = bytes(source_code, "utf-8")
    tree = parser.parse(bytes_source_code)
    
    for node in tree.root_node.children:
        if node.type != "comment":
            ret_source_code = list(source_code)
            ret_source_code.insert(node.start_byte, "\n".join(include_statements) + "\n\n")
            return ''.join(ret_source_code)
        
    return bytes_source_code.decode().strip()

def add_before_class(functions_signature, source_code):
    """
    在类定义之前添加函数签名。
    如果没有类定义，则在注释和预处理指令之后添加函数签名。
    """
    if not functions_signature or all(not sig.strip() for sig in functions_signature):
        return source_code
    
    byte_source_code = bytes(source_code, "utf-8")
    tree = parser.parse(byte_source_code)
    target_statements = []
    traverse_target_node(tree.root_node, byte_source_code, "class_specifier", target_statements)
    
    signatures_str = "\n".join(sig for sig in functions_signature if sig.strip())
    
    if len(target_statements) == 0:
        # 没有类定义，找到合适的插入位置（注释和预处理指令之后）
        lines = source_code.split('\n')
        insert_line = 0
        
        # 跳过文件开头的注释、空行和预处理指令
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('/*') or stripped.startswith('*') or stripped.startswith('//'):
                insert_line = i + 1
            elif stripped.startswith('#'):
                insert_line = i + 1
            elif stripped == '':
                continue
            elif stripped.startswith('extern') and '"C"' in stripped:
                # 找到 extern "C" 块，在其之前插入
                insert_line = i
                break
            elif stripped.startswith('namespace') or stripped.startswith('class') or stripped.startswith('struct'):
                insert_line = i
                break
            elif stripped and not stripped.startswith('/*') and not stripped.startswith('*') and not stripped.startswith('//') and not stripped.startswith('#'):
                # 遇到第一个有效代码行
                if i > 0:
                    insert_line = i
                break
        
        # 在找到的位置插入函数签名
        lines.insert(insert_line, '\n' + signatures_str + '\n')
        return '\n'.join(lines)
    
    insert_position = source_code.find(target_statements[0])
    new_source_code = source_code[:insert_position] + signatures_str + "\n\n" + source_code[insert_position:]
    return new_source_code

def extract_code(content, file_type="rust"):
    """
    从 LLM 响应中提取代码
    Args:
        content: LLM 响应内容
        file_type: 文件类型，支持 "rust" 或 "toml"
    """
    translated_result = content
    translated_code = None

    # 根据文件类型选择匹配模式
    if file_type == "toml":
        patterns = [r'```toml(.*?)```', r'```TOML(.*?)```', r'```(.*?)```']
    else:
        patterns = [r'```rust(.*?)```', r'```Rust(.*?)```', r'```(.*?)```']

    for pattern in patterns:
        if translated_code is None:
            try:
                matches = re.findall(pattern, translated_result, re.DOTALL)
                if matches:
                    translated_code = matches[0].strip()
            except (re.error, TypeError, AttributeError) as e:
                logger.debug(f"Pattern match failed: {e}")
                translated_code = None
        else:
            break
    
    if translated_code is None:
        translated_code = translated_result
    
    # 后处理：修复骨架翻译中的常见问题
    if file_type == "rust" and translated_code:
        translated_code = _postprocess_rust_skeleton(translated_code)
    
    return translated_code


def _postprocess_rust_skeleton(code: str) -> str:
    """
    后处理 Rust 骨架代码，修复 LLM 生成的常见问题
    """
    if not code:
        return code
    
    # 1. 移除循环类型别名（如 pub type i32 = i32;）
    invalid_type_patterns = [
        r'pub\s+type\s+(\w+)\s*=\s*\1\s*;',  # pub type i32 = i32;
        r'type\s+(\w+)\s*=\s*\1\s*;',         # type i32 = i32;
    ]
    for pattern in invalid_type_patterns:
        code = re.sub(pattern, '', code)
    
    # 2. 移除明显无效的 crate 导入（仅移除语法上明显错误的，不针对特定项目）
    # 只移除导入了不存在的 crate 子模块的情况（如 use crate::SomeUndefinedModule;）
    # 注意：不再针对特定项目的函数名进行硬编码过滤
    
    # 3. 移除重复的前向声明（如果后面有完整定义）
    forward_decl_pattern = r'^(\s*)(pub\s+)?(struct|enum)\s+(\w+)\s*;'
    full_def_pattern = r'^(\s*)(?:pub\s+)?(struct|enum)\s+(\w+)\s*[{\(]'
    
    # 收集所有有完整定义的类型名
    full_defined_types = set()
    for match in re.finditer(full_def_pattern, code, re.MULTILINE):
        full_defined_types.add(match.group(3))
    
    # 移除这些类型的前向声明
    def remove_forward_decl(match):
        type_name = match.group(4)
        if type_name in full_defined_types:
            return ''  # 移除前向声明
        return match.group(0)  # 保留
    
    code = re.sub(forward_decl_pattern, remove_forward_decl, code, flags=re.MULTILINE)
    
    # 4. 去重和合并导入语句（增强版：支持多种导入格式）
    code = _merge_duplicate_imports(code)
    
    # 5. （可选）保守兜底：修复 LLM 生成的无效函数体（把参数声明语法当作函数调用）
    #
    # 历史策略：直接将该函数体替换为 unimplemented!()，以“保证骨架可编译”为第一优先级。
    # 但这会抹平语法错误信号，削弱后续增量修复的有效性，因此默认关闭。
    #
    # 如需恢复旧行为，可设置：
    #   C2R_SANITIZE_INVALID_FUNCTION_BODIES=1
    try:
        sanitize_invalid = str(os.environ.get("C2R_SANITIZE_INVALID_FUNCTION_BODIES", "0")).lower() in (
            "1",
            "true",
            "yes",
        )
    except Exception:
        sanitize_invalid = False
    if sanitize_invalid:
        code = _fix_invalid_function_bodies(code)
    
    # 6. 修复 struct 和函数同名冲突
    # 单元 struct (pub struct X;) 与同名函数 (pub fn X()) 会冲突
    code = _fix_struct_fn_name_conflict(code)
    
    # 7. 修复 trait 和 impl 块中的错误
    code = _fix_trait_and_impl_errors(code)
    
    # 8. 修复孤立的 pub 声明（pub 后面没有 item）
    code = _fix_orphan_pub_declarations(code)
    
    # 9. 扫描并补充未声明的全局变量（兜底机制）
    code = _fix_undeclared_global_variables(code)
    
    # 10. 修复宏定义变体不完整的问题
    code = _fix_incomplete_macro_variants(code)
    
    # 11. 清理多余的空行
    code = re.sub(r'\n{3,}', '\n\n', code)
    
    return code


def _fix_orphan_pub_declarations(code: str) -> str:
    """
    修复孤立的 pub 声明（pub 后面应该跟着一个 item，如 fn, struct, const 等）
    例如: "pub message_handle: ..." 应该被移除或包装成 struct 字段
    """
    lines = code.split('\n')
    result_lines = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        # 检测孤立的 pub 字段声明（不在 struct 内）
        # 模式: pub identifier: Type
        orphan_field_pattern = r'^(\s*)pub\s+(\w+)\s*:\s*(.+)$'
        match = re.match(orphan_field_pattern, line)
        
        if match:
            # 检查这是否在 struct/impl 块内
            # 简单检测：检查前面是否有未闭合的 struct/impl 块
            prev_code = '\n'.join(result_lines[-20:]) if len(result_lines) >= 20 else '\n'.join(result_lines)
            
            # 计算是否在块内
            in_block = False
            brace_count = 0
            for prev_line in result_lines[-50:] if len(result_lines) >= 50 else result_lines:
                if re.search(r'(struct|impl|enum)\s+\w+.*\{', prev_line):
                    brace_count += prev_line.count('{') - prev_line.count('}')
                    in_block = brace_count > 0
                else:
                    brace_count += prev_line.count('{') - prev_line.count('}')
                    if brace_count <= 0:
                        in_block = False
            
            if not in_block:
                # 这是一个孤立的字段声明，跳过它
                # 或者将其转换为注释
                indent = match.group(1)
                field_name = match.group(2)
                field_type = match.group(3)
                # 将其转换为注释，保留信息以供手动修复
                result_lines.append(f'{indent}// TODO: Orphan field declaration removed: pub {field_name}: {field_type}')
                i += 1
                continue
        
        result_lines.append(line)
        i += 1
    
    return '\n'.join(result_lines)


def _fix_undeclared_global_variables(code: str) -> str:
    """
    扫描代码中使用但未声明的全局变量（g_xxx 或 G_XXX 模式），
    并在模块级别补充占位声明。
    
    这是一个兜底机制，确保即使 LLM 遗漏了变量声明，代码也能编译。
    """
    if not code:
        return code
    
    # 1. 收集已声明的全局变量
    declared_vars = set()
    
    # 匹配 static mut xxx 声明
    static_decl_pattern = r'static\s+(?:mut\s+)?(\w+)\s*:'
    for match in re.finditer(static_decl_pattern, code):
        declared_vars.add(match.group(1))
    
    # 匹配 const XXX 声明
    const_decl_pattern = r'const\s+(\w+)\s*:'
    for match in re.finditer(const_decl_pattern, code):
        declared_vars.add(match.group(1))
    
    # 匹配 let xxx 声明（函数内）
    let_decl_pattern = r'let\s+(?:mut\s+)?(\w+)\s*[=:]'
    for match in re.finditer(let_decl_pattern, code):
        declared_vars.add(match.group(1))
    
    # 2. 扫描代码中使用的全局变量（g_xxx 或 G_XXX 模式）
    # 排除函数定义和类型定义中的同名情况
    used_vars = set()
    
    # 匹配 g_xxx 或 G_XXX 形式的标识符
    global_var_pattern = r'\b(g_\w+|G_[A-Z_][A-Z0-9_]*)\b'
    for match in re.finditer(global_var_pattern, code):
        var_name = match.group(1)
        # 排除在函数签名中作为参数名的情况
        # 检查前面是否有 fn 关键字（函数定义）
        start = match.start()
        prev_text = code[max(0, start-50):start]
        if re.search(r'fn\s+\w+\s*\([^)]*$', prev_text):
            continue  # 这是函数参数，跳过
        used_vars.add(var_name)
    
    # 3. 找出使用但未声明的变量
    undeclared_vars = used_vars - declared_vars
    
    if not undeclared_vars:
        return code
    
    # 4. 生成占位声明
    declarations = []
    declarations.append('\n// Auto-generated placeholder declarations for undeclared global variables')
    declarations.append('// TODO: Review and fix these declarations with correct types')
    
    for var_name in sorted(undeclared_vars):
        # 根据命名模式推断可能的类型
        if 'Interface' in var_name or 'Callback' in var_name or 'Handler' in var_name:
            # 接口指针
            decl = f'static mut {var_name}: *mut std::ffi::c_void = std::ptr::null_mut();'
        elif var_name.startswith('G_') and var_name.isupper():
            # 常量风格，可能是数组或简单值
            decl = f'static mut {var_name}: i32 = 0;  // TODO: Determine correct type'
        elif 'SIZE' in var_name.upper() or 'COUNT' in var_name.upper() or 'NUM' in var_name.upper():
            # 数量相关
            decl = f'static mut {var_name}: usize = 0;'
        else:
            # 默认使用指针类型（更安全）
            decl = f'static mut {var_name}: *mut std::ffi::c_void = std::ptr::null_mut();'
        
        declarations.append(decl)
    
    declarations.append('')
    
    # 5. 将声明插入到代码中适当的位置（在第一个 static 声明之前，或在 use 语句之后）
    lines = code.split('\n')
    insert_idx = 0
    
    # 找到最后一个 use 语句的位置
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('use '):
            insert_idx = i + 1
        elif stripped.startswith('static ') and insert_idx > 0:
            # 在第一个 static 声明之前插入
            break
    
    # 如果没找到 use 语句，在开头的注释之后插入
    if insert_idx == 0:
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped and not stripped.startswith('//') and not stripped.startswith('/*'):
                insert_idx = i
                break
    
    # 插入声明
    lines.insert(insert_idx, '\n'.join(declarations))
    
    return '\n'.join(lines)


def _fix_incomplete_macro_variants(code: str) -> str:
    """
    修复宏定义变体不完整的问题。
    
    常见问题：宏只定义了带可变参数的变体 ($($arg:tt)*)，
    但调用时只传了固定数量的参数，导致编译错误。
    
    解决方案：为常见的日志宏添加多个变体。
    """
    if not code:
        return code
    
    # 1. 找到所有宏定义
    # 匹配 macro_rules! NAME { ... }
    macro_pattern = r'macro_rules!\s+(\w+)\s*\{'
    
    macros_to_fix = []
    for match in re.finditer(macro_pattern, code):
        macro_name = match.group(1)
        # 只处理看起来像日志宏的名称
        if any(log_keyword in macro_name.upper() for log_keyword in ['LOG', 'DISC_', 'HDF_', 'HILOG', 'PRINT', 'DEBUG', 'INFO', 'WARN', 'ERROR']):
            macros_to_fix.append((macro_name, match.start()))
    
    if not macros_to_fix:
        return code
    
    # 2. 检查每个宏是否只有单一变体（带可变参数）
    for macro_name, start_pos in macros_to_fix:
        # 找到宏定义的完整范围
        # 简单方法：找到配对的 }
        brace_count = 0
        end_pos = start_pos
        in_macro = False
        
        for i in range(start_pos, len(code)):
            if code[i] == '{':
                brace_count += 1
                in_macro = True
            elif code[i] == '}':
                brace_count -= 1
                if in_macro and brace_count == 0:
                    end_pos = i + 1
                    break
        
        macro_def = code[start_pos:end_pos]
        
        # 检查是否只有一个变体且包含 $($arg:tt)*
        # 如果是，替换为多变体版本
        if '$($arg:tt)*' in macro_def or '$( $arg:tt )*' in macro_def:
            # 检查是否只有一个规则（一个 => ）
            arrow_count = macro_def.count('=>')
            
            if arrow_count == 1:
                # 只有一个变体，需要添加更多变体
                # 生成新的多变体宏定义
                new_macro = f'''macro_rules! {macro_name} {{
    ($format:expr) => {{}};
    ($module:expr, $format:expr) => {{}};
    ($module:expr, $format:expr, $($arg:tt)*) => {{}};
}}'''
                # 替换原有宏定义
                code = code[:start_pos] + new_macro + code[end_pos:]
    
    return code


def _fix_struct_fn_name_conflict(code: str) -> str:
    """
    修复单元 struct 和同名函数的冲突
    当存在 'pub struct X;' 和 'pub fn X()' 时，将函数重命名为 'new_x' 或移除
    """
    # 找到所有单元 struct 定义
    unit_struct_pattern = r'^(\s*)(?:pub\s+)?struct\s+(\w+)\s*;'
    unit_structs = set()
    for match in re.finditer(unit_struct_pattern, code, re.MULTILINE):
        unit_structs.add(match.group(2))
    
    if not unit_structs:
        return code
    
    # 找到与单元 struct 同名的函数并重命名
    for struct_name in unit_structs:
        # 匹配与 struct 同名的函数定义
        # pub fn StructName(...) -> ReturnType {
        fn_pattern = rf'^(\s*)(pub\s+)?fn\s+{re.escape(struct_name)}\s*(\([^)]*\))\s*(->.*?)?\s*\{{'
        
        def rename_fn(match):
            indent = match.group(1) or ''
            pub = match.group(2) or ''
            params = match.group(3)
            ret_type = match.group(4) or ''
            # 将函数重命名为 new_xxx
            new_name = f'new_{struct_name.lower()}'
            return f'{indent}{pub}fn {new_name}{params} {ret_type}{{'
        
        code = re.sub(fn_pattern, rename_fn, code, flags=re.MULTILINE)
    
    return code


def _fix_trait_and_impl_errors(code: str) -> str:
    """
    修复 trait 和 impl 块中的常见错误
    """
    # 0. 修复孤立的 &self 方法（不在 impl 块内的方法使用了 &self）
    code = _fix_orphan_self_methods(code)
    
    # 1. 修复 trait 定义中的 pub fn（trait 方法不能有 pub）
    # 匹配 trait 块内的 pub fn
    trait_pattern = r'(pub\s+)?trait\s+\w+\s*\{'
    
    lines = code.split('\n')
    result_lines = []
    in_trait = False
    brace_count = 0
    
    for line in lines:
        if not in_trait:
            if re.search(trait_pattern, line):
                in_trait = True
                brace_count = line.count('{') - line.count('}')
            result_lines.append(line)
        else:
            brace_count += line.count('{') - line.count('}')
            # 在 trait 内部，移除 pub fn 中的 pub
            if 'pub fn' in line or 'pub async fn' in line:
                line = re.sub(r'\bpub\s+(async\s+)?fn\b', r'\1fn', line)
            result_lines.append(line)
            
            if brace_count <= 0:
                in_trait = False
    
    code = '\n'.join(result_lines)
    
    # 2. 修复 impl 块内的 static 变量（移到模块级别）
    impl_static_pattern = r'(\s*)impl\s+\w+\s*\{[^}]*?\n(\s*)(pub\s+)?static\s+(mut\s+)?(\w+)\s*:\s*([^=;]+)\s*=\s*([^;]+);'
    
    # 收集需要移动的 static 变量
    statics_to_move = []
    for match in re.finditer(impl_static_pattern, code, re.DOTALL):
        pub = match.group(3) or ''
        mut = match.group(4) or ''
        name = match.group(5)
        type_ = match.group(6).strip()
        value = match.group(7).strip()
        statics_to_move.append(f'{pub}static {mut}{name}: {type_} = {value};')
    
    # 移除 impl 块内的 static 变量声明
    code = re.sub(r'\n\s*(pub\s+)?static\s+(mut\s+)?\w+\s*:\s*[^=;]+\s*=\s*[^;]+;\s*(?=\n)', '', code)
    
    # 将 static 变量添加到文件开头（在导入之后）
    if statics_to_move:
        # 找到最后一个 use 语句的位置
        lines = code.split('\n')
        insert_idx = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('use '):
                insert_idx = i + 1
        
        # 插入 static 变量
        for static_decl in statics_to_move:
            lines.insert(insert_idx, static_decl)
            insert_idx += 1
        
        code = '\n'.join(lines)
    
    return code


def _fix_orphan_self_methods(code: str) -> str:
    """
    修复孤立的 &self 方法（不在 impl/trait 块内但使用了 &self 参数的函数）
    这种情况通常是 LLM 生成了一个应该在 impl 块内的方法但放在了顶层
    """
    lines = code.split('\n')
    result_lines = []
    
    # 追踪是否在 impl/trait 块内
    in_impl_or_trait = False
    brace_count = 0
    
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        # 检测 impl/trait 块开始
        if re.search(r'^(pub\s+)?(impl|trait)\s+', stripped):
            in_impl_or_trait = True
            brace_count = line.count('{') - line.count('}')
            result_lines.append(line)
            i += 1
            continue
        
        # 追踪大括号
        if in_impl_or_trait:
            brace_count += line.count('{') - line.count('}')
            if brace_count <= 0:
                in_impl_or_trait = False
            result_lines.append(line)
            i += 1
            continue
        
        # 检测孤立的 &self 方法
        # 模式: fn name(&self, ...) -> ... {
        if not in_impl_or_trait and re.search(r'^(\s*)(pub\s+)?fn\s+\w+\s*\(\s*&self', line):
            # 这是一个孤立的方法，需要移除 &self 或跳过
            # 我们选择将其转换为普通函数（移除 &self）
            
            # 收集整个函数
            fn_lines = [line]
            fn_brace_count = line.count('{') - line.count('}')
            
            # 如果函数体在同一行
            if fn_brace_count == 0 and '{' in line:
                # 单行函数
                # 移除 &self 参数
                fixed_line = re.sub(r'\(\s*&self\s*,?\s*', '(', line)
                fixed_line = re.sub(r'\(\s*\)', '()', fixed_line)
                result_lines.append(f'// NOTE: Converted from method to function (removed &self)')
                result_lines.append(fixed_line)
                i += 1
                continue
            
            # 多行函数，收集整个函数
            j = i + 1
            while j < len(lines) and fn_brace_count > 0:
                fn_lines.append(lines[j])
                fn_brace_count += lines[j].count('{') - lines[j].count('}')
                j += 1
            
            # 修复函数签名
            fn_code = '\n'.join(fn_lines)
            fixed_fn = re.sub(r'\(\s*&self\s*,?\s*', '(', fn_code)
            fixed_fn = re.sub(r'\(\s*\)', '()', fixed_fn)
            
            result_lines.append('// NOTE: Converted from method to function (removed &self)')
            result_lines.extend(fixed_fn.split('\n'))
            i = j
            continue
        
        result_lines.append(line)
        i += 1
    
    return '\n'.join(result_lines)


def _merge_duplicate_imports(code: str) -> str:
    """
    合并重复的导入语句（支持多种导入格式）
    """
    lines = code.split('\n')
    seen_imports = {}  # 完整路径 -> 导入项集合
    exact_imports = set()  # 完全相同的导入行（用于检测完全重复）
    import_lines = []  # (行号, 原始行)
    non_import_lines = []  # (行号, 非导入行)
    
    # 第一遍：收集所有导入
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('use '):
            import_lines.append((i, line))
            
            # 检测完全重复
            if stripped in exact_imports:
                continue
            exact_imports.add(stripped)
            
            # 解析导入路径和项
            # 支持格式：
            # use path::item;
            # use path::{item1, item2};
            # use path::*;
            # use path;
            
            # 尝试匹配 use path::{items};
            match = re.match(r'use\s+([\w:]+)::\{([^}]+)\}\s*;', stripped)
            if match:
                base_path = match.group(1)
                items_str = match.group(2)
                items = set(item.strip() for item in items_str.split(',') if item.strip())
                if base_path not in seen_imports:
                    seen_imports[base_path] = set()
                seen_imports[base_path].update(items)
                continue
            
            # 尝试匹配 use path::item; 或 use path::*;
            match = re.match(r'use\s+([\w:]+)::(\w+|\*)\s*;', stripped)
            if match:
                base_path = match.group(1)
                item = match.group(2)
                if base_path not in seen_imports:
                    seen_imports[base_path] = set()
                seen_imports[base_path].add(item)
                continue
            
            # 尝试匹配 use path; (无具体项)
            match = re.match(r'use\s+([\w:]+)\s*;', stripped)
            if match:
                full_path = match.group(1)
                # 将完整路径作为单独项存储
                if full_path not in seen_imports:
                    seen_imports[full_path] = set()
                seen_imports[full_path].add('__self__')  # 标记为自身导入
        else:
            non_import_lines.append((i, line))
    
    # 生成合并后的导入语句
    merged_imports = []
    for base_path in sorted(seen_imports.keys()):
        items = seen_imports[base_path]
        
        # 如果有 * 通配符，只使用 *
        if '*' in items:
            merged_imports.append(f'use {base_path}::*;')
        elif '__self__' in items and len(items) == 1:
            # 只有自身导入
            merged_imports.append(f'use {base_path};')
        elif '__self__' in items:
            # 有自身导入和其他项
            other_items = items - {'__self__'}
            if len(other_items) == 1:
                merged_imports.append(f'use {base_path}::{list(other_items)[0]};')
            else:
                items_str = ", ".join(sorted(other_items))
                merged_imports.append(f'use {base_path}::{{{items_str}}};')
        elif len(items) == 1:
            merged_imports.append(f'use {base_path}::{list(items)[0]};')
        else:
            items_str = ", ".join(sorted(items))
            merged_imports.append(f'use {base_path}::{{{items_str}}};')
    
    # 重建代码：找到第一个导入行的位置，将所有合并后的导入放在那里
    result_lines = []
    import_inserted = False
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('use '):
            if not import_inserted:
                # 在第一个导入位置插入所有合并后的导入
                result_lines.extend(merged_imports)
                import_inserted = True
            # 跳过原始导入行
            continue
        result_lines.append(line)
    
    # 如果没有导入行，直接返回原始代码
    if not import_inserted:
        return code
    
    return '\n'.join(result_lines)


def _fix_invalid_function_bodies(code: str) -> str:
    """
    修复 LLM 生成的无效函数体，例如把参数声明语法当作函数调用
    """
    lines = code.split('\n')
    result_lines = []
    in_function = False
    brace_count = 0
    current_function_lines = []
    
    for i, line in enumerate(lines):
        if not in_function:
            # 检查是否是函数开始（在 impl 块内或独立函数）
            if re.search(r'(?:pub\s+)?(?:unsafe\s+)?(?:extern\s+"C"\s+)?fn\s+\w+', line) and '{' in line:
                in_function = True
                brace_count = line.count('{') - line.count('}')
                current_function_lines = [line]
                if brace_count == 0:
                    # 单行函数，直接保留
                    result_lines.append(line)
                    in_function = False
                    current_function_lines = []
            else:
                result_lines.append(line)
        else:
            current_function_lines.append(line)
            brace_count += line.count('{') - line.count('}')
            
            if brace_count <= 0:
                # 函数结束，检查函数体
                fn_code = '\n'.join(current_function_lines)
                
                # 检查是否有无效的参数声明语法
                body_match = re.search(r'\{([\s\S]*)\}$', fn_code)
                if body_match:
                    body = body_match.group(1)
                    # 检查是否有 "func_name(\n  param: Type," 模式
                    invalid_call = re.search(r'\w+\s*\(\s*\n?\s*\w+\s*:\s*[&*]?(?:mut\s+)?[\w<>:]+', body)
                    if invalid_call:
                        # 找到无效语法，替换整个函数体
                        fn_sig_match = re.match(r'([\s\S]*?)\{', fn_code)
                        if fn_sig_match:
                            fn_sig = fn_sig_match.group(1).rstrip()
                            # 计算缩进
                            first_line = current_function_lines[0]
                            indent = len(first_line) - len(first_line.lstrip())
                            indent_str = ' ' * indent
                            body_indent = ' ' * (indent + 4)
                            fixed_fn = f'{fn_sig} {{\n{body_indent}// TODO: Invalid function body - needs manual implementation\n{body_indent}unimplemented!()\n{indent_str}}}'
                            result_lines.append(fixed_fn)
                        else:
                            result_lines.extend(current_function_lines)
                    else:
                        result_lines.extend(current_function_lines)
                else:
                    result_lines.extend(current_function_lines)
                
                in_function = False
                current_function_lines = []
    
    # 处理未结束的函数
    if current_function_lines:
        result_lines.extend(current_function_lines)
    
    return '\n'.join(result_lines)

def get_function_mapping(code_under_translation, translated_result, current_file_name):
    byte_source_code = bytes(code_under_translation, "utf-8")
    tree = parser.parse(byte_source_code)
    source_code_function_signature = []
    traverse_target_node(tree.root_node, byte_source_code, "function_declarator", source_code_function_signature)
    # source_code_function_signature = [x.split("(")[0].strip() for x in source_code_function_signature]

    # tree-sitter-rust 不可用时：退化为从 auto_test_rust 的解析器提取 fn 签名
    if not _HAS_TREE_SITTER_RUST:
        try:
            from auto_test_rust import read_translated_function
            translated_sigs, _, _ = read_translated_function(translated_result)
            translated_code_function_signature = [
                s.split("{")[0].strip().split(";")[0] for s in (translated_sigs or [])
            ]
            return source_code_function_signature, translated_code_function_signature
        except Exception:
            # 最保守兜底：正则提取
            sigs = re.findall(r'(?m)^[ \\t]*(?:pub(?:\\([^)]*\\))?\\s+)?(?:unsafe\\s+)?(?:extern\\s+\"C\"\\s+)?fn\\s+\\w+[^\\{;]*', translated_result or "")
            sigs = [s.split("{")[0].strip().split(";")[0] for s in sigs]
            return source_code_function_signature, sigs

    try:
        RS_LANGUAGE = Language(tsrust.language(), "rust")
    except TypeError:
        RS_LANGUAGE = Language(tsrust.language())
    rust_parser = Parser()
    try:
        rust_parser.set_language(RS_LANGUAGE)
    except Exception:
        rust_parser = Parser(RS_LANGUAGE)

    byte_source_code = bytes(translated_result, "utf-8")
    tree = rust_parser.parse(byte_source_code)
    translated_code_function_signature = []
    traverse_target_node(tree.root_node, byte_source_code, "function_item", translated_code_function_signature)
    if not "dlp_file" in current_file_name :
        traverse_target_node(tree.root_node, byte_source_code, "function_signature_item", translated_code_function_signature)
    
    translated_code_function_signature = [x.split("{")[0].strip().split(";")[0] for x in translated_code_function_signature]

    return source_code_function_signature, translated_code_function_signature


def get_struct_code(source_code, dependency_code_signature):
    if not _HAS_TREE_SITTER_RUST:
        return dependency_code_signature

    try:
        RS_LANGUAGE = Language(tsrust.language(), "rust")
    except TypeError:
        RS_LANGUAGE = Language(tsrust.language())
    rust_parser = Parser()
    try:
        rust_parser.set_language(RS_LANGUAGE)
    except Exception:
        rust_parser = Parser(RS_LANGUAGE)

    source_code = bytes(source_code, "utf-8")
    tree = rust_parser.parse(source_code)
    target_statements = []
    traverse_target_node(tree.root_node, source_code, "impl_item", target_statements)
    # print(target_statements)
    for target_statement in target_statements:
        if dependency_code_signature in target_statement:
            return target_statement.split("{")[0] + "{\n" + dependency_code_signature + "{unimplemented!()}" + "\n}" 
        
    return dependency_code_signature


            

# 用于线程安全的打印锁
print_lock = Lock()
progress_lock = Lock()

def process_single_file(args):
    """
    处理单个文件的翻译任务（用于并行处理）
    """
    source_file, header_file, project_root, skeleton_dir, source_skeleton_dir, all_source_stems, system_prompot = args
    
    source_path = Path(source_file)
    current_file_name = source_path.stem  # 不含扩展名的文件名
    
    output_path = skeleton_dir / f"{current_file_name}.rs"
    if os.path.exists(output_path):
        with print_lock:
            print(f"{current_file_name}: already exists, skip")
        return {"status": "skipped", "file": current_file_name}
    
    try:
        # 精简输出：移除每个文件的详细处理信息
        # with print_lock:
        #     print(f"处理: {current_file_name}")

        # 读取源文件内容
        src_content = read_file(source_file)
        
        # 读取头文件内容（如果存在）
        header_content = ""
        if header_file:
            try:
                header_content = read_file(header_file)
            except Exception as e:
                with print_lock:
                    print(f"  {current_file_name}: 警告: 无法读取头文件: {e}")

        src_content_includes, src_content_content = split_include_and_content(src_content)
        src_content_includes = [x.replace('.h', '.cpp') for x in src_content_includes]
        
        # 处理头文件内容（如果存在）
        header_content_includes = []
        header_content_content = ""
        if header_content:
            header_content_includes, header_content_content = split_include_and_content(header_content)
            header_content_includes = [x.replace('.h', '.cpp') for x in header_content_includes]

        # 合并所有 include 语句
        total_includes = set(src_content_includes + header_content_includes)
        
        # 移除自引用
        tmp = set(total_includes)
        for include_statement in total_includes:
            if current_file_name in include_statement:
                tmp.remove(include_statement)
        total_includes = list(tmp)
        
        translated_includes = []
        for include_statement in total_includes:
            if "\"" in include_statement:
                tmp.remove(include_statement)
                file_name = include_statement.split("\"")[1]
                file_stem = Path(file_name).stem
                # 检查是否在项目源文件中
                if file_stem in all_source_stems:
                    translated_includes.append(f"use crate::{file_stem}::*;")
        # 注意：不再添加硬编码的 dlp_log_* 导入，这些在实际项目中通常不存在
        # 如果需要日志功能，应该使用 Rust 的 log crate
        total_includes = list(tmp)
        
        # 确定要翻译的代码内容
        # 检查内容是否有函数实现（有函数体），而不仅仅是函数声明
        def has_function_implementations(content):
            """检查内容是否有函数实现（有函数体 {...}），而不仅仅是函数声明"""
            if not content:
                return False
            
            # 简单检测：函数实现通常有 ") {" 或 ")\n{" 模式
            # 而函数声明只有 ");" 模式
            import re
            
            # 检查是否有函数实现（函数体）
            # 匹配模式：函数名(参数) { 或 函数名(参数)\n{
            function_impl_pattern = r'\)\s*\{'
            has_impl = bool(re.search(function_impl_pattern, content))
            
            # 如果没有函数实现，检查是否有类/结构体定义（这些在头文件中也是有意义的）
            if not has_impl:
                # 检查是否有类定义（class X { ... }）
                class_pattern = r'class\s+\w+[^;]*\{'
                struct_pattern = r'struct\s+\w+[^;]*\{'
                has_class_or_struct = bool(re.search(class_pattern, content)) or bool(re.search(struct_pattern, content))
                return has_class_or_struct
            
            return has_impl
        
        # 优先使用有函数实现的内容
        header_has_impl = header_content_content and has_function_implementations(header_content_content)
        src_has_impl = src_content_content and has_function_implementations(src_content_content)
        
        # 决策逻辑：
        # 1. 如果源文件有函数实现，优先使用源文件
        # 2. 如果头文件有函数实现（如 inline 函数），使用头文件
        # 3. 如果都没有实现，使用源文件（可能有其他内容）
        if src_has_impl:
            use_header = False
        elif header_has_impl:
            use_header = True
        else:
            use_header = False  # 默认使用源文件
        
        if use_header:
            code_under_translation = add_after_comment(total_includes, header_content_content)
        else:
            if header_content_content:
                with print_lock:
                    print(f"  {current_file_name}: 注意: 头文件无实质内容，将使用源文件内容")
            else:
                with print_lock:
                    print(f"  {current_file_name}: 注意: 没有头文件，将直接使用源文件内容")
            code_under_translation = add_after_comment(total_includes, src_content_content)
        
        # get function signature from src file 
        total_functions_undertranslated = get_function(src_content_content)
        functions_undertranslated = {x:y for x, y in total_functions_undertranslated.items() if "::" not in x.split("(")[0]}
        functions_undertranslated_signature = [x.split("{")[0].strip() + ";" for x in functions_undertranslated.values()]

        # get variable statement from src file
        tmp = src_content_content
        for function_understranslated in total_functions_undertranslated.values():
            tmp = tmp.replace(function_understranslated, "")
        variable_undertranslated = get_variable(tmp)
        
        # 提取静态变量和全局变量信息（用于明确告知 LLM 需要声明哪些变量）
        static_vars = get_static_variables(src_content_content)
        if static_vars:
            with print_lock:
                print(f"  {current_file_name}: 检测到 {len(static_vars)} 个静态/全局变量")
        
        # 格式化变量列表，用于 prompt
        static_vars_hint = ""
        if static_vars:
            static_vars_hint = "\n## DETECTED GLOBAL/STATIC VARIABLES (MUST BE DECLARED)\n"
            static_vars_hint += "The following variables were detected in the source code. You MUST declare ALL of them at module level:\n"
            static_vars_hint += "```\n"
            for var in static_vars:
                var_name = var['name']
                var_type = var['type']
                is_ptr = "*" if var['is_pointer'] else ""
                is_arr = "[]" if var['is_array'] else ""
                static_vars_hint += f"- {var_name}: {var_type}{is_ptr}{is_arr}\n"
            static_vars_hint += "```\n"
            static_vars_hint += "**CRITICAL**: If any of these variables are used in functions but not declared, compilation WILL FAIL.\n"
        
        code_under_translation = add_before_class(functions_undertranslated_signature + ["\n"] + variable_undertranslated, code_under_translation)

        # 保存源代码骨架
        with open(source_skeleton_dir / f"{current_file_name}.txt", 'w', encoding='utf-8') as f:
            f.write(code_under_translation)
        
        user_prompt = f"""
# C++ to Rust Skeleton Translation

## Source Code
```cpp
{code_under_translation}
```
{static_vars_hint}
## PHASE 1: Analysis (Do this mentally before generating code)

### 1.1 Static/Global Variables Analysis
Identify ALL file-level static and global variables in the source code:
- `static Type var;` or `static Type var = value;`
- `static Type *var;` or `static Type* var = NULL;`
- `static Type arr[SIZE];`
- Global variables: `Type g_varName;` or `Type* g_interface = nullptr;`
- Class static members: `static inline Type member;`
**Each identified variable MUST appear in your Rust output as a module-level static.**

### 1.2 Type Completeness Analysis  
Identify ALL types used in function signatures and variable declarations:
- Struct/class types
- Enum types
- Typedef types
- Template instantiations (e.g., `std::shared_ptr<T>` → `Option<Box<T>>`)
**Each identified type MUST have a definition in your Rust output (real or placeholder).**

### 1.3 External C Function Analysis
Identify ALL external C function calls that are NOT defined in the current source file:
- Functions from C headers (included via `#include`)
- System library functions
- Logging functions, utility functions, etc.
**For external C functions, declare them in an `extern "C"` block so the Rust compiler knows they exist externally.**
**DO NOT fabricate `extern` declarations for macros or `static inline` helpers.** In this pipeline, function bodies are translated from preprocessed `.i` (macros expanded, inline bodies visible). If macro identifiers still appear, that indicates incomplete build context / preprocessing, not something to “fix” by stubbing.

## PHASE 2: Translation Rules

### 2.1 Static Variables Translation
| C++ Pattern | Rust Translation |
|-------------|------------------|
| `static int g_count = 0;` | `static mut g_count: i32 = 0;` |
| `static Type* g_ptr = NULL;` | `static mut g_ptr: *mut Type = std::ptr::null_mut();` |
| `static Type arr[N];` | `static mut arr: [Type; N] = [Default::default(); N];` |
| `static Type* arr[N];` | `static mut arr: [*mut Type; N] = [std::ptr::null_mut(); N];` |
| Class `static inline shared_ptr<T>` | `static INSTANCE: OnceLock<Option<Box<dyn T>>> = OnceLock::new();` |

### 2.2 Type Translation
| C++ Pattern | Rust Translation |
|-------------|------------------|
| Undefined struct `TypeName` | `#[repr(C)] pub struct TypeName;` (placeholder) |
| Class with methods | `pub struct ClassName {{ ... }}` + `impl ClassName {{ ... }}` |
| Abstract class (pure virtual) | `pub trait TraitName {{ ... }}` |
| `std::shared_ptr<T>` | `Option<Box<T>>` or `Arc<T>` |
| `std::unique_ptr<T>` | `Box<T>` |
| `std::string` | `String` |
| `std::vector<T>` | `Vec<T>` |

### 2.3 External C Functions (FFI)
For external C functions that are called but NOT defined in the source code, use `extern "C"` block:
```rust
// Declare external C functions - compiler will assume they exist at link time
extern "C" {{
    fn external_c_function(arg: i32) -> i32;
    fn another_c_function(msg: *const i8);
    // Add declarations for ALL external C functions used in the code
}}
```

### 2.4 Macros / static inline (IMPORTANT)
In this pipeline, function bodies are translated from preprocessed `.i` where macros are expanded and `static inline` bodies are visible.
- **DO NOT** declare function-like macros as `extern "C"` functions.
- **DO NOT** add no-op `macro_rules!` stubs just to make compilation pass.
- If macro identifiers remain in the input, treat it as a build-context/preprocessing gap (missing `#include` / `-D` / wrong TU), not something to “patch” in Rust.

### 2.5 Function Translation
- Functions with body → `fn name(...) {{ unimplemented!() }}`
- `static` functions → Regular Rust `fn` (NOT `extern "C"`)
- Pure virtual (`= 0`) → Trait method
- Callback functions (used as fn pointers) → `pub extern "C" fn`

## PHASE 3: Output Requirements

### 3.1 Code Organization Order
Your output MUST follow this order:
1. **Standard library imports** (`use std::...`)
2. **Constant definitions** (`const NAME: Type = value;`)
3. **extern "C" block** (declarations for external C functions)
4. **No-op macros** (for C macros that can't be translated)
5. **Static/global variables** (`static mut ...`)
6. **Type definitions** (structs, enums, type aliases)
7. **Trait definitions**
8. **Impl blocks**
9. **Function definitions**

### 3.2 Critical Rules
- **NEVER use `use crate::xxx;`** for modules that don't exist in the current project
- **Use `extern "C" {{ fn ...; }}`** for external C functions (NOT defined in current file)
- **DO NOT use `extern "C"` for functions that HAVE implementations in source code**
- **ALL types must be defined** (use placeholder if needed)
- **ALL static variables must be at module level**
- **Use `unimplemented!()` for function bodies**

### 3.3 Common C→Rust Type Mapping
`int`→`i32`, `unsigned int`→`u32`, `long`→`i64`, `size_t`→`usize`,
`char`→`i8`, `unsigned char`→`u8`, `void*`→`*mut std::ffi::c_void`,
`bool`→`bool`, `float`→`f32`, `double`→`f64`

## Output Format
Output ONLY valid Rust code in a single code block:
```rust
// Complete translated Rust code
```
    """

        messages = [
            {"role": "system", "content": system_prompot},
            {"role": "user", "content": user_prompt}
        ]
        
        # 保存提示词
        try:
            from save_llm_prompts import save_llm_prompt, save_llm_prompt_text
            # os 已在文件顶部导入，不需要重复导入
            llm_name = os.environ.get("LLM_NAME", "default")
            metadata = {
                "file_name": current_file_name,
                "source_file": str(source_file),
                "has_header": bool(header_file),
                "code_length": len(code_under_translation)
            }
            save_llm_prompt(
                messages=messages,
                project_name=PROJECT_NAME,
                llm_name=llm_name,
                task_type="skeleton_translate",
                function_name=current_file_name,
                metadata=metadata
            )
            save_llm_prompt_text(
                messages=messages,
                project_name=PROJECT_NAME,
                llm_name=llm_name,
                task_type="skeleton_translate",
                function_name=current_file_name,
                metadata=metadata
            )
        except Exception as e:
            import logging
            logging.warning(f"保存提示词失败: {e}")

        response = generation(message=messages)

        translated_result = extract_code(response)
        
        # 验证提取的代码是否有效
        if not translated_result or len(translated_result.strip()) < 20:
            with print_lock:
                print(f"⚠ {current_file_name}: LLM返回的代码过短，可能只有占位符")
                print(f"   响应长度: {len(response) if response else 0} 字符")
                print(f"   提取代码长度: {len(translated_result) if translated_result else 0} 字符")
        
        # 检查是否只包含占位符文本
        placeholder_text = "Your translated Rust code here"
        if translated_result and placeholder_text in translated_result and len(translated_result.strip()) < 100:
            with print_lock:
                print(f"⚠ {current_file_name}: LLM只返回了占位符文本，没有实际代码")
                print(f"   原始响应预览: {response[:200] if response else 'None'}...")
        
        skeleton = add_after_comment(translated_includes, translated_result)
        # 对最终骨架进行后处理（清理重复导入、虚假导入等）
        skeleton = _postprocess_rust_skeleton(skeleton)
        with open(output_path, 'w') as f:
            f.write(skeleton)

        with print_lock:
            # 验证最终骨架文件
            if os.path.exists(output_path):
                with open(output_path, 'r', encoding='utf-8', errors='ignore') as f:
                    final_content = f.read()
                fn_count = len(re.findall(r'\bfn\s+\w+\s*\(', final_content))
                if fn_count == 0:
                    print(f"⚠ {current_file_name}: 骨架文件已保存但无函数定义 (文件长度: {len(final_content)})")
                else:
                    print(f"✓ {current_file_name}: successfully get {output_path} ({fn_count} 个函数)")
            else:
                print(f"✗ {current_file_name}: 文件保存失败")
        
        return {"status": "success", "file": current_file_name}
    
    except Exception as e:
        with print_lock:
            print(f"✗ {current_file_name}: 处理失败 - {str(e)}")
        return {"status": "error", "file": current_file_name, "error": str(e)}


# =========================================================================
# 新增：分层骨架构建方法（基于论文方案）
# =========================================================================

def translate_skeleton_layered(use_bindgen=True, use_llm_for_signatures=False, use_type_mapper=True, use_llm_type_mapper=False, use_llm_fallback=True, use_libclang=False, verbose=False):
    """
    分层骨架翻译 - 从"单次生成"转向"分层构建"
    
    基于论文方法：
    - 阶段 A (Truth Layer): 使用 bindgen 生成绝对正确的类型定义
    - 阶段 B (State Layer): 使用 tree-sitter 精确提取全局/静态变量
    - 阶段 C (Logic Skeleton): 生成仅包含签名的 unimplemented!() 桩代码（使用确定性规则引擎）
    - 阶段 D (Validation): 编译验证
    
    Args:
        use_bindgen: 是否使用 bindgen 生成类型（默认 True）
        use_llm_for_signatures: 是否使用 LLM 翻译函数签名（默认 False，推荐使用 TypeMapper）
        use_type_mapper: 是否使用 TypeMapper（确定性规则引擎），默认 True（推荐）
        use_llm_fallback: 当 tree-sitter 解析失败时是否使用 LLM 兜底（默认 True）
        use_libclang: 使用 libclang 替代 tree-sitter 提取函数（更准确，能处理复杂宏和属性）
        verbose: 是否输出详细日志（默认 False）
    """
    truth_mode = os.environ.get("C2R_TRUTH_MODE", "0").strip().lower() in ("1", "true", "yes", "y", "on")
    if truth_mode:
        # Truth-mode: avoid heuristic/derived layers and LLM fallbacks in skeleton stage.
        use_llm_for_signatures = False
        use_llm_type_mapper = False
        use_llm_fallback = False

    project_path = PROJECT_NAME
    project_root = Path(PROJECT_ROOT)
    
    # 使用新的工作空间路径
    skeleton_dir = get_skeleton_path(project_path)
    source_skeleton_dir = get_source_skeleton_path(project_path)
    
    # 确保目录存在
    ensure_dirs(project_path)
    skeleton_dir.mkdir(parents=True, exist_ok=True)
    (skeleton_dir / "src").mkdir(parents=True, exist_ok=True)
    source_skeleton_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"分层骨架构建开始 (Layered Skeleton Building)")
    print(f"{'='*60}")
    print(f"项目: {project_path}")
    print(f"项目根目录: {project_root}")
    print(f"使用 bindgen: {use_bindgen}")
    print(f"使用 LLM 翻译签名: {use_llm_for_signatures}")
    print(f"使用 TypeMapper（确定性规则引擎）: {use_type_mapper}")
    print(f"Truth mode (C2R_TRUTH_MODE): {truth_mode}")
    print(f"骨架输出目录: {skeleton_dir}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    # ---------------------------------------------------------------------
    # compile_commands.json / OpenHarmony build profile 选择（out_dir 主键）
    # ---------------------------------------------------------------------
    print("\n[查询编译文件] 开始查找 compile_commands.json / build profile...")
    # 默认路径（仅兜底；开源版优先使用 his2trans/data/ohos/ohos_root_min）
    DEFAULT_OHOS_ROOT = (Path(__file__).resolve().parent.parent.parent / "data" / "ohos" / "ohos_root_min")
    DEFAULT_COMPILE_COMMANDS = DEFAULT_OHOS_ROOT / "out" / "rk3568" / "compile_commands.json"

    # 查找 compile_commands.json（如果可用）
    compile_commands_path = None
    ohos_root = None

    # 优先级0: 用户/上游显式指定（允许覆盖硬编码默认）
    env_cc = os.environ.get("OHOS_COMPILE_COMMANDS", "").strip() or os.environ.get("COMPILE_COMMANDS_PATH", "").strip()
    if env_cc:
        p = Path(env_cc).expanduser()
        if p.exists():
            compile_commands_path = p.resolve()
            ohos_root_env = os.environ.get("OHOS_ROOT", "").strip()
            if ohos_root_env and Path(ohos_root_env).exists():
                ohos_root = Path(ohos_root_env).expanduser().resolve()
            else:
                # 尝试从 out/<board>/compile_commands.json 推断
                try:
                    ohos_root = compile_commands_path.parent.parent.parent.resolve()
                except Exception:
                    ohos_root = None
            print(f"  ✓ 使用环境变量指定的 compile_commands.json: {compile_commands_path}")
            if ohos_root:
                print(f"    OHOS 根目录: {ohos_root}")

    # 优先级0.25: 复用阶段1(get_dependencies.py) 已选择的 profile（避免重复推断，确保 bindgen/types 与 .i 预处理一致）
    if not compile_commands_path:
        try:
            candidates = []
            pre_dir = os.environ.get("PREPROCESS_OUTPUT_DIR", "").strip()
            if pre_dir:
                pre_path = Path(pre_dir).expanduser()
                if pre_path.is_dir():
                    candidates.append(pre_path / "tu_context_map.json")
                else:
                    candidates.append(pre_path)

            ws_root_env = os.environ.get("C2R_WORKSPACE_ROOT", "").strip()
            if ws_root_env:
                candidates.append(Path(ws_root_env).expanduser().resolve() / ".preprocessed" / "tu_context_map.json")
            else:
                candidates.append(Path("workspace").resolve() / ".preprocessed" / "tu_context_map.json")

            for tu_map_path in candidates:
                if not tu_map_path.exists():
                    continue
                try:
                    tu_map = json.loads(tu_map_path.read_text(encoding="utf-8", errors="ignore"))
                except Exception:
                    continue
                sel = tu_map.get("selected_profile") or {}
                cc = (sel.get("compile_commands_json") or "").strip()
                if not cc:
                    continue
                cc_path = Path(cc).expanduser()
                if not cc_path.exists():
                    continue
                compile_commands_path = cc_path.resolve()
                oh = (sel.get("ohos_root") or "").strip()
                if oh and Path(oh).exists():
                    ohos_root = Path(oh).expanduser().resolve()
                os.environ.setdefault("OHOS_COMPILE_COMMANDS", str(compile_commands_path))
                if ohos_root:
                    os.environ.setdefault("OHOS_ROOT", str(ohos_root))
                out_dir = (sel.get("out_dir") or "").strip()
                label = (sel.get("label") or "").strip()
                print(f"  ✓ 复用 TU profile(out_dir): {out_dir or '(unknown)'}")
                if label:
                    print(f"    label(product@vendor): {label}")
                print(f"    compile_commands.json: {compile_commands_path}")
                break
        except Exception as e:
            print(f"  ⚠ 复用 tu_context_map 的 profile 失败（将继续自动选择）: {e}")

    # 优先级0.5: OpenHarmony registry 自动选择 profile（AUTO/缓存/Pin）
    if not compile_commands_path:
        try:
            # 只在 SelfContained 模块（有 original_path.txt）或显式配置 registry 时启用，避免对普通项目引入额外开销
            has_original_path = (project_root / "original_path.txt").exists()
            has_registry_env = bool(os.environ.get("OHOS_CC_REGISTRY") or os.environ.get("OHOS_COMPILE_COMMANDS_REGISTRY"))
            if has_original_path or has_registry_env:
                from ohos_build_profile import select_profile_for_project

                selected = select_profile_for_project(
                    project_name=str(project_path),
                    project_root=project_root,
                    require_registry=False,
                )
                if selected:
                    compile_commands_path = Path(selected.compile_commands_json).resolve()
                    ohos_root = Path(selected.ohos_root).resolve() if selected.ohos_root else None
                    os.environ.setdefault("OHOS_COMPILE_COMMANDS", str(compile_commands_path))
                    if ohos_root:
                        os.environ.setdefault("OHOS_ROOT", str(ohos_root))
                    print(f"  ✓ 选择 build profile(out_dir): {selected.profile.out_dir}")
                    print(f"    label(product@vendor): {selected.profile.label}")
                    print(f"    compile_commands.json: {compile_commands_path}")
        except Exception as e:
            print(f"  ⚠ build profile 自动选择失败，回退到传统查找逻辑: {e}")
    
    # 优先级1-4: 传统查找逻辑（仅在前面没有选中时启用）
    if not compile_commands_path:
        # 优先级1: 使用硬编码的默认路径
        print(f"  [优先级1] 检查硬编码路径: {DEFAULT_COMPILE_COMMANDS}")
        if DEFAULT_COMPILE_COMMANDS.exists():
            compile_commands_path = DEFAULT_COMPILE_COMMANDS.resolve()
            ohos_root = DEFAULT_OHOS_ROOT
            print(f"  ✓ 使用硬编码路径找到 compile_commands.json: {compile_commands_path}")
            print(f"    OpenHarmony 根目录: {ohos_root}")
        else:
            print(f"  ✗ 硬编码路径不存在，继续查找...")

            # 优先级2: 尝试从环境变量获取 OpenHarmony 根目录
            ohos_root_env = os.environ.get("OHOS_ROOT")
            if ohos_root_env:
                print(f"  [优先级2] 检查环境变量 OHOS_ROOT: {ohos_root_env}")
                ohos_root = Path(ohos_root_env).expanduser().resolve()
                if ohos_root.exists():
                    print(f"    OpenHarmony 根目录存在，查找 compile_commands.json...")
                    compile_commands_path = get_compile_commands_path(ohos_root)
                    if compile_commands_path:
                        print(f"  ✓ 从环境变量找到: {compile_commands_path}")
                else:
                    print(f"    ✗ OpenHarmony 根目录不存在")
            else:
                print(f"  [优先级2] 环境变量 OHOS_ROOT 未设置")
                # 优先级3: 尝试自动检测 OpenHarmony 根目录（从项目根目录向上查找）
                print(f"  [优先级3] 从项目根目录向上查找 OpenHarmony 根目录...")
                current = project_root.resolve()
                for level in range(5):  # 最多向上查找5层
                    print(f"    检查层级 {level+1}: {current}")
                    if (current / "out" / "rk3568" / "build.ninja").exists():
                        ohos_root = current
                        print(f"    ✓ 找到 OpenHarmony 根目录: {ohos_root}")
                        compile_commands_path = get_compile_commands_path(ohos_root)
                        if compile_commands_path:
                            print(f"  ✓ 找到 compile_commands.json: {compile_commands_path}")
                        break
                    parent = current.parent
                    if parent == current:  # 到达根目录
                        print(f"    到达根目录，停止查找")
                        break
                    current = parent

            # 优先级4: 如果还没找到，尝试全局搜索
            if not compile_commands_path:
                print(f"  [优先级4] 尝试全局搜索 compile_commands.json...")
                compile_commands_path = get_compile_commands_path()
                # 如果找到了，尝试确定 ohos_root
                if compile_commands_path:
                    print(f"  ✓ 全局搜索找到: {compile_commands_path}")
                    # 从 compile_commands.json 路径推断 ohos_root
                    if "out/rk3568" in str(compile_commands_path):
                        ohos_root = compile_commands_path.parent.parent.parent.resolve()
                        print(f"    推断 OpenHarmony 根目录: {ohos_root}")
                    elif "out/sdk" in str(compile_commands_path):
                        ohos_root = compile_commands_path.parent.parent.parent.resolve()
                        print(f"    推断 OpenHarmony 根目录: {ohos_root}")
        
        if compile_commands_path:
            print(f"\n[完成] 找到 compile_commands.json: {compile_commands_path}")
            if ohos_root:
                print(f"  OpenHarmony 根目录: {ohos_root}")
        else:
            print(f"\n[警告] 未找到 compile_commands.json，将使用项目内头文件搜索路径")
            print(f"  提示: 硬编码路径不存在: {DEFAULT_COMPILE_COMMANDS}")
            print("  提示: 设置环境变量 OHOS_COMPILE_COMMANDS 或 OHOS_ROOT 以启用优化")
    
    # 创建骨架构建器（传入 compile_commands.json 路径）
    print(f"\n[初始化] 创建骨架构建器...")
    if compile_commands_path:
        print(f"  使用 compile_commands.json: {compile_commands_path}")
    else:
        print(f"  未使用 compile_commands.json（将使用项目内头文件搜索路径）")
    
    builder = SkeletonBuilder(
        project_root, 
        skeleton_dir,
        compile_commands_path=compile_commands_path,
        ohos_root=ohos_root
    )

    # ---------------------------------------------------------------------
    # TU-closure gate (strict): if stage1 preprocessing is incomplete, stop here.
    #
    # Rationale:
    # - We rely on the stage1-pinned preprocessed `.i` (tu_context_map.json) as the single truth entry
    #   for signatures/types/globals/function bodies.
    # - If `.i` is missing or preprocessing failed for any *included* TU, later stages would silently fall back
    #   to non-expanded C or heuristic TU selection, which creates systematic mismatches.
    # ---------------------------------------------------------------------
    try:
        require_tu = (os.environ.get("C2R_REQUIRE_TU_CLOSURE", "1") or "1").strip().lower() not in (
            "0",
            "false",
            "no",
        )
    except Exception:
        require_tu = True
    if require_tu:
        # Strict mode requires stage1 TU mapping; otherwise we'd have to re-pick a TU (non-truth).
        if not getattr(builder, "_tu_context_files", None):
            print("✗ 未发现/未能加载 tu_context_map.json（C2R_REQUIRE_TU_CLOSURE=1），跳过本项目以保持 TU 真值一致性")
            print("  提示: 请先运行阶段1预处理(get_dependencies.py)生成 tu_context_map.json，或设置 C2R_REQUIRE_TU_CLOSURE=0 关闭严格模式")
            return 1

    if require_tu and getattr(builder, "_tu_context_map_path", None) and getattr(builder, "_tu_context_files", None):
        try:
            bad = []
            for safe_name, rec in (getattr(builder, "_tu_context_files", {}) or {}).items():
                if not isinstance(rec, dict):
                    continue
                if rec.get("excluded_by_compile_commands"):
                    continue
                if rec.get("compile_commands_entry") is None:
                    bad.append((safe_name, "missing_compile_commands_entry"))
                    continue
                err = str(rec.get("error") or "").strip()
                if err:
                    bad.append((safe_name, "preprocess_failed"))
                    continue
                pre = rec.get("preprocessed_file")
                if not pre:
                    bad.append((safe_name, "preprocessed_file_not_recorded"))
                    continue
                try:
                    if not Path(str(pre)).expanduser().exists():
                        bad.append((safe_name, "preprocessed_file_missing"))
                        continue
                except Exception:
                    bad.append((safe_name, "preprocessed_file_invalid_path"))
                    continue
            if bad:
                print("✗ TU 闭包不完整（缺 TU 条目/预处理失败/.i 缺失），跳过本项目以保持 TU 真值一致性")
                # Print a compact sample; the full list is already written by batch_test_staged.sh gate.
                sample = ", ".join(f"{n}:{r}" for n, r in bad[:8])
                print(f"  样例: {sample}")
                try:
                    print(f"  tu_context_map: {getattr(builder, '_tu_context_map_path', '')}")
                except Exception:
                    pass
                return 1
        except Exception:
            # If we can't validate, prefer failing closed in strict mode.
            print("✗ TU 闭包校验失败，跳过本项目以避免使用非真值输入")
            return 1
    
    if builder.compile_commands_parser:
        print(f"  ✓ compile_commands.json 已加载")
    else:
        print(f"  ⚠ compile_commands.json 未加载（将使用项目内头文件搜索路径）")
    
    # 收集源文件和头文件
    source_header_pairs = get_all_source_files_with_headers(project_root)

    # ---------------------------------------------------------------------
    # Route A (truth-first): Source set is defined by compile_commands.json.
    #
    # If stage1 (get_dependencies.py) already produced `tu_context_map.json`,
    # respect its per-file `excluded_by_compile_commands` markers so we do NOT
    # generate skeleton modules for sources that are not built in the selected profile.
    # ---------------------------------------------------------------------
    try:
        source_set_from_cc = os.environ.get("C2R_SOURCE_SET_FROM_COMPILE_COMMANDS", "1").strip().lower() not in ("0", "false", "no")
        if source_set_from_cc:
            tu_map_candidates = []
            pre_dir = (os.environ.get("PREPROCESS_OUTPUT_DIR", "") or "").strip()
            if pre_dir:
                pre_path = Path(pre_dir).expanduser()
                if pre_path.is_dir():
                    tu_map_candidates.append(pre_path / "tu_context_map.json")
                else:
                    tu_map_candidates.append(pre_path)
            ws_root_env = os.environ.get("C2R_WORKSPACE_ROOT", "").strip()
            if ws_root_env:
                tu_map_candidates.append(Path(ws_root_env).expanduser().resolve() / ".preprocessed" / "tu_context_map.json")
            else:
                tu_map_candidates.append(Path("workspace").resolve() / ".preprocessed" / "tu_context_map.json")

            included_abs = set()
            excluded_abs = set()
            for tu_map_path in tu_map_candidates:
                if not tu_map_path.exists():
                    continue
                try:
                    tu_map = json.loads(tu_map_path.read_text(encoding="utf-8", errors="ignore") or "{}")
                except Exception:
                    continue
                files = tu_map.get("files") if isinstance(tu_map, dict) else None
                if not isinstance(files, dict) or not files:
                    continue
                for _k, rec in files.items():
                    if not isinstance(rec, dict):
                        continue
                    src_abs = rec.get("source_file_abs")
                    if not src_abs:
                        continue
                    try:
                        p = Path(str(src_abs)).resolve()
                    except Exception:
                        continue
                    if rec.get("excluded_by_compile_commands"):
                        excluded_abs.add(p)
                    else:
                        included_abs.add(p)
                break

            if included_abs:
                before = len(source_header_pairs)
                source_header_pairs = [
                    (src, hdr)
                    for (src, hdr) in source_header_pairs
                    if Path(src).resolve() in included_abs
                ]
                after = len(source_header_pairs)
                if excluded_abs:
                    print(
                        f"[RouteA] 已按 compile_commands 过滤源文件：included={after}, excluded(not-built)={len(excluded_abs)}, scanned={before}"
                    )
    except Exception:
        pass

    source_files = [Path(src) for src, _ in source_header_pairs]
    header_files = [Path(hdr) for _, hdr in source_header_pairs if hdr]
    
    # 额外收集所有头文件（但默认排除 test/unittest 目录，避免把单元测试的 C++ 头链喂给 bindgen）
    def _exclude_from_truth_inputs(p: Path) -> bool:
        if os.environ.get("C2R_EXCLUDE_TEST_DIRS", "1").strip().lower() in ("0", "false", "no"):
            return False
        excluded = {
            x.strip().lower()
            for x in os.environ.get("C2R_EXCLUDED_DIR_NAMES", "test,tests,unittest,unittests").split(",")
            if x.strip()
        }
        try:
            rel = p.resolve().relative_to(project_root.resolve())
        except Exception:
            rel = p
        return any(str(part).lower() in excluded for part in rel.parts)

    all_headers = {h for h in header_files if not _exclude_from_truth_inputs(h)}
    for h in project_root.glob("**/*.h"):
        if not _exclude_from_truth_inputs(h):
            all_headers.add(h)
    for h in project_root.glob("**/*.hpp"):
        if not _exclude_from_truth_inputs(h):
            all_headers.add(h)
    header_files = list(all_headers)
    
    print(f"发现 {len(source_files)} 个源文件, {len(header_files)} 个头文件")
    print(f"自动检测到 {len(builder.include_dirs)} 个头文件搜索目录\n")
    
    # =========================================================================
    # 阶段 A: 类型骨架 (Truth Layer)
    # =========================================================================
    print(f"{'='*60}")
    print("阶段 A: 生成类型骨架 (bindgen)")
    print(f"{'='*60}")
    
    if use_bindgen and header_files:
        phase_start = time.time()
        # 增强：传递源文件以分析外部依赖（例如 #include "softbus_error_code.h"）
        bindgen_success = builder.generate_type_skeleton(header_files, source_files=source_files)
        phase_time = time.time() - phase_start
        
        if bindgen_success:
            # 兼容：generate_type_skeleton 可能会走 stub/clang-preprocessed 回退，不能一律写“bindgen 成功”
            mode = "unknown"
            report_path = skeleton_dir / "types_generation_report.json"
            if report_path.exists():
                try:
                    report = json.loads(report_path.read_text(encoding="utf-8", errors="ignore"))
                    mode = report.get("mode") or mode
                except Exception:
                    pass
            print(f"✓ 类型骨架生成完成: mode={mode} (耗时: {phase_time:.1f}秒)")
        else:
            print(f"⚠ bindgen 失败，使用占位类型 (耗时: {phase_time:.1f}秒)")
            print(f"  提示: 可能是缺少外部依赖的头文件，这不影响后续流程")
    else:
        # 创建占位类型文件
        builder._create_placeholder_types(skeleton_dir / "src" / "types.rs")
        print("跳过 bindgen（无头文件或已禁用）")
    
    print()
    
    # =========================================================================
    # 阶段 A.5: Libclang 统一提取（Single Source of Truth）+ 覆盖率度量
    # =========================================================================
    libclang_functions = None
    libclang_global_vars = None
    libclang_extractor = None  # 复用提取器（避免重复加载 compile_commands）
    libclang_json_path = source_skeleton_dir / "libclang_extracted.json"
    libclang_coverage_path = source_skeleton_dir / "libclang_coverage.json"
    coverage_threshold = 0.6  # 覆盖率阈值，低于此值触发回退
    per_file_coverage = {}  # {file_path: coverage_ratio} - 按文件覆盖率
    low_coverage_files = set()  # 低覆盖率文件集合（用于 Phase C 按文件回退）

    if use_libclang:
        print(f"{'='*60}")
        print("阶段 A.5: Libclang 统一提取 (函数 + 全局变量 + 覆盖率度量)")
        print(f"{'='*60}")

        phase_start = time.time()
        try:
            from unified_function_extractor import (
                UnifiedFunctionExtractor, find_source_files as libclang_find_files,
                UnifiedFunction, CoverageReport
            )

            # 复用前面已经找到的 compile_commands_path（在 translate_skeleton_layered 开头已查找）
            # 变量 compile_commands_path 在 line 1561-1620 已经设置
            if compile_commands_path:
                print(f"  📍 复用 compile_commands.json: {compile_commands_path}")
            else:
                print(f"  ⚠ 未找到 compile_commands.json，使用 builder.include_dirs 增强")

            # 初始化提取器（带覆盖率阈值 + builder.include_dirs 增强）
            # 将 builder.include_dirs 转换为字符串列表传入 libclang
            extra_include_dirs = [str(d) for d in builder.include_dirs] if hasattr(builder, 'include_dirs') else []
            if extra_include_dirs:
                print(f"  📂 合并 {len(extra_include_dirs)} 个 include 目录到 libclang")

            libclang_extractor = UnifiedFunctionExtractor(
                compile_commands_path=compile_commands_path,
                project_root=project_root,
                coverage_threshold=coverage_threshold,
                extra_include_dirs=extra_include_dirs
            )

            # 使用带覆盖率跟踪的提取方式
            libclang_functions_list, coverage_report = libclang_extractor.extract_from_project_with_coverage(
                source_files, include_body=True
            )

            # 保存覆盖率报告
            libclang_extractor.save_coverage_report(libclang_coverage_path, coverage_threshold)
            # 保存每文件 diagnostics（便于定位 include/macro/生成头文件缺失等问题）
            try:
                libclang_diag_path = source_skeleton_dir / "libclang_diagnostics.json"
                libclang_extractor.save_diagnostics_report(libclang_diag_path)
                print(f"  📄 diagnostics 已保存: {libclang_diag_path}")
            except Exception as e:
                print(f"  ⚠ 保存 diagnostics 失败: {e}")

            # 提取全局变量（单独提取，因为 coverage 方法只提取函数）
            libclang_global_vars = []
            for src_file in source_files:
                try:
                    gvars = libclang_extractor.extract_global_variables(src_file)
                    libclang_global_vars.extend([v.to_dict() for v in gvars])
                except Exception:
                    pass

            # 构建完整数据
            complete_data = {
                'version': '2.0',
                'extractor': 'libclang',
                'total_functions': len(libclang_functions_list),
                'total_global_vars': len(libclang_global_vars),
                'coverage': coverage_report.to_dict(),
                'functions': [f.to_dict() for f in libclang_functions_list],
                'global_vars': libclang_global_vars,
                'stats': libclang_extractor.get_stats()
            }

            # 保存为 JSON（Single Source of Truth）
            with open(libclang_json_path, 'w', encoding='utf-8') as f:
                json.dump(complete_data, f, indent=2, ensure_ascii=False)

            # 设置最终函数列表
            libclang_functions = libclang_functions_list
            total_functions = len(libclang_functions)

            phase_time = time.time() - phase_start
            print(f"✓ libclang 提取: {total_functions} 函数, {len(libclang_global_vars)} 变量 ({phase_time:.1f}s)")
            print(f"  📊 覆盖率: {coverage_report.overall_coverage:.1%} ({coverage_report.total_libclang}/{coverage_report.total_expected})")
            if verbose:
                print(f"  结果保存到: {libclang_json_path}")
                print(f"  覆盖率报告: {libclang_coverage_path}")

            # ========== 按文件覆盖率存储（供 Phase C 使用）==========
            # 不再全局回退，而是保留 libclang 结果，在 Phase C 按文件决定使用哪个
            per_file_coverage = {}  # {file_path: coverage_ratio}
            low_coverage_files = set()  # 低覆盖率文件集合
            for fc in coverage_report.per_file_coverage:
                per_file_coverage[fc.file_path] = fc.coverage_ratio
                if fc.coverage_ratio < coverage_threshold and fc.expected_count > 0:
                    low_coverage_files.add(fc.file_path)

            # 打印覆盖率摘要
            good_coverage_count = len(per_file_coverage) - len(low_coverage_files)
            print(f"  📊 文件覆盖率: {good_coverage_count}/{len(per_file_coverage)} 个文件达标 (>={coverage_threshold:.0%})")

            if low_coverage_files:
                print(f"  ⚠ {len(low_coverage_files)} 个文件将回退到 tree-sitter:")
                for f in list(low_coverage_files)[:5]:
                    ratio = per_file_coverage.get(f, 0)
                    print(f"    - {f} ({ratio:.0%})")
                if len(low_coverage_files) > 5:
                    print(f"    ... 还有 {len(low_coverage_files) - 5} 个")

            # 只有当 libclang 提取完全失败（0 个函数）时才全局回退
            if total_functions == 0:
                print(f"  ⚠ libclang 提取到 0 个函数，全局回退到 tree-sitter 模式")
                libclang_functions = None
                per_file_coverage = {}
                low_coverage_files = set()

            # ============================================================
            # 阶段 A.5.2: 写回 extracted/（让阶段2/3使用同一事实源）
            # ============================================================
            # batch_test_staged.sh 在 --use-libclang 时会跳过 get_dependencies.py，
            # 因此这里需要把 libclang 的提取结果写回 workspace/extracted/<proj>/functions
            # 并生成 workspace/extracted/<proj>/call_graph.json（schema v2.0）。
            export_extracted = os.environ.get("C2R_EXPORT_EXTRACTED_FROM_LIBCLANG", "true").lower() in ("true", "1", "yes")
            if export_extracted and libclang_functions is not None and libclang_extractor is not None:
                try:
                    from collections import defaultdict
                    import hashlib
                    from call_graph import generate_function_uid

                    extracted_dir = get_extracted_path(project_path)
                    functions_dir = get_functions_path(project_path)
                    dependencies_dir = get_dependencies_path(project_path)
                    dependencies_not_in_file_dir = get_dependencies_not_in_file_path(project_path)

                    extracted_dir.mkdir(parents=True, exist_ok=True)
                    functions_dir.mkdir(parents=True, exist_ok=True)
                    dependencies_dir.mkdir(parents=True, exist_ok=True)
                    dependencies_not_in_file_dir.mkdir(parents=True, exist_ok=True)

                    # 清理旧的 .txt（避免复用上次 run 的陈旧函数编号/内容）
                    for d in (functions_dir, dependencies_dir, dependencies_not_in_file_dir):
                        try:
                            for p in d.glob("*.txt"):
                                p.unlink()
                        except Exception:
                            pass
                    try:
                        (extracted_dir / "call_graph.json").unlink()
                    except Exception:
                        pass

                    # (1) 生成 extracted/functions/*.txt
                    funcs_by_file: dict[str, list] = defaultdict(list)
                    for func in libclang_functions:
                        funcs_by_file[str(Path(func.source_file_abs).resolve())].append(func)

                    func_meta: dict[str, dict] = {}
                    name_to_func_files: dict[str, list[str]] = defaultdict(list)
                    name_and_file_to_func_files: dict[tuple[str, str], list[str]] = defaultdict(list)
                    func_file_contents: dict[str, str] = {}

                    total_exported = 0
                    for abs_file, funcs in sorted(funcs_by_file.items()):
                        src_file = Path(abs_file)
                        safe_name = builder._get_safe_module_name(src_file)

                        # 稳定排序：按 start_line 决定 idx（与 tree-sitter 的“文件顺序”一致）
                        funcs_sorted = sorted(
                            funcs,
                            key=lambda f: (int(getattr(f, "start_line", 0) or 0), str(getattr(f, "name", ""))),
                        )
                        for idx, func in enumerate(funcs_sorted, 1):
                            func_file_name = f"{safe_name}_{idx}"

                            # 用 libclang 的 c_signature 替换原始宏签名（例如 STATIC/INLINE），确保 tree-sitter 后续可解析函数名
                            body = getattr(func, "body", "") or ""
                            brace_idx = body.find("{")
                            if brace_idx != -1:
                                body_from_brace = body[brace_idx:].lstrip()
                                content = f"{func.c_signature}\n{body_from_brace}"
                            else:
                                content = f"{func.c_signature}\n{{\n    // Function body unavailable\n}}\n"

                            # 统一换行
                            if not content.endswith("\n"):
                                content += "\n"

                            out_path = functions_dir / f"{func_file_name}.txt"
                            out_path.write_text(content, encoding="utf-8", errors="ignore")

                            func_file_contents[func_file_name] = content
                            name_to_func_files[func.name].append(func_file_name)
                            name_and_file_to_func_files[(func.name, safe_name)].append(func_file_name)

                            func_meta[func_file_name] = {
                                "name": func.name,
                                "file": safe_name,
                                "start_line": int(getattr(func, "start_line", 0) or idx),
                                "end_line": int(getattr(func, "end_line", 0) or 0),
                                "signature": getattr(func, "c_signature", "") or "",
                                "mangled_name": getattr(func, "mangled_name", "") or "",
                                "source_file": str(getattr(func, "source_file", "") or ""),
                            }
                            total_exported += 1

                    # (2) 用 libclang 再扫一遍调用（只为生成 call_graph.json / dependencies/*.txt）
                    deps = []
                    try:
                        deps = libclang_extractor.extract_dependencies(libclang_functions, source_files)
                    except Exception:
                        deps = []

                    # 初始化：每个函数都有一个空 entry，保证 call_graph 节点数一致
                    call_graph: dict[str, list[str]] = {k: [] for k in func_meta.keys()}
                    # 同时创建空的 dependencies 文件（与 tree-sitter 模式输出结构保持一致）
                    for func_file in call_graph.keys():
                        try:
                            (dependencies_dir / f"{func_file}.txt").write_text("", encoding="utf-8")
                            (dependencies_not_in_file_dir / f"{func_file}.txt").write_text("", encoding="utf-8")
                        except Exception:
                            pass

                    for dep in deps:
                        caller_name = getattr(dep, "caller", "") or ""
                        caller_file_rel = getattr(dep, "caller_file", "") or ""
                        callees = list(getattr(dep, "callees", []) or [])

                        caller_safe = ""
                        if caller_file_rel:
                            try:
                                caller_src = (project_root / caller_file_rel).resolve()
                                caller_safe = builder._get_safe_module_name(caller_src)
                            except Exception:
                                caller_safe = ""

                        caller_candidates = []
                        if caller_name and caller_safe:
                            caller_candidates = name_and_file_to_func_files.get((caller_name, caller_safe), [])
                        if not caller_candidates and caller_name:
                            caller_candidates = name_to_func_files.get(caller_name, [])
                        if not caller_candidates:
                            continue
                        caller_func_file = caller_candidates[0]

                        callee_files: list[str] = []
                        for callee_name in callees:
                            for f in name_to_func_files.get(callee_name, []):
                                if f not in callee_files:
                                    callee_files.append(f)

                        call_graph[caller_func_file] = callee_files

                        # 同时生成 dependencies/*.txt（拼接被调用函数源码，便于后续上下文）
                        if callee_files:
                            called_code = [func_file_contents.get(f, "") for f in callee_files if func_file_contents.get(f, "")]
                            (dependencies_dir / f"{caller_func_file}.txt").write_text("\n\n".join(called_code), encoding="utf-8", errors="ignore")

                            caller_prefix = caller_func_file.rsplit("_", 1)[0] if "_" in caller_func_file else caller_func_file
                            not_in_current = []
                            for f in callee_files:
                                callee_prefix = f.rsplit("_", 1)[0] if "_" in f else f
                                if callee_prefix != caller_prefix:
                                    code = func_file_contents.get(f, "")
                                    if code:
                                        not_in_current.append(code)
                            (dependencies_not_in_file_dir / f"{caller_func_file}.txt").write_text("\n\n".join(not_in_current), encoding="utf-8", errors="ignore")

                    # (3) 写出 call_graph.json（schema v2.0，与 get_dependencies.py 对齐）
                    functions_dict = {}
                    name_index = defaultdict(list)
                    func_file_name_to_uid = {}

                    for func_file_name, meta in func_meta.items():
                        file_stem = meta.get("file", "")
                        start_line = int(meta.get("start_line") or 0)
                        name = meta.get("name", "")
                        mangled = meta.get("mangled_name") or ""

                        uid = generate_function_uid(file_stem, start_line, name, mangled_name=mangled or None)
                        func_file_name_to_uid[func_file_name] = uid

                        functions_dict[uid] = {
                            "name": name,
                            "file": file_stem,
                            "start_line": start_line,
                            "end_line": int(meta.get("end_line") or 0),
                            "signature": meta.get("signature") or "",
                            "callees": [],
                            "uid": uid,
                            "mangled_name": mangled or "",
                        }
                        name_index[name].append(uid)

                    call_graph_by_uid = {}
                    for caller_func_file, callee_func_files in call_graph.items():
                        caller_uid = func_file_name_to_uid.get(caller_func_file)
                        if not caller_uid:
                            continue
                        callee_uids = []
                        for callee_func_file in callee_func_files or []:
                            cu = func_file_name_to_uid.get(callee_func_file)
                            if cu and cu not in callee_uids:
                                callee_uids.append(cu)
                        call_graph_by_uid[caller_uid] = callee_uids
                        functions_dict[caller_uid]["callees"] = callee_uids

                    reverse_call_graph = defaultdict(list)
                    for caller_uid, callee_uids in call_graph_by_uid.items():
                        for callee_uid in callee_uids:
                            reverse_call_graph[callee_uid].append(caller_uid)

                    call_graph_data = {
                        "schema_version": "2.0",
                        "functions": functions_dict,
                        "call_graph": call_graph_by_uid,
                        "reverse_call_graph": dict(reverse_call_graph),
                        "name_index": dict(name_index),
                        "global_vars": {},
                        "type_defs": {},
                        "stats": {
                            "total_functions": len(functions_dict),
                            "total_calls": sum(len(v) for v in call_graph_by_uid.values()),
                            "total_files": len(set(v.get("file", "") for v in functions_dict.values())),
                            "cyclic_deps": 0,
                            "leaf_functions": 0,
                            "root_functions": 0,
                        },
                    }

                    (extracted_dir / "call_graph.json").write_text(
                        json.dumps(call_graph_data, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                        errors="ignore",
                    )

                    # (4) 写出 functions_manifest.json（供后续确定性映射/调试）
                    manifest_entries = []
                    for func_file_name, meta in sorted(func_meta.items()):
                        code = func_file_contents.get(func_file_name, "")
                        sha1 = hashlib.sha1(code.encode("utf-8", errors="ignore")).hexdigest() if code else ""
                        manifest_entries.append(
                            {
                                "func_file": func_file_name,
                                "name": meta.get("name", ""),
                                "source_file": meta.get("file", ""),
                                "start_line": int(meta.get("start_line") or 0),
                                "end_line": int(meta.get("end_line") or 0),
                                "uid": func_file_name_to_uid.get(func_file_name, ""),
                                "sha1": sha1,
                            }
                        )
                    (extracted_dir / "functions_manifest.json").write_text(
                        json.dumps(
                            {
                                "version": "1.0",
                                "project": project_path,
                                "total_functions": len(manifest_entries),
                                "functions": manifest_entries,
                            },
                            indent=2,
                            ensure_ascii=False,
                        ),
                        encoding="utf-8",
                        errors="ignore",
                    )

                    print(f"  ✓ 写回 extracted 完成: {total_exported} 个函数, call_graph.json 已生成")
                except Exception as e:
                    print(f"  ⚠ 写回 extracted/ 调用图失败: {e}")

        except ImportError as e:
            print(f"  ⚠ libclang 模块未安装: {e}")
            print(f"  ⚠ 回退到 tree-sitter 模式")
            use_libclang = False
        except Exception as e:
            print(f"  ⚠ libclang 提取失败: {e}")
            import traceback
            if verbose:
                traceback.print_exc()
            print(f"  ⚠ 回退到 tree-sitter 模式")
            use_libclang = False

        print()
    
    # =========================================================================
    # 阶段 B: 变量骨架 (State Layer) - 基于 Rustine/EvoC2Rust 的静态变量提升
    # =========================================================================
    print(f"{'='*60}")
    
    phase_start = time.time()
    all_variables = []
    variable_lifting_map = {}  # 变量提升映射: {原变量名: 新变量名}
    preprocessed_sources = {}  # 缓存预处理结果: {完整路径: 预处理后代码}
    processed_files = 0
    
    if libclang_global_vars:
        print("阶段 B: 提取全局变量 + 静态变量提升 (libclang)")
        print(f"{'='*60}")
        print(f"  📍 使用 libclang 提取的 {len(libclang_global_vars)} 个全局变量")
    else:
        print("阶段 B: 提取全局变量 + 静态变量提升 (tree-sitter)")
        print(f"{'='*60}")
    
    if libclang_global_vars:
        # 使用 libclang 提取的全局变量
        from skeleton_builder import ExtractedVariable

        # 保存所有源文件到 source_skeletons 目录（用于签名匹配）
        # 这与 tree-sitter 分支保持一致，确保 get_dependencies_match.py 能找到所有需要的源文件
        for src_file in source_files:
            try:
                if not src_file.exists():
                    continue

                # 获取安全的模块名用于文件保存
                safe_name = builder._get_safe_module_name(src_file)

                # 修复：与 tree-sitter 分支保持一致，获取并保存预处理后的代码
                # 如果 .c 文件只有 #include "xxx.h"，后续阶段需要预处理版本
                preprocessed = None
                try:
                    rec = getattr(builder, "_tu_context_files", {}).get(safe_name)
                    pre_path = (rec or {}).get("preprocessed_file") if isinstance(rec, dict) else None
                    if pre_path:
                        p = Path(str(pre_path)).expanduser()
                        if p.exists():
                            preprocessed = p.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    preprocessed = None

                if not preprocessed:
                    preprocessed = builder.preprocess_source(src_file)

                # 填充预处理缓存（供后续阶段使用）
                preprocessed_sources[str(src_file)] = preprocessed

                # 保存预处理后的源码
                preprocessed_path = source_skeleton_dir / "src" / f"{safe_name}.preprocessed.txt"
                preprocessed_path.parent.mkdir(parents=True, exist_ok=True)
                with open(preprocessed_path, 'w', encoding='utf-8') as f:
                    f.write(preprocessed)

                # 同时保存原始源码（用于签名匹配）
                with open(src_file, 'r', encoding='utf-8', errors='ignore') as f:
                    original = f.read()
                original_path = source_skeleton_dir / "src" / f"{safe_name}.txt"
                with open(original_path, 'w', encoding='utf-8') as f:
                    f.write(original)

            except Exception as e:
                print(f"  ⚠ 保存源文件骨架失败 {src_file.name}: {e}")

        # 处理全局变量
        for gvar in libclang_global_vars:
            # 跳过 extern 声明（只是声明，不是定义）
            if gvar.get('is_extern', False):
                continue
            
            # 类型映射
            c_type = gvar.get('c_type', 'int')
            is_pointer = '*' in c_type
            is_array = gvar.get('array_size') is not None
            rust_type = builder._c_type_to_rust(c_type, is_pointer, is_array)
            
            var = ExtractedVariable(
                name=gvar.get('name', ''),
                c_type=c_type,
                rust_type=rust_type,
                initial_value=gvar.get('initial_value'),
                is_const=gvar.get('is_const', False),
                is_static=gvar.get('is_static', False),
                is_extern=gvar.get('is_extern', False),
                is_pointer=is_pointer,
                from_function=None,  # 全局变量不在函数内
                array_size=gvar.get('array_size')
            )
            all_variables.append(var)
        
        processed_files = len(source_files)
    else:
        # 回退到 tree-sitter 模式
        for src_file in source_files:
            try:
                # 预处理源码（展开宏）：
                # - Truth-first: MUST reuse stage1(get_dependencies.py) pinned TU `.i` (same flags/macros/include order)
                # - Only when strict TU closure is disabled, fall back to Rustine-style lightweight preprocessing.
                preprocessed = None
                used_pinned_i = False
                # 使用完整路径作为 key，避免同名文件覆盖
                
                # 获取安全的模块名用于文件保存
                safe_name = builder._get_safe_module_name(src_file)

                try:
                    rec = getattr(builder, "_tu_context_files", {}).get(safe_name)
                    pre_path = (rec or {}).get("preprocessed_file") if isinstance(rec, dict) else None
                    if pre_path:
                        p = Path(str(pre_path)).expanduser()
                        if p.exists():
                            preprocessed = p.read_text(encoding="utf-8", errors="ignore")
                            used_pinned_i = True
                except Exception:
                    preprocessed = None

                if not preprocessed:
                    # In strict TU-closure mode, do NOT fall back to non-TU preprocessing: it breaks the
                    # single-truth-input guarantee and is a common cause of tree-sitter ERROR nodes.
                    strict_tu = os.environ.get("C2R_REQUIRE_TU_CLOSURE", "1").strip().lower() not in (
                        "0",
                        "false",
                        "no",
                    )
                    if strict_tu and getattr(builder, "_tu_context_files", None):
                        raise RuntimeError(
                            f"missing pinned .i for {safe_name} (C2R_REQUIRE_TU_CLOSURE=1); "
                            "refusing to use non-TU preprocessing"
                        )
                    preprocessed = builder.preprocess_source(src_file)

                preprocessed_sources[str(src_file)] = preprocessed
                
                # 保存预处理后的源码（用于调试和后续参考）
                preprocessed_path = source_skeleton_dir / "src" / f"{safe_name}.preprocessed.txt"
                preprocessed_path.parent.mkdir(parents=True, exist_ok=True)
                with open(preprocessed_path, 'w', encoding='utf-8') as f:
                    f.write(preprocessed)
                
                # 同时保存原始源码
                with open(src_file, 'r', encoding='utf-8', errors='ignore') as f:
                    original = f.read()
                original_path = source_skeleton_dir / "src" / f"{safe_name}.txt"
                with open(original_path, 'w', encoding='utf-8') as f:
                    f.write(original)
                
                # 提取变量（包括函数内 static 变量的提升）
                variables = builder.extract_variables_with_treesitter(preprocessed, safe_name)
                all_variables.extend(variables)
                
                # 记录变量提升映射
                for var in variables:
                    if var.from_function:
                        # 这是从函数内提升的 static 变量
                        # 原名: count, 新名: func_count
                        original_name = var.name.replace(f"{var.from_function}_", "")
                        variable_lifting_map[original_name] = var.name
                        print(f"    ⬆ 提升: {var.from_function}::{original_name} → {var.name}")
                
                processed_files += 1
                
            except Exception as e:
                print(f"  ⚠ 处理 {src_file.name} 失败: {e}")
    
    # 生成 globals.rs
    # 默认启用 Scheme A：bindgen-truth + pub static mut storage（不做 Mutex / safe wrapper）
    globals_mode = (os.environ.get("C2R_GLOBALS_MODE", "bindgen_static") or "bindgen_static").strip().lower()
    if globals_mode in {"rustmap", "safe"}:
        builder.generate_globals_rs(all_variables, use_safe_wrappers=True)
    elif globals_mode in {"plain", "unsafe", "legacy_static", "static"} or globals_mode in {"0", "false", "off", "no"}:
        builder.generate_globals_rs(all_variables, use_safe_wrappers=False)
    else:
        # bindgen_static / bindgen / default
        builder.generate_globals_rs_bindgen_static(all_variables)
    
    # 保存变量提升映射（供后续函数体翻译使用）
    lifting_map_path = source_skeleton_dir / "variable_lifting_map.json"
    with open(lifting_map_path, 'w', encoding='utf-8') as f:
        json.dump({
            "lifting_map": variable_lifting_map,
            "all_variables": [
                {
                    "name": v.name,
                    "c_type": v.c_type,
                    "rust_type": v.rust_type,
                    "from_function": v.from_function,
                    "is_static": v.is_static,
                    "is_pointer": v.is_pointer
                }
                for v in all_variables
            ]
        }, f, indent=2, ensure_ascii=False)
    
    phase_time = time.time() - phase_start
    lifted_count = len([v for v in all_variables if v.from_function])
    print(f"✓ 提取了 {len(all_variables)} 个变量 (其中 {lifted_count} 个从函数内提升)")
    print(f"  变量映射保存到: {lifting_map_path}")
    print(f"  ({processed_files} 个文件, 耗时: {phase_time:.1f}秒)")
    print()
    
    # =========================================================================
    # 阶段 C: 函数骨架 (Logic Skeleton) - 基于 EvoC2Rust 的桩代码生成
    # =========================================================================
    print(f"{'='*60}")
    print("阶段 C: 生成函数骨架 (仅签名 + unimplemented!)")
    print(f"{'='*60}")
    
    # 显示提取方式（libclang_functions 已在阶段 A.5 提取）
    if libclang_functions:
        print(f"  📍 使用 libclang 函数列表（{len(libclang_functions)} 个函数，已在阶段 A.5 提取）")
    else:
        print(f"  📍 使用 tree-sitter 提取函数（默认）")
    
    phase_start = time.time()
    
    # 准备 LLM 翻译函数（可选）
    llm_translate_fn = None
    if use_llm_for_signatures:
        def translate_signature(c_sig: str) -> str:
            prompt = create_signature_translation_prompt(c_sig)
            messages = [
                {"role": "system", "content": "You are a C to Rust translation expert. Translate function signatures accurately."},
                {"role": "user", "content": prompt}
            ]
            try:
                response = generation(messages)
                return parse_llm_signature_response(response)
            except Exception as e:
                print(f"  ⚠ LLM 翻译失败: {e}")
                return None
        
        llm_translate_fn = translate_signature
    
    total_functions = 0
    function_info_list = []  # 收集函数信息供后续使用
    module_results = []  # 收集并行处理结果
    module_name_to_src: Dict[str, Path] = {}
    module_signatures_by_module: Dict[str, List[Any]] = {}
    
    # ========== 并行处理源文件 ==========
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import multiprocessing
    
    # 获取 CPU 核心数，限制最大并行度
    max_workers = min(multiprocessing.cpu_count(), len(source_files), 8)
    
    # ========== LLM 兜底配置 ==========
    # 复用 generation.py 的配置，支持 vLLM 和外部 API 切换
    llm_client = None
    from generate.generation import (
        USE_VLLM, VLLM_BASE_URL, VLLM_API_KEY, VLLM_MODEL_NAME, VLLM_REQUEST_TIMEOUT,
        EXTERNAL_API_BASE_URL, EXTERNAL_API_KEY, EXTERNAL_API_MODEL, EXTERNAL_API_TIMEOUT
    )
    if USE_VLLM:
        llm_model = VLLM_MODEL_NAME
        api_base = VLLM_BASE_URL
        api_key = VLLM_API_KEY
        timeout = VLLM_REQUEST_TIMEOUT
    else:
        llm_model = EXTERNAL_API_MODEL
        api_base = EXTERNAL_API_BASE_URL
        api_key = EXTERNAL_API_KEY
        timeout = EXTERNAL_API_TIMEOUT

    if use_llm_fallback:
        try:
            from openai import OpenAI
            llm_client = OpenAI(base_url=api_base, api_key=api_key, timeout=timeout)
            logger.info(f"LLM 兜底已启用: {api_base}, 模型: {llm_model}")
        except Exception as e:
            logger.warning(f"LLM 兜底初始化失败: {e}，将仅使用 Tree-sitter")
            llm_client = None

    # ========== Deterministic signature source (bindgen on `.i` TU truth) ==========
    sig_source = (os.environ.get("C2R_SIGNATURE_SOURCE", "bindgen") or "bindgen").strip().lower()
    prefer_bindgen_signatures = sig_source in ("bindgen", "auto", "1", "true", "yes", "on")
    manifest_funcs_by_safe: Dict[str, List[Dict[str, Any]]] = {}
    functions_dir = get_functions_path(project_path)
    try:
        manifest_path = get_extracted_path(project_path) / "functions_manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8", errors="ignore") or "{}")
            funcs = manifest.get("functions") if isinstance(manifest, dict) else None
            if isinstance(funcs, list):
                for meta in funcs:
                    if not isinstance(meta, dict):
                        continue
                    func_file = str(meta.get("func_file") or "").strip()
                    if not func_file:
                        continue
                    safe = func_file.rsplit("_", 1)[0] if "_" in func_file else func_file
                    manifest_funcs_by_safe.setdefault(safe, []).append(meta)
                for safe, metas in list(manifest_funcs_by_safe.items()):
                    try:
                        manifest_funcs_by_safe[safe] = sorted(
                            metas,
                            key=lambda m: (
                                int(m.get("start_line") or 0),
                                str(m.get("func_file") or ""),
                                str(m.get("name") or ""),
                            ),
                        )
                    except Exception:
                        manifest_funcs_by_safe[safe] = metas
    except Exception:
        manifest_funcs_by_safe = {}

    def _extract_c_signature_from_func_file(func_file: str) -> str:
        if not func_file:
            return ""
        try:
            p = functions_dir / f"{func_file}.txt"
            if not p.exists():
                return ""
            code = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return ""
        brace = code.find("{")
        sig = (code[:brace] if brace != -1 else code.splitlines()[0] if code else "").strip()
        # collapse whitespace
        return " ".join(sig.replace("\n", " ").replace("\r", " ").split()).strip()
    
    def process_source_file(src_file):
        """处理单个源文件（可并行执行）"""
        try:
            # 优先使用预处理后的代码（使用完整路径作为 key）
            src_key = str(src_file)
            if src_key in preprocessed_sources:
                source_code = preprocessed_sources[src_key]
            else:
                with open(src_file, 'r', encoding='utf-8', errors='ignore', newline='') as f:
                    source_code = f.read()

            # ========== 按文件决定提取方式（per-file fallback）==========
            # 1. 检查是否使用 libclang 模式
            # 2. 检查当前文件是否在低覆盖率列表中
            # 3. 如果在低覆盖率列表中，回退到 tree-sitter
            use_libclang_for_this_file = False
            fallback_reason = None

            if use_libclang and libclang_functions is not None:
                # 计算相对路径（与 coverage 报告中的 file_path 格式匹配）
                try:
                    rel_path = str(src_file.resolve().relative_to(project_root))
                except ValueError:
                    rel_path = src_file.name

                # 检查是否在低覆盖率列表中
                if rel_path in low_coverage_files:
                    use_libclang_for_this_file = False
                    fallback_reason = f"低覆盖率 ({per_file_coverage.get(rel_path, 0):.0%})"
                else:
                    use_libclang_for_this_file = True

            if use_libclang_for_this_file:
                # 从 libclang 结果中提取该文件的函数
                from skeleton_builder import FunctionSignature
                signatures = []
                src_file_abs = str(src_file.resolve())
                for func in libclang_functions:
                    if func.source_file_abs == src_file_abs:
                        sig = FunctionSignature(
                            name=func.name,
                            c_signature=func.c_signature,
                            rust_signature="",
                            return_type=func.return_type,
                            parameters=func.parameters,
                            is_static=func.is_static
                        )
                        signatures.append(sig)
            else:
                # 使用 Tree-sitter 提取函数签名
                signatures = builder.extract_function_signatures(source_code)
            
            # 查找并处理对应的头文件
            header_signatures = []
            possible_headers = [
                src_file.with_suffix('.h'),
                src_file.parent.parent / 'include' / (src_file.stem + '.h'),
                src_file.parent / 'include' / (src_file.stem + '.h'),
            ]
            
            for header_file in possible_headers:
                if header_file.exists():
                    try:
                        with open(header_file, 'r', encoding='utf-8', errors='ignore', newline='') as f:
                            header_code = f.read()
                        header_signatures = builder.extract_function_signatures(header_code)
                        break
                    except Exception:
                        pass
            
            # 合并签名并去重
            sig_map = {sig.name: sig for sig in header_signatures}
            sig_map.update({sig.name: sig for sig in signatures})
            signatures = list(sig_map.values())

            # 返回额外信息用于 LLM 兜底判断和 per-file fallback 日志
            return (src_file, signatures, len(header_signatures) if header_signatures else 0, source_code, fallback_reason)
        except Exception as e:
            return (src_file, None, str(e), None, None)
    
    # 并行提取函数签名
    print(f"  使用 {max_workers} 个线程并行处理 {len(source_files)} 个源文件...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_source_file, sf): sf for sf in source_files}
        
        for future in as_completed(futures):
            result = future.result()
            module_results.append(result)
    
    # 串行生成模块文件（避免文件写入冲突）+ LLM 兜底
    llm_fallback_count = 0
    ts_fallback_count = 0  # 按文件 tree-sitter 回退计数

    for result in module_results:
        # 处理返回格式（5 个元素: src_file, signatures, header_count, source_code, fallback_reason）
        if len(result) == 5:
            src_file, signatures, header_count, source_code, fallback_reason = result
        elif len(result) == 4:
            src_file, signatures, header_count, source_code = result
            fallback_reason = None
        else:
            src_file, signatures, header_count = result
            source_code = None
            fallback_reason = None

        if signatures is None:
            print(f"  ⚠ {src_file.name} 失败: {header_count}")  # header_count 此时是错误信息
            continue

        # ========== bindgen allowlist signatures (preferred) ==========
        used_bindgen_sigs = False
        safe_name = None
        try:
            safe_name = builder._get_safe_module_name(src_file)
        except Exception:
            safe_name = None
        if prefer_bindgen_signatures and safe_name and safe_name in manifest_funcs_by_safe:
            metas = manifest_funcs_by_safe.get(safe_name) or []
            fn_names = [str(m.get("name") or "").strip() for m in metas if isinstance(m, dict)]
            fn_names = [n for n in fn_names if n]
            bindgen_map = {}
            try:
                bindgen_map = builder.get_bindgen_function_signatures_from_tu(
                    safe_name=safe_name,
                    fn_names=fn_names,
                    source_file=src_file,
                )
            except Exception:
                bindgen_map = {}
            if bindgen_map:
                existing_by_name = {s.name: s for s in (signatures or []) if getattr(s, "name", None)}
                new_sigs = []
                for m in metas:
                    if not isinstance(m, dict):
                        continue
                    name = str(m.get("name") or "").strip()
                    func_file = str(m.get("func_file") or "").strip()
                    if not name:
                        continue
                    rust_sig = str(bindgen_map.get(name) or "").strip()
                    if rust_sig:
                        c_sig = _extract_c_signature_from_func_file(func_file) or str(getattr(existing_by_name.get(name), "c_signature", "") or "")
                        is_static = bool(re.search(r"\bstatic\b", c_sig)) if c_sig else False
                        from skeleton_builder import FunctionSignature

                        existing = existing_by_name.get(name)
                        ret_ty = str(getattr(existing, "return_type", "") or "") if existing else ""
                        params = getattr(existing, "parameters", None) if existing else None
                        if not isinstance(params, list):
                            params = []

                        new_sigs.append(
                            FunctionSignature(
                                name=name,
                                c_signature=c_sig or f"{name}(...)",
                                rust_signature=rust_sig,
                                return_type=ret_ty,
                                parameters=params,
                                is_static=is_static,
                            )
                        )
                    else:
                        # Fall back to extracted signature (tree-sitter/libclang) for missing bindgen items.
                        if name in existing_by_name:
                            new_sigs.append(existing_by_name[name])
                if new_sigs:
                    signatures = new_sigs
                    used_bindgen_sigs = True

        # 日志：如果此文件回退到 tree-sitter
        if fallback_reason:
            ts_fallback_count += 1
            if verbose:
                print(f"  🔄 {src_file.name}: 回退到 tree-sitter ({fallback_reason})")
        
        # ========== LLM 兜底逻辑（确定性条件）==========
        # 仅在以下确定性条件下触发 LLM：
        # 1. 使用 tree-sitter 模式（非 libclang 或 per-file 回退）
        # 2. 提取结果为空，但源文件有实质性内容（> 500 字节）
        # 3. 不使用不可靠的正则估算
        # fallback_reason 不为 None 表示此文件回退到了 tree-sitter
        using_treesitter = (not use_libclang) or (fallback_reason is not None)
        if source_code and llm_client and using_treesitter:
            actual_count = len(signatures)
            file_has_content = len(source_code.strip()) > 500
            
            # 确定性触发条件：提取为空但文件有内容
            should_fallback = (actual_count == 0 and file_has_content)
            
            if should_fallback and not used_bindgen_sigs:
                print(f"  🤖 {src_file.name}: 提取 0 个函数但文件有内容，尝试 LLM 兜底...")
                try:
                    from llm_signature_extractor import LLMSignatureExtractor, ExtractedSignature
                    from skeleton_builder import FunctionSignature
                    
                    extractor = LLMSignatureExtractor(llm_client, llm_model)
                    llm_sigs = extractor.extract(source_code)
                    
                    if llm_sigs:
                        # 合并结果：添加 LLM 发现但 Tree-sitter 遗漏的函数
                        existing_names = {sig.name for sig in signatures}
                        added = 0
                        for llm_sig in llm_sigs:
                            if llm_sig.name not in existing_names:
                                sig = FunctionSignature(
                                    name=llm_sig.name,
                                    c_signature=llm_sig.raw_signature or f"{llm_sig.return_type} {llm_sig.name}(...)",
                                    rust_signature="",
                                    return_type=llm_sig.return_type,
                                    parameters=llm_sig.parameters,
                                    is_static=llm_sig.is_static
                                )
                                signatures.append(sig)
                                existing_names.add(llm_sig.name)
                                added += 1
                        
                        if added > 0:
                            print(f"    ✅ LLM 补充了 {added} 个函数（总计 {len(signatures)} 个）")
                            llm_fallback_count += added
                        else:
                            print(f"    ℹ️ LLM 未发现新函数")
                except Exception as e:
                    logger.warning(f"LLM 兜底失败: {e}")
        
        if header_count > 0:
            print(f"    📄 从头文件提取了 {header_count} 个函数")
        
        if signatures:
            # 生成桩代码（仅 unimplemented!()）
            stubs = builder.generate_function_stubs(
                signatures, 
                llm_translate_fn if use_llm_for_signatures else None,
                use_type_mapper=use_type_mapper,
                use_llm_type_mapper=use_llm_type_mapper
            )
            
            # 生成模块文件（使用安全模块名，避免文件名冲突）
            module_name = builder._generate_module_file(src_file, stubs)

            # Record per-module signatures for TU-truth local types generation.
            try:
                module_name_to_src[module_name] = src_file
                module_signatures_by_module[module_name] = list(signatures or [])
            except Exception:
                pass
            
            total_functions += len(stubs)
            print(f"  ✓ {src_file.name}: {len(stubs)} 个函数 -> {module_name}.rs")
            
            # 记录函数信息
            for sig in signatures:
                function_info_list.append({
                    "name": sig.name,
                    "c_signature": sig.c_signature,
                    "rust_signature": sig.rust_signature,
                    "source_file": module_name,
                    "is_static": sig.is_static
                })
        else:
            # 创建空模块
            module_name = builder._get_safe_module_name(src_file)
            empty_path = skeleton_dir / "src" / f"{module_name}.rs"
            with open(empty_path, 'w', encoding='utf-8') as f:
                f.write(f'//! Module: {module_name}\n//! No functions found.\n')
            try:
                module_name_to_src[module_name] = src_file
                module_signatures_by_module[module_name] = []
            except Exception:
                pass

    # =========================================================================
    # 阶段 C.2.5: TU 真值类型补全（header==0 时从 .i 生成 + 私有类型本地化）
    # =========================================================================
    try:
        tu_types_info = builder.ensure_types_and_local_types_from_tu(
            module_name_to_src=module_name_to_src,
            module_signatures=module_signatures_by_module,
            header_files_present=bool(header_files),
        )
        if tu_types_info.get("generated_global_units") or tu_types_info.get("generated_local_units"):
            print("  🔧 TU 真值类型补全完成:")
            if tu_types_info.get("generated_global_units"):
                print(f"    - 全局补全单元: {len(tu_types_info.get('generated_global_units') or [])}")
            if tu_types_info.get("generated_local_units"):
                print(f"    - 本地补全单元: {len(tu_types_info.get('generated_local_units') or [])}")
    except Exception as e:
        logger.debug(f"TU 真值类型补全失败（将由后续编译错误暴露）: {e}")
    
    # 保存函数信息（供后续函数体翻译使用）
    function_info_path = source_skeleton_dir / "function_signatures.json"
    with open(function_info_path, 'w', encoding='utf-8') as f:
        json.dump(function_info_list, f, indent=2, ensure_ascii=False)

    # =========================================================================
    # 阶段 C.3: 生成确定性签名映射 (func_file -> rust_signature)
    # =========================================================================
    # 目的：避免阶段2对 skeleton signatures 进行 BM25 猜测匹配导致错配/遗漏（典型：Ipv4/Ivp6 数字被分词丢失）。
    # 输入：
    # - extracted/<proj>/functions_manifest.json (func_file,name,source_file,uid,sha1...)
    # - source_skeletons/<proj>/function_signatures.json (list[{name,c_signature,rust_signature,source_file,is_static}])
    # 输出：
    # - source_skeletons/<proj>/func_file_to_rust_sig.json (dict: func_file -> rust_signature)
    # - source_skeletons/<proj>/rust_signature_by_uid.json (dict: uid -> rust_signature)
    try:
        manifest_path = get_extracted_path(project_path) / "functions_manifest.json"
        func_file_to_sig_path = source_skeleton_dir / "func_file_to_rust_sig.json"
        uid_to_sig_path = source_skeleton_dir / "rust_signature_by_uid.json"

        if not manifest_path.exists():
            if verbose:
                print(f"  ⚠ functions_manifest.json 不存在，跳过确定性签名映射: {manifest_path}")
        else:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8", errors="ignore"))
            manifest_funcs = manifest.get("functions") or []

            def _norm_sig(s: str) -> str:
                s = (s or "").strip()
                if not s:
                    return ""
                # collapse whitespace
                s = " ".join(s.replace("\n", " ").replace("\r", " ").split())
                # strip a trailing '{' if any
                if s.endswith("{"):
                    s = s[:-1].strip()
                return s

            # Build indices from skeleton signatures
            sig_by_name_file = {}
            sig_by_csig = {}
            sig_by_name_only = {}

            for it in function_info_list:
                name = (it.get("name") or "").strip()
                src = (it.get("source_file") or "").strip()
                rust_sig = (it.get("rust_signature") or "").strip()
                c_sig = _norm_sig(it.get("c_signature") or "")
                if not name or not rust_sig:
                    continue
                if src:
                    sig_by_name_file[(name, src)] = rust_sig
                if c_sig:
                    sig_by_csig[c_sig] = rust_sig
                sig_by_name_only.setdefault(name, []).append(rust_sig)

            functions_dir = get_functions_path(project_path)
            funcfile_to_sig = {}
            uid_to_sig = {}
            matched = 0
            for meta in manifest_funcs:
                func_file = (meta.get("func_file") or "").strip()
                name = (meta.get("name") or "").strip()
                src = (meta.get("source_file") or "").strip()
                uid = (meta.get("uid") or "").strip()

                rust_sig = ""
                # 1) Best: (name, source_file) direct hit
                if name and src:
                    rust_sig = sig_by_name_file.get((name, src), "") or ""

                    # 1.5) Fallback: 将 source_file 转换为安全名称再匹配
                    # 处理 manifest 中 source_file 是路径格式（如 "src/url.h"）
                    # 而 function_signatures.json 中是安全名称格式（如 "src_url"）的情况
                    if not rust_sig and src:
                        try:
                            # 转换为安全名称格式：src/url.h -> src_url, src/url.c -> src_url
                            src_path = Path(src)
                            safe_src = src_path.stem  # url
                            if src_path.parent and str(src_path.parent) != '.':
                                safe_src = str(src_path.parent).replace('/', '_').replace('\\', '_') + '_' + safe_src  # src_url
                            rust_sig = sig_by_name_file.get((name, safe_src), "") or ""
                        except Exception:
                            pass

                # 2) Fallback: match by exact C signature (from extracted function file)
                if not rust_sig and functions_dir.exists() and func_file:
                    try:
                        p = functions_dir / f"{func_file}.txt"
                        if p.exists():
                            code = p.read_text(encoding="utf-8", errors="ignore")
                            brace = code.find("{")
                            c_sig = _norm_sig(code[:brace] if brace != -1 else code.splitlines()[0])
                            if c_sig:
                                rust_sig = sig_by_csig.get(c_sig, "") or ""
                    except Exception:
                        pass
                # 3) Last resort: name-only if unique
                if not rust_sig and name:
                    cands = sig_by_name_only.get(name) or []
                    cands = [c for c in cands if c]
                    if len(set(cands)) == 1:
                        rust_sig = cands[0]

                if func_file:
                    funcfile_to_sig[func_file] = rust_sig if rust_sig else None
                    if rust_sig:
                        matched += 1
                if uid:
                    uid_to_sig[uid] = rust_sig if rust_sig else ""

            func_file_to_sig_path.write_text(
                json.dumps(funcfile_to_sig, indent=2, ensure_ascii=False),
                encoding="utf-8",
                errors="ignore",
            )
            uid_to_sig_path.write_text(
                json.dumps(uid_to_sig, indent=2, ensure_ascii=False),
                encoding="utf-8",
                errors="ignore",
            )
            print(f"  ✓ 确定性签名映射已生成: {matched}/{len(funcfile_to_sig)} ({func_file_to_sig_path})")
    except Exception as e:
        print(f"  ⚠ 生成确定性签名映射失败（将回退到 BM25）：{e}")
    
    phase_time = time.time() - phase_start
    print(f"✓ 生成了 {total_functions} 个函数骨架 (耗时: {phase_time:.1f}秒)")
    print(f"  函数签名保存到: {function_info_path}")
    print()
    
    # =========================================================================
    # 阶段 C.5: 补全缺失类型 (必须在所有文件解析完之后)
    # =========================================================================
    print(f"{'='*60}")
    print("阶段 C.5: 补全缺失类型 (防御性不透明类型生成)")
    print(f"{'='*60}")

    if truth_mode:
        print("⏭️ Truth mode: 跳过防御性类型补全（保持 types.rs 为 bindgen 真值输出）")
        print()
    else:
        phase_start = time.time()
        builder.finalize_types_rs()
        phase_time = time.time() - phase_start
        print(f"✓ 类型补全完成 (耗时: {phase_time:.1f}秒)")
        print()
    
    # =========================================================================
    # 阶段 C.6: 构建函数调用图 (用于翻译顺序优化和上下文提供)
    # =========================================================================
    call_graph_builder = None
    if CALL_GRAPH_AVAILABLE:
        print(f"{'='*60}")
        print("阶段 C.6: 构建函数调用图 (用于翻译优化)")
        print(f"{'='*60}")
        
        phase_start = time.time()
        try:
            call_graph_builder = CallGraphBuilder(max_workers=4)
            call_graph_builder.analyze_files(source_files)
            
            # 保存调用图
            call_graph_path = source_skeleton_dir / "call_graph.json"
            call_graph_builder.save(call_graph_path)
            
            # 保存推荐翻译顺序
            translation_order = call_graph_builder.get_translation_order()
            order_path = source_skeleton_dir / "translation_order.json"
            with open(order_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "order": translation_order,
                    "total_functions": call_graph_builder.stats.total_functions,
                    "total_calls": call_graph_builder.stats.total_calls,
                    "cyclic_deps": call_graph_builder.stats.cyclic_deps
                }, f, indent=2, ensure_ascii=False)
            
            phase_time = time.time() - phase_start
            print(f"✓ 调用图构建完成 (耗时: {phase_time:.1f}秒)")
            print(f"  - 函数数: {call_graph_builder.stats.total_functions}")
            print(f"  - 调用关系: {call_graph_builder.stats.total_calls}")
            print(f"  - 循环依赖: {call_graph_builder.stats.cyclic_deps}")
            print(f"  - 调用图保存到: {call_graph_path}")
            print(f"  - 翻译顺序保存到: {order_path}")
            
            # 可选：生成 DOT 可视化文件
            dot_path = source_skeleton_dir / "call_graph.dot"
            call_graph_builder.generate_dot(dot_path)
            print(f"  - DOT 文件保存到: {dot_path}")
            
        except Exception as e:
            print(f"⚠ 调用图构建失败: {e}")
            import traceback
            traceback.print_exc()
        
        print()
    else:
        print("跳过调用图构建 (call_graph 模块不可用)")
    
    # =========================================================================
    # 生成 main.rs 和 Cargo.toml
    # =========================================================================
    # Ensure compat layer exists before generating main.rs.
    # Module stubs import `crate::compat::*;`, so missing compat will cause E0432 on first compile.
    try:
        builder._generate_compat_rs()
    except Exception as e:
        logger.debug(f"生成 compat.rs 失败（将由后续精准修复兜底）: {e}")
    builder._generate_main_rs()
    builder._generate_cargo_toml()
    
    # =========================================================================
    # 阶段 D: 编译验证
    # =========================================================================
    print(f"{'='*60}")
    print("阶段 D: 骨架编译验证")
    print(f"{'='*60}")

    enable_skeleton_repairs = (not truth_mode) and (
        os.environ.get("C2R_ENABLE_SKELETON_REPAIRS", "true").strip().lower() in ("1", "true", "yes", "y", "on")
    )

    # 预先净化 types.rs，修复常见结构性问题（如孤立 attributes / 重复定义等）
    if enable_skeleton_repairs:
        try:
            types_path = skeleton_dir / "src" / "types.rs"
            modified, fix_count = builder._sanitize_types_rs(types_path)
            if modified and fix_count:
                print(f"  🧹 types.rs 净化器修复了 {fix_count} 个结构性错误")
        except Exception as e:
            logger.debug(f"types.rs 预净化失败: {e}")
    else:
        if truth_mode:
            print("⏭️ Truth mode: 跳过 types.rs 净化/修复（保持真值，不做派生层修补）")

    phase_start = time.time()
    success, error_msg = builder.cargo_check()
    phase_time = time.time() - phase_start
    
    if success:
        print(f"✓ 骨架编译成功！(耗时: {phase_time:.1f}秒)")
    else:
        print(f"⚠ 骨架编译失败 (耗时: {phase_time:.1f}秒)")
        print(f"  错误信息（完整）:")
        if error_msg:
            print(error_msg.rstrip("\n"))
        else:
            print("Unknown error")
        try:
            error_path = skeleton_dir / "compile_error_full.log"
            error_path.write_text(error_msg or "", encoding="utf-8", errors="ignore")
            print(f"  完整错误已保存到: {error_path}")
        except Exception as e:
            logger.debug(f"保存完整编译错误失败: {e}")
        
        if not enable_skeleton_repairs:
            print("\n⏭️ 已禁用骨架修复（C2R_TRUTH_MODE 或 C2R_ENABLE_SKELETON_REPAIRS=false）")
            # Truth-first: if the skeleton itself cannot compile and we are not allowed to mutate the truth layer,
            # fail-closed so later stages don't waste time on impossible repair loops.
            fail_closed = truth_mode or (
                os.environ.get("C2R_FAIL_ON_SKELETON_COMPILE_ERROR", "0").strip().lower() in ("1", "true", "yes", "y", "on")
            )
            if fail_closed:
                raise SystemExit(1)
        else:
            # ========== 第一轮修复：基于规则的自动修复 ==========
            print("\n尝试基于规则的自动修复骨架...")
            max_rule_rounds = int(os.environ.get("SKELETON_RULE_FIX_ROUNDS", "5"))
            did_any_rule_fix = False
            for round_idx in range(1, max_rule_rounds + 1):
                did_fix = _auto_fix_missing_types(skeleton_dir, error_msg)
                if not did_fix:
                    break
                did_any_rule_fix = True

                # 每轮修复后再净化一次 types.rs，避免引入重复/结构性问题
                try:
                    types_path = skeleton_dir / "src" / "types.rs"
                    modified, fix_count = builder._sanitize_types_rs(types_path)
                    if modified and fix_count:
                        print(f"  🧹 types.rs 净化器修复了 {fix_count} 个结构性错误 (round {round_idx})")
                except Exception as e:
                    logger.debug(f"types.rs 修复后净化失败 (round {round_idx}): {e}")

                # 重新检查编译
                success_after_fix, error = builder.cargo_check(log_suffix="after_rule_fix")
                if success_after_fix:
                    print(f"\n✓ 基于规则的修复后编译成功！")
                    success = True
                    break

                error_msg = error

            if did_any_rule_fix and not success:
                print(f"\n⚠ 基于规则的修复后仍有错误")
                print(f"  错误信息（完整）:")
                if error_msg:
                    print(error_msg.rstrip("\n"))
                else:
                    print("Unknown error")
                try:
                    error_path = skeleton_dir / "compile_error_after_rule_fix.log"
                    error_path.write_text(error_msg or "", encoding="utf-8", errors="ignore")
                    print(f"  完整错误已保存到: {error_path}")
                except Exception as e:
                    logger.debug(f"保存规则修复后编译错误失败: {e}")
            elif not did_any_rule_fix:
                print("  未发现可自动修复的类型缺失问题")

            # ========== 第二轮修复：精准修复（规则 + LLM）==========
            if not success:
                print(f"\n{'='*60}")
                print("阶段 D.2: 精准修复（规则优先，LLM 兜底）")
                print(f"{'='*60}")

                try:
                    from llm_precise_fixer import LLMPreciseFixer

                    # ★★★ 增强版：传入 C 源码目录和 compile_commands 用于精准搜索 ★★★
                    # 优先使用 compile_commands.json 中的文件和 include 路径
                    c_source_dirs = [Path(PROJECT_ROOT)] if PROJECT_ROOT else []

                    # 获取 compile_commands_parser（如果可用）
                    cc_parser = builder.compile_commands_parser if hasattr(builder, 'compile_commands_parser') else None

                    # 即使没有 LLM，规则修复也能处理很多常见错误
                    fixer = LLMPreciseFixer(
                        skeleton_dir,
                        llm_client,
                        llm_model,
                        c_source_dirs=c_source_dirs,
                        compile_commands_parser=cc_parser  # ★★★ 传入精准搜索支持 ★★★
                    )
                    # LLM 最多重试 3 次（用户可通过参数调整）
                    fix_success, fixed_count, total_errors = fixer.fix_all_errors(max_rounds=5)

                    if fix_success:
                        print(f"\n✓ 精准修复成功！修复了 {fixed_count} 个错误")
                        success = True
                    else:
                        print(f"\n⚠ 精准修复后仍有错误")
                        print(f"  修复统计: {fixed_count}/{total_errors}")
                        if not llm_client:
                            print(f"  提示: 启用 LLM 可能能修复更多错误")
                        # 输出最终编译错误（完整）
                        try:
                            final_ok, final_err = builder.cargo_check()
                            if not final_ok:
                                print(f"\n  最终错误信息（完整）:")
                                if final_err:
                                    print(final_err.rstrip("\n"))
                                else:
                                    print("Unknown error")
                                try:
                                    final_error_path = skeleton_dir / "compile_error_after_precise_fix.log"
                                    final_error_path.write_text(final_err or "", encoding="utf-8", errors="ignore")
                                    print(f"  完整错误已保存到: {final_error_path}")
                                except Exception as e:
                                    logger.debug(f"保存精准修复后编译错误失败: {e}")
                        except Exception as e:
                            logger.debug(f"获取精准修复后的完整编译错误失败: {e}")

                    # ★★★ 生成诊断报告 ★★★
                    try:
                        from config.predefines import get_predefine_manager
                        manager = get_predefine_manager()

                        # 检查是否有占位符
                        placeholders = manager.get_placeholder_items()
                        placeholder_types = placeholders.get('types', [])
                        placeholder_constants = placeholders.get('constants', [])

                        if placeholder_types or placeholder_constants:
                            print(f"\n{'='*60}")
                            print("⚠️ 诊断报告：以下符号使用了占位符，需要人工审查")
                            print(f"{'='*60}")

                            if placeholder_types:
                                print(f"\n📋 占位符类型 ({len(placeholder_types)} 个):")
                                for t in placeholder_types[:10]:  # 最多显示10个
                                    print(f"  - {t.name}: {t.failure_reason}")
                                    if t.c_source_file:
                                        print(f"      来源: {t.c_source_file}:{t.c_source_line}")

                            if placeholder_constants:
                                print(f"\n📋 占位符常量 ({len(placeholder_constants)} 个):")
                                for c in placeholder_constants[:10]:
                                    print(f"  - {c.name} = {c.value}: {c.failure_reason}")
                                    if c.c_source_file:
                                        print(f"      来源: {c.c_source_file}:{c.c_source_line}")

                            # 保存完整诊断报告
                            report_path = skeleton_dir / "diagnostic_report.txt"
                            manager.generate_diagnostic_report(report_path)
                            print(f"\n📄 完整诊断报告已保存到: {report_path}")
                        else:
                            print("\n✅ 所有学习的类型和常量都有真实定义")
                        
                    except Exception as e:
                        logger.debug(f"生成诊断报告失败: {e}")

                except ImportError as e:
                    logger.warning(f"精准修复模块未加载: {e}")
                except Exception as e:
                    logger.error(f"精准修复失败: {e}")
    
    # 最终统计
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"分层骨架构建完成！")
    print(f"总耗时: {total_time:.1f}秒 ({total_time/60:.1f}分钟)")
    print(f"输出目录: {skeleton_dir}")
    print(f"{'='*60}\n")
    
    return skeleton_dir


def _auto_fix_missing_types(skeleton_dir: Path, error_msg: str) -> bool:
    """
    自动修复缺失的类型定义
    
    对于外部库类型，生成占位符定义以通过编译
    
    Returns:
        是否有修复
    """
    if not error_msg:
        return False
    
    types_file = skeleton_dir / "src" / "types.rs"
    if not types_file.exists():
        return False
    
    # 从错误信息中提取缺失的类型
    missing_types = set()
    
    # 匹配 Rust 编译器的类型缺失错误（支持多种格式）
    type_patterns = [
        r"cannot find type `(\w+)`",  # 最宽松的模式，匹配任何 "cannot find type `XXX`"
        r"cannot find value `(\w+)`",
        r"not found in this scope.*`(\w+)`",
        r"failed to resolve: use of undeclared type `(\w+)`",
        r"cannot find type or namespace `(\w+)`",
        r"use of undeclared type `(\w+)`",
    ]
    
    for pattern in type_patterns:
        for match in re.finditer(pattern, error_msg):
            type_name = match.group(1)
            # 排除 Rust 内置类型和常见关键字
            if type_name not in ['i32', 'u32', 'i64', 'u64', 'usize', 'isize', 
                                  'bool', 'char', 'str', 'String', 'Vec', 'Option',
                                  'Result', 'Box', 'Rc', 'Arc', 'self', 'Self',
                                  'true', 'false', 'None', 'Some', 'Ok', 'Err']:
                missing_types.add(type_name)
    
    if not missing_types:
        return False
    
    print(f"  发现 {len(missing_types)} 个缺失类型，自动生成占位符...")
    
    # 读取现有 types.rs
    with open(types_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 生成占位符定义
    placeholders = []
    placeholders.append('\n// ==========================================')
    placeholders.append('// Auto-generated placeholders for external types')
    placeholders.append('// These allow compilation without the actual headers')
    placeholders.append('// ==========================================\n')
    
    # === C2R FIX: 已知的常量名（不应被生成为结构体）===
    known_constants = {
        'PTHREAD_MUTEX_INITIALIZER', 'PTHREAD_COND_INITIALIZER',
        'PTHREAD_RWLOCK_INITIALIZER', 'PTHREAD_ONCE_INIT',
        '__PTHREAD_MUTEX_INITIALIZER', '__PTHREAD_COND_INITIALIZER',
        '__PTHREAD_RWLOCK_INITIALIZER', '__PTHREAD_ONCE_INIT',
        'NULL', 'TRUE', 'FALSE', 'EOF',
        'EINVAL', 'ENOMEM', 'ENOENT', 'EEXIST', 'EBUSY', 'EAGAIN',
        'ETIMEDOUT', 'ENODEV', 'EFAULT', 'ENOSYS', 'ERANGE', 'ENOTCONN',
    }

    for type_name in sorted(missing_types):
        # 检查是否已经定义（包括 const 定义）
        if (f'type {type_name}' in content or
            f'struct {type_name}' in content or
            f'enum {type_name}' in content or
            f'const {type_name}' in content):
            continue

        # 跳过已知的常量名
        if type_name in known_constants:
            continue

        # === C2R FIX: 处理标准 C FFI 类型 ===
        # 这些类型应该使用 core::ffi 中的定义，而不是 i32
        c_ffi_types = {
            'c_void': 'core::ffi::c_void',
            'c_char': 'core::ffi::c_char',
            'c_schar': 'core::ffi::c_schar',
            'c_uchar': 'core::ffi::c_uchar',
            'c_short': 'core::ffi::c_short',
            'c_ushort': 'core::ffi::c_ushort',
            'c_int': 'core::ffi::c_int',
            'c_uint': 'core::ffi::c_uint',
            'c_long': 'core::ffi::c_long',
            'c_ulong': 'core::ffi::c_ulong',
            'c_longlong': 'core::ffi::c_longlong',
            'c_ulonglong': 'core::ffi::c_ulonglong',
            'c_float': 'core::ffi::c_float',
            'c_double': 'core::ffi::c_double',
        }

        if type_name in c_ffi_types:
            placeholders.append(f'/// Standard C FFI type `{type_name}`')
            placeholders.append(f'pub use {c_ffi_types[type_name]};')
        elif type_name.endswith('_t') or type_name.endswith('_T'):
            # 可能是 typedef，生成类型别名
            placeholders.append(f'/// Placeholder for external type `{type_name}`')
            placeholders.append(f'pub type {type_name} = *mut core::ffi::c_void;')
        elif type_name.startswith('g_') or type_name.startswith('G_'):
            # 可能是全局变量，跳过（由 globals.rs 处理）
            continue
        elif type_name[0].isupper():
            # 可能是结构体或类
            placeholders.append(f'/// Opaque placeholder for external struct `{type_name}`')
            placeholders.append(f'#[repr(C)]')
            placeholders.append(f'pub struct {type_name} {{')
            placeholders.append(f'    _unused: [u8; 0],')
            placeholders.append(f'}}')
        else:
            # 其他情况，生成类型别名（使用 c_int 而不是 i32，更符合 FFI 语义）
            placeholders.append(f'/// Placeholder for external type `{type_name}`')
            placeholders.append(f'pub type {type_name} = core::ffi::c_int;')
        
        placeholders.append('')
    
    if len(placeholders) > 4:  # 超过开头的注释行
        # 添加到 types.rs
        with open(types_file, 'a', encoding='utf-8') as f:
            f.write('\n'.join(placeholders))
        
        print(f"  ✓ 已添加 {len(missing_types)} 个类型占位符到 types.rs")
        return True
    
    return False


def repair_layered_skeleton(skeleton_dir: Path, error_msg: str, max_rounds: int = 3):
    """
    修复分层骨架的编译错误
    
    专门针对签名错误进行修复，不修改函数体
    """
    from auto_test_rust import run_tests
    
    print(f"\n开始修复骨架（最多 {max_rounds} 轮）")
    
    # 首先尝试自动修复缺失的类型
    if _auto_fix_missing_types(skeleton_dir, error_msg):
        # 重新检查编译
        output, error, success = run_tests(str(skeleton_dir), ["cargo", "check", "--offline"])
        if success:
            print(f"\n✓ 自动类型修复后编译成功！")
            return True
        error_msg = error if error else output
    
    for round_num in range(1, max_rounds + 1):
        print(f"\n--- 修复轮次 {round_num}/{max_rounds} ---")
        
        # 每轮都尝试自动修复缺失类型
        _auto_fix_missing_types(skeleton_dir, error_msg)
        
        # 解析错误信息，找出需要修复的文件
        error_files = _parse_skeleton_errors(error_msg)
        
        if not error_files:
            print("无法解析错误信息，跳过修复")
            break
        
        print(f"发现 {len(error_files)} 个文件需要修复")
        
        # 逐个修复
        for error_file, errors in error_files.items():
            file_path = skeleton_dir / "src" / error_file
            if not file_path.exists():
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 构建修复提示
                repair_prompt = f"""Fix the Rust compilation errors in this skeleton code.

## Current Code
```rust
{content}
```

## Compilation Errors
{chr(10).join(errors[:5])}

## Rules
1. ONLY fix the errors, keep everything else unchanged
2. DO NOT implement function bodies - keep unimplemented!()
3. Fix type errors, missing imports, syntax errors
4. For missing external types, use placeholder types:
   - Use `*mut std::ffi::c_void` for unknown pointer types
   - Use `i32` or appropriate primitive for unknown value types
   - Define opaque structs if needed: `#[repr(C)] pub struct TypeName {{ _unused: [u8; 0] }}`
5. For external C functions, declare them in an `extern "C" {{ }}` block
6. DO NOT add `use crate::xxx;` for modules that don't exist
7. Output the complete fixed code

## Output
```rust
(fixed code)
```
"""
                
                messages = [
                    {"role": "system", "content": "You are a Rust expert. Fix compilation errors while preserving the skeleton structure."},
                    {"role": "user", "content": repair_prompt}
                ]
                
                response = generation(messages)
                fixed_code = extract_code(response)
                
                if fixed_code and len(fixed_code) > 50:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(fixed_code)
                    print(f"  ✓ 已修复: {error_file}")
                    
            except Exception as e:
                print(f"  ⚠ 修复 {error_file} 失败: {e}")
        
        # 重新检查编译
        output, error, success = run_tests(str(skeleton_dir), ["cargo", "check", "--offline"])
        
        if success:
            print(f"\n✓ 第 {round_num} 轮修复后编译成功！")
            return True
        else:
            error_msg = error if error else output
            print(f"  第 {round_num} 轮修复后仍有错误")
    
    print(f"\n⚠ {max_rounds} 轮修复后仍有错误，需要手动检查")
    return False


def _parse_skeleton_errors(error_msg: str) -> dict:
    """解析骨架编译错误，返回 {文件名: [错误列表]}"""
    errors = {}
    
    if not error_msg:
        return errors
    
    # 按错误块分割
    current_file = None
    current_errors = []
    
    for line in error_msg.split('\n'):
        # 查找文件路径
        if '-->' in line and '.rs:' in line:
            parts = line.split('-->')
            if len(parts) > 1:
                file_part = parts[1].strip().split(':')[0]
                file_name = file_part.split('/')[-1]
                if file_name.endswith('.rs'):
                    if current_file and current_errors:
                        if current_file not in errors:
                            errors[current_file] = []
                        errors[current_file].extend(current_errors)
                    current_file = file_name
                    current_errors = []
        
        if current_file and (line.strip().startswith('error') or line.strip().startswith('  ')):
            current_errors.append(line)
    
    # 处理最后一个文件
    if current_file and current_errors:
        if current_file not in errors:
            errors[current_file] = []
        errors[current_file].extend(current_errors)
    
    return errors


def translate_skeleton(max_workers=48, use_layered=None):
    """
    翻译代码骨架，支持灵活的项目结构和并行处理
    
    Args:
        max_workers: 最大并行工作线程数，默认48
        use_layered: 是否使用分层构建模式（None=自动检测, True=强制使用, False=使用传统模式）
    """
    truth_mode = os.environ.get("C2R_TRUTH_MODE", "0").strip().lower() in ("1", "true", "yes", "y", "on")

    # 检查是否使用分层构建模式
    if use_layered is None:
        # 从环境变量读取配置
        use_layered = os.environ.get("USE_LAYERED_SKELETON", "false").lower() == "true"

    # Truth-mode: force layered skeleton to avoid legacy prompt-based skeleton generation.
    if truth_mode and not use_layered:
        print("Truth mode: 强制启用分层骨架构建（USE_LAYERED_SKELETON=true），跳过传统 LLM 骨架提示词路径")
        use_layered = True
    
    if use_layered:
        print("使用分层骨架构建模式 (Layered Skeleton Building)")
        use_bindgen = os.environ.get("USE_BINDGEN", "true").lower() == "true"
        use_llm_signatures = os.environ.get("USE_LLM_SIGNATURES", "false").lower() == "true"  # 默认 False，使用 TypeMapper
        use_type_mapper = os.environ.get("USE_TYPE_MAPPER", "true").lower() == "true"  # 默认 True
        use_llm_type_mapper = os.environ.get("USE_LLM_TYPE_MAPPER", "false").lower() == "true"  # LLMTypeMapper
        
        if use_llm_type_mapper:
            print("  ✓ 启用 LLMTypeMapper（TypeMapper + LLM 验证）")
        
        return translate_skeleton_layered(
            use_bindgen=use_bindgen,
            use_llm_for_signatures=use_llm_signatures,
            use_type_mapper=use_type_mapper,
            use_llm_type_mapper=use_llm_type_mapper
        )
    
    # 传统模式
    project_path = PROJECT_NAME
    project_root = PROJECT_ROOT

    # 使用新的工作空间路径
    skeleton_dir = get_skeleton_path(project_path) / "src"
    source_skeleton_dir = get_source_skeleton_path(project_path) / "src"
    
    # 确保目录存在
    ensure_dirs(project_path)
    skeleton_dir.mkdir(parents=True, exist_ok=True)
    source_skeleton_dir.mkdir(parents=True, exist_ok=True)

    # 获取所有源文件和对应的头文件
    source_header_pairs = get_all_source_files_with_headers(project_root)
    
    # 获取所有源文件的基础名称（不含扩展名），用于匹配 include
    all_source_stems = {Path(f).stem for f, _ in get_all_source_files_with_headers(project_root)}
    
    # 读取系统提示词（所有线程共享）
    system_promopt_path = "generate/system_gen_translate.txt"
    system_prompot = read_file(system_promopt_path)
    
    print(f"\n{'='*60}")
    print(f"骨架翻译开始")
    print(f"{'='*60}")
    print(f"项目: {project_path}")
    print(f"源文件数: {len(source_header_pairs)}")
    print(f"并行工作线程: {max_workers}")
    print(f"骨架输出目录: {skeleton_dir}")
    print(f"{'='*60}\n")
    
    # 准备任务参数
    tasks = [
        (source_file, header_file, project_root, skeleton_dir, source_skeleton_dir, all_source_stems, system_prompot)
        for source_file, header_file in source_header_pairs
    ]
    
    # 统计信息
    success_count = 0
    skipped_count = 0
    error_count = 0
    start_time = time.time()
    
    # 使用线程池并行处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_task = {executor.submit(process_single_file, task): task for task in tasks}
        
        # 处理完成的任务
        for future in as_completed(future_to_task):
            try:
                result = future.result()
                if result["status"] == "success":
                    success_count += 1
                elif result["status"] == "skipped":
                    skipped_count += 1
                elif result["status"] == "error":
                    error_count += 1
                
                # 显示进度
                completed = success_count + skipped_count + error_count
                if completed % 10 == 0 or completed == len(tasks):
                    elapsed = time.time() - start_time
                    remaining = len(tasks) - completed
                    if completed > 0:
                        avg_time = elapsed / completed
                        eta = avg_time * remaining
                        with print_lock:
                            print(f"\n[进度] {completed}/{len(tasks)} ({completed*100//len(tasks)}%) | "
                                  f"成功: {success_count} | 跳过: {skipped_count} | 错误: {error_count} | "
                                  f"预计剩余: {int(eta)}秒\n")
            except Exception as e:
                error_count += 1
                with print_lock:
                    print(f"✗ 任务执行异常: {str(e)}")
    
    # 最终统计
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"翻译完成！")
    print(f"总文件数: {len(tasks)}")
    print(f"成功: {success_count}")
    print(f"跳过: {skipped_count}")
    print(f"错误: {error_count}")
    print(f"总耗时: {int(elapsed)}秒 ({elapsed/60:.1f}分钟)")
    print(f"平均速度: {len(tasks)/elapsed:.2f} 文件/秒")
    print("=" * 60)


    # get_dependencies_match()


def create_main():
    project_path = PROJECT_NAME
    target_project_path = get_skeleton_path(project_path)

    main_file_name = "main.rs"
    output_path = target_project_path / "src" / main_file_name
    if output_path.exists():
        print(f"{output_path} already exists, skip")
        return

    mod_statements = []
    src_files_path = target_project_path / "src"
    src_files = list(src_files_path.glob("*.rs"))
    for src_file in src_files:
        current_file_name = src_file.stem  # 不含扩展名
        mod_statements.append(f"pub mod {current_file_name};")
    
    body_statements = "fn main() {\n\n\n}"

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(mod_statements))
        f.write("\n\n")
        f.write(body_statements)

def add_dependencies():
    project_path = PROJECT_NAME
    target_project_path = get_skeleton_path(project_path)
    target_project_files = list((target_project_path / "src").glob("*.rs"))

    total_dependencies = set()

    for target_project_file in target_project_files:
        file_content = read_file(str(target_project_file))


        dependencies = set([x.split(" ")[1].split(';')[0].split("::")[0] for x in extract_import(bytes(file_content, "utf-8"))])

        rust_content = file_content.splitlines()
        for line in rust_content:
            if "extern crate" in line:
                dependencies.add(line.split("extern crate")[1].split(";")[0].strip())

        dependencies = [f"{x} = \"*\"" for x in dependencies if x != "std" and x != "crate" and x != "super"]

        total_dependencies.update(dependencies)

    # with open(os.path.join(target_project_path, "Cargo.toml"), "r", encoding="utf-8") as file:
    #     content = file.read()

    content = f"""
[package]
name = "rust_test"
version = "0.1.0"
edition = "2021"


[dependencies]
"""
    content += "\n".join(total_dependencies)

    with open(target_project_path / "Cargo.toml", "w", encoding="utf-8") as file:
        file.write(content)

def copy_tree_replace(src, dst):
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

def repair():
    project_path = PROJECT_NAME
    target_project_path = get_skeleton_path(project_path)
    previous_project_path = str(target_project_path)
    copy_tree_replace(previous_project_path, str(target_project_path) + "_before_repair")
    previous_project_path = str(target_project_path) + "_before_repair"
    
    print(f"\n{'='*60}")
    print(f"骨架编译修复开始")
    print(f"{'='*60}")
    print(f"项目: {project_path}")
    print(f"骨架目录: {target_project_path}")
    print(f"最多修复轮数: 3")
    print(f"{'='*60}\n")
    
    for i in range(3):
        import time
        round_start = time.time()
        print(f"\n{'='*60}")
        print(f"修复轮次 {i+1}/3")
        print(f"{'='*60}")
        print(f"正在编译项目: {previous_project_path}")
        
        output, error, key = run_tests(previous_project_path, ["cargo", "build"])
        if key:
            round_time = time.time() - round_start
            print(f"✓ 编译成功！(耗时: {round_time:.1f}秒)")
        else:
            print(f"✗ 编译失败，开始修复...")
            # 调试：显示错误信息的前500个字符
            if error:
                print(f"错误信息（完整）:\n{error}")
            else:
                print(f"警告: 错误信息为空，使用输出信息")
                error = output if output else ""
        
        error_files_to_messages = {}
        if not key:
            # 修复结果目录就是完整的项目目录（包含 src/ 和 Cargo.toml）
            repair_result_dir = get_repair_results_path(project_path, "default", i+1)
            
            if not os.path.exists(repair_result_dir):
                repair_result_dir.mkdir(parents=True, exist_ok=True)
            else:
                # 检查是否已有修复结果（检查 src 目录是否存在且非空）
                if (repair_result_dir / "src").exists() and len(list((repair_result_dir / "src").glob("*.rs"))) > 0:
                    print(f"already exist {repair_result_dir}")
                    previous_project_path = str(repair_result_dir)
                    continue

            # 复制整个项目到修复结果目录
            copy_tree_replace(previous_project_path, str(repair_result_dir))
            
            # 尝试从 error 或 output 中提取错误信息
            error_text = error if error else output
            if not error_text:
                print(f"警告: 无法获取错误信息，跳过修复")
                continue
            
            # 查找错误信息的起始位置
            start1 = error_text.find("error:")
            start2 = error_text.find("error[E")
            if start1 == -1 and start2 == -1:
                # 如果没有找到标准错误格式，尝试查找其他错误标识
                start1 = error_text.find("Compiling")
                start2 = error_text.find("error")
                if start1 != -1 or start2 != -1:
                    start = min(start1, start2) if start1 != -1 and start2 != -1 else (start1 if start1 != -1 else start2)
                else:
                    print(f"警告: 无法找到错误信息起始位置，使用全部错误信息")
                    start = 0
            else:
                start = start1 if (start1 != -1 and (start2 == -1 or start1 < start2)) else start2
            
            error_text = error_text[start:]

            # 首先检查是否是 Cargo.toml 依赖错误（格式：error: no matching package named 'xxx' found）
            if "no matching package named" in error_text or "required by package" in error_text:
                # 这是 Cargo.toml 依赖错误
                if "Cargo.toml" not in error_files_to_messages:
                    error_files_to_messages["Cargo.toml"] = []
                # 提取错误信息（从 "error:" 开始到下一个空行或下一个 "error:" 之前）
                error_start = error_text.find("error:")
                if error_start != -1:
                    error_end = error_text.find("\n\n", error_start)
                    if error_end == -1:
                        error_end = error_text.find("\nerror:", error_start + 1)
                    if error_end == -1:
                        error_end = len(error_text)
                    cargo_error = error_text[error_start:error_end].strip()
                    error_files_to_messages["Cargo.toml"].append(cargo_error)
            
            # 解析错误信息：按行分割，查找包含文件路径的错误
            error_lines = error_text.splitlines()
            current_file = None
            current_error = []
            
            for idx, line in enumerate(error_lines):
                # 查找文件路径行（通常格式：  --> src/file.rs:line:col 或 src/file.rs:line:col 或 --> Cargo.toml:line:col）
                if (">" in line or ":" in line) and (".rs:" in line or "Cargo.toml:" in line):
                    # 尝试提取文件名
                    file_name = None
                    # 格式1: --> src/file.rs:line:col 或 --> Cargo.toml:line:col
                    if "-->" in line:
                        parts = line.split("-->")
                        if len(parts) > 1:
                            file_part = parts[1].strip().split(":")[0]
                            file_name = file_part.split("/")[-1]
                    # 格式2: src/file.rs:line:col
                    elif ".rs:" in line:
                        file_part = line.split(".rs:")[0]
                        file_name = file_part.split("/")[-1] + ".rs"
                    # 格式3: Cargo.toml:line:col
                    elif "Cargo.toml:" in line:
                        file_name = "Cargo.toml"
                    
                    if file_name and (file_name.endswith(".rs") or file_name == "Cargo.toml"):
                        # 保存之前的错误
                        if current_file and current_error:
                            if current_file not in error_files_to_messages:
                                error_files_to_messages[current_file] = []
                            error_files_to_messages[current_file].append("\n".join(current_error))
                        current_file = file_name
                        current_error = [line]
                        continue
                
                # 查找 error[E 或 error: 开头的错误行
                if line.strip().startswith("error[E") or line.strip().startswith("error:"):
                    if current_error:
                        current_error.append(line)
                    elif idx > 0:
                        # 尝试从上一行提取文件名
                        prev_line = error_lines[idx - 1]
                        if (">" in prev_line or ":" in prev_line) and (".rs:" in prev_line or "Cargo.toml:" in prev_line):
                            file_name = None
                            if "-->" in prev_line:
                                parts = prev_line.split("-->")
                                if len(parts) > 1:
                                    file_part = parts[1].strip().split(":")[0]
                                    file_name = file_part.split("/")[-1]
                            elif ".rs:" in prev_line:
                                file_part = prev_line.split(".rs:")[0]
                                file_name = file_part.split("/")[-1] + ".rs"
                            elif "Cargo.toml:" in prev_line:
                                file_name = "Cargo.toml"
                            
                            if file_name and (file_name.endswith(".rs") or file_name == "Cargo.toml"):
                                current_file = file_name
                                current_error = [prev_line, line]
                    else:
                        # 如果没有上下文，创建一个通用错误
                        if current_error:
                            current_error.append(line)
                        else:
                            current_error = [line]
                elif current_error and (line.strip().startswith("  ") or line.strip().startswith("help:") or line.strip().startswith("note:")):
                    # 错误信息的续行
                    current_error.append(line)
            
            # 处理最后一个文件
            if current_file and current_error:
                if current_file not in error_files_to_messages:
                    error_files_to_messages[current_file] = []
                error_files_to_messages[current_file].append("\n".join(current_error))
            
            # 如果上面的方法没有找到文件，尝试按错误块分割
            if not error_files_to_messages:
                # 尝试按空行分割错误块
                error_blocks = error_text.split("\n\n")
                for block in error_blocks:
                    if "error[E" in block or "error:" in block:
                        block_lines = block.splitlines()
                        for line in block_lines:
                            if (">" in line or ":" in line) and (".rs:" in line or "Cargo.toml:" in line):
                                file_name = None
                                if "-->" in line:
                                    parts = line.split("-->")
                                    if len(parts) > 1:
                                        file_part = parts[1].strip().split(":")[0]
                                        file_name = file_part.split("/")[-1]
                                elif ".rs:" in line:
                                    file_part = line.split(".rs:")[0]
                                    file_name = file_part.split("/")[-1] + ".rs"
                                elif "Cargo.toml:" in line:
                                    file_name = "Cargo.toml"
                                
                                if file_name and (file_name.endswith(".rs") or file_name == "Cargo.toml"):
                                    if file_name not in error_files_to_messages:
                                        error_files_to_messages[file_name] = []
                                    error_files_to_messages[file_name].append(block)
                                    break
            
            print(f"发现 {len(error_files_to_messages)} 个文件需要修复:")
            for error_file in error_files_to_messages.keys():
                error_count = len(error_files_to_messages[error_file])
                print(f"  - {error_file} ({error_count} 个错误)")
            print()
            
            for error_file, error_messages in error_files_to_messages.items():
                # 从修复结果目录读取文件（如果存在），否则从骨架目录读取
                # Cargo.toml 在项目根目录，不在 src 目录
                if error_file == "Cargo.toml":
                    error_file_path = repair_result_dir / error_file
                    if not error_file_path.exists():
                        error_file_path = target_project_path / error_file
                else:
                    error_file_path = repair_result_dir / "src" / error_file
                    if not error_file_path.exists():
                        error_file_path = target_project_path / "src" / error_file
                if not error_file_path.exists():
                    print(f"警告: 文件不存在: {error_file_path}，跳过")
                    continue
                with open(error_file_path, 'r', encoding='utf-8') as f:
                    previous_response = f.read()
                
                # 读取对应的 C++ 源文件（仅对 .rs 文件）
                corpus_func = ""
                if error_file.endswith(".rs"):
                    source_skeleton_src = get_source_skeleton_path(PROJECT_NAME) / "src"
                    if source_skeleton_src.exists():
                        corpus_func_path = source_skeleton_src / error_file.replace(".rs", ".txt")
                        if corpus_func_path.exists():
                            with open(corpus_func_path, 'r', encoding='utf-8') as f:
                                corpus_func = f.read()
                
                combined_error_messages = "\n\n".join(error_messages)
                corpus_lang = "C++" if error_file.endswith(".rs") else "TOML"
                query_lang = "Rust" if error_file.endswith(".rs") else "TOML"

                system_prompt = f"""
Hello. You are a talented Rust programmer. Here you're going to help the user fix the translated Rust file from C++ with error.

You have known enough for understanding and using the Rust compilation rule and syntax. Please follow the user's instructions and requirements to fix the Rust code provided by the user

"""
                if error_file == "Cargo.toml":
                    user_prompt = f"""
# Cargo.toml Repair
## Task Description
Fix the syntax error in the Cargo.toml file.

## Basic Details about the code
<previous response>\n{previous_response}\n</previous response>\n<error message>\n{combined_error_messages}\n</error message>\n

## Note
The Repair process must adhere to the following rules:
    - **ONLY fix the error code, repeat completely the rest of the code.**
    - Fix invalid dependency syntax (e.g., "some_napi crate = "*"" should be "some_napi = "*"")
    - **CRITICAL**: If the error says "no matching package named 'XXX' found", you MUST DELETE that entire dependency line, as the package does not exist in crates.io
    - If there are empty keys (like " = "*""), delete that line completely
    - Keep all other valid dependencies unchanged

## Output Format
```toml
** (only reply with the fixed Cargo.toml content) **
```

"""
                else:
                    user_prompt = f"""
# Code Repair
## Task Description
you were asked to translate the given {corpus_lang} function to {query_lang}. Some errors occurred when executing your code. Fix the error in the Rust code from previous response.

## Basic Details about the code
Here are the basic details about the code under repair
<previous response>\n{previous_response}\n</previous response>\n<error message>\n{combined_error_messages}\n</error message>\n<{corpus_lang} source code>\n{corpus_func}\n</{corpus_lang} source code>\n

## Note
The Repair process must adhere to the following rules:
    - **ONLY fix the error code, repeat completely the rest of the code.**

## Output Format
```rust
** (only reply with the translated result of the focal code) **
```

"""
                messages = [
                    {
                        "role" : "system",
                        "content" : system_prompt
                    },
                    {
                        "role" : "user",
                        "content" : user_prompt
                    }
                ]
                # 保存提示词
                try:
                    from save_llm_prompts import save_llm_prompt, save_llm_prompt_text
                    # os 已在文件顶部导入，不需要重复导入
                    llm_name = os.environ.get("LLM_NAME", "default")
                    metadata = {
                        "file_name": error_file,
                        "round": i+1,
                        "error_type": corpus_lang,
                        "error_count": len(error_messages) if error_messages else 0
                    }
                    save_llm_prompt(
                        messages=messages,
                        project_name=project_path,
                        llm_name=llm_name,
                        task_type="skeleton_repair",
                        function_name=f"{error_file}_round_{i+1}",
                        metadata=metadata
                    )
                    save_llm_prompt_text(
                        messages=messages,
                        project_name=project_path,
                        llm_name=llm_name,
                        task_type="skeleton_repair",
                        function_name=f"{error_file}_round_{i+1}",
                        metadata=metadata
                    )
                except Exception as e:
                    import logging
                    logging.warning(f"保存提示词失败: {e}")
                
                # print(user_prompt)
                # sys.exit()
                response = generation(messages)
                # 根据文件类型提取代码
                file_type = "toml" if error_file == "Cargo.toml" else "rust"
                repair_result = extract_code(response, file_type=file_type)
                # 写入修复结果目录（Cargo.toml 在根目录，.rs 文件在 src 目录）
                if error_file == "Cargo.toml":
                    repair_result_file = repair_result_dir / error_file
                else:
                    repair_result_file = repair_result_dir / "src" / error_file
                repair_result_file.parent.mkdir(parents=True, exist_ok=True)
                with open(repair_result_file, 'w', encoding='utf-8') as f:
                    f.write(repair_result)

                print(f"  ✓ 已修复: {error_file}")
                
                # 下一轮使用修复结果目录
                previous_project_path = str(repair_result_dir)
            
            round_time = time.time() - round_start
            print(f"\n轮次 {i+1} 修复完成，共修复 {len(error_files_to_messages)} 个文件 (耗时: {round_time:.1f}秒)")
            # 统计输出
            if len(error_files_to_messages) > 0:
                print(f"  - 修复的文件列表: {', '.join(error_files_to_messages.keys())}")
        else:
            print(f"\n{'='*60}")
            print(f"✓ 编译成功！正在复制项目文件...")
            copy_tree_replace(previous_project_path, str(target_project_path))
            print(f"✓ 修复完成！")
            print(f"{'='*60}\n")
            return True
    
    # 如果3次修复都失败，将最后一次修复结果复制到目标项目路径
    print(f"\n{'='*60}")
    print(f"⚠ 3次修复后仍有错误，将最后一次修复结果复制到目标路径...")
    final_repair_path = get_repair_results_path(project_path, "default", 3)
    if os.path.exists(final_repair_path) and (final_repair_path / "src").exists():
        copy_tree_replace(str(final_repair_path), str(target_project_path))
        print(f"✓ 已复制最后一次修复结果到: {target_project_path}")
    else:
        print(f"✗ 警告: 最后一次修复结果不存在: {final_repair_path}")
    print(f"{'='*60}\n")
    return False

if __name__ == "__main__":
    import time
    import argparse
    
    # ========== 日志配置：静默 httpx/openai 的冗余日志 ==========
    from log_config import setup_logging, silence_http_logs
    setup_logging(level=logging.INFO, quiet_mode=True, simple_format=True)
    silence_http_logs()  # 额外确保 HTTP 日志被静默
    
    # 解析命令行参数
    # 注意：使用 arg_parser 而不是 parser，避免覆盖全局的 tree-sitter parser
    arg_parser = argparse.ArgumentParser(description="C++ to Rust 骨架翻译")
    arg_parser.add_argument("--layered", action="store_true", 
                        help="使用分层骨架构建模式 (bindgen + tree-sitter)")
    arg_parser.add_argument("--no-bindgen", action="store_true",
                        help="禁用 bindgen 类型生成")
    arg_parser.add_argument("--no-llm-signatures", action="store_true",
                        help="禁用 LLM 签名翻译（使用 TypeMapper 规则引擎）")
    arg_parser.add_argument("--no-type-mapper", action="store_true",
                        help="禁用 TypeMapper（使用 LLM 或简单规则）")
    arg_parser.add_argument("--use-libclang", action="store_true",
                        help="使用 libclang 替代 tree-sitter 提取函数（更准确，能处理复杂宏和属性）")
    arg_parser.add_argument("--max-workers", type=int, default=48,
                        help="最大并行线程数（传统模式）")
    args = arg_parser.parse_args()
    
    start_time = time.time()
    
    truth_mode = os.environ.get("C2R_TRUTH_MODE", "0").strip().lower() in ("1", "true", "yes", "y", "on")

    # 检查是否使用分层模式
    use_layered = args.layered or os.environ.get("USE_LAYERED_SKELETON", "false").lower() == "true"
    if truth_mode and not use_layered:
        print("Truth mode: 强制启用分层骨架构建（USE_LAYERED_SKELETON=true），跳过传统 LLM 骨架提示词路径")
        use_layered = True
    
    if use_layered:
        print("\n" + "="*60)
        print("分层骨架翻译流程开始 (Layered Mode)")
        print("="*60 + "\n")
        
        print("步骤 1/2: 获取依赖...")
        step_start = time.time()
        get_dependencies()
        step_time = time.time() - step_start
        print(f"✓ 依赖获取完成 (耗时: {step_time:.1f}秒)\n")
        
        print("步骤 2/2: 分层骨架构建...")
        step_start = time.time()
        
        # 设置参数
        use_bindgen = not args.no_bindgen
        use_llm_signatures = not args.no_llm_signatures and not args.no_type_mapper  # 如果禁用 TypeMapper，可以使用 LLM
        use_type_mapper = not args.no_type_mapper  # 默认启用 TypeMapper
        use_llm_type_mapper = os.environ.get("USE_LLM_TYPE_MAPPER", "false").lower() == "true"  # LLMTypeMapper
        use_libclang = args.use_libclang or os.environ.get("USE_LIBCLANG", "false").lower() == "true"
        
        if use_llm_type_mapper:
            print("  ✓ 启用 LLMTypeMapper（TypeMapper + LLM 验证）")
        
        translate_skeleton_layered(
            use_bindgen=use_bindgen,
            use_llm_for_signatures=use_llm_signatures,
            use_type_mapper=use_type_mapper,
            use_llm_type_mapper=use_llm_type_mapper,
            use_libclang=use_libclang
        )
        step_time = time.time() - step_start
        print(f"✓ 分层骨架构建完成 (耗时: {step_time:.1f}秒)\n")
        
    else:
        # 传统模式
        print("\n" + "="*60)
        print("骨架翻译与修复流程开始 (Traditional Mode)")
        print("="*60 + "\n")
        
        print("步骤 1/5: 获取依赖...")
        step_start = time.time()
        get_dependencies()
        step_time = time.time() - step_start
        print(f"✓ 依赖获取完成 (耗时: {step_time:.1f}秒)\n")
        
        print("步骤 2/5: 翻译代码骨架...")
        step_start = time.time()
        # 支持通过环境变量配置并行线程数
        max_workers = args.max_workers or int(os.environ.get("TRANSLATE_MAX_WORKERS", "48"))
        translate_skeleton(max_workers=max_workers, use_layered=False)
        step_time = time.time() - step_start
        print(f"✓ 骨架翻译完成 (耗时: {step_time:.1f}秒)\n")
        
        print("步骤 3/5: 创建主文件...")
        step_start = time.time()
        create_main()
        step_time = time.time() - step_start
        print(f"✓ 主文件创建完成 (耗时: {step_time:.1f}秒)\n")
        
        print("步骤 4/5: 添加依赖...")
        step_start = time.time()
        add_dependencies()
        step_time = time.time() - step_start
        print(f"✓ 依赖添加完成 (耗时: {step_time:.1f}秒)\n")
        
        print("步骤 5/5: 自动修复...")
        step_start = time.time()
        result = repair()
        step_time = time.time() - step_start
        if not result:
            print(f"⚠ 需要手动修复 (耗时: {step_time:.1f}秒)\n")
        else:
            print(f"✓ 修复完成 (耗时: {step_time:.1f}秒)\n")
    
    total_time = time.time() - start_time
    print("="*60)
    print(f"骨架翻译流程完成！总耗时: {total_time:.1f}秒 ({total_time/60:.1f}分钟)")
    print("="*60 + "\n")
