import os
import sys
import json
import argparse
import logging
from collections import defaultdict
from typing import Dict, Any

from tree_sitter import Language, Parser
import tree_sitter_cpp as tscpp
from project_config import PROJECT_NAME, PROJECT_ROOT, get_all_source_files_with_headers
from workspace_config import (
    get_functions_path, get_dependencies_path, get_dependencies_not_in_file_path,
    ensure_dirs, safe_module_name, get_compile_commands_path
)
from pathlib import Path
from call_graph import generate_function_uid

# 导入日志配置
from log_config import ensure_logging_setup, LogPrinter
ensure_logging_setup()
logger = logging.getLogger(__name__)
log = LogPrinter(__name__)

# 导入预处理相关模块
from compile_commands_parser import (
    CompileCommandsParser,
    ContextSelectionStrategy,
    PreprocessingContext
)

# ========== 预处理配置 ==========
# 是否启用预处理上下文选择（默认启用）
USE_PREPROCESSING = os.environ.get("USE_PREPROCESSING", "true").lower() == "true"
# 预处理策略（active/best/union，默认 active）
PREPROCESSING_STRATEGY = os.environ.get("PREPROCESSING_STRATEGY", "active")
# 目标配置（用于 ACTIVE 策略，默认 rk3568）
TARGET_CONFIG = os.environ.get("TARGET_CONFIG", "rk3568")
# 目标 out_dir（优先级高于 TARGET_CONFIG；例如 out/rk3568）
TARGET_OUT_DIR = os.environ.get("TARGET_OUT_DIR", "").strip()

# compile_commands.json 路径（自动查找或从环境变量获取）
def _find_compile_commands_json():
    """自动查找 compile_commands.json 文件

    查找优先级：
    1. 环境变量 COMPILE_COMMANDS_PATH（兼容旧变量）
    2. 环境变量 OHOS_COMPILE_COMMANDS（与 workspace_config 保持一致）
    3. workspace_config.get_compile_commands_path()（包含硬编码默认路径：OpenHarmony-v5.0.1-Release/out/rk3568）
    4. 仓库内/相对脚本目录的常见路径
    5. 返回空字符串（禁用预处理；开源最小输入默认不依赖全量 OpenHarmony compile DB）
    """
    # 优先使用环境变量
    env_path = os.environ.get("COMPILE_COMMANDS_PATH", "").strip()
    if env_path and Path(env_path).exists():
        return env_path

    env_path = os.environ.get("OHOS_COMPILE_COMMANDS", "").strip()
    if env_path and Path(env_path).exists():
        return env_path

    # 使用 workspace_config 的确定性查找逻辑（带默认路径）
    try:
        cc = get_compile_commands_path()
        if cc and Path(cc).exists():
            print(f"[默认] 使用 workspace_config 找到 compile_commands.json: {cc}")
            return str(cc)
    except Exception:
        pass

    # 自动查找常见路径（仅限仓库/相对路径；避免隐式依赖宿主机外部目录）
    possible_paths = [
        # 相对于当前脚本目录的可能路径
        Path(__file__).resolve().parent.parent.parent / "out" / "rk3568" / "compile_commands.json",
        Path(__file__).resolve().parent.parent.parent / "openharmony" / "out" / "rk3568" / "compile_commands.json",
    ]

    for path in possible_paths:
        if path.exists() and path.is_file():
            print(f"[自动发现] 找到 compile_commands.json: {path}")
            return str(path)

    # 未找到，返回空字符串
    return ""

COMPILE_COMMANDS_PATH = _find_compile_commands_json()
# 预处理输出目录
PREPROCESS_OUTPUT_DIR = os.environ.get("PREPROCESS_OUTPUT_DIR", "./translation_outputs/.preprocessed")

# tree-sitter 初始化（兼容新版 bindings：Language(..., name)）
try:
    CPP_LANGUAGE = Language(tscpp.language(), "cpp")
except TypeError:
    CPP_LANGUAGE = Language(tscpp.language())
parser = Parser()
try:
    parser.set_language(CPP_LANGUAGE)
except Exception:
    parser = Parser(CPP_LANGUAGE)

# 是否使用 libclang 模式
USE_LIBCLANG = False



def read_file(path):
    with open(path, 'r', encoding='utf-8', errors='ignore') as input_file:
        content = input_file.read() 

    return content


def traverse_target_node(node, source_code, target_node, target_statements):
    if node.type == target_node:
        # call_functions.append(node)
        target_statement = source_code[node.start_byte:node.end_byte].decode('utf-8').strip()
        # 同时返回节点对象和文本（用于获取行号）
        target_statements.append((target_statement, node))
        return

    for child in node.children:
        traverse_target_node(child, source_code, target_node, target_statements)


def get_function(source_code):
    """
    提取函数定义，返回包含元数据的字典

    Returns:
        dict: {function_name: {'code': str, 'start_line': int, 'end_line': int}}
    """
    import re
    source_code_bytes = bytes(source_code, "utf-8")
    tree = parser.parse(source_code_bytes)
    target_statements = []
    traverse_target_node(tree.root_node, source_code_bytes, "function_definition", target_statements)

    function_name_to_info = {}
    for function_statements, func_node in target_statements:
        function_name = []
        bytes_source_code = bytes(function_statements, "utf-8")
        tree = parser.parse(bytes_source_code)
        traverse_target_node(tree.root_node, bytes_source_code, "function_declarator", function_name)
        
        # 检查是否成功提取到函数名
        # 需要确保列表不为空且第一个元素不是空字符串
        # 先检查列表是否为空，避免 IndexError
        # 注意：function_name 现在是 (text, node) 元组的列表
        has_valid_function_name = False
        func_name_text = None
        if function_name and len(function_name) > 0:
            func_name_text, _ = function_name[0]  # 提取元组的第一个元素（文本）
            if func_name_text and func_name_text.strip():
                has_valid_function_name = True

        # 获取行号（从原始节点，不是重新解析的节点）
        start_line = func_node.start_point[0] + 1  # tree-sitter 行号从0开始，转换为1开始
        end_line = func_node.end_point[0] + 1
        
        if not has_valid_function_name:
            # 如果无法提取函数名，尝试从函数定义中直接提取
            # 使用更智能的正则表达式匹配
            try:
                # 移除注释和预处理指令
                clean_code = re.sub(r'//.*?$|/\*.*?\*/', '', function_statements, flags=re.MULTILINE | re.DOTALL)
                clean_code = re.sub(r'^\s*#.*?$', '', clean_code, flags=re.MULTILINE)
                
                # 尝试多种函数定义模式
                # 先尝试更精确的模式，再尝试通用模式
                patterns = [
                    # 析构函数: ~ClassName()
                    (r'~\s*(\w+)\s*\([^)]*\)\s*{', 'destructor'),
                    # 运算符重载: operator+(参数)
                    (r'operator\s*([^\s(]+)\s*\([^)]*\)\s*{', 'operator'),
                    # 类成员函数: ClassName::functionName(参数)
                    (r'(\w+(?:::\w+)*)::(\w+)\s*\([^)]*\)\s*{', 'member'),
                    # 普通函数: functionName(参数) {
                    # 匹配函数名，但排除关键字
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
                            # 类::函数
                            func_name = f"{match.group(1)}::{match.group(2)}"
                        elif pattern_type == 'function':
                            # 普通函数
                            candidate = match.group(1)
                            # 排除关键字
                            if candidate not in excluded_keywords:
                                func_name = candidate
                        
                        if func_name and func_name not in excluded_keywords:
                            break
                        func_name = None
                
                if func_name:
                    # 如果函数名已存在，添加后缀
                    original_func_name = func_name
                    counter = 1
                    while func_name in function_name_to_info:
                        func_name = f"{original_func_name}_{counter}"
                        counter += 1
                    function_name_to_info[func_name] = {
                        'code': function_statements,
                        'start_line': start_line,
                        'end_line': end_line
                    }
                    continue
            except Exception as e:
                # 如果正则提取也失败，继续到匿名函数处理
                pass
            
            # 如果仍然无法提取，使用索引作为函数名（带文件哈希以避免冲突）
            index = len(function_name_to_info) + 1
            # 使用函数体的前几个字符作为标识，避免完全相同的匿名函数被覆盖
            func_hash = hash(function_statements[:100]) % 10000
            function_name_to_info[f"anonymous_function_{index}_{func_hash}"] = {
                'code': function_statements,
                'start_line': start_line,
                'end_line': end_line
            }
            continue

        # 正常情况：使用提取到的函数名
        # func_name_text 已经是字符串（从元组中提取的第一个元素）
        func_name = func_name_text.strip()

        # 处理函数名可能包含参数的情况
        # 从函数声明器中提取函数名（可能包含命名空间、类名等）
        # 例如: "MyClass::foo" 或 "foo(int, int)"
        func_name = func_name.split('(')[0].strip()

        # 如果函数名包含命名空间分隔符，只保留最后一部分
        # 例如: "MyClass::foo" -> "foo"，但保留完整的用于区分
        if '::' in func_name:
            # 保留完整的命名空间路径，但同时也保存短名称
            short_name = func_name.split('::')[-1]
            # 如果短名称不为空，使用短名称作为主键
            if short_name:
                func_name = func_name  # 保留完整路径用于区分

        # 清理函数名（移除可能的模板参数）
        if '<' in func_name:
            func_name = func_name.split('<')[0].strip()

        # 如果函数名已存在，添加后缀
        original_func_name = func_name
        counter = 1
        while func_name in function_name_to_info:
            func_name = f"{original_func_name}_{counter}"
            counter += 1

        function_name_to_info[func_name] = {
            'code': function_statements,
            'start_line': start_line,
            'end_line': end_line
        }

    return function_name_to_info

def get_call_function(source_code):
    source_code = bytes(source_code, "utf-8")
    tree = parser.parse(source_code)
    target_statements = []
    traverse_target_node(tree.root_node, source_code, "call_expression", target_statements)

    # 只返回文本部分（忽略节点）
    return [text for text, _ in target_statements]

def get_total_function():
    """
    递归查找项目中的所有函数，不依赖特定的目录结构
    """
    project_root = PROJECT_ROOT
    
    # 获取所有源文件和对应的头文件
    source_header_pairs = get_all_source_files_with_headers(project_root)

    # ---------------------------------------------------------------------
    # Route A (truth-first): Source set is defined by compile_commands.json
    #
    # Many SelfContained modules contain multiple build-variant source files
    # (e.g. *_virtual.c, *_liteos.c, *_linux.c) but a *single* selected OHOS
    # profile only compiles a subset. In this mode, we only consider sources
    # that have an exact compile_commands entry in the selected profile.
    #
    # NOTE: This is a "file selection" gate, NOT a heuristic proxy-TU closure.
    # ---------------------------------------------------------------------
    source_set_from_cc = os.environ.get("C2R_SOURCE_SET_FROM_COMPILE_COMMANDS", "1").strip().lower() not in ("0", "false", "no")
    if USE_PREPROCESSING and source_set_from_cc:
        try:
            from ohos_build_profile import select_profile_for_project, create_ohos_path_mapper

            # Prefer an explicitly pinned compile_commands.json (e.g. OSS suite: project-root/compile_commands.json).
            # If not explicitly set, fall back to OHOS profile auto-selection (compile_commands_all registry).
            selected = None
            compile_db_path = None
            ohos_root_for_mapping = None
            env_cc = (os.environ.get("COMPILE_COMMANDS_PATH") or "").strip() or (os.environ.get("OHOS_COMPILE_COMMANDS") or "").strip()
            if env_cc and Path(env_cc).exists():
                compile_db_path = Path(env_cc)
            else:
                selected = select_profile_for_project(
                    project_name=str(PROJECT_NAME),
                    project_root=Path(project_root),
                    require_registry=False,
                )
                compile_db_path = Path(selected.compile_commands_json) if selected else None
                ohos_root_for_mapping = Path(selected.ohos_root) if (selected and selected.ohos_root) else None
            map_to_ohos_path = None
            if ohos_root_for_mapping and ohos_root_for_mapping.exists():
                map_to_ohos_path = create_ohos_path_mapper(Path(project_root), ohos_root_for_mapping)

            if compile_db_path and compile_db_path.exists():
                inferred_ohos_root = ohos_root_for_mapping
                if not inferred_ohos_root:
                    try:
                        # Only infer OHOS root for "real" OHOS layouts (avoid mis-inference for OSS projects).
                        parts = set(compile_db_path.resolve().parts)
                        if "OpenHarmony" in parts and "out" in parts:
                            inferred_ohos_root = compile_db_path.parent.parent.parent.resolve()
                    except Exception:
                        inferred_ohos_root = None

                cc_parser = CompileCommandsParser(compile_db_path, ohos_root=inferred_ohos_root)

                built_pairs = []
                excluded = 0
                for src, hdr in source_header_pairs:
                    src_path = Path(src)
                    src_for_cc = src_path
                    if map_to_ohos_path:
                        try:
                            mapped = map_to_ohos_path(src_path)
                            if mapped and Path(mapped).exists():
                                src_for_cc = Path(mapped)
                        except Exception:
                            src_for_cc = src_path

                    entry, info = cc_parser.get_entry_for_file_with_reason(src_for_cc)
                    # Route A requires exact-path hit to avoid filename collisions.
                    if entry and (info.get("reason") == "filename_index_exact_path"):
                        built_pairs.append((src, hdr))
                    else:
                        excluded += 1

                if excluded > 0:
                    print(
                        f"[RouteA] 使用 compile_commands 定义源文件集合: built={len(built_pairs)}, excluded(not-built)={excluded}, total={len(source_header_pairs)}"
                    )
                source_header_pairs = built_pairs
        except Exception:
            # Best-effort: fall back to directory scan if profile/cc discovery fails.
            pass
    
    functions_undertranslated = {}
    
    print(f"正在扫描项目: {project_root}")
    print(f"找到 {len(source_header_pairs)} 个源文件")
    
    for source_file, header_file in source_header_pairs:
        source_path = Path(source_file)
        current_file_name = source_path.stem  # 不含扩展名的文件名
        
        # 精简输出：移除每个文件的详细处理信息
        # print(f"处理: {current_file_name} ({source_path.relative_to(project_root)})")
        
        # 读取源文件内容
        src_content = read_file(source_file)
        src_functions = get_function(src_content)
        functions_undertranslated.update(src_functions)
        
        # 如果有匹配的头文件，也读取头文件内容
        if header_file:
            try:
                header_content = read_file(header_file)
                header_functions = get_function(header_content)
                functions_undertranslated.update(header_functions)
                # 精简输出：移除头文件匹配信息
                # print(f"  -> 找到匹配的头文件: {Path(header_file).relative_to(project_root)}")
            except Exception as e:
                # 只保留错误信息
                print(f"  -> 警告: 读取头文件失败: {e}")
        else:
            # 精简输出：移除未找到头文件的信息
            # print(f"  -> 未找到匹配的头文件")
            pass
    
    print(f"总共提取了 {len(functions_undertranslated)} 个函数")
    return functions_undertranslated

def get_call_function_total(project_total_functions):
    """
    处理所有源文件，提取函数和依赖关系，不依赖特定的目录结构

    支持预处理上下文选择（Preprocess-First Approach）：
    - 如果启用预处理且提供了 compile_commands.json，将使用 clang -E 预处理源文件
    - 预处理后的 .i 文件包含宏展开后的完整代码，确保函数签名和调用图的一致性
    """
    project_path = PROJECT_NAME
    project_root = PROJECT_ROOT

    # 使用新的工作空间路径
    functions_dir = get_functions_path(project_path)
    dependencies_dir = get_dependencies_path(project_path)
    dependencies_not_in_file_dir = get_dependencies_not_in_file_path(project_path)

    # 确保目录存在
    ensure_dirs(project_path)

    # ========== 预处理初始化 ==========
    preprocess_parser = None
    preprocessing_enabled = False
    ohos_root_for_mapping = None
    map_to_ohos_path = None
    selected_profile_label = None
    selected_profile_out_dir = None
    # 记录每个源文件选择到的 TU 上下文（用于下游阶段复用同一套 flags/宏/include 顺序）
    tu_context_map: Dict[str, Any] = {
        "schema_version": 1,
        "project_name": str(project_path),
        "project_root": str(Path(project_root).resolve()),
        "preprocess_output_dir": str(Path(PREPROCESS_OUTPUT_DIR).resolve()),
        "use_preprocessing": bool(USE_PREPROCESSING),
        "preprocessing_enabled": False,
        "preprocessing_strategy": str(PREPROCESSING_STRATEGY),
        "target_config": str(TARGET_OUT_DIR or TARGET_CONFIG or ""),
        "selected_profile": None,
        "files": {},
    }

    # 优先：使用 OpenHarmony compile_commands_all registry 自动选择 profile（out_dir 主键）
    compile_db_path = None
    if USE_PREPROCESSING:
        try:
            from ohos_build_profile import select_profile_for_project, create_ohos_path_mapper

            # 0) Prefer an explicitly pinned compile_commands.json if provided.
            # This is required for OSS suite runs (each project has its own compile_commands.json),
            # and also allows users to override OHOS auto-selection deterministically.
            env_cc = (os.environ.get("COMPILE_COMMANDS_PATH") or "").strip() or (os.environ.get("OHOS_COMPILE_COMMANDS") or "").strip()
            if env_cc and Path(env_cc).exists():
                compile_db_path = Path(env_cc)
                # Record the "single truth" compile_commands used for this run.
                tu_context_map["selected_profile"] = {
                    "label": None,
                    "out_dir": None,
                    "compile_commands_json": str(compile_db_path),
                    "ohos_root": None,
                    "selection_reason": "env",
                }
            else:
                selected = select_profile_for_project(
                    project_name=str(project_path),
                    project_root=Path(project_root),
                    require_registry=False,
                )
                if selected:
                    compile_db_path = Path(selected.compile_commands_json)
                    ohos_root_for_mapping = Path(selected.ohos_root) if selected.ohos_root else None
                    selected_profile_label = selected.profile.label
                    selected_profile_out_dir = selected.profile.out_dir
                    if ohos_root_for_mapping and ohos_root_for_mapping.exists():
                        map_to_ohos_path = create_ohos_path_mapper(Path(project_root), ohos_root_for_mapping)
                    # 让下游（如 translate.py/skeleton_builder）也能复用本次选择结果
                    os.environ.setdefault("OHOS_COMPILE_COMMANDS", str(compile_db_path))
                    if ohos_root_for_mapping:
                        os.environ.setdefault("OHOS_ROOT", str(ohos_root_for_mapping))
                    tu_context_map["selected_profile"] = {
                        "label": selected_profile_label,
                        "out_dir": selected_profile_out_dir,
                        "compile_commands_json": str(compile_db_path),
                        "ohos_root": str(ohos_root_for_mapping) if ohos_root_for_mapping else None,
                    }
        except Exception as e:
            print(f"\n[预处理] ⚠ profile 自动选择失败，将回退到单一 compile_commands.json: {e}")

    # 回退：传统单一 compile_commands.json 路径
    if USE_PREPROCESSING and not compile_db_path and COMPILE_COMMANDS_PATH:
        compile_db_path = Path(COMPILE_COMMANDS_PATH)

    if USE_PREPROCESSING and compile_db_path:
        if compile_db_path.exists():
            print(f"\n[预处理] 启用预处理上下文选择")
            print(f"  - 策略: {PREPROCESSING_STRATEGY}")
            if PREPROCESSING_STRATEGY.lower() in ("active", "auto"):
                tc = TARGET_OUT_DIR or TARGET_CONFIG
                if selected_profile_out_dir:
                    tc = selected_profile_out_dir
                print(f"  - 目标配置 (active/auto): {tc}")
            if selected_profile_out_dir:
                print(f"  - profile(out_dir): {selected_profile_out_dir}")
            if selected_profile_label:
                print(f"  - profile(label): {selected_profile_label}")
            print(f"  - compile_commands.json: {compile_db_path}")
            print(f"  - 输出目录: {PREPROCESS_OUTPUT_DIR}")

            try:
                # 推断 OpenHarmony 根目录（仅用于路径规范化/映射）
                inferred_ohos_root = None
                if ohos_root_for_mapping and ohos_root_for_mapping.exists():
                    inferred_ohos_root = ohos_root_for_mapping
                else:
                    # 传统推断：.../OpenHarmony/out/<board>/compile_commands.json
                    try:
                        # Avoid mis-inference for OSS projects where compile_commands.json lives under project root.
                        parts = set(compile_db_path.resolve().parts)
                        if "OpenHarmony" in parts and "out" in parts:
                            inferred_ohos_root = compile_db_path.parent.parent.parent.resolve()
                    except Exception:
                        inferred_ohos_root = None
                if inferred_ohos_root:
                    print(f"  - OHOS 根目录: {inferred_ohos_root}")

                preprocess_parser = CompileCommandsParser(compile_db_path, ohos_root=inferred_ohos_root)
                preprocessing_enabled = True
                tu_context_map["preprocessing_enabled"] = True

                # 解析策略
                strategy = ContextSelectionStrategy.BEST
                if PREPROCESSING_STRATEGY.lower() in ("active", "auto"):
                    strategy = ContextSelectionStrategy.ACTIVE
                elif PREPROCESSING_STRATEGY.lower() == "union":
                    strategy = ContextSelectionStrategy.UNION

                print(f"  - ✓ 预处理解析器初始化成功")
            except Exception as e:
                print(f"  - ✗ 预处理解析器初始化失败: {e}")
                preprocessing_enabled = False
        else:
            print(f"\n[预处理] compile_commands.json 不存在: {compile_db_path}")
            print(f"  - 将使用原始源文件（不预处理）")
    else:
        print(f"\n[预处理] 未启用预处理（USE_PREPROCESSING={USE_PREPROCESSING}）")

    # 获取所有源文件和对应的头文件
    source_header_pairs = get_all_source_files_with_headers(project_root)

    # 精简输出：只在开始时输出一次
    print(f"\n开始分析 {len(source_header_pairs)} 个源文件的依赖...")
    print(f"项目总函数数: {len(project_total_functions)}")

    # ========== 优化：构建函数名索引（O(1) 查找替代 O(n) 遍历）==========
    # 参考 sactor 框架：使用哈希索引加速函数匹配
    # function_name_index: short_name -> [(full_key, info), ...]
    function_name_index = defaultdict(list)
    for full_key, info in project_total_functions.items():
        # 提取短函数名（去除命名空间和类名）
        short_name = full_key.split("::")[-1].split("(")[0].strip()
        # 同时保留完整键名作为备选
        function_name_index[short_name].append((full_key, info))
        # 如果有命名空间分隔，也保留完整路径
        if "::" in full_key:
            function_name_index[full_key.split("(")[0].strip()].append((full_key, info))
    log.info(f"构建函数名索引完成: {len(function_name_index)} 个唯一函数名")

    # 新增：构建调用图数据结构
    call_graph = {}  # func_file_name -> [callee_file_names]
    function_names = {}  # func_file_name -> actual_function_name
    # 构建反向映射：函数代码 -> 函数文件名（用于查找被调用函数对应的文件名）
    # 注意：project_total_functions 的 key 是函数名，value 可能是函数代码(str)或包含元数据的 dict
    # 我们需要建立函数代码到文件名的映射，但文件名是在循环中生成的
    # 所以我们需要在循环中建立这个映射
    function_code_to_file_name = {}

    # ========== 新增：全局 manifest 收集 ==========
    all_functions_manifest = []  # 收集所有函数的元数据

    total_functions_processed = 0
    total_call_expressions = 0
    total_preprocessed_files = 0

    def get_safe_file_name(file_path: Path, project_root: Path) -> str:
        """
        生成安全的文件名，避免同名文件覆盖

        调用公共函数 safe_module_name()，确保与 skeleton_builder.py 命名一致
        """
        return safe_module_name(project_root, file_path)
    
    # Route A: collect "not built in selected profile" sources for end-of-run reporting.
    excluded_by_cc_files = []

    for file_idx, (source_file, header_file) in enumerate(source_header_pairs):
        source_path = Path(source_file)
        # 使用安全的文件名，避免同名文件覆盖
        current_file_name = get_safe_file_name(source_path, project_root)

        # 跳过特殊文件
        if "dependency_from_other_project" in current_file_name:
            continue

        # 每10个文件输出一次进度
        if file_idx > 0 and file_idx % 10 == 0:
            print(f"[进度] 文件: {file_idx}/{len(source_header_pairs)} | 已处理函数: {total_functions_processed} | 已分析调用: {total_call_expressions}")

        # 读取源文件内容
        print(f"[文件 {file_idx+1}/{len(source_header_pairs)}] 处理: {current_file_name}")

        # ========== 预处理上下文选择 ==========
        preprocess_context = None
        actual_source_file = source_file  # 实际要分析的文件（原始或预处理后）
        line_mapping = {}  # 行号映射：.i文件行号 -> (原文件, 原行号)
        source_for_cc = source_path
        target_cfg = None
        excluded_by_compile_commands = False
        exclude_reason = None

        if preprocessing_enabled and preprocess_parser:
            try:
                print(f"  [预处理] 正在选择编译上下文...")
                # 优先使用映射到 OHOS 源码树的路径（精确匹配 compile_commands.json）
                source_for_cc = source_path
                if map_to_ohos_path:
                    try:
                        mapped = map_to_ohos_path(source_path)
                        if mapped and Path(mapped).exists():
                            source_for_cc = Path(mapped)
                    except Exception:
                        source_for_cc = source_path

                target_cfg = None
                if strategy == ContextSelectionStrategy.ACTIVE:
                    # AUTO/ACTIVE: 如果 profile 已选定，则用 out_dir 作为过滤 key；否则回退到用户的 TARGET_OUT_DIR/TARGET_CONFIG
                    target_cfg = selected_profile_out_dir or TARGET_OUT_DIR or TARGET_CONFIG

                preprocess_context = preprocess_parser.select_preprocessing_context(
                    source_for_cc,
                    strategy=strategy,
                    target_config=target_cfg,
                    output_dir=Path(PREPROCESS_OUTPUT_DIR)
                )

                # Route A: missing compile_commands entry => this file is NOT built in the selected profile.
                # Do NOT treat it as a closure gap; just exclude it from analysis/translation.
                if preprocess_context and str(preprocess_context.error or "").strip().startswith("compile_commands_missing_entry"):
                    excluded_by_compile_commands = True
                    exclude_reason = "not_built_in_selected_profile"
                    print(f"  [RouteA] ⏭️ 源文件不在所选 profile 的 compile_commands 中，跳过该文件: {source_path.name}")
                elif preprocess_context and not preprocess_context.error and preprocess_context.preprocessed_file:
                    actual_source_file = preprocess_context.preprocessed_file
                    line_mapping = preprocess_context.line_mapping
                    total_preprocessed_files += 1
                    print(f"  [预处理] ✓ 使用预处理文件: {preprocess_context.preprocessed_file.name}")
                    print(f"  [预处理]   - 函数数: {preprocess_context.function_count}, 宏数: {preprocess_context.macro_count}")
                else:
                    if preprocess_context and preprocess_context.error:
                        print(f"  [预处理] ✗ 预处理失败: {preprocess_context.error}")
                    print(f"  [预处理] → 回退到原始源文件")
                    actual_source_file = source_file
            except Exception as e:
                print(f"  [预处理] ✗ 预处理异常: {e}")
                print(f"  [预处理] → 回退到原始源文件")
                actual_source_file = source_file

        # ========== 记录 TU 上下文映射（用于后续阶段严格复用） ==========
        try:
            file_record: Dict[str, Any] = {
                "safe_file_name": current_file_name,
                "source_file_rel": str(source_path.relative_to(project_root))
                if Path(project_root) in source_path.parents or source_path == Path(project_root)
                else str(source_path),
                "source_file_abs": str(source_path.resolve()),
                "source_for_cc_abs": str(Path(source_for_cc).resolve())
                if preprocessing_enabled and preprocess_context
                else str(source_path.resolve()),
                "preprocessed_file": str(preprocess_context.preprocessed_file.resolve())
                if (preprocess_context and preprocess_context.preprocessed_file)
                else None,
                "strategy": str(strategy.value) if preprocessing_enabled and preprocess_parser else None,
                "target_config": str(target_cfg) if preprocessing_enabled and preprocess_parser else None,
                "function_count": int(getattr(preprocess_context, "function_count", 0) or 0) if preprocess_context else 0,
                "macro_count": int(getattr(preprocess_context, "macro_count", 0) or 0) if preprocess_context else 0,
                "error": str(getattr(preprocess_context, "error", ""))
                if preprocess_context and getattr(preprocess_context, "error", None)
                else None,
                "proxy_used": bool(getattr(preprocess_context, "proxy_used", False)) if preprocess_context else False,
                "proxy_entry_file": str(getattr(preprocess_context, "proxy_entry_file", "") or "") if preprocess_context else "",
                "proxy_reason": str(getattr(preprocess_context, "proxy_reason", "") or "") if preprocess_context else "",
                # Route A: treat compile_commands as the truth of "what is built".
                # If a file has no entry in the selected profile, it is excluded (not a closure bug).
                "excluded_by_compile_commands": bool(excluded_by_compile_commands),
                "exclude_reason": str(exclude_reason) if exclude_reason else None,
                "compile_commands_entry": None,
                "entry_hash": None,
            }
            if preprocess_context and getattr(preprocess_context, "entry", None):
                entry = preprocess_context.entry
                # 只保留必要字段（避免 json 过大，同时能完整复用 TU flags）
                file_record["compile_commands_entry"] = {
                    "file": entry.get("file"),
                    "directory": entry.get("directory"),
                    "command": entry.get("command"),
                    "arguments": entry.get("arguments"),
                }
                try:
                    import hashlib as _hashlib

                    entry_hash = (
                        _hashlib.md5(json.dumps(file_record["compile_commands_entry"], sort_keys=True).encode())
                        .hexdigest()[:8]
                    )
                    file_record["entry_hash"] = entry_hash
                except Exception:
                    file_record["entry_hash"] = None
            tu_context_map["files"][current_file_name] = file_record
        except Exception:
            # 记录失败不影响主流程
            pass

        # Route A: if this source file isn't built in the selected profile, skip extraction for this file.
        if excluded_by_compile_commands:
            excluded_by_cc_files.append(current_file_name)
            continue

        src_content = read_file(actual_source_file)
        
        # 读取头文件内容（如果存在）
        header_content = ""
        if header_file:
            try:
                header_content = read_file(header_file)
            except Exception as e:
                print(f"  警告: 无法读取头文件 {header_file}: {e}")
        
        # 合并头文件和源文件中的所有函数
        functions_undertranslated = {}
        if header_content:
            header_functions = get_function(header_content)
            functions_undertranslated.update(header_functions)
            print(f"  [头文件] 提取了 {len(header_functions)} 个函数")
        src_functions = get_function(src_content)
        functions_undertranslated.update(src_functions)
        print(f"  [源文件] 提取了 {len(src_functions)} 个函数，总计 {len(functions_undertranslated)} 个函数")

        # 处理每个函数
        i = 1
        func_count_in_file = len(functions_undertranslated)
        # 记录函数元数据（用于生成 manifest）
        functions_metadata = []

        for func_idx, (function_name, func_info) in enumerate(functions_undertranslated.items(), 1):
            # 提取函数数据
            function_code = func_info['code']
            func_start_line = func_info['start_line']
            func_end_line = func_info['end_line']

            # ========== 行号映射：.i 文件行号 -> 原文件行号 ==========
            original_start_line = func_start_line
            original_end_line = func_end_line
            original_source_file = str(source_path.relative_to(project_root))

            if line_mapping and func_start_line in line_mapping:
                mapped_file, mapped_line = line_mapping[func_start_line]
                # 允许映射到项目内的任何文件（包括 .h 头文件）
                # 因为有些项目将实现放在头文件中（如 urlparser 的 url.h 包含全部代码）
                try:
                    mapped_path = Path(mapped_file)
                    project_root_abs = project_root.resolve()

                    # 检查映射的文件是否在项目目录内
                    # 注意：.i 文件中的路径可能指向原始项目目录（而非 workspace 副本）
                    is_in_project = False
                    relative_path = None

                    if mapped_path.is_absolute():
                        # 检查是否在 workspace 项目目录内
                        if str(mapped_path).startswith(str(project_root_abs)):
                            is_in_project = True
                            try:
                                relative_path = str(mapped_path.relative_to(project_root_abs))
                            except ValueError:
                                relative_path = mapped_file
                        else:
                            # 检查是否是项目文件（通过文件名匹配）
                            # 支持原始项目路径（.i 文件中的行号指令可能指向原始目录）
                            mapped_basename = mapped_path.name
                            if mapped_basename.endswith(('.c', '.h', '.cpp', '.hpp', '.cc', '.cxx')):
                                # 检查这个文件名是否存在于当前项目中
                                for candidate in project_root.rglob(mapped_basename):
                                    if candidate.is_file():
                                        is_in_project = True
                                        try:
                                            relative_path = str(candidate.relative_to(project_root_abs))
                                        except ValueError:
                                            relative_path = str(candidate)
                                        break
                    else:
                        # 相对路径：检查是否存在于项目中
                        candidate = project_root / mapped_path
                        if candidate.exists():
                            is_in_project = True
                            relative_path = mapped_file
                        else:
                            # 尝试用 basename 查找
                            for ext in ['.c', '.h', '.cpp', '.hpp']:
                                if mapped_path.name.endswith(ext):
                                    is_in_project = True
                                    relative_path = mapped_file
                                    break

                    if is_in_project:
                        original_start_line = mapped_line
                        original_source_file = relative_path if relative_path else mapped_file
                except Exception:
                    pass  # 保持原始值

            if line_mapping and func_end_line in line_mapping:
                mapped_file, mapped_line = line_mapping[func_end_line]
                try:
                    mapped_path = Path(mapped_file)
                    project_root_abs = project_root.resolve()

                    is_in_project = False
                    if mapped_path.is_absolute():
                        if str(mapped_path).startswith(str(project_root_abs)):
                            is_in_project = True
                        else:
                            # 检查是否是项目文件（通过文件名匹配）
                            mapped_basename = mapped_path.name
                            if mapped_basename.endswith(('.c', '.h', '.cpp', '.hpp', '.cc', '.cxx')):
                                for candidate in project_root.rglob(mapped_basename):
                                    if candidate.is_file():
                                        is_in_project = True
                                        break
                    else:
                        candidate = project_root / mapped_path
                        if candidate.exists():
                            is_in_project = True
                        else:
                            for ext in ['.c', '.h', '.cpp', '.hpp']:
                                if mapped_path.name.endswith(ext):
                                    is_in_project = True
                                    break

                    if is_in_project:
                        original_end_line = mapped_line
                except Exception:
                    pass

            # 每10个函数输出一次进度
            if func_idx > 1 and func_idx % 10 == 0:
                print(f"    [函数 {func_idx}/{func_count_in_file}] 处理中... (已处理: {total_functions_processed}, 已分析调用: {total_call_expressions})")
            func_file_name = f"{current_file_name}_{i}"

            # 保存函数代码
            with open(functions_dir / f"{func_file_name}.txt", 'w', encoding='utf-8') as f:
                f.write(function_code)

            # 记录函数名映射
            function_names[func_file_name] = function_name

            # 新增：记录函数元数据（使用映射后的行号）
            functions_metadata.append({
                'func_file': func_file_name,
                'name': function_name,
                'source_file': original_source_file,
                'start_line': original_start_line,
                'end_line': original_end_line,
                'is_static': False,  # TODO: 检测是否为static
            })

            # 建立函数代码到文件名的映射（用于后续查找被调用函数）
            function_code_to_file_name[function_code] = func_file_name

            # 分析函数调用
            call_functions = get_call_function(function_code)
            call_functions_in_current_project = []
            call_functions_not_in_current_project = set()
            call_functions_not_in_current_file = set()
            
            # 新增：记录被调用的函数文件名
            callee_file_names = []
            
            if len(call_functions) > 0:
                print(f"    [函数 {func_idx}/{func_count_in_file}] {function_name}: 找到 {len(call_functions)} 个函数调用，开始匹配...")
            
            call_match_start_time = None
            import time
            if len(call_functions) > 50:  # 如果调用很多，记录开始时间
                call_match_start_time = time.time()

            for call_idx, call_function in enumerate(call_functions):
                # 如果调用很多，每50个输出一次进度
                if len(call_functions) > 50 and call_idx > 0 and call_idx % 50 == 0:
                    elapsed = time.time() - call_match_start_time if call_match_start_time else 0
                    print(f"      [调用匹配] {call_idx}/{len(call_functions)} (已用时: {elapsed:.1f}秒, 已匹配: {len(call_functions_in_current_project)}, 外部: {len(call_functions_not_in_current_project)})")
                key = False
                # 提取调用函数名（用于匹配）
                call_func_name = call_function.split("(")[0].split("->")[-1].split(":")[-1].strip()

                # ========== 优化：使用哈希索引 O(1) 查找替代 O(n) 遍历 ==========
                # 直接从索引中查找匹配的函数
                candidates = function_name_index.get(call_func_name, [])

                for project_total_function, called_info in candidates:
                    called_function_code = called_info.get('code') if isinstance(called_info, dict) else called_info
                    if not isinstance(called_function_code, str):
                        # 兜底：异常情况下跳过当前匹配，继续检查其他候选函数
                        continue
                    call_functions_in_current_project.append(called_function_code)

                    # 新增：找到被调用函数对应的文件名
                    if called_function_code in function_code_to_file_name:
                        callee_file_name = function_code_to_file_name[called_function_code]
                        if callee_file_name not in callee_file_names:
                            callee_file_names.append(callee_file_name)

                    # 检查函数是否在当前文件（头文件或源文件）中
                    if called_function_code not in header_content and called_function_code not in src_content:
                        call_functions_not_in_current_file.add(called_function_code)
                    key = True
                    break

                if not key:
                    call_functions_not_in_current_project.add(call_function)

            # 保存依赖关系
            call_functions_not_in_current_project = list(call_functions_not_in_current_project)
            call_functions_not_in_current_file = list(call_functions_not_in_current_file)
            
            with open(dependencies_dir / f"{func_file_name}.txt", 'w', encoding='utf-8') as f:
                f.write("\n\n".join(call_functions_in_current_project))
            
            with open(dependencies_not_in_file_dir / f"{func_file_name}.txt", 'w', encoding='utf-8') as f:
                f.write("\n\n".join(call_functions_not_in_current_file))
            
            # 新增：记录调用关系
            call_graph[func_file_name] = callee_file_names
            
            total_functions_processed += 1
            total_call_expressions += len(call_functions)
            
            if len(call_functions) > 0:
                print(f"    [完成] {function_name}: 匹配到 {len(call_functions_in_current_project)} 个内部调用, {len(call_functions_not_in_current_project)} 个外部调用")

            i += 1

        # 将本文件的函数元数据添加到全局 manifest
        all_functions_manifest.extend(functions_metadata)

        print(f"  [完成] {current_file_name}: 处理了 {func_count_in_file} 个函数")
    
    # 新增：保存调用图到 JSON 文件
    print(f"\n[汇总] 依赖分析完成:")
    print(f"  - 处理文件数: {len(source_header_pairs)}")
    if excluded_by_cc_files:
        print(f"  - [RouteA] 被排除(所选 profile 未编译)的文件组数: {len(excluded_by_cc_files)}")
    print(f"  - 处理函数数: {total_functions_processed}")
    print(f"  - 分析调用表达式数: {total_call_expressions}")
    print(f"  - 调用图节点数: {len(call_graph)}")

    if preprocessing_enabled:
        print(f"\n[预处理统计]:")
        print(f"  - 成功预处理文件数: {total_preprocessed_files}/{len(source_header_pairs)}")
        print(f"  - 预处理成功率: {100.0 * total_preprocessed_files / len(source_header_pairs):.1f}%")

    call_graph_path = dependencies_dir.parent / "call_graph.json"
    print(f"\n[保存] 正在保存调用图到: {call_graph_path}")
    try:
        # 构建符合 CallGraphBuilder.load() 期望的新格式 (schema v2.0)
        # 使用 uid (file:line:name) 作为主键，避免函数名冲突
        functions_dict = {}  # uid -> 函数信息
        name_index = defaultdict(list)  # name -> [uid...]
        call_graph_by_uid = {}  # caller_uid -> [callee_uid...]

        # 第一步：为每个函数生成 uid 并建立索引
        func_file_name_to_uid = {}  # func_file_name -> uid
        for func_file_name, actual_function_name in function_names.items():
            # 获取函数代码以提取签名等信息
            func_code = ""
            func_file_path = functions_dir / f"{func_file_name}.txt"
            if func_file_path.exists():
                with open(func_file_path, 'r', encoding='utf-8') as f:
                    func_code = f.read()

            # 从函数代码中提取文件路径（从 func_file_name 推断）
            # func_file_name 格式: {file_name}_{index}
            file_stem = func_file_name.rsplit('_', 1)[0] if '_' in func_file_name else func_file_name

            # 尝试从 func_file_name 推断行号（格式中的 index 作为行号占位）
            start_line = 0
            if '_' in func_file_name:
                try:
                    start_line = int(func_file_name.rsplit('_', 1)[1])
                except ValueError:
                    pass

            # 生成唯一 uid
            uid = generate_function_uid(file_stem, start_line, actual_function_name)
            func_file_name_to_uid[func_file_name] = uid

            # 构建函数信息
            functions_dict[uid] = {
                "name": actual_function_name,
                "file": file_stem,
                "start_line": start_line,
                "end_line": 0,
                "signature": "",
                "callees": [],  # 将在下面填充 (callee uid)
                "uid": uid,
                "mangled_name": "",
            }

            # 更新 name_index
            name_index[actual_function_name].append(uid)

        # 第二步：转换调用图为 uid 格式
        for func_file_name, callee_file_names in call_graph.items():
            caller_uid = func_file_name_to_uid.get(func_file_name)
            if not caller_uid:
                continue

            callee_uids = []
            for callee_file_name in callee_file_names:
                callee_uid = func_file_name_to_uid.get(callee_file_name)
                if callee_uid:
                    callee_uids.append(callee_uid)
                else:
                    # 未找到定义，生成外部函数占位 uid
                    callee_name = function_names.get(callee_file_name, callee_file_name)
                    callee_uid = f"external:0:{callee_name}"
                    callee_uids.append(callee_uid)

            call_graph_by_uid[caller_uid] = callee_uids
            functions_dict[caller_uid]["callees"] = callee_uids

        # 第三步：构建反向调用图
        reverse_call_graph = defaultdict(list)
        for caller_uid, callee_uids in call_graph_by_uid.items():
            for callee_uid in callee_uids:
                reverse_call_graph[callee_uid].append(caller_uid)

        # 保存符合 CallGraphBuilder.load() 期望的新格式 (schema v2.0)
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
                "total_calls": sum(len(callees) for callees in call_graph_by_uid.values()),
                "total_files": len(set(f["file"] for f in functions_dict.values())),
                "cyclic_deps": 0,
                "leaf_functions": 0,
                "root_functions": 0,
            }
        }

        # 检查并报告同名函数（用于调试）
        duplicate_names = {name: uids for name, uids in name_index.items() if len(uids) > 1}
        if duplicate_names:
            print(f"  - 检测到 {len(duplicate_names)} 个同名函数（已通过 uid 区分）:")
            for name, uids in list(duplicate_names.items())[:3]:
                print(f"    '{name}': {len(uids)} 个实例")

        with open(call_graph_path, 'w', encoding='utf-8') as f:
            json.dump(call_graph_data, f, indent=2, ensure_ascii=False)
        print(f"[完成] 调用图已保存 (schema v2.0)")
        print(f"  - 文件路径: {call_graph_path}")
        print(f"  - 包含 {len(functions_dict)} 个函数的详细信息")
        print(f"  - 包含 {len(call_graph_by_uid)} 个函数的调用关系")
    except Exception as e:
        print(f"[错误] 保存调用图失败: {e}")
        import traceback
        traceback.print_exc()

    # ========== 新增：保存函数清单 (functions_manifest.json) ==========
    # 注意：dependencies_dir.parent 已经是 workspace/extracted/<project>/
    # 不需要再添加 "extracted" / project_path，否则会产生双重嵌套路径
    manifest_path = dependencies_dir.parent / "functions_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n[保存] 正在保存函数清单到: {manifest_path}")
    try:
        # 为每个函数生成 uid
        for func_meta in all_functions_manifest:
            func_meta['uid'] = generate_function_uid(
                func_meta['source_file'],
                func_meta['start_line'],
                func_meta['name']
            )
            # 生成函数代码的 SHA1 哈希（用于校验）
            import hashlib
            func_file_path = functions_dir / f"{func_meta['func_file']}.txt"
            if func_file_path.exists():
                with open(func_file_path, 'r', encoding='utf-8') as f:
                    func_code = f.read()
                    func_meta['sha1'] = hashlib.sha1(func_code.encode('utf-8')).hexdigest()
            else:
                func_meta['sha1'] = ""

        manifest_data = {
            "version": "1.0",
            "project": project_path,
            "total_functions": len(all_functions_manifest),
            "functions": all_functions_manifest
        }

        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest_data, f, indent=2, ensure_ascii=False)

        print(f"[完成] 函数清单已保存")
        print(f"  - 文件路径: {manifest_path}")
        print(f"  - 包含 {len(all_functions_manifest)} 个函数的元数据")
        print(f"  - 每个函数包含: func_file, name, source_file, start_line, end_line, uid, sha1")
    except Exception as e:
        print(f"[错误] 保存函数清单失败: {e}")
        import traceback
        traceback.print_exc()

    # ========== 写出 TU 上下文映射（供 translate.py / incremental_translate.py / post_run_analysis 复用） ==========
    try:
        tu_out_dir = Path(PREPROCESS_OUTPUT_DIR)
        tu_out_dir.mkdir(parents=True, exist_ok=True)
        tu_map_path = tu_out_dir / "tu_context_map.json"
        with open(tu_map_path, "w", encoding="utf-8") as f:
            json.dump(tu_context_map, f, indent=2, ensure_ascii=False)
        print(f"\n[TU上下文] 已保存 TU 上下文映射: {tu_map_path}")
        print(f"  - 文件组数: {len(tu_context_map.get('files', {}) or {})}")
        print(f"  - preprocessing_enabled: {tu_context_map.get('preprocessing_enabled')}")
    except Exception as e:
        print(f"\n[TU上下文] ⚠ 保存 TU 上下文映射失败: {e}")

def get_dependencies_libclang():
    """
    使用 libclang 模式进行依赖分析
    
    这是全链路 libclang 架构的入口点，提供更精确的代码解析。
    """
    from unified_function_extractor import (
        UnifiedFunctionExtractor, 
        find_source_files,
        save_complete_json
    )
    
    project_root = Path(PROJECT_ROOT)
    project_path = PROJECT_NAME
    
    # 确保目录存在
    ensure_dirs(project_path)
    functions_dir = get_functions_path(project_path)
    dependencies_dir = get_dependencies_path(project_path)
    
    print("=" * 60)
    print("🔬 Libclang 模式：全链路精确提取")
    print("=" * 60)
    
    # 查找 compile_commands.json
    compile_commands_path = None
    possible_paths = [
        project_root / 'compile_commands.json',
        project_root / 'build' / 'compile_commands.json',
    ]
    for p in possible_paths:
        if p.exists():
            compile_commands_path = p
            print(f"[配置] 使用 compile_commands.json: {p}")
            break
    
    if not compile_commands_path:
        print("[警告] 未找到 compile_commands.json，使用默认编译参数")
    
    # 创建提取器
    extractor = UnifiedFunctionExtractor(
        compile_commands_path=compile_commands_path,
        project_root=project_root
    )
    
    # 查找源文件
    source_files = find_source_files(project_root)
    print(f"[发现] 找到 {len(source_files)} 个源文件")
    
    # 完整提取
    print("\n[步骤 1] 提取函数、全局变量和依赖关系...")
    complete_data = extractor.extract_all(source_files, include_body=True)
    
    # 保存完整 JSON（Single Source of Truth）
    output_json = dependencies_dir.parent / "libclang_extracted.json"
    save_complete_json(complete_data, output_json)
    print(f"[保存] 完整提取结果已保存到: {output_json}")
    
    # 生成与 tree-sitter 模式兼容的文件（用于后续流程）
    print("\n[步骤 2] 生成兼容格式的函数文件...")
    from unified_function_extractor import UnifiedFunction
    functions = [UnifiedFunction.from_dict(f) for f in complete_data['functions']]
    
    # 按源文件分组
    files_funcs = {}
    for func in functions:
        source_file = func.source_file
        if source_file not in files_funcs:
            files_funcs[source_file] = []
        files_funcs[source_file].append(func)
    
    file_count = 0
    function_code_to_file_name = {}
    function_names = {}
    
    for source_file, funcs in files_funcs.items():
        # 生成安全的文件名（使用公共函数，确保与 skeleton_builder.py 一致）
        safe_name = safe_module_name(project_root, Path(source_file))
        
        for i, func in enumerate(funcs, 1):
            func_file_name = f"{safe_name}_{i}"
            output_file = functions_dir / f"{func_file_name}.txt"
            
            # 使用函数体（如果有），否则生成签名
            if func.body:
                content = func.body
            else:
                param_str = ', '.join([f'{ptype} {pname}' for pname, ptype in func.parameters])
                content = f"{func.return_type} {func.name}({param_str})\n{{\n    // Function body\n}}"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            function_code_to_file_name[content] = func_file_name
            function_names[func_file_name] = func.name
            file_count += 1
    
    print(f"[完成] 生成了 {file_count} 个函数文件")
    
    # 生成调用图
    print("\n[步骤 3] 生成调用图...")
    call_graph = {}
    for dep in complete_data['dependencies']:
        caller = dep['caller']
        # 找到 caller 对应的文件名
        caller_file_name = None
        for fn, name in function_names.items():
            if name == caller:
                caller_file_name = fn
                break
        
        if caller_file_name:
            callee_file_names = []
            for callee in dep['callees']:
                for fn, name in function_names.items():
                    if name == callee and fn not in callee_file_names:
                        callee_file_names.append(fn)
                        break
            call_graph[caller_file_name] = callee_file_names

    # 保存调用图（使用符合 CallGraphBuilder.load() 期望的新格式 schema v2.0）
    call_graph_path = dependencies_dir.parent / "call_graph.json"

    # 构建符合 CallGraphBuilder.load() 期望的新格式 (schema v2.0)
    functions_dict = {}  # uid -> 函数信息
    name_index = defaultdict(list)  # name -> [uid...]
    call_graph_by_uid = {}  # caller_uid -> [callee_uid...]

    # 第一步：为每个函数生成 uid 并建立索引
    func_file_name_to_uid = {}  # func_file_name -> uid
    func_name_to_info = {}  # 临时存储函数名到 complete_data 信息的映射

    # 从 complete_data 中获取函数信息
    for func_dict in complete_data['functions']:
        func_name = func_dict['name']
        func_name_to_info[func_name] = func_dict

    for func_file_name, actual_function_name in function_names.items():
        func_info = func_name_to_info.get(actual_function_name, {})
        source_file = func_info.get('source_file', '')
        file_stem = Path(source_file).stem if source_file else 'unknown'
        start_line = func_info.get('start_line', 0)
        end_line = func_info.get('end_line', 0)
        signature = func_info.get('c_signature', '')

        # 生成唯一 uid
        uid = generate_function_uid(file_stem, start_line, actual_function_name)
        func_file_name_to_uid[func_file_name] = uid

        # 构建函数信息
        functions_dict[uid] = {
            "name": actual_function_name,
            "file": file_stem,
            "start_line": start_line,
            "end_line": end_line,
            "signature": signature,
            "callees": [],  # 将在下面填充 (callee uid)
            "uid": uid,
            "mangled_name": "",
        }

        # 更新 name_index
        name_index[actual_function_name].append(uid)

    # 第二步：转换调用图为 uid 格式
    for func_file_name, callee_file_names in call_graph.items():
        caller_uid = func_file_name_to_uid.get(func_file_name)
        if not caller_uid:
            continue

        callee_uids = []
        for callee_file_name in callee_file_names:
            callee_uid = func_file_name_to_uid.get(callee_file_name)
            if callee_uid:
                callee_uids.append(callee_uid)
            else:
                # 未找到定义，生成外部函数占位 uid
                callee_name = function_names.get(callee_file_name, callee_file_name)
                callee_uid = f"external:0:{callee_name}"
                callee_uids.append(callee_uid)

        call_graph_by_uid[caller_uid] = callee_uids
        functions_dict[caller_uid]["callees"] = callee_uids

    # 第三步：构建反向调用图
    reverse_call_graph = defaultdict(list)
    for caller_uid, callee_uids in call_graph_by_uid.items():
        for callee_uid in callee_uids:
            reverse_call_graph[callee_uid].append(caller_uid)

    # 保存符合 CallGraphBuilder.load() 期望的新格式 (schema v2.0)
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
            "total_calls": sum(len(callees) for callees in call_graph_by_uid.values()),
            "total_files": len(set(f["file"] for f in functions_dict.values())),
            "cyclic_deps": 0,
            "leaf_functions": 0,
            "root_functions": 0,
        }
    }

    # 检查并报告同名函数（用于调试）
    duplicate_names = {name: uids for name, uids in name_index.items() if len(uids) > 1}
    if duplicate_names:
        print(f"  - 检测到 {len(duplicate_names)} 个同名函数（已通过 uid 区分）:")
        for name, uids in list(duplicate_names.items())[:3]:
            print(f"    '{name}': {len(uids)} 个实例")

    with open(call_graph_path, 'w', encoding='utf-8') as f:
        json.dump(call_graph_data, f, indent=2, ensure_ascii=False)
    print(f"[保存] 调用图已保存到: {call_graph_path} (schema v2.0)")
    print(f"  - 包含 {len(functions_dict)} 个函数的详细信息")
    print(f"  - 包含 {len(call_graph_by_uid)} 个函数的调用关系")
    
    # 打印汇总
    print("\n" + "=" * 60)
    print("📊 提取汇总:")
    print(f"  - 函数数量: {complete_data['total_functions']}")
    print(f"  - 全局变量: {complete_data['total_global_vars']}")
    print(f"  - 依赖关系: {complete_data['total_dependencies']}")
    print("=" * 60)


def get_dependencies():
    """
    使用 tree-sitter 模式进行依赖分析（传统模式）
    """
    print("=" * 60)
    print("步骤 1: 提取所有函数")
    print("=" * 60)
    project_total_functions = get_total_function()
    print(f"[完成] 总共提取了 {len(project_total_functions)} 个函数\n")
    
    print("=" * 60)
    print("步骤 2: 分析函数调用关系")
    print("=" * 60)
    get_call_function_total(project_total_functions)
    print("\n[完成] 依赖分析全部完成！")


def main():
    """主入口函数"""
    global USE_LIBCLANG
    
    arg_parser = argparse.ArgumentParser(
        description='C/C++ 依赖分析工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
    # 传统模式 (tree-sitter)
    python get_dependencies.py
    
    # Libclang 模式 (更精确)
    python get_dependencies.py --use-libclang
        """
    )
    
    arg_parser.add_argument(
        '--use-libclang', 
        action='store_true',
        help='使用 libclang 进行精确的代码解析（推荐）'
    )
    
    args = arg_parser.parse_args()
    USE_LIBCLANG = args.use_libclang
    
    if USE_LIBCLANG:
        print("🔬 使用 libclang 模式进行依赖分析")
        get_dependencies_libclang()
    else:
        print("🌳 使用 tree-sitter 模式进行依赖分析")
        get_dependencies()


if __name__ == "__main__":
    main()
