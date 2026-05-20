#!/usr/bin/env python3
"""
类型工具模块 (Type Utilities)

提供类型处理、清洗和验证的辅助函数。
用于解决 C 到 Rust 翻译过程中的常见问题：
- 过滤非法标识符（如 "case 'i'", "struct(ram"）
- 解决参数重名问题（E0415）
- 提取和收集类型依赖
- 清洗非法类型名（如宏解析失败产生的垃圾字符串）
"""

import re


# =============================================================================
# 增强型类型名称清洗器 (Enhanced Type Sanitizer)
# 解决 pack 项目因签名解析异常导致的编译失败
# =============================================================================

# 类型名称中的非法字符（通常由宏解析失败产生）
# 注意：* [] () 在 C 类型中是合法的（指针、数组、函数指针），不应视为非法
ILLEGAL_TYPE_CHARS = re.compile(r"[;'\"\{\}=+\-/<>!@#$%^&|\\`~]")

# 类型名称白名单：允许字母、数字、下划线、空格、冒号（命名空间）、*（指针）、[]（数组）
# 注意：此正则用于基本格式检查，后续有更详细的语义检查
TYPE_NAME_WHITELIST = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_: *\[\]]*$')

# 合法的 C 类型修饰符和关键字（允许出现在空格分隔的类型名中）
VALID_TYPE_KEYWORDS = {
    'const', 'volatile', 'struct', 'enum', 'union', 'static', 'extern', 'inline',
    'unsigned', 'signed', 'short', 'long', 'int', 'char', 'void', 'float', 'double',
    'bool', '_Bool', 'size_t', 'ssize_t', 'ptrdiff_t', 'intptr_t', 'uintptr_t',
    'uint8_t', 'uint16_t', 'uint32_t', 'uint64_t', 'int8_t', 'int16_t', 'int32_t', 'int64_t',
}


def is_valid_type_name(type_str: str) -> bool:
    """
    检查类型名称是否合法。
    
    合法的类型名应该只包含：
    - 字母、数字、下划线
    - 空格（用于多单词类型如 "unsigned int"）
    - 冒号（用于命名空间如 "std::string"）
    - 指针符号 *（如 "int *", "void *"）  ★★★ 2025-12-11 修复 ★★★
    - 数组符号 []（如 "int[]", "char[32]"）
    
    特殊检测：
    - 如果包含空格，检查是否是合法的组合（如 "unsigned int", "struct MyStruct"）
    - 过滤掉明显的解析碎片（如 "ackHnp star", "fo h pCfg"）
    
    Args:
        type_str: 待检查的类型名称
        
    Returns:
        如果是合法的类型名返回 True，否则返回 False
    """
    if not type_str or len(type_str) > 100:
        return False
    
    # 清理两端空白
    clean = type_str.strip()
    
    if not clean:
        return False
    
    # 检查是否包含非法字符
    if ILLEGAL_TYPE_CHARS.search(clean):
        return False
    
    # ★★★ 2025-12-11 修复：先处理指针和数组符号 ★★★
    # 将指针和数组符号视为合法的类型修饰符，在后续检查前移除
    # 但要保留它们的存在性信息
    has_pointer = '*' in clean
    has_array = '[' in clean or ']' in clean
    
    # 移除指针和数组符号后检查基础类型
    clean_for_check = clean.replace('*', ' ').replace('[', ' ').replace(']', ' ')
    clean_for_check = ' '.join(clean_for_check.split())  # 规范化空格
    
    # ★★★ 如果只有指针/数组符号（如 "*" 或 "[]"），认为是合法的占位符 ★★★
    if not clean_for_check:
        return has_pointer or has_array
    
    if clean_for_check and not TYPE_NAME_WHITELIST.match(clean_for_check):
        return False
    
    # ========== 增强检测：空格分隔的类型名 ==========
    # 使用清理后的类型名进行检查（不含 * 和 []）
    words = clean_for_check.split()
    
    if len(words) > 1:
        # 多单词类型名，检查是否是合法的 C 类型组合
        # 合法的组合：
        # 1. 所有单词都是类型关键字（如 "unsigned long int"）
        # 2. 最后一个单词是类型名，前面是修饰符（如 "struct MyStruct"）
        # 3. 如果有多个非关键字单词，可能是解析错误（如 "ackHnp star"）
        
        non_keyword_words = [w for w in words if w not in VALID_TYPE_KEYWORDS]
        
        if len(non_keyword_words) > 1:
            # 多个非关键字单词 -> 很可能是解析错误
            # 例如："ackHnp star" -> ["ackHnp", "star"] -> 2个非关键字
            # 例如："struct MyStruct" -> ["MyStruct"] -> 1个非关键字 -> OK
            return False
        
        # 检查单词顺序：修饰符应该在类型名前面
        # 如果最后一个单词是关键字，而前面有非关键字，那是错误的
        # 例如："MyStruct const" 是错误的（应该是 "const MyStruct"）
        if words[-1] in VALID_TYPE_KEYWORDS and non_keyword_words:
            # 最后一个是关键字，但前面有非关键字 -> 顺序错误
            return False
    
    # 移除 const/volatile/struct/enum/union 等关键字后检查
    base_type = clean_for_check
    for keyword in VALID_TYPE_KEYWORDS:
        base_type = re.sub(rf'\b{keyword}\b', '', base_type).strip()
    
    # 清理多余空格
    base_type = ' '.join(base_type.split())
    
    if not base_type:
        return True  # 只有关键字（如 "void", "unsigned int", "int *"）是合法的
    
    # 基础类型应该是单个合法标识符（不含空格）
    return is_valid_c_identifier(base_type)


def sanitize_type_name(type_str: str, fallback: str = "*mut std::ffi::c_void") -> str:
    """
    清洗类型名称，处理宏解析失败产生的垃圾字符串。
    
    这是解决 pack 项目 'ackHnp star', 'fgInf ;' 等非法类型名的关键函数。
    
    策略：
    1. 如果包含非法字符（;, ', ", {, } 等），说明解析失败
    2. 直接降级为 void* 指针，确保编译通过
    
    Args:
        type_str: 原始类型字符串
        fallback: 检测到非法类型时的降级类型，默认为 void*
        
    Returns:
        清洗后的类型名称，或降级类型
    """
    if not type_str:
        return fallback
    
    # 1. 检查是否包含明显的非法字符（通常是解析失败的标志）
    if ILLEGAL_TYPE_CHARS.search(type_str):
        print(f"⚠️ Warning: Detected garbage type '{type_str}', fallback to {fallback}")
        return fallback
    
    # 2. 清理多余空白
    clean = ' '.join(type_str.split())
    
    # 3. 检查是否是合法的类型名
    if not is_valid_type_name(clean):
        print(f"⚠️ Warning: Invalid type name '{type_str}', fallback to {fallback}")
        return fallback
    
    # 4. 额外检查：类型名不应该太短（单字符，除非是常见缩写）
    base_type = extract_base_type(clean) if clean else ""
    if base_type and len(base_type) == 1 and base_type not in ['T', 'K', 'V']:
        # 单字符类型名（除了常见泛型参数）很可能是解析错误
        print(f"⚠️ Warning: Suspiciously short type '{base_type}', fallback to {fallback}")
        return fallback
    
    return clean


def sanitize_rust_type_name(rust_type: str) -> str:
    """
    清洗 Rust 类型名称中的非法部分。
    
    主要处理 crate::types::xxx 格式中 xxx 部分的非法字符。
    
    Args:
        rust_type: Rust 类型字符串
        
    Returns:
        清洗后的 Rust 类型名称
    """
    if not rust_type:
        return "*mut std::ffi::c_void"
    
    # 检查是否是 crate::types:: 引用
    if "crate::types::" in rust_type:
        prefix = "crate::types::"
        idx = rust_type.find(prefix)
        type_name = rust_type[idx + len(prefix):]
        
        # 清洗类型名
        if ILLEGAL_TYPE_CHARS.search(type_name):
            print(f"⚠️ Warning: Detected garbage in Rust type '{rust_type}', fallback to void*")
            return "*mut std::ffi::c_void"
        
        # 清洗类型名中的空格
        clean_type_name = type_name.replace(' ', '_')
        return f"{rust_type[:idx + len(prefix)]}{clean_type_name}"
    
    return rust_type


def is_valid_c_identifier(name: str) -> bool:
    """
    检查是否是合法的 C 标识符。
    
    用于过滤掉 Tesseract/Tree-sitter 解析出的垃圾代码（如 "case 'i'", "struct(ram"）。
    
    Args:
        name: 待检查的标识符名称
        
    Returns:
        如果是合法的 C 标识符返回 True，否则返回 False
    """
    if not name or len(name) > 100:  # 太长通常也是错的
        return False
    
    # 标准 C 标识符：字母或下划线开头，后面跟字母、数字或下划线
    # 严格模式：不允许包含空格、括号、引号
    return re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', name) is not None


def sanitize_parameter_names(params: list) -> list:
    """
    解决参数重名问题 (E0415) 和非法参数名问题。
    
    增强功能：
    - 处理包含非法字符的参数名（如 "hnpPack-", "HNP_LOGI("）
    - 处理参数重名
    
    输入: [('arg', 'int'), ('arg', 'float')]
    输出: [('arg_0', 'int'), ('arg_1', 'float')]
    
    对于通用名字（如 'arg'）或空名字，第一个也会添加 _0 后缀。
    对于其他名字，第一个保持原样，重名时添加后缀。
    
    Args:
        params: 参数列表，格式为 [(param_name, param_type), ...]
        
    Returns:
        清洗后的参数列表，重名参数会被添加后缀
    """
    seen_names = {}
    new_params = []
    
    for i, (name, type_str) in enumerate(params):
        # 1. 清洗参数名：移除非法字符，只保留合法的标识符字符
        clean_name = sanitize_param_name(name)
        
        # 2. 如果清洗后为空或只是 "arg"，使用通用名字
        if not clean_name or clean_name == "arg":
            clean_name = "arg"
        
        # 3. 解决冲突
        if clean_name in seen_names:
            # 重名：添加递增后缀
            seen_names[clean_name] += 1
            final_name = f"{clean_name}_{seen_names[clean_name]}"
        else:
            # 第一次出现：对于通用名字（如 'arg'），添加 _0 后缀
            # 对于其他名字，保持原样（但如果后续有重名，会添加后缀）
            if clean_name == "arg":
                seen_names[clean_name] = 0
                final_name = f"{clean_name}_0"
            else:
                seen_names[clean_name] = 0  # 初始化为 0，如果后续有重名，第一个会变成 _1
                final_name = clean_name
            
        new_params.append((final_name, type_str))
        
    return new_params


def sanitize_param_name(name: str) -> str:
    """
    清洗参数名，移除非法字符。
    
    处理的情况：
    - 参数名包含非法字符（如 "hnpPack-", "HNP_LOGI("）
    - 空参数名
    - 以数字开头的参数名
    
    Args:
        name: 原始参数名
        
    Returns:
        清洗后的参数名
    """
    if not name:
        return "arg"
    
    # 移除两端空白
    clean = name.strip()
    
    if not clean:
        return "arg"
    
    # 移除可能的指针/引用符号
    clean = clean.strip('*&')
    
    # 只保留合法的标识符字符（字母、数字、下划线）
    # 移除所有非法字符
    cleaned_chars = []
    for char in clean:
        if char.isalnum() or char == '_':
            cleaned_chars.append(char)
    
    clean = ''.join(cleaned_chars)
    
    if not clean:
        return "arg"
    
    # 如果以数字开头，添加下划线前缀
    if clean[0].isdigit():
        clean = '_' + clean
    
    # 如果太短（单字符除了常见缩写），可能是垃圾，但保留
    # 常见的单字符参数名：i, j, k, n, x, y, z 等
    
    return clean


def extract_base_type(type_str: str) -> str:
    """
    从复杂类型字符串中提取核心类型名，用于收集依赖。
    
    增强版：能处理 Rust 路径 (crate::types::...) 和 C 类型修饰符。
    
    例如: 
    - "struct PermissionService *" -> "PermissionService"
    - "const napi_env" -> "napi_env"
    - "int *" -> "int"
    - "void" -> "void"
    - "*mut crate::types::MessageParcel" -> "MessageParcel"  # 新增：处理 Rust 路径
    - "crate::types::MessageOption" -> "MessageOption"  # 新增：处理 Rust 路径
    
    Args:
        type_str: C 类型字符串或 Rust 类型字符串
        
    Returns:
        提取出的基础类型名（纯净的类型名，不含路径和修饰符）
    """
    if not type_str:
        return ""
    
    # 1. 移除 Rust 特有的修饰符
    clean = re.sub(r'\b(mut|const|pub|ref|extern)\b', '', type_str)
    
    # 2. 移除指针、引用、数组符号
    clean = clean.replace('*', '').replace('&', '').replace('[', '').replace(']', '')
    
    # 3. 关键修复：移除路径前缀 (crate::types:: 或 std::ffi::)
    # 例如: "crate::types::MessageParcel" -> "MessageParcel"
    #      "std::ffi::c_void" -> "c_void"
    if "::" in clean:
        # 只取最后一部分，如 MessageParcel
        clean = clean.split("::")[-1]
    
    # 4. 移除 C 的修饰符 (struct/enum/union/volatile)
    clean = re.sub(r'\b(struct|enum|union|volatile)\b', '', clean)
    
    # 5. 清理多余空格
    clean = ' '.join(clean.split())
    
    return clean.strip()

