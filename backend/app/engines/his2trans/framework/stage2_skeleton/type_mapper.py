#!/usr/bin/env python3
"""
确定性类型映射器：替代 LLM 进行 C -> Rust 类型转换

基于规则的纯工具链方案，确保类型映射的绝对确定性。

增强功能：
- 集成类型清洗器，自动处理宏解析失败产生的垃圾类型名
- 提供更安全的降级机制
- ★★★ 2025-12-11 增强：遇到无法处理的类型时，自动调用 LLM 兜底 ★★★
"""

import re
import os
import logging
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

# 尝试导入类型清洗器
try:
    from type_utils import (
        sanitize_type_name, 
        sanitize_rust_type_name, 
        is_valid_type_name,
        sanitize_param_name
    )
    TYPE_SANITIZER_AVAILABLE = True
except ImportError:
    TYPE_SANITIZER_AVAILABLE = False

# =========================================================================
# LLM 类型映射兜底支持
# =========================================================================

# 全局 LLM 客户端（懒加载）
_llm_client = None
_llm_model = "qwen3_coder"

def _get_llm_client():
    """懒加载 LLM 客户端 - 复用 generation.py 的配置"""
    global _llm_client
    if _llm_client is None:
        try:
            import os
            from openai import OpenAI
            try:
                from generate.generation import (
                    USE_VLLM, VLLM_BASE_URL, VLLM_API_KEY, VLLM_MODEL_NAME, VLLM_REQUEST_TIMEOUT,
                    EXTERNAL_API_BASE_URL, EXTERNAL_API_KEY, EXTERNAL_API_MODEL, EXTERNAL_API_TIMEOUT
                )
                global _llm_model
                if USE_VLLM:
                    _llm_model = VLLM_MODEL_NAME
                    api_base = VLLM_BASE_URL
                    api_key = VLLM_API_KEY
                    timeout = VLLM_REQUEST_TIMEOUT
                else:
                    _llm_model = EXTERNAL_API_MODEL
                    api_base = EXTERNAL_API_BASE_URL
                    api_key = EXTERNAL_API_KEY
                    timeout = EXTERNAL_API_TIMEOUT
            except ImportError:
                api_base = os.environ.get("OPENAI_API_BASE", "http://localhost:8000/v1")
                api_key = "dummy"
                timeout = 600.0
            _llm_client = OpenAI(base_url=api_base, api_key=api_key, timeout=timeout)
            logger.info(f"[TypeMapper] LLM 兜底已启用: {api_base}")
        except Exception as e:
            logger.warning(f"[TypeMapper] LLM 兜底不可用: {e}")
            _llm_client = False  # 标记为不可用
    return _llm_client if _llm_client else None


def _llm_map_type(c_type: str) -> Optional[str]:
    """
    使用 LLM 将 C 类型映射为 Rust 类型
    
    这是最后的兜底策略，当规则引擎无法处理时调用
    
    Args:
        c_type: C 类型字符串
        
    Returns:
        Rust 类型字符串，或 None（LLM 不可用时）
    """
    client = _get_llm_client()
    if not client:
        return None
    
    prompt = f'''Convert this C type to Rust FFI type.

C type: {c_type}

Rules:
1. Basic types: int->i32, unsigned int->u32, char->c_char, void*->*mut c_void
2. Pointers: T* -> *mut T (or *const T if const)
3. Arrays: T[] or T[N] -> *mut T
4. Unknown structs: struct X -> crate::types::X
5. Function pointers: simplify to Option<unsafe extern "C" fn()>

Output ONLY the Rust type, nothing else. Example outputs:
- i32
- *mut std::ffi::c_char
- *const crate::types::MyStruct
- Option<unsafe extern "C" fn()>

Rust type:'''

    try:
        kwargs = dict(
            model=_llm_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.0,
        )
        if "deepseek-v4" in str(_llm_model).lower():
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        response = client.chat.completions.create(**kwargs)
        
        result = response.choices[0].message.content.strip()
        
        # 清洗 LLM 输出：只保留第一行，移除解释
        result = result.split('\n')[0].strip()
        result = result.strip('`').strip()
        
        # 验证输出是否看起来像 Rust 类型
        if result and not any(c in result for c in [';', '{', '}', '=', '//']):
            logger.info(f"[TypeMapper-LLM] '{c_type}' -> '{result}'")
            return result
        else:
            logger.warning(f"[TypeMapper-LLM] 无效输出: '{result}'")
            return None
            
    except Exception as e:
        logger.warning(f"[TypeMapper-LLM] 调用失败: {e}")
        return None


class TypeMapper:
    """
    确定性类型映射器：替代 LLM 进行 C -> Rust 类型转换
    
    核心原则：
    1. 基础类型使用硬编码映射表（绝对确定性）
    2. 自定义类型引用 crate::types::*
    3. 指针统一使用 *mut（最通用且兼容 bindgen）
    """
    
    # 1. 基础类型映射表 (Hard Rules)
    PRIMITIVE_MAP = {
        # void 类型
        "void": "()",  # 仅作为返回值时，参数中需特殊处理

        # 布尔类型
        "bool": "bool",
        "_Bool": "bool",  # 👈 彻底解决 _Bool 映射问题

        # 字符类型
        "char": "std::ffi::c_char",
        "unsigned char": "u8",
        "signed char": "i8",
        "wchar_t": "i32",  # 宽字符

        # 整数类型
        "short": "i16",
        "short int": "i16",
        "int": "i32",
        "long": "i64",  # 注意：在 64 位系统通常是 i64
        "long int": "i64",
        "long long": "i64",
        "long long int": "i64",

        # 无符号整数类型
        "unsigned short": "u16",
        "unsigned short int": "u16",
        "unsigned": "u32",
        "unsigned int": "u32",
        "unsigned long": "u64",
        "unsigned long int": "u64",
        "unsigned long long": "u64",
        "unsigned long long int": "u64",

        # 浮点类型
        "float": "f32",
        "double": "f64",
        "long double": "f64",

        # 标准库类型
        "size_t": "usize",
        "ssize_t": "isize",
        "ptrdiff_t": "isize",
        "intptr_t": "isize",
        "uintptr_t": "usize",

        # 固定宽度整数类型 (stdint.h)
        "int8_t": "i8",
        "uint8_t": "u8",
        "int16_t": "i16",
        "uint16_t": "u16",
        "int32_t": "i32",
        "uint32_t": "u32",
        "int64_t": "i64",
        "uint64_t": "u64",

        # Linux kernel 整数类型 (types.h)
        # 这些在 kernel 代码中经常使用
        "u8": "u8",
        "u16": "u16",
        "u32": "u32",
        "u64": "u64",
        "s8": "i8",
        "s16": "i16",
        "s32": "i32",
        "s64": "i64",
        "__u8": "u8",
        "__u16": "u16",
        "__u32": "u32",
        "__u64": "u64",
        "__s8": "i8",
        "__s16": "i16",
        "__s32": "i32",
        "__s64": "i64",

        # Rust 原生类型（防止被添加 crate::types:: 前缀）
        "i8": "i8",
        "i16": "i16",
        "i32": "i32",
        "i64": "i64",
        "i128": "i128",
        "isize": "isize",
        "u8": "u8",
        "u16": "u16",
        "u32": "u32",
        "u64": "u64",
        "u128": "u128",
        "usize": "usize",
        "f32": "f32",
        "f64": "f64",

        # 常见的 POSIX 类型
        "off_t": "i64",
        "pid_t": "i32",
        "uid_t": "u32",
        "gid_t": "u32",
        "mode_t": "u32",
        "time_t": "i64",
        "clock_t": "i64",
        "socklen_t": "u32",
        "dev_t": "u64",
        "ino_t": "u64",
        "nlink_t": "u64",
        "blksize_t": "i64",
        "blkcnt_t": "i64",

        # 特殊指针类型
        "void*": "*mut std::ffi::c_void",

        # ★ 2025-12-23 增强：更多嵌入式/OpenHarmony 类型
        "BOOL": "i32",  # Windows/OHOS 风格布尔
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
        "BYTE": "u8",
        "WORD": "u16",
        "DWORD": "u32",
        "QWORD": "u64",
        "HANDLE": "*mut std::ffi::c_void",
        "LPVOID": "*mut std::ffi::c_void",
        "LPCVOID": "*const std::ffi::c_void",
        "LPSTR": "*mut std::ffi::c_char",
        "LPCSTR": "*const std::ffi::c_char",
        "HRESULT": "i32",

        # FILE 类型
        "FILE": "std::ffi::c_void",  # 通常通过指针使用

        # va_list
        "va_list": "*mut std::ffi::c_void",
        "__va_list": "*mut std::ffi::c_void",
        "__gnuc_va_list": "*mut std::ffi::c_void",
    }
    
    # Rust 关键字列表（需要转义）
    # Rust 保留关键字 - 需要 r# 转义
    RUST_KEYWORDS = {
        # 类型和模块关键字
        "type", "struct", "enum", "trait", "impl", "mod", "use", "fn",
        # 控制流关键字
        "if", "else", "match", "loop", "while", "for", "in", "break", "continue", "return",
        # 变量声明关键字
        "let", "const", "static", "mut", "ref", "move",
        # 可见性和修饰符关键字
        "pub", "priv", "unsafe", "extern", "async", "await", "dyn",
        # 其他关键字
        "as", "where", "box", "true", "false", "abstract", "become", "do",
        "final", "macro", "override", "typeof", "unsized", "virtual", "yield", "try"
    }
    
    # 特殊关键字 - 不能用 r# 转义，需要重命名
    SPECIAL_KEYWORDS = {"self", "Self", "super", "crate"}

    @staticmethod
    def _extract_type_from_param(param: str) -> str:
        """
        从 C 参数声明中提取类型（移除参数名）

        例如：
        - "int x" -> "int"
        - "const char *name" -> "const char *"
        - "void *" -> "void *"
        - "struct MyStruct data" -> "struct MyStruct"

        Args:
            param: C 参数声明字符串

        Returns:
            类型字符串
        """
        param = param.strip()
        if not param:
            return param

        # 处理指针类型：找到最后一个 * 后的标识符（如果有的话）
        if '*' in param:
            # 找到最后一个 *
            last_star_idx = param.rfind('*')
            # * 后面的部分可能是参数名
            after_star = param[last_star_idx + 1:].strip()
            # 如果 * 后面是纯标识符，移除它
            if after_star and after_star.replace('_', '').isalnum() and not after_star[0].isdigit():
                return param[:last_star_idx + 1].strip()
            return param

        # 处理数组类型：int arr[10] -> int
        if '[' in param:
            bracket_idx = param.find('[')
            before_bracket = param[:bracket_idx].strip()
            # before_bracket 可能是 "int arr"，需要移除 arr
            parts = before_bracket.split()
            if len(parts) > 1 and parts[-1].replace('_', '').isalnum():
                return ' '.join(parts[:-1])
            return before_bracket

        # 普通类型：分割空格，最后一个可能是参数名
        parts = param.split()
        if len(parts) > 1:
            # 检查最后一个是否像参数名（纯标识符）
            last = parts[-1]
            if last.replace('_', '').isalnum() and not last[0].isdigit():
                # 检查是否是类型修饰符（如 long, short, unsigned 等）
                type_keywords = {'const', 'volatile', 'struct', 'enum', 'union', 'signed',
                                 'unsigned', 'long', 'short', 'int', 'char', 'float', 'double',
                                 'void', 'bool', '_Bool'}
                if last.lower() not in type_keywords:
                    return ' '.join(parts[:-1])
        return param

    @staticmethod
    def map_c_type(c_type_str: str, is_pointer: bool = False, is_const: bool = False) -> str:
        """
        递归解析 C 类型并转换为 Rust 类型

        ★★★ 2025-12-23 增强：支持多级指针 (char **, int ***) ★★★
        参考 Simcrat 框架的递归 Type::Ptr(Box<Type>, bool) 方法

        Args:
            c_type_str: C 类型字符串（如 "int", "struct MyStruct", "const char*", "char **"）
            is_pointer: 是否为指针类型（从外部传入，用于处理复杂指针声明）
            is_const: 是否为 const 类型

        Returns:
            Rust 类型字符串（如 "i32", "crate::types::MyStruct", "*const std::ffi::c_char", "*mut *mut c_char"）
        """
        s = c_type_str.strip()

        # 处理空字符串
        if not s:
            return "()"

        import re

        # ========== ★★★ 2025-12-23 增强：函数指针处理 ★★★ ==========
        # 检测函数指针模式：
        # - void (*)(int, char*) - 匿名函数指针
        # - int (*callback)(void) - 命名函数指针
        # - void (*)(void) - 无参数函数指针
        func_ptr_pattern = re.match(
            r'^(.+?)\s*\(\s*\*\s*(\w*)\s*\)\s*\(([^)]*)\)$', s
        )
        if func_ptr_pattern:
            return_type = func_ptr_pattern.group(1).strip()
            # func_name = func_ptr_pattern.group(2)  # 函数指针名称（可忽略）
            params_str = func_ptr_pattern.group(3).strip()

            # 转换返回类型
            rust_ret = TypeMapper.map_c_type(return_type, False, False)
            if rust_ret == "()":
                rust_ret_str = ""
            else:
                rust_ret_str = f" -> {rust_ret}"

            # 转换参数类型
            if not params_str or params_str == "void":
                rust_params = ""
            else:
                param_types = []
                # 简单分割参数（不处理嵌套情况，那些用 LLM 兜底）
                for param in params_str.split(','):
                    param = param.strip()
                    if param and param != "...":
                        # 移除参数名，只保留类型
                        param_type = TypeMapper._extract_type_from_param(param)
                        rust_param = TypeMapper.map_c_type(param_type, False, False)
                        param_types.append(rust_param)
                rust_params = ", ".join(param_types)

            # 生成 Rust 函数指针类型
            # 使用 Option 包装，因为 C 函数指针可能为 NULL
            return f"Option<unsafe extern \"C\" fn({rust_params}){rust_ret_str}>"

        # 0. 处理数组类型 (Array) - 必须在类型清洗之前处理
        # C 语法：type[] 或 type[N] -> Rust: *mut type 或 [type; N]

        # 空数组 (如 usart_pin_map_t[])
        if s.endswith("[]"):
            inner_type = s[:-2].strip()
            rust_inner = TypeMapper.map_c_type(inner_type, False, False)
            return f"*mut {rust_inner}"

        # 带大小的数组 (如 type[10])
        array_match = re.match(r'^(.+?)\[(\d+)\]$', s)
        if array_match:
            inner_type = array_match.group(1).strip()
            array_size = array_match.group(2)
            rust_inner = TypeMapper.map_c_type(inner_type, False, False)
            return f"[{rust_inner}; {array_size}]"

        # ========== 增强: 类型清洗 (解决 pack 项目崩溃问题) ==========
        # 检测并处理宏解析失败产生的垃圾类型名
        if TYPE_SANITIZER_AVAILABLE:
            s = sanitize_type_name(s, fallback="void*")
            # 如果清洗后变成了 void*，直接返回
            if s == "void*":
                return "*mut std::ffi::c_void"

        # ========== ★★★ 关键增强：多级指针处理 ★★★ ==========
        # 参考 Simcrat: Type::Ptr(Box::new(Type::Ptr(...)), bool) 递归结构
        # 统计指针层数，然后递归处理
        pointer_count = 0
        const_at_level = []  # 记录每层是否为 const

        temp_s = s
        while temp_s.endswith("*"):
            pointer_count += 1
            temp_s = temp_s[:-1].strip()
            # 检查这一层是否有 const 修饰
            if temp_s.endswith("const"):
                const_at_level.append(True)
                temp_s = temp_s[:-5].strip()
            else:
                const_at_level.append(False)

        # 如果外部标记了 is_pointer 但字符串里没有 *，补上一层
        if is_pointer and pointer_count == 0:
            pointer_count = 1
            const_at_level.append(is_const)

        # 1. 处理指针 (Pointer) - 递归构建多级指针类型
        if pointer_count > 0:
            # 先处理最内层类型（去掉所有 * 和 const）
            inner_type = temp_s.replace("const", "").strip()

            # 递归处理内层类型（非指针）
            if inner_type == "void" or inner_type == "":
                rust_inner = "std::ffi::c_void"
            else:
                rust_inner = TypeMapper.map_c_type(inner_type, False, False)

            # 从内向外包装指针层
            # const_at_level[0] 是最外层的 const 信息（最后剥离的）
            # 但我们是从内向外构建，所以需要反向
            const_at_level.reverse()

            # 判断最终的 const 属性（最外层）
            # 如果原始调用带有 is_const，或者内层有 const 修饰
            has_inner_const = "const" in s

            for i in range(pointer_count):
                # 每层决定是 *mut 还是 *const
                # 规则：最外层（i == pointer_count - 1）根据 is_const 或类型内 const 判断
                # 内层默认用 *mut
                if i == pointer_count - 1 and (is_const or has_inner_const):
                    rust_inner = f"*const {rust_inner}"
                else:
                    rust_inner = f"*mut {rust_inner}"

            return rust_inner
        
        # 2. 处理 const（非指针情况）
        if s.startswith("const "):
            inner = s[6:].strip()
            return TypeMapper.map_c_type(inner, False, True)
        
        # 3. 处理 volatile（通常可以忽略）
        if s.startswith("volatile "):
            inner = s[9:].strip()
            return TypeMapper.map_c_type(inner, False, False)
        
        # 4. 处理基础类型映射
        if s in TypeMapper.PRIMITIVE_MAP:
            return TypeMapper.PRIMITIVE_MAP[s]
        
        # 5. 处理自定义类型 (Struct/Enum/Union)
        # 如果不是基础类型，默认认为是 bindgen 生成的类型
        # 移除 struct/enum/union 前缀
        clean_name = s.replace("struct ", "").replace("enum ", "").replace("union ", "").strip()
        
        # 移除可能的 const/volatile 修饰符
        clean_name = clean_name.replace("const", "").replace("volatile", "").strip()
        
        # 移除多余空格
        clean_name = " ".join(clean_name.split())
        
        # 如果清理后为空，返回不透明类型
        if not clean_name:
            return "*mut std::ffi::c_void"
        
        # ========== 增强: 清洗自定义类型名 + LLM 兜底 ==========
        # 确保类型名是有效的 Rust 标识符
        if TYPE_SANITIZER_AVAILABLE:
            if not is_valid_type_name(clean_name):
                print(f"⚠️ Warning: Invalid type name '{clean_name}', trying LLM fallback...")
                
                # ★★★ 优先尝试 LLM 兜底 ★★★
                truth_mode = (os.environ.get("C2R_TRUTH_MODE", "0") or "0").strip().lower() in (
                    "1",
                    "true",
                    "yes",
                    "y",
                    "on",
                )
                flag = (os.environ.get("C2R_ENABLE_TYPEMAPPER_LLM_FALLBACK", "") or "").strip().lower()
                if flag:
                    enable_llm_fallback = flag in ("1", "true", "yes", "y", "on")
                else:
                    # Default: allow in non-truth mode, disable in truth mode.
                    enable_llm_fallback = not truth_mode

                if enable_llm_fallback:
                    llm_result = _llm_map_type(s)  # 使用原始类型字符串
                    if llm_result:
                        return llm_result
                
                # LLM 也失败了，降级为 void*
                print(f"⚠️ Warning: LLM fallback also failed for '{clean_name}', fallback to void*")
                return "*mut std::ffi::c_void"
        
        # 替换空格为下划线（Rust 标识符不能有空格）
        rust_type_name = clean_name.replace(' ', '_')
        
        # 确保类型名以字母或下划线开头
        if rust_type_name and not rust_type_name[0].isalpha() and rust_type_name[0] != '_':
            rust_type_name = '_' + rust_type_name
        
        truth_mode = (os.environ.get("C2R_TRUTH_MODE", "0") or "0").strip().lower() in (
            "1",
            "true",
            "yes",
            "y",
            "on",
        )
        prefix_flag = (os.environ.get("C2R_TYPEMAPPER_PREFIX_CUSTOM_TYPES", "") or "").strip().lower()
        if prefix_flag:
            prefix_custom_types = prefix_flag not in ("0", "false", "no", "off")
        else:
            # Default: non-truth mode keeps the legacy `crate::types::` contract;
            # truth mode keeps custom types unqualified to avoid hiding TU gaps.
            prefix_custom_types = not truth_mode

        return f"crate::types::{rust_type_name}" if prefix_custom_types else rust_type_name
    
    @staticmethod
    def sanitize_identifier(name: str) -> str:
        """
        清理标识符名称，避免 Rust 关键字冲突和非法字符
        
        增强功能：
        - 清洗非法字符（如 "hnpPack-", "HNP_LOGI("）
        - 避免 Rust 关键字冲突
        
        Args:
            name: 原始标识符名称
        
        Returns:
            清理后的标识符名称
        """
        if not name:
            return "arg"
        
        # ========== 增强: 使用参数名清洗器 (解决 pack 项目问题) ==========
        clean_name = None
        if TYPE_SANITIZER_AVAILABLE:
            clean_name = sanitize_param_name(name)
        
        if clean_name is None:
            # 回退: 简单清洗
            clean_name = name.strip("*&").strip()
            # 只保留合法字符
            clean_name = ''.join(c for c in clean_name if c.isalnum() or c == '_')
            if not clean_name:
                clean_name = "arg"
        
        # 特殊关键字 - 不能用 r# 转义，需要重命名
        if clean_name in TypeMapper.SPECIAL_KEYWORDS:
            return f"{clean_name}_"  # self -> self_, Self -> Self_, etc.
        
        # 如果是 Rust 关键字，添加 r# 前缀
        if clean_name in TypeMapper.RUST_KEYWORDS:
            return f"r#{clean_name}"
        
        # 如果以数字开头，添加下划线前缀
        if clean_name and clean_name[0].isdigit():
            return f"_{clean_name}"
        
        return clean_name
    
    @staticmethod
    def process_function_signature(
        c_ret_type: str, 
        c_params: List[Tuple[str, str]],
        is_static: bool = False
    ) -> Tuple[str, str, str]:
        """
        生成 Rust 函数签名
        
        Args:
            c_ret_type: C 返回类型字符串
            c_params: C 参数列表，每个元素为 (参数名, 参数类型)
            is_static: 是否为 static 函数
        
        Returns:
            (函数修饰符, 参数列表字符串, 返回值字符串)
            例如: ("pub extern \"C\"", "arg1: i32, arg2: *mut u8", "-> i32")
        """
        # 1. 处理返回值
        c_ret_type_clean = c_ret_type.strip()
        if c_ret_type_clean == "void" or not c_ret_type_clean:
            rust_ret = ""  # Rust 默认返回 ()
        else:
            rust_ret_type = TypeMapper.map_c_type(c_ret_type_clean)
            rust_ret = f"-> {rust_ret_type}"
        
        # 2. 处理参数
        rust_params_list = []
        for name, c_type in c_params:
            c_type_clean = c_type.strip()
            
            # 处理 func(void) 情况
            if c_type_clean == "void" and not name:
                continue
            
            # ⚠️ 重要：不要在这里“先把 * 去掉再传 is_pointer=True”。
            # 否则会把 `char *[]` 这种“指针数组（参数退化为 char **）”错误降级成 `*mut c_char`。
            # 直接把原始类型字符串交给 map_c_type 递归解析（指针/数组/const 都在 map_c_type 内处理）。
            is_ptr_from_name = "*" in name  # 兼容极端提取器把 * 放进参数名的情况
            rust_type = TypeMapper.map_c_type(c_type_clean, is_ptr_from_name, False)
            
            # 清理参数名
            clean_param_name = TypeMapper.sanitize_identifier(name)
            
            rust_params_list.append(f"{clean_param_name}: {rust_type}")
        
        # 3. 确定函数修饰符
        if is_static:
            # static 函数：普通 Rust 函数（不导出）
            func_modifier = "fn"
        else:
            # 非 static C 函数：可能被外部调用，使用 extern "C"
            func_modifier = "pub extern \"C\" fn"
        
        params_str = ", ".join(rust_params_list) if rust_params_list else ""
        
        return func_modifier, params_str, rust_ret
    
    @staticmethod
    def generate_function_stub(
        func_name: str,
        c_ret_type: str,
        c_params: List[Tuple[str, str]],
        is_static: bool = False
    ) -> str:
        """
        生成完整的 Rust 函数桩代码
        
        Args:
            func_name: 函数名
            c_ret_type: C 返回类型
            c_params: C 参数列表
            is_static: 是否为 static 函数
        
        Returns:
            完整的 Rust 函数定义字符串
        """
        func_modifier, params_str, rust_ret = TypeMapper.process_function_signature(
            c_ret_type, c_params, is_static
        )
        
        # 清理函数名（避免关键字冲突）
        clean_func_name = TypeMapper.sanitize_identifier(func_name)
        
        # 生成函数体
        stub = f"""{func_modifier} {clean_func_name}({params_str}){rust_ret} {{
    unimplemented!()
}}"""
        
        return stub


# 便捷函数
def map_c_to_rust(c_type: str, is_pointer: bool = False, is_const: bool = False) -> str:
    """
    便捷函数：将 C 类型映射为 Rust 类型
    
    Args:
        c_type: C 类型字符串
        is_pointer: 是否为指针
        is_const: 是否为 const
    
    Returns:
        Rust 类型字符串
    """
    return TypeMapper.map_c_type(c_type, is_pointer, is_const)


if __name__ == "__main__":
    # 测试代码
    print("=" * 60)
    print("TypeMapper 测试")
    print("=" * 60)
    
    # 测试基础类型映射
    test_cases = [
        ("int", False, False, "i32"),
        ("_Bool", False, False, "bool"),  # 关键测试
        ("unsigned int", False, False, "u32"),
        ("char*", True, False, "*mut std::ffi::c_char"),
        ("const char*", True, True, "*const std::ffi::c_char"),
        ("void*", True, False, "*mut std::ffi::c_void"),
        ("struct MyStruct", False, False, "crate::types::MyStruct"),
        ("size_t", False, False, "usize"),
    ]
    
    print("\n1. 基础类型映射测试：")
    for c_type, is_ptr, is_const, expected in test_cases:
        result = TypeMapper.map_c_type(c_type, is_ptr, is_const)
        status = "✓" if result == expected else "✗"
        print(f"  {status} {c_type} -> {result} (期望: {expected})")
    
    # 测试函数签名生成
    print("\n2. 函数签名生成测试：")
    test_signatures = [
        ("int", [("arg1", "int"), ("arg2", "char*")], False),
        ("_Bool", [("capability", "uint32_t")], True),  # 关键测试
        ("void", [("option", "const PublishOption*")], False),
    ]
    
    for ret_type, params, is_static in test_signatures:
        func_mod, params_str, ret_str = TypeMapper.process_function_signature(
            ret_type, params, is_static
        )
        print(f"  C: {ret_type} func({', '.join(f'{n}: {t}' for n, t in params)})")
        print(f"  Rust: {func_mod} func({params_str}){ret_str}")
        print()
    
    # 测试完整函数桩生成
    print("\n3. 完整函数桩生成测试：")
    stub = TypeMapper.generate_function_stub(
        "ApproachBleIsConcern",
        "_Bool",
        [("capability", "uint32_t")],
        is_static=True
    )
    print(stub)
    print()
    
    # 验证 _Bool 映射
    print("\n4. _Bool 映射验证（关键测试）：")
    bool_result = TypeMapper.map_c_type("_Bool", False, False)
    if bool_result == "bool":
        print(f"  ✓ _Bool 正确映射为: {bool_result}")
    else:
        print(f"  ✗ _Bool 映射错误: {bool_result} (期望: bool)")
