"""
规则修复模块 (rule_fix.py)

提供基于规则的代码修复功能，用于修复常见的编译错误。
这些规则修复比 LLM 修复更快速、更可靠。

设计原则：
1. 每个修复器专注于一类特定的错误模式
2. 修复应该是确定性的、可预测的
3. 如果规则修复失败，再使用 LLM 修复
"""

import re
import logging
from typing import List, Tuple, Optional, Callable

logger = logging.getLogger(__name__)


class RuleFixer:
    """基于规则的代码修复器基类"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.fix_count = 0
    
    def can_fix(self, code: str, error_msg: str) -> bool:
        """检查是否可以修复该错误"""
        raise NotImplementedError
    
    def apply(self, code: str, error_msg: str) -> List[str]:
        """
        应用修复，返回修复后的代码候选列表
        
        Args:
            code: 原始代码
            error_msg: 编译错误消息
            
        Returns:
            修复后的代码候选列表（可能有多个）
        """
        raise NotImplementedError
    
    def try_fix(self, code: str, error_msg: str) -> Tuple[Optional[str], bool]:
        """
        尝试修复代码
        
        Returns:
            (修复后的代码, 是否成功)
        """
        if not self.can_fix(code, error_msg):
            return None, False
        
        candidates = self.apply(code, error_msg)
        if candidates:
            self.fix_count += 1
            return candidates[0], True
        return None, False


# ============================================================
# 具体的修复器实现
# ============================================================

class BracketMismatchFixer(RuleFixer):
    """修复括号不匹配错误"""
    
    def __init__(self):
        super().__init__(
            name="BracketMismatchFixer",
            description="修复括号/花括号/方括号不匹配的错误"
        )
    
    def can_fix(self, code: str, error_msg: str) -> bool:
        return any(keyword in error_msg.lower() for keyword in [
            "unclosed delimiter",
            "mismatched closing delimiter",
            "unexpected closing",
            "expected `}`",
            "expected `)`",
            "expected `]`"
        ])
    
    def apply(self, code: str, error_msg: str) -> List[str]:
        candidates = []
        
        # 策略 1: 统计括号，尝试补全
        bracket_pairs = [('(', ')'), ('{', '}'), ('[', ']')]
        
        for open_b, close_b in bracket_pairs:
            open_count = code.count(open_b)
            close_count = code.count(close_b)
            
            if open_count > close_count:
                # 缺少闭合括号，在末尾添加
                diff = open_count - close_count
                new_code = code.rstrip() + '\n' + (close_b * diff)
                candidates.append(new_code)
            elif close_count > open_count:
                # 多余的闭合括号，尝试删除最后一个
                diff = close_count - open_count
                new_code = code
                for _ in range(diff):
                    idx = new_code.rfind(close_b)
                    if idx != -1:
                        new_code = new_code[:idx] + new_code[idx+1:]
                candidates.append(new_code)
        
        # 策略 2: 从错误信息中提取行号，尝试修复特定行
        line_match = re.search(r'line\s*(\d+)', error_msg)
        if line_match:
            error_line = int(line_match.group(1))
            lines = code.split('\n')
            if 0 < error_line <= len(lines):
                # 检查该行是否缺少括号
                line = lines[error_line - 1]
                for open_b, close_b in bracket_pairs:
                    if line.count(open_b) > line.count(close_b):
                        lines[error_line - 1] = line + close_b
                        candidates.append('\n'.join(lines))
        
        return candidates


class KeywordIdentifierFixer(RuleFixer):
    """
    Fix identifiers that clash with Rust keywords, typically coming from C field names.

    Our type recovery sanitizes keyword field names using the bindgen-like convention:
      `super` -> `super_`, `type` -> `type_`, ...

    When the LLM still emits `.super` or `r#super`, rustc reports:
      "expected identifier, found keyword `super`"
    This fixer rewrites the obvious field-access forms to the sanitized version.
    """

    def __init__(self):
        super().__init__(
            name="KeywordIdentifierFixer",
            description="修复与 Rust 关键字冲突的标识符（例如 `.super` -> `.super_`）",
        )

    def can_fix(self, code: str, error_msg: str) -> bool:
        if not error_msg:
            return False
        msg = error_msg.lower()
        if "expected identifier" not in msg or "found keyword" not in msg:
            return False
        return re.search(r"keyword `([A-Za-z_][A-Za-z0-9_]*)`", error_msg) is not None

    def apply(self, code: str, error_msg: str) -> List[str]:
        m = re.search(r"keyword `([A-Za-z_][A-Za-z0-9_]*)`", error_msg)
        if not m:
            return []
        kw = m.group(1)

        new_code = code
        # Field access: `.super` -> `.super_`
        new_code = re.sub(rf"\\.{re.escape(kw)}\\b", f".{kw}_", new_code)
        # Raw identifier (LLM guess): `r#type` -> `type_` (we use suffix convention)
        new_code = re.sub(rf"\\br#{re.escape(kw)}\\b", f"{kw}_", new_code)
        # Struct literal field: `{ super: ... }` -> `{ super_: ... }`
        new_code = re.sub(rf"\\b{re.escape(kw)}\\s*:", f"{kw}_:", new_code)

        if new_code == code:
            return []
        return [new_code]


class TypeCastFixer(RuleFixer):
    """修复类型转换错误"""

    def __init__(self):
        super().__init__(
            name="TypeCastFixer",
            description="修复类型不匹配错误，添加 'as' 转换"
        )

    def can_fix(self, code: str, error_msg: str) -> bool:
        # 基本检查
        if not any(keyword in error_msg.lower() for keyword in [
            "mismatched types",
            "expected",
            "found",
            "e0308"  # Rust 类型不匹配错误码
        ]):
            return False

        # 排除指针类型错误（由 PointerTypeCastFixer 处理）
        pointer_keywords = ['*mut c_void', '*mut i8', '*const c_void', '*const i8',
                           '*mut ::core::ffi::c_void', '*mut ::core::ffi::c_char']
        if any(kw in error_msg for kw in pointer_keywords):
            return False

        # 排除 size 类型错误（由 SizeTypeMismatchFixer 处理）
        size_keywords = ['u64', 'usize', 'size_t', 'c_ulong', 'i64', 'isize', 'ssize_t']
        if any(kw in error_msg for kw in size_keywords):
            return False

        return True
    
    def apply(self, code: str, error_msg: str) -> List[str]:
        candidates = []
        
        # 解析错误信息，提取期望类型和实际类型
        # 格式: "expected `i32`, found `u32`"
        expected_match = re.search(r"expected\s+`([^`]+)`.*found\s+`([^`]+)`", error_msg)
        if not expected_match:
            return candidates
        
        expected_type = expected_match.group(1)
        found_type = expected_match.group(2)
        
        # 解析错误行号
        line_match = re.search(r":(\d+):", error_msg)
        if not line_match:
            return candidates
        
        error_line = int(line_match.group(1))
        lines = code.split('\n')
        
        if 0 < error_line <= len(lines):
            line = lines[error_line - 1]
            
            # 策略 1: 在行尾添加 as 转换
            if expected_type in ['i32', 'u32', 'i64', 'u64', 'usize', 'isize']:
                # 查找可能需要转换的表达式
                # 简单策略：在分号前添加类型转换
                if line.strip().endswith(';'):
                    new_line = re.sub(r'(\S+)\s*;$', rf'\1 as {expected_type};', line)
                    lines[error_line - 1] = new_line
                    candidates.append('\n'.join(lines))
        
        return candidates


class BorrowCheckerFixer(RuleFixer):
    """修复借用检查问题 - 索引相关"""
    
    def __init__(self):
        super().__init__(
            name="BorrowCheckerFixer",
            description="修复结构体字段互相索引导致的借用检查错误"
        )
    
    def can_fix(self, code: str, error_msg: str) -> bool:
        return any(keyword in error_msg.lower() for keyword in [
            "cannot borrow",
            "borrowed as",
            "while",
            "e0502",  # 借用冲突
            "e0499"   # 多次可变借用
        ])
    
    def apply(self, code: str, error_msg: str) -> List[str]:
        """
        修复结构体自引用问题，如:
        s.arr[s.idx] → let tmp = s.idx; s.arr[tmp]
        """
        candidates = []
        
        # 匹配模式: something.field[something.other_field]
        pattern = r'(\w+)\.(\w+)\[(\1)\.(\w+)\]'
        
        lines = code.split('\n')
        modified = False
        
        for i, line in enumerate(lines):
            matches = list(re.finditer(pattern, line))
            if matches:
                indent = len(line) - len(line.lstrip())
                new_lines = []
                curr_start = 0
                
                for idx, match in enumerate(matches):
                    # 提取临时变量
                    var_name = match.group(1)
                    field1 = match.group(2)
                    field2 = match.group(4)
                    
                    tmp_var = f"_tmp_idx_{idx}"
                    
                    # 在当前行前插入临时变量声明
                    new_lines.append(' ' * indent + f"let {tmp_var} = {var_name}.{field2};")
                    
                    # 修改当前行
                    line = line[:match.start()] + f"{var_name}.{field1}[{tmp_var}]" + line[match.end():]
                
                if new_lines:
                    lines[i] = '\n'.join(new_lines) + '\n' + line
                    modified = True
        
        if modified:
            candidates.append('\n'.join(lines))
        
        return candidates


class MacroConfusionFixer(RuleFixer):
    """修复宏/常量混淆错误"""
    
    def __init__(self):
        super().__init__(
            name="MacroConfusionFixer",
            description="修复宏与常量混淆的错误 (MY_MACRO vs MY_MACRO!())"
        )
    
    def can_fix(self, code: str, error_msg: str) -> bool:
        return any(keyword in error_msg.lower() for keyword in [
            "cannot find macro",
            "not a macro",
            "cannot find value",
            "expected macro",
            "e0433"  # 找不到宏
        ])
    
    def apply(self, code: str, error_msg: str) -> List[str]:
        candidates = []
        
        # 从错误信息中提取标识符名称
        # 格式: "cannot find macro `MY_MACRO`"
        macro_match = re.search(r"cannot find macro\s+`(\w+)`", error_msg)
        if macro_match:
            macro_name = macro_match.group(1)
            # 将 MACRO!() 改为 MACRO (移除宏调用语法)
            new_code = re.sub(rf'\b{macro_name}!\s*\(\s*\)', macro_name, code)
            if new_code != code:
                candidates.append(new_code)
        
        # 反向：尝试将常量改为宏调用
        value_match = re.search(r"cannot find value\s+`(\w+)`", error_msg)
        if value_match:
            value_name = value_match.group(1)
            # 如果是全大写标识符，可能是宏
            if value_name.isupper() and '_' in value_name:
                new_code = re.sub(rf'\b{value_name}\b(?!\s*!)', f'{value_name}!()', code)
                if new_code != code:
                    candidates.append(new_code)
        
        return candidates


class ConstShadowingLetBindingFixer(RuleFixer):
    """修复 let 绑定与常量/静态同名导致的模式匹配错误"""

    def __init__(self):
        super().__init__(
            name="ConstShadowingLetBindingFixer",
            description="修复 `let CONST = ...` 被解析为常量模式 (E0005) 或 let 绑定 shadow 静态变量 (E0530)"
        )

    def can_fix(self, code: str, error_msg: str) -> bool:
        lowered = error_msg.lower()
        # E0005: 常量模式导致 refutable pattern
        if "refutable pattern in local binding" in lowered and "interpreted as a constant pattern" in lowered:
            return True
        # E0530: let 绑定不能 shadow 静态
        if "let bindings cannot shadow statics" in lowered or "error[e0530]" in lowered:
            return True
        return False

    def _extract_conflict_name(self, error_msg: str) -> Optional[str]:
        # E0005 的典型提示：because `TELNET` is interpreted as a constant pattern
        m = re.search(r'`(\w+)`\s+is interpreted as a constant pattern', error_msg)
        if m:
            return m.group(1)

        # E0530 的典型提示：the static `g_xxx` is imported here
        m = re.search(r'the static\s+`(\w+)`', error_msg)
        if m:
            return m.group(1)

        return None

    def apply(self, code: str, error_msg: str) -> List[str]:
        candidates: List[str] = []
        name = self._extract_conflict_name(error_msg)
        if not name:
            return candidates

        replacement = f"{name}_var"
        if replacement == name:
            return candidates

        # 仅修复 `let NAME ...` 这种最常见的情况，避免全局替换影响常量用法
        pattern = rf'(\blet\s+)(mut\s+)?{re.escape(name)}\b'
        new_code = re.sub(pattern, '\\1\\2' + replacement, code)

        if new_code != code:
            candidates.append(new_code)

        return candidates


class FunctionCallSyntaxFixer(RuleFixer):
    """修复函数调用语法错误"""
    
    def __init__(self):
        super().__init__(
            name="FunctionCallSyntaxFixer",
            description="修复结构体字段函数指针调用语法错误"
        )
    
    def can_fix(self, code: str, error_msg: str) -> bool:
        return any(keyword in error_msg.lower() for keyword in [
            "no method named",
            "method not found",
            "e0599"  # 方法未找到
        ])
    
    def apply(self, code: str, error_msg: str) -> List[str]:
        """
        修复函数指针调用语法:
        s.callback(arg) → (s.callback)(arg)
        """
        candidates = []
        
        # 从错误信息提取方法名
        method_match = re.search(r"no method named\s+`(\w+)`", error_msg)
        if not method_match:
            return candidates
        
        method_name = method_match.group(1)
        
        # 模式: obj.method(args) → (obj.method)(args)
        pattern = rf'(\w+)\.({method_name})\s*\('
        
        def replace_func(m):
            return f"({m.group(1)}.{m.group(2)})("
        
        new_code = re.sub(pattern, replace_func, code)
        if new_code != code:
            candidates.append(new_code)
        
        return candidates


class NullPointerInitFixer(RuleFixer):
    """修复指针初始化问题"""
    
    def __init__(self):
        super().__init__(
            name="NullPointerInitFixer",
            description="修复指针初始化为 NULL 的问题"
        )
    
    def can_fix(self, code: str, error_msg: str) -> bool:
        return any(keyword in error_msg.lower() for keyword in [
            "cannot find value `null`",
            "cannot find value `NULL`",
            "use of undeclared identifier"
        ])
    
    def apply(self, code: str, error_msg: str) -> List[str]:
        candidates = []
        
        # 替换 NULL 为 std::ptr::null_mut()
        new_code = code
        new_code = re.sub(r'\bNULL\b', 'std::ptr::null_mut()', new_code)
        new_code = re.sub(r'\bnull\b', 'std::ptr::null_mut()', new_code)
        
        if new_code != code:
            candidates.append(new_code)
        
        # 替换 NULL 为 std::ptr::null() (const 版本)
        new_code2 = code
        new_code2 = re.sub(r'\bNULL\b', 'std::ptr::null()', new_code2)
        if new_code2 != code and new_code2 != new_code:
            candidates.append(new_code2)
        
        return candidates


class CharLiteralFixer(RuleFixer):
    """修复字符字面量问题"""
    
    def __init__(self):
        super().__init__(
            name="CharLiteralFixer",
            description="修复 char 字面量类型不匹配 ('\\0' vs b'\\0')"
        )
    
    def can_fix(self, code: str, error_msg: str) -> bool:
        return "expected `u8`, found `char`" in error_msg
    
    def apply(self, code: str, error_msg: str) -> List[str]:
        candidates = []
        
        # 替换 'x' 为 b'x'
        # 匹配单字符字面量
        pattern = r"(?<![b'])'(\\.|[^'\\])'"
        new_code = re.sub(pattern, r"b'\1' as u8", code)
        
        if new_code != code:
            candidates.append(new_code)
        
        return candidates


class DereferenceOperatorFixer(RuleFixer):
    """修复解引用操作符问题"""
    
    def __init__(self):
        super().__init__(
            name="DereferenceOperatorFixer",
            description="修复解引用和引用操作符问题"
        )
    
    def can_fix(self, code: str, error_msg: str) -> bool:
        return any(keyword in error_msg.lower() for keyword in [
            "type `*mut",
            "type `*const",
            "cannot be dereferenced",
            "cannot be applied"
        ])
    
    def apply(self, code: str, error_msg: str) -> List[str]:
        # 这个修复比较复杂，暂时返回空
        # 后续可以根据具体错误模式添加规则
        return []


class ExternCABIFixer(RuleFixer):
    """修复函数指针 ABI 不匹配问题 - 添加 extern "C" 声明"""

    def __init__(self):
        super().__init__(
            name="ExternCABIFixer",
            description="修复 extern \"C\" ABI 不匹配错误，为函数添加 extern \"C\" 声明"
        )

    def can_fix(self, code: str, error_msg: str) -> bool:
        # 匹配错误模式：expected "C" fn, found "Rust" fn
        return (
            ('expected "C" fn' in error_msg and 'found "Rust" fn' in error_msg) or
            ("expected `extern \"C\" fn" in error_msg) or
            ('expected "C" fn, found "Rust" fn' in error_msg.replace("'", '"'))
        )

    def apply(self, code: str, error_msg: str) -> List[str]:
        candidates = []

        # 从错误信息中提取函数名
        # 格式: "fn(*mut c_void, i32, i32) -> *mut c_void {default_bzalloc}"
        fn_name_match = re.search(r'\{(\w+)\}', error_msg)
        if not fn_name_match:
            # 尝试另一种格式
            fn_name_match = re.search(r'found "Rust" fn.*?`(\w+)`', error_msg)

        if fn_name_match:
            fn_name = fn_name_match.group(1)

            # 在代码中查找并修复函数定义
            # 匹配模式：
            # 1. fn func_name(
            # 2. pub fn func_name(
            # 3. unsafe fn func_name(
            # 4. pub unsafe fn func_name(
            patterns = [
                # pub unsafe fn -> pub unsafe extern "C" fn
                (rf'^(\s*)(pub\s+)(unsafe\s+)(fn\s+{fn_name}\s*\()',
                 r'\1\2\3extern "C" \4'),
                # pub fn -> pub extern "C" fn
                (rf'^(\s*)(pub\s+)(fn\s+{fn_name}\s*\()',
                 r'\1\2extern "C" \3'),
                # unsafe fn -> unsafe extern "C" fn
                (rf'^(\s*)(unsafe\s+)(fn\s+{fn_name}\s*\()',
                 r'\1\2extern "C" \3'),
                # fn -> extern "C" fn
                (rf'^(\s*)(fn\s+{fn_name}\s*\()',
                 r'\1extern "C" \2'),
            ]

            for pattern, replacement in patterns:
                new_code = re.sub(pattern, replacement, code, flags=re.MULTILINE)
                if new_code != code:
                    candidates.append(new_code)
                    break

        # 如果没有找到特定函数名，尝试从错误位置推断
        if not candidates:
            # 解析错误行号
            line_match = re.search(r':(\d+):', error_msg)
            if line_match:
                error_line = int(line_match.group(1))
                lines = code.split('\n')

                # 向上查找函数定义（错误可能在函数调用处，定义在前面）
                for i in range(error_line - 1, -1, -1):
                    if i < len(lines):
                        line = lines[i]
                        # 检测是否是函数定义行
                        if re.match(r'\s*(pub\s+)?(unsafe\s+)?fn\s+\w+\s*\(', line):
                            # 添加 extern "C"
                            if 'extern "C"' not in line:
                                new_line = re.sub(
                                    r'^(\s*)(pub\s+)?(unsafe\s+)?(fn\s+)',
                                    r'\1\2\3extern "C" \4',
                                    line
                                )
                                lines[i] = new_line
                                candidates.append('\n'.join(lines))
                            break

        return candidates


class PointerTypeCastFixer(RuleFixer):
    """修复指针类型不匹配问题 (如 *mut c_void vs *mut i8)"""

    def __init__(self):
        super().__init__(
            name="PointerTypeCastFixer",
            description="修复指针类型不匹配错误，添加 as 类型转换"
        )

    def can_fix(self, code: str, error_msg: str) -> bool:
        # 检查是否是类型不匹配错误
        if 'E0308' not in error_msg and 'mismatched types' not in error_msg.lower():
            return False

        # 检查是否涉及指针类型
        pointer_keywords = ['*mut c_void', '*mut i8', '*const c_void', '*const i8',
                           '*mut ::core::ffi::c_void', '*mut ::core::ffi::c_char']
        return any(kw in error_msg for kw in pointer_keywords)

    def apply(self, code: str, error_msg: str) -> List[str]:
        candidates = []

        # 解析错误信息，提取期望类型和实际类型
        # 格式: "expected `*mut c_void`, found `*mut i8`"
        expected_match = re.search(r"expected\s+`([^`]+)`.*found\s+`([^`]+)`", error_msg)
        if not expected_match:
            return candidates

        expected_type = expected_match.group(1)
        found_type = expected_match.group(2)

        # 确定需要的类型转换
        cast_to = None
        if 'c_void' in expected_type:
            # 需要转换为 *mut c_void 或 *const c_void
            if '*mut' in expected_type:
                cast_to = 'as *mut ::core::ffi::c_void'
            elif '*const' in expected_type:
                cast_to = 'as *const ::core::ffi::c_void'
        elif 'i8' in expected_type or 'c_char' in expected_type:
            # 需要转换为 *mut i8 或 *const i8
            if '*mut' in expected_type:
                cast_to = 'as *mut i8'
            elif '*const' in expected_type:
                cast_to = 'as *const i8'

        if not cast_to:
            return candidates

        # 解析错误行号
        line_match = re.search(r':(\d+):(\d+)', error_msg)
        if not line_match:
            line_match = re.search(r':(\d+):', error_msg)

        if line_match:
            error_line = int(line_match.group(1))
            lines = code.split('\n')

            if 0 < error_line <= len(lines):
                line = lines[error_line - 1]

                # 策略 1: 在 free/malloc 等函数调用参数处添加类型转换
                # 匹配 free(xxx) 并在参数后添加类型转换
                func_patterns = [
                    (r'(free\s*\()([^)]+)(\))', rf'\1\2 {cast_to}\3'),
                    (r'(malloc\s*\()([^)]+)(\))', rf'\1\2 {cast_to}\3'),
                    (r'(realloc\s*\()([^,]+)(,)', rf'\1\2 {cast_to}\3'),
                ]

                for pattern, replacement in func_patterns:
                    if re.search(pattern, line):
                        # 检查是否已经有类型转换
                        if cast_to not in line:
                            new_line = re.sub(pattern, replacement, line, count=1)
                            if new_line != line:
                                lines[error_line - 1] = new_line
                                candidates.append('\n'.join(lines))
                                break

                # 策略 2: 通用参数转换
                if not candidates:
                    # 尝试在错误位置的变量后添加转换
                    # 匹配函数调用参数
                    param_patterns = [
                        (r'(\(\s*)(\w+)(\s*\))', rf'\1\2 {cast_to}\3'),  # (var) -> (var as *mut c_void)
                        (r'(\(\s*)(\w+)(\s*,)', rf'\1\2 {cast_to}\3'),   # (var, -> (var as *mut c_void,
                        (r'(,\s*)(\w+)(\s*\))', rf'\1\2 {cast_to}\3'),   # , var) -> , var as *mut c_void)
                        (r'(,\s*)(\w+)(\s*,)', rf'\1\2 {cast_to}\3'),    # , var, -> , var as *mut c_void,
                    ]

                    for pattern, replacement in param_patterns:
                        test_line = re.sub(pattern, replacement, line, count=1)
                        if test_line != line and cast_to not in line:
                            lines[error_line - 1] = test_line
                            candidates.append('\n'.join(lines))
                            break

        return candidates


class SizeTypeMismatchFixer(RuleFixer):
    """修复 size_t/usize 类型不匹配问题"""

    def __init__(self):
        super().__init__(
            name="SizeTypeMismatchFixer",
            description="修复 size_t/usize/u64 类型不匹配错误，添加类型转换"
        )

        # 定义类型转换映射
        self.type_conversions = [
            # (expected, found, cast_to)
            ('u64', 'usize', 'as u64'),
            ('usize', 'u64', 'as usize'),
            ('c_ulong', 'usize', 'as ::core::ffi::c_ulong'),
            ('::core::ffi::c_ulong', 'usize', 'as ::core::ffi::c_ulong'),
            ('size_t', 'usize', 'as crate::types::size_t'),
            ('usize', 'size_t', 'as usize'),
            ('i64', 'isize', 'as i64'),
            ('isize', 'i64', 'as isize'),
            ('u32', 'usize', 'as u32'),
            ('usize', 'u32', 'as usize'),
            ('i32', 'isize', 'as i32'),
            ('isize', 'i32', 'as isize'),
            ('i32', 'usize', 'as i32'),
            ('usize', 'i32', 'as usize'),
        ]

        # libc 函数返回 usize 的列表
        self.libc_usize_funcs = [
            'strlen', 'wcslen', 'strnlen', 'wcsnlen',
            'strspn', 'strcspn', 'strftime', 'wcsftime',
        ]

    def can_fix(self, code: str, error_msg: str) -> bool:
        # 检查是否是类型不匹配错误
        if 'E0308' not in error_msg and 'mismatched types' not in error_msg.lower():
            return False

        # 检查是否涉及 size 相关类型
        size_keywords = ['u64', 'usize', 'size_t', 'c_ulong', 'i64', 'isize', 'ssize_t']
        return any(kw in error_msg for kw in size_keywords)

    def apply(self, code: str, error_msg: str) -> List[str]:
        candidates = []

        # 解析错误信息，提取期望类型和实际类型
        # 格式: "expected `u64`, found `usize`"
        expected_match = re.search(r"expected\s+`([^`]+)`.*found\s+`([^`]+)`", error_msg)
        if not expected_match:
            return candidates

        expected_type = expected_match.group(1)
        found_type = expected_match.group(2)

        # 确定需要的类型转换
        cast_to = None
        for exp, fnd, cast in self.type_conversions:
            if exp in expected_type and fnd in found_type:
                cast_to = cast
                break

        if not cast_to:
            # 默认转换
            if 'u64' in expected_type:
                cast_to = 'as u64'
            elif 'usize' in expected_type:
                cast_to = 'as usize'
            elif 'size_t' in expected_type:
                cast_to = 'as crate::types::size_t'
            else:
                return candidates

        # 解析错误行号和列号
        line_match = re.search(r':(\d+):(\d+)', error_msg)
        if not line_match:
            line_match = re.search(r':(\d+):', error_msg)

        if line_match:
            error_line = int(line_match.group(1))
            lines = code.split('\n')

            if 0 < error_line <= len(lines):
                line = lines[error_line - 1]

                # 策略 1: 检查是否有 libc 函数调用，在其后添加类型转换
                for func in self.libc_usize_funcs:
                    # 匹配 libc::func(...) 或 func(...)
                    pattern = rf'((?:libc::)?{func}\s*\([^)]*\))'
                    match = re.search(pattern, line)
                    if match:
                        # 检查是否已有类型转换
                        full_match_end = match.end()
                        remaining = line[full_match_end:full_match_end + 10] if full_match_end < len(line) else ""
                        if not remaining.strip().startswith('as '):
                            new_line = line[:match.end()] + f' {cast_to}' + line[match.end():]
                            lines[error_line - 1] = new_line
                            candidates.append('\n'.join(lines))
                            break

                # 策略 2: 在函数调用参数处添加类型转换
                if not candidates:
                    # 尝试在错误位置前的标识符后添加转换
                    # 匹配变量名或简单表达式
                    new_line = line

                    # 如果行中有函数调用参数，尝试在参数处添加转换
                    # 例如: func(len) -> func(len as u64)
                    # 匹配 (var) 或 , var) 或 , var,
                    param_patterns = [
                        (r'(\(\s*)(\w+)(\s*\))', rf'\1\2 {cast_to}\3'),  # (var) -> (var as u64)
                        (r'(\(\s*)(\w+)(\s*,)', rf'\1\2 {cast_to}\3'),   # (var, -> (var as u64,
                        (r'(,\s*)(\w+)(\s*\))', rf'\1\2 {cast_to}\3'),   # , var) -> , var as u64)
                        (r'(,\s*)(\w+)(\s*,)', rf'\1\2 {cast_to}\3'),    # , var, -> , var as u64,
                    ]

                    for pattern, replacement in param_patterns:
                        test_line = re.sub(pattern, replacement, new_line, count=1)
                        if test_line != new_line:
                            lines[error_line - 1] = test_line
                            candidates.append('\n'.join(lines))
                            break

        return candidates


# ============================================================
# 规则修复管理器
# ============================================================

class RuleFixManager:
    """规则修复管理器，管理所有修复器"""
    
    def __init__(self):
        # 注意：专用修复器应放在通用修复器之前
        # 这样即使通用修复器的排除条件不完整，专用修复器也能优先匹配
        self.fixers: List[RuleFixer] = [
            BracketMismatchFixer(),
            KeywordIdentifierFixer(),
            # 专用类型修复器 - 必须放在通用 TypeCastFixer 之前
            PointerTypeCastFixer(),      # 处理 *mut c_void vs *mut i8
            SizeTypeMismatchFixer(),     # 处理 u64 vs usize
            CharLiteralFixer(),          # 处理 u8 vs char
            # 通用类型修复器
            TypeCastFixer(),             # 处理其他类型转换
            BorrowCheckerFixer(),
            MacroConfusionFixer(),
            ConstShadowingLetBindingFixer(),
            FunctionCallSyntaxFixer(),
            NullPointerInitFixer(),
            DereferenceOperatorFixer(),
            ExternCABIFixer(),
        ]
        
        self.total_fixes = 0
        self.fix_history: List[Tuple[str, str, str]] = []  # (fixer_name, before, after)
    
    def add_fixer(self, fixer: RuleFixer):
        """添加自定义修复器"""
        self.fixers.append(fixer)
    
    def try_fix(self, code: str, error_msg: str) -> Tuple[Optional[str], Optional[str]]:
        """
        尝试使用规则修复代码
        
        Args:
            code: 原始代码
            error_msg: 编译错误消息
            
        Returns:
            (修复后的代码, 使用的修复器名称) 或 (None, None)
        """
        for fixer in self.fixers:
            fixed_code, success = fixer.try_fix(code, error_msg)
            if success and fixed_code:
                self.total_fixes += 1
                self.fix_history.append((fixer.name, code[:100], fixed_code[:100]))
                logger.info(f"规则修复成功: {fixer.name}")
                return fixed_code, fixer.name
        
        return None, None
    
    def get_all_candidates(self, code: str, error_msg: str) -> List[Tuple[str, str]]:
        """
        获取所有可能的修复候选
        
        Returns:
            [(修复后的代码, 修复器名称), ...]
        """
        candidates = []
        for fixer in self.fixers:
            if fixer.can_fix(code, error_msg):
                for fixed in fixer.apply(code, error_msg):
                    candidates.append((fixed, fixer.name))
        return candidates
    
    def get_stats(self) -> dict:
        """获取修复统计信息"""
        return {
            "total_fixes": self.total_fixes,
            "fixers": {fixer.name: fixer.fix_count for fixer in self.fixers},
            "history_count": len(self.fix_history)
        }


# ============================================================
# 便捷函数
# ============================================================

_default_manager: Optional[RuleFixManager] = None

def get_rule_fix_manager() -> RuleFixManager:
    """获取默认的规则修复管理器（单例）"""
    global _default_manager
    if _default_manager is None:
        _default_manager = RuleFixManager()
    return _default_manager


def try_rule_fix(code: str, error_msg: str) -> Tuple[Optional[str], Optional[str]]:
    """
    便捷函数：尝试规则修复
    
    Args:
        code: 原始代码
        error_msg: 编译错误消息
        
    Returns:
        (修复后的代码, Agent 名称) 或 (None, None)
    """
    return get_rule_fix_manager().try_fix(code, error_msg)


def apply_all_rule_fixes(code: str, error_msgs: List[str]) -> str:
    """
    应用所有可能的规则修复
    
    Args:
        code: 原始代码
        error_msgs: 编译错误消息列表
        
    Returns:
        修复后的代码（可能多次修复）
    """
    manager = get_rule_fix_manager()
    current_code = code
    
    for error_msg in error_msgs:
        fixed, agent_name = manager.try_fix(current_code, error_msg)
        if fixed:
            current_code = fixed
            logger.info(f"应用规则修复: {agent_name}")
    
    return current_code


# ============================================================
# 测试代码
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 测试括号修复
    code1 = """
    pub fn test() {
        if x > 0 {
            println!("hello"
        }
    }
    """
    error1 = "error: unclosed delimiter at line 4"
    
    manager = RuleFixManager()
    fixed, agent = manager.try_fix(code1, error1)
    print(f"Agent: {agent}")
    print(f"Fixed: {fixed}")
    
    # 测试借用检查修复
    code2 = """
    pub fn test(s: &mut MyStruct) {
        s.arr[s.idx] = 10;
    }
    """
    error2 = "cannot borrow `s.idx` as immutable"
    
    fixed, agent = manager.try_fix(code2, error2)
    print(f"Agent: {agent}")
    print(f"Fixed: {fixed}")
    
    print("\n统计信息:", manager.get_stats())
