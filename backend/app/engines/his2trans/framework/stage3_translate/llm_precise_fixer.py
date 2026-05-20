#!/usr/bin/env python3
"""
LLM 精准修复器 (LLM Precise Fixer)

针对骨架编译错误进行精准修复：
1. 使用 cargo check --message-format=json 获取结构化错误
2. 提取报错的文件、行号、错误信息
3. 只把相关代码 + 完整上下文发给 LLM
4. 用 LLM 返回的代码替换原代码

设计原则：
- 最小化 LLM 输入：只发送必要的上下文
- 严格限制输出：只要求返回修复后的代码行
- 保留 FFI 语义：确保修复后的代码仍然是有效的 FFI 代码
- 自适应学习：从编译错误中学习新的类型/常量定义
"""

import json
import subprocess
import re
import os
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# 尝试导入自适应预定义管理器
try:
    from config.predefines import get_predefine_manager, PredefineManager
    PREDEFINES_AVAILABLE = True
except ImportError:
    PREDEFINES_AVAILABLE = False
    logger.warning("config.predefines 模块不可用，将使用内置的硬编码定义")


@dataclass
class CompileError:
    """编译错误的结构化表示"""
    file_path: str           # 文件路径
    line_number: int         # 行号 (1-based)
    column: int              # 列号 (1-based)
    error_code: str          # 错误代码 (如 E0308)
    message: str             # 错误消息
    rendered: str            # 完整的渲染错误信息
    suggestion: Optional[str] = None  # 编译器建议的修复（如果有）
    

@dataclass  
class FixContext:
    """修复上下文"""
    error: CompileError                    # 错误信息
    source_line: str                       # 出错的代码行
    context_before: List[str]              # 前面几行上下文
    context_after: List[str]               # 后面几行上下文
    related_types: Dict[str, str]          # 相关类型定义 (类型名 -> 定义)
    original_c_code: Optional[str] = None  # 原始 C 代码（如果能找到）


# =============================================================================
# LLM 提示词模板
# =============================================================================

PRECISE_FIX_PROMPT = '''You are fixing a Rust FFI compilation error.

## Error Information
- Error Code: {error_code}
- Message: {message}
- File: {file_path}
- Line: {line_number}

## Code Context
```rust
{context_before}
>>> {source_line}  // <-- ERROR HERE
{context_after}
```

{type_definitions}

{original_c_code}

## Your Task
Fix ONLY the error line. Output ONLY the corrected single line of code.

Rules:
1. Keep FFI compatibility (use raw pointers, C types)
2. For type mismatches: use correct Rust type or add `as` cast
3. For initialization errors: use `std::mem::zeroed()` for structs, `std::ptr::null_mut()` for pointers
4. For bool vs u8: use `0u8` or `1u8` instead of `false`/`true`
5. For string literals: use `b"string\\0".as_ptr() as *mut _` or `std::ptr::null_mut()`

Output the fixed line ONLY, no explanation, no markdown:
'''

PRECISE_FIX_PROMPT_MULTI = '''You are fixing multiple Rust FFI compilation errors in one file.

## Errors to Fix
{error_list}

## Full File Content
```rust
{file_content}
```

{type_definitions}

## Your Task
Output the COMPLETE fixed file content.

Rules:
1. Keep FFI compatibility (use raw pointers, C types)
2. For type mismatches: use correct Rust type or add `as` cast
3. For initialization errors: use `unsafe {{ std::mem::zeroed() }}` for structs, `std::ptr::null_mut()` for pointers
4. For bool vs u8: use `0u8` or `1u8` instead of `false`/`true`
5. For string literals: use `std::ptr::null_mut()` for *mut pointers

Output the complete fixed Rust code ONLY, no explanation, no markdown:
'''


class LLMPreciseFixer:
    """LLM 精准修复器"""
    
    # 常见错误的规则修复映射
    RULE_FIXES = {
        # E0308: mismatched types
        # 布尔值 -> 整数类型
        r'expected `u8`, found `bool`': lambda line: re.sub(r'\bfalse\b', '0u8', re.sub(r'\btrue\b', '1u8', line)),
        r'expected `i32`, found `bool`': lambda line: re.sub(r'\bfalse\b', '0', re.sub(r'\btrue\b', '1', line)),
        r'expected `u32`, found `bool`': lambda line: re.sub(r'\bfalse\b', '0u32', re.sub(r'\btrue\b', '1u32', line)),
        
        # 字符串字面量 -> 空指针
        r'expected raw pointer .\*mut': lambda line: re.sub(r'=\s*"[^"]*"', '= std::ptr::null_mut()', line),
        r'expected raw pointer .\*const': lambda line: re.sub(r'=\s*"[^"]*"', '= std::ptr::null()', line),
        
        # 十六进制前缀大写 -> 小写 (0X -> 0x)
        r'invalid base prefix for number literal': lambda line: re.sub(r'\b0X', '0x', line),
        
        # 整数溢出 (0x80000000 对 i32 来说太大)
        r'literal out of range for `i32`': lambda line: re.sub(r'(0x[0-9A-Fa-f]+)\s*;', r'\1u32 as i32;', line),
        r'literal out of range for `i64`': lambda line: re.sub(r'(0x[0-9A-Fa-f]+)\s*;', r'\1u64 as i64;', line),
    }
    
    # E0425: cannot find value - 使用 zeroed() 替换未定义的初始化常量
    VALUE_REPLACEMENT_RULES = {
        'PTHREAD_MUTEX_INITIALIZER': 'unsafe { std::mem::zeroed() }',
        'PTHREAD_COND_INITIALIZER': 'unsafe { std::mem::zeroed() }',
        'PTHREAD_RWLOCK_INITIALIZER': 'unsafe { std::mem::zeroed() }',
        'PTHREAD_ONCE_INIT': 'unsafe { std::mem::zeroed() }',
        'AUDIO_FORMAT_TYPE_PCM_16_BIT': '1',
        'AUDIO_FORMAT_TYPE_PCM_8_BIT': '0',
        'NULL': 'std::ptr::null_mut()',
    }
    
    # 类型缺失的自动补全定义（动态从 PredefineManager 获取）
    @classmethod
    def _get_type_definitions(cls) -> Dict[str, str]:
        """获取类型定义（优先使用可配置的 PredefineManager）"""
        if PREDEFINES_AVAILABLE:
            return get_predefine_manager().get_all_types()
        # 回退到硬编码定义
        return {
            'file': '#[repr(C)]\npub struct file { _opaque: [u8; 0] }',
            'FILE': '#[repr(C)]\npub struct FILE { _opaque: [u8; 0] }',
            'AudioPort': '#[repr(C)]\npub struct AudioPort { _opaque: [u8; 0] }',
            'AudioPortCapability': '#[repr(C)]\npub struct AudioPortCapability { _opaque: [u8; 0] }',
            'AudioAdapter': '#[repr(C)]\npub struct AudioAdapter { _opaque: [u8; 0] }',
            'pthread_mutex_t': '#[repr(C)]\npub struct pthread_mutex_t { _opaque: [u8; 40] }',
            'pthread_cond_t': '#[repr(C)]\npub struct pthread_cond_t { _opaque: [u8; 48] }',
        }
    
    @classmethod
    def _get_const_definitions(cls) -> Dict[str, str]:
        """获取常量定义（优先使用可配置的 PredefineManager）"""
        if PREDEFINES_AVAILABLE:
            manager = get_predefine_manager()
            result = {}
            for name, rust_type, value in manager.get_all_constants():
                result[name] = f"pub const {name}: {rust_type} = {value};"
            return result
        # 回退到硬编码定义
        return {
            'PTHREAD_MUTEX_INITIALIZER': 'pub const PTHREAD_MUTEX_INITIALIZER: pthread_mutex_t = pthread_mutex_t { _opaque: [0u8; 40] };',
            'PTHREAD_COND_INITIALIZER': 'pub const PTHREAD_COND_INITIALIZER: pthread_cond_t = pthread_cond_t { _opaque: [0u8; 48] };',
            'AUDIO_FORMAT_TYPE_PCM_16_BIT': 'pub const AUDIO_FORMAT_TYPE_PCM_16_BIT: i32 = 1;',
            'AUDIO_FORMAT_TYPE_PCM_8_BIT': 'pub const AUDIO_FORMAT_TYPE_PCM_8_BIT: i32 = 0;',
        }
    
    # 保留硬编码作为回退 (兼容性)
    MISSING_TYPE_DEFINITIONS = None  # 使用 _get_type_definitions() 替代
    MISSING_CONST_DEFINITIONS = None  # 使用 _get_const_definitions() 替代
    
    def __init__(
        self, 
        skeleton_dir: Path, 
        llm_client=None, 
        model_name: str = "qwen3_coder",
        c_source_dirs: List[Path] = None,
        include_dirs: List[Path] = None,
        compile_commands_parser=None  # ★ 新增：compile_commands.json 解析器
    ):
        """
        初始化修复器
        
        Args:
            skeleton_dir: 骨架目录（包含 Cargo.toml 的目录）
            llm_client: OpenAI 兼容的 LLM 客户端
            model_name: 模型名称
            c_source_dirs: C 源码目录（用于诊断分析，搜索原始定义）
            include_dirs: 头文件搜索目录
            compile_commands_parser: CompileCommandsParser 实例
                                     ★★★ 优先使用此参数进行精准搜索 ★★★
        """
        self.skeleton_dir = Path(skeleton_dir)
        self.llm_client = llm_client
        self.model_name = model_name
        self.src_dir = self.skeleton_dir / "src"
        
        # 诊断分析相关
        self.c_source_dirs = [Path(d) for d in (c_source_dirs or [])]
        self.include_dirs = [Path(d) for d in (include_dirs or [])]
        
        # ★ 新增：compile_commands.json 支持
        self.compile_commands_parser = compile_commands_parser
        # Lazy-initialized C source searcher (can be expensive to build if backed by compile_commands).
        self._c_source_searcher = None
        self._c_source_searcher_init_failed = False
        # Cached index for `src/types.rs` (constants + enum variants).
        self._types_index_built = False
        self._types_const_names = set()
        self._types_enum_variants = {}  # enum_name -> [variant_name, ...]

    # ---------------------------------------------------------------------
    # types.rs indexing helpers (deterministic, no network)
    # ---------------------------------------------------------------------

    def _types_rs_path(self) -> Path:
        return self.src_dir / "types.rs"

    def _ensure_types_index(self) -> None:
        """Parse `src/types.rs` once to index constants and enum variants."""
        if self._types_index_built:
            return
        self._types_index_built = True
        types_path = self._types_rs_path()
        if not types_path.exists():
            return
        try:
            text = types_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return

        const_names = set()
        enum_variants = {}

        lines = text.splitlines()
        i = 0
        while i < len(lines):
            line = lines[i]
            m = re.match(r"\s*pub\s+const\s+([A-Za-z_][A-Za-z0-9_]*)\s*:", line)
            if m:
                const_names.add(m.group(1))
                i += 1
                continue
            m = re.match(r"\s*pub\s+enum\s+([A-Za-z_][A-Za-z0-9_]*)\s*\{", line)
            if m:
                enum_name = m.group(1)
                variants = []
                brace_level = line.count("{") - line.count("}")
                i += 1
                while i < len(lines) and brace_level > 0:
                    l = lines[i]
                    brace_level += l.count("{") - l.count("}")
                    stripped = l.strip()
                    if not stripped or stripped.startswith(("#", "//", "/*", "*")):
                        i += 1
                        continue
                    vm = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*(=|,)", stripped)
                    if vm:
                        variants.append(vm.group(1))
                    i += 1
                if variants:
                    enum_variants[enum_name] = variants
                continue
            i += 1

        self._types_const_names = const_names
        self._types_enum_variants = enum_variants

    def _types_has_const(self, name: str) -> bool:
        if not name:
            return False
        self._ensure_types_index()
        return name in self._types_const_names

    def _types_enum_has_variant(self, enum_name: str, variant: str) -> bool:
        if not enum_name or not variant:
            return False
        self._ensure_types_index()
        return variant in self._types_enum_variants.get(enum_name, [])

    def _types_enum_first_variant(self, enum_name: str) -> Optional[str]:
        if not enum_name:
            return None
        self._ensure_types_index()
        variants = self._types_enum_variants.get(enum_name) or []
        return variants[0] if variants else None

    def _extract_static_decl_parts(self, line: str) -> Optional[Tuple[str, str, str]]:
        """
        Parse `... static ... NAME: TYPE = EXPR;` into (name, type, expr) for simple cases.
        Returns None if not a recognizable static declaration.
        """
        if not line or "static" not in line or "=" not in line or ":" not in line:
            return None
        # Keep it intentionally simple/robust: do not try to parse complex generics.
        m = re.search(r"\bstatic\s+mut\s+([A-Za-z_][A-Za-z0-9_]*)\s*:\s*([^=;]+?)\s*=\s*(.+?)\s*;", line)
        if not m:
            m = re.search(r"\bstatic\s+([A-Za-z_][A-Za-z0-9_]*)\s*:\s*([^=;]+?)\s*=\s*(.+?)\s*;", line)
        if not m:
            return None
        return (m.group(1), m.group(2).strip(), m.group(3).strip())

    def _extract_array_len_from_type(self, type_str: str) -> Optional[str]:
        if not type_str:
            return None
        m = re.search(r"\[\s*[^;\]]+\s*;\s*([0-9_]+)\s*(?:usize)?\s*\]", type_str)
        return m.group(1) if m else None

    def _get_c_source_searcher(self):
        """
        获取（并缓存）C 源码搜索器。

        说明：
        - 基于 compile_commands.json 的索引构建可能很重（数万源文件/数千 include 目录）。
        - 在多轮修复中重复初始化会造成明显卡顿/超时。
        """
        if self._c_source_searcher is not None:
            return self._c_source_searcher
        if self._c_source_searcher_init_failed:
            return None
        if not PREDEFINES_AVAILABLE:
            self._c_source_searcher_init_failed = True
            return None
        try:
            from config.predefines import CSourceSearcher
        except Exception:
            self._c_source_searcher_init_failed = True
            return None

        try:
            if self.compile_commands_parser:
                self._c_source_searcher = CSourceSearcher(
                    self.c_source_dirs,
                    self.include_dirs,
                    compile_commands_parser=self.compile_commands_parser,
                )
                logger.info("已启用 C 源码搜索器（使用 compile_commands.json 精准搜索）")
            elif self.c_source_dirs:
                self._c_source_searcher = CSourceSearcher(self.c_source_dirs, self.include_dirs)
                logger.info(f"已启用 C 源码搜索器（传统搜索），目录: {[str(d) for d in self.c_source_dirs]}")
        except Exception as e:
            self._c_source_searcher_init_failed = True
            logger.debug(f"CSourceSearcher 初始化失败: {e}")
            return None

        return self._c_source_searcher
        
    def get_compile_errors_json(self) -> Optional[List[CompileError]]:
        """
        使用 cargo check --message-format=json 获取结构化错误
        
        Returns:
            编译错误列表
        """
        # 设置 RUSTFLAGS 抑制无害警告
        env = os.environ.copy()
        env["RUSTFLAGS"] = "-A unused_imports -A dead_code -A unused_variables -A unused_mut"

        try:
            timeout_s = int(os.environ.get("C2R_CARGO_CHECK_TIMEOUT", "").strip() or "300")
        except Exception:
            timeout_s = 300
        timeout_s = max(60, timeout_s)

        try:
            result = subprocess.run(
                ["cargo", "check", "--message-format=json"],
                cwd=self.skeleton_dir,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                env=env,
            )
        except subprocess.TimeoutExpired as e:
            logger.error(f"cargo check(json) 超时({timeout_s}s): {e}")
            return None
        except Exception as e:
            logger.error(f"获取编译错误失败: {e}")
            return None

        errors: List[CompileError] = []
        for line in result.stdout.split("\n"):
            if not line.strip():
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if msg.get("reason") != "compiler-message":
                continue
            message_data = msg.get("message", {})
            if message_data.get("level") != "error":
                continue
            error = self._parse_error_message(message_data)
            if error:
                errors.append(error)

        if result.returncode == 0:
            return errors

        # If cargo failed but emitted no structured errors, avoid false "no errors" success.
        if errors:
            return errors

        # Fallback: attempt to recover at least some locations from the plain output.
        plain = self._parse_plain_cargo_errors((result.stderr or "") + "\n" + (result.stdout or ""))
        if plain:
            return plain

        logger.error("cargo check 失败，但未解析到任何错误（JSON/Plain 都为空）")
        return None

    def _parse_plain_cargo_errors(self, text: str) -> List[CompileError]:
        """
        Best-effort fallback when `--message-format=json` yields no parseable errors.

        This must never report "no errors" on a failing build; it prefers returning a few
        coarse errors over returning an empty list.
        """
        if not text:
            return []

        err_header = re.compile(r"^error(?:\\[(?P<code>E\\d{4})\\])?:\\s*(?P<msg>.*)$")
        span = re.compile(r"^\\s*-->\\s+(?P<file>[^:]+):(?P<line>\\d+):(?P<col>\\d+)\\s*$")

        lines = text.splitlines()
        out: List[CompileError] = []
        for i, line in enumerate(lines):
            m = err_header.match(line.strip())
            if not m:
                continue
            code = (m.group("code") or "E0000").strip()
            msg = (m.group("msg") or "").strip() or "cargo check failed"

            file_path = ""
            line_no = 1
            col = 1
            rendered = "\n".join(lines[i : min(len(lines), i + 30)])

            for j in range(i + 1, min(len(lines), i + 30)):
                sm = span.match(lines[j].rstrip("\n"))
                if not sm:
                    continue
                file_path = sm.group("file")
                try:
                    line_no = int(sm.group("line"))
                except Exception:
                    line_no = 1
                try:
                    col = int(sm.group("col"))
                except Exception:
                    col = 1
                break

            # Keep only src/* by default (same as JSON parser), otherwise it becomes unfixable noise.
            if file_path and not file_path.startswith("src/"):
                continue

            if not file_path:
                # Still return a coarse sentinel so the caller doesn't misinterpret this as success.
                file_path = "src/main.rs"

            out.append(
                CompileError(
                    file_path=file_path,
                    line_number=line_no,
                    column=col,
                    error_code=code,
                    message=msg,
                    rendered=rendered,
                )
            )

        return out
    
    def _parse_error_message(self, message: Dict) -> Optional[CompileError]:
        """解析 JSON 格式的错误消息"""
        try:
            # 提取错误代码
            code = message.get('code', {})
            error_code = code.get('code', 'E0000') if code else 'E0000'
            
            # 提取主要位置信息
            spans = message.get('spans', [])
            if not spans:
                return None
            
            # 找到主要的 span（is_primary=True 或第一个）
            primary_span = None
            for span in spans:
                if span.get('is_primary', False):
                    primary_span = span
                    break
            if not primary_span:
                primary_span = spans[0]
            
            file_path = primary_span.get('file_name', '')
            line_start = primary_span.get('line_start', 0)
            column_start = primary_span.get('column_start', 0)
            
            # 过滤非源文件错误
            if not file_path.startswith('src/'):
                return None
            
            # 提取建议（如果有）
            suggestion = None
            children = message.get('children', [])
            for child in children:
                if child.get('level') == 'help':
                    suggestion = child.get('message', '')
                    break
            
            return CompileError(
                file_path=file_path,
                line_number=line_start,
                column=column_start,
                error_code=error_code,
                message=message.get('message', ''),
                rendered=message.get('rendered', ''),
                suggestion=suggestion
            )
            
        except Exception as e:
            logger.debug(f"解析错误消息失败: {e}")
            return None
    
    def build_fix_context(self, error: CompileError, context_lines: int = 5) -> Optional[FixContext]:
        """
        构建修复上下文
        
        Args:
            error: 编译错误
            context_lines: 上下文行数
            
        Returns:
            修复上下文
        """
        file_path = self.src_dir.parent / error.file_path
        if not file_path.exists():
            logger.warning(f"文件不存在: {file_path}")
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # 行号是 1-based
            line_idx = error.line_number - 1
            if line_idx < 0 or line_idx >= len(lines):
                return None
            
            source_line = lines[line_idx].rstrip('\n')
            
            # 获取上下文
            start_idx = max(0, line_idx - context_lines)
            end_idx = min(len(lines), line_idx + context_lines + 1)
            
            context_before = [l.rstrip('\n') for l in lines[start_idx:line_idx]]
            context_after = [l.rstrip('\n') for l in lines[line_idx + 1:end_idx]]
            
            # 获取相关类型定义
            related_types = self._extract_related_types(error, source_line)
            
            return FixContext(
                error=error,
                source_line=source_line,
                context_before=context_before,
                context_after=context_after,
                related_types=related_types
            )
            
        except Exception as e:
            logger.error(f"构建修复上下文失败: {e}")
            return None
    
    def _extract_related_types(self, error: CompileError, source_line: str) -> Dict[str, str]:
        """从 types.rs 和 globals.rs 提取相关类型定义"""
        related_types = {}
        
        # 从错误消息和源代码中提取类型名
        type_names = set()
        
        # 从错误消息中提取类型
        type_patterns = [
            r"expected `([^`]+)`",
            r"found `([^`]+)`",
            r"type `([^`]+)`",
            r"crate::types::(\w+)",
        ]
        for pattern in type_patterns:
            for match in re.finditer(pattern, error.message + ' ' + error.rendered):
                type_name = match.group(1)
                # 提取基本类型名（去除指针、泛型等）
                base_type = re.sub(r'[*&\[\]<>]', '', type_name).split('::')[-1].strip()
                if base_type and len(base_type) > 1:
                    type_names.add(base_type)
        
        # 从源代码行中提取类型
        for match in re.finditer(r'crate::types::(\w+)', source_line):
            type_names.add(match.group(1))
        
        # 读取 types.rs
        types_rs = self.src_dir / "types.rs"
        if types_rs.exists():
            try:
                with open(types_rs, 'r', encoding='utf-8') as f:
                    types_content = f.read()
                
                for type_name in type_names:
                    # 查找 struct/enum/type 定义
                    patterns = [
                        rf'(pub\s+struct\s+{type_name}\s*\{{[^}}]*\}})',
                        rf'(pub\s+struct\s+{type_name}\s*;)',
                        rf'(pub\s+type\s+{type_name}\s*=[^;]+;)',
                        rf'(pub\s+enum\s+{type_name}\s*\{{[^}}]*\}})',
                    ]
                    for pattern in patterns:
                        match = re.search(pattern, types_content, re.DOTALL)
                        if match:
                            related_types[type_name] = match.group(1)
                            break
            except Exception as e:
                logger.debug(f"读取 types.rs 失败: {e}")
        
        # 读取 globals.rs
        globals_rs = self.src_dir / "globals.rs"
        if globals_rs.exists() and 'globals' in source_line:
            try:
                with open(globals_rs, 'r', encoding='utf-8') as f:
                    globals_content = f.read()
                
                # 查找相关变量定义
                for match in re.finditer(r'(pub\s+static\s+(?:mut\s+)?(\w+)\s*:[^;]+;)', globals_content):
                    var_def, var_name = match.groups()
                    if var_name in source_line or var_name in error.message:
                        related_types[f"global_{var_name}"] = var_def
            except Exception as e:
                logger.debug(f"读取 globals.rs 失败: {e}")
        
        return related_types

    # -------------------------------------------------------------------------
    # Structural fixes (non-LLM, project-wide)
    # -------------------------------------------------------------------------

    def _ensure_compat_layer_files(self) -> int:
        """Ensure `src/compat.rs` and `src/compatibility.rs` exist and are minimally valid.

        Deterministically fixes noisy framework-level errors like:
        - E0432: unresolved import `crate::compat`
        - E0433: could not find `compat` in the crate root
        """
        changes = 0
        self.src_dir.mkdir(parents=True, exist_ok=True)

        compat_path = self.src_dir / "compat.rs"
        compat_template = "\n".join(
            [
                "//! Compatibility / Fallback Layer",
                "//!",
                "//! Auto-generated to keep the translated project compiling.",
                "",
                "#![allow(dead_code)]",
                "#![allow(unused)]",
                "#![allow(non_snake_case)]",
                "#![allow(non_camel_case_types)]",
                "",
                "/// Common C FFI types (c_int, c_char, c_void, ...).",
                "pub use core::ffi::*;",
                "",
                "/// Minimal FFI prelude (for legacy skeletons that import `crate::compat::ffi::*`).",
                "pub mod ffi {",
                "    pub use core::ffi::*;",
                "}",
                "",
                "// === C2R_COMPAT_PLACEHOLDERS_BEGIN ===",
                "// (auto-appended placeholders will be inserted here)",
                "// === C2R_COMPAT_PLACEHOLDERS_END ===",
                "",
                "// === C2R_ACCESSOR_SHIMS_BEGIN ===",
                "// (auto-appended accessor shim declarations will be inserted here)",
                "// === C2R_ACCESSOR_SHIMS_END ===",
                "",
            ]
        )

        if not compat_path.exists():
            compat_path.write_text(compat_template, encoding="utf-8")
            changes += 1
        else:
            try:
                existing = compat_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                existing = ""
            updated = existing
            if "pub use core::ffi::*;" not in updated:
                updated = (
                    updated.rstrip()
                    + "\n\n/// Common C FFI types (c_int, c_char, c_void, ...).\n"
                    + "pub use core::ffi::*;\n"
                )
            if "pub mod ffi" not in updated:
                updated = (
                    updated.rstrip()
                    + "\n\n/// Minimal FFI prelude (for legacy skeletons that import `crate::compat::ffi::*`).\n"
                    + "pub mod ffi {\n    pub use core::ffi::*;\n}\n"
                )
            if "C2R_COMPAT_PLACEHOLDERS_BEGIN" not in updated:
                updated = (
                    updated.rstrip()
                    + "\n\n// === C2R_COMPAT_PLACEHOLDERS_BEGIN ===\n"
                    + "// (auto-appended placeholders will be inserted here)\n"
                    + "// === C2R_COMPAT_PLACEHOLDERS_END ===\n"
                )
            if "C2R_ACCESSOR_SHIMS_BEGIN" not in updated:
                updated = (
                    updated.rstrip()
                    + "\n\n// === C2R_ACCESSOR_SHIMS_BEGIN ===\n"
                    + "// (auto-appended accessor shim declarations will be inserted here)\n"
                    + "// === C2R_ACCESSOR_SHIMS_END ===\n"
                )
            if updated != existing and updated:
                compat_path.write_text(updated, encoding="utf-8")
                changes += 1

        compatibility_path = self.src_dir / "compatibility.rs"
        if not compatibility_path.exists():
            compatibility_path.write_text(
                "\n".join(
                    [
                        "//! Backward-compatibility alias for legacy skeleton outputs.",
                        "//!",
                        "//! Prefer `crate::compat` for new code.",
                        "",
                        "#![allow(dead_code)]",
                        "#![allow(unused)]",
                        "",
                        "pub use crate::compat::*;",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            changes += 1

        return changes

    def _ensure_root_module_decls(self, modules: List[str]) -> int:
        """Ensure crate root declares the given modules (main.rs or lib.rs)."""
        root = None
        main_rs = self.src_dir / "main.rs"
        lib_rs = self.src_dir / "lib.rs"
        if main_rs.exists():
            root = main_rs
        elif lib_rs.exists():
            root = lib_rs
        if root is None:
            return 0

        try:
            text = root.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return 0

        def has_decl(mod: str) -> bool:
            pattern = re.compile(rf"^\s*(pub\s+)?mod\s+{re.escape(mod)}\s*;", re.M)
            return bool(pattern.search(text))

        lines = text.splitlines()

        # Insert before `fn main` if present; otherwise append.
        insert_at = None
        for idx, line in enumerate(lines):
            if re.match(r"^\s*fn\s+main\s*\(", line):
                insert_at = idx
                break
        if insert_at is None:
            insert_at = len(lines)

        inserted = False
        for mod in modules:
            if has_decl(mod):
                continue
            lines.insert(insert_at, f"pub mod {mod};")
            insert_at += 1
            inserted = True

        if not inserted:
            return 0

        new_text = "\n".join(lines) + ("\n" if text.endswith("\n") else "")
        root.write_text(new_text, encoding="utf-8")
        return 1

    def fix_missing_compat_modules(self, errors: List[CompileError]) -> int:
        """Fix missing `crate::compat` / `crate::compatibility` modules deterministically."""
        merged = ""
        for e in errors:
            if e.error_code not in ("E0432", "E0433"):
                continue
            merged += (e.message or "") + "\n" + (e.rendered or "") + "\n"

        if (
            "crate::compat" not in merged
            and "crate::compatibility" not in merged
            and "`compat` in the crate root" not in merged
            and "`compatibility` in the crate root" not in merged
        ):
            return 0

        changes = 0
        changes += self._ensure_compat_layer_files()
        changes += self._ensure_root_module_decls(["compat", "compatibility"])
        return changes
    
    def fix_missing_types_and_consts(self, errors: List[CompileError]) -> int:
        """
        修复缺失的类型和常量（直接追加到 types.rs）
        
        ★★★ 增强版：使用诊断分析器搜索 C 源码中的原始定义 ★★★
        - 如果找到真实定义：使用真实定义
        - 如果找不到：使用占位符，但记录失败原因
        
        Returns:
            修复的错误数量
        """
        types_rs = self.src_dir / "types.rs"
        if not types_rs.exists():
            return 0
        
        # 读取现有内容
        with open(types_rs, 'r', encoding='utf-8') as f:
            content = f.read()
        
        additions = []
        fixed_count = 0
        
        added_types = set()  # 避免重复添加
        added_consts = set()
        
        # 创建（并缓存）C 源码搜索器
        c_searcher = self._get_c_source_searcher()
        
        for error in errors:
            # E0412: cannot find type (支持多种格式)
            if error.error_code == 'E0412':
                # 尝试多种匹配模式
                patterns = [
                    r"cannot find type `(\w+)`",
                    r"type `(\w+)` not found",
                ]
                type_name = None
                for pattern in patterns:
                    match = re.search(pattern, error.message)
                    if match:
                        type_name = match.group(1)
                        break
                
                if type_name and type_name not in added_types:
                    # 使用正则精确匹配，避免匹配到 file_handle 之类的类型
                    type_defined = bool(re.search(rf'\bstruct\s+{type_name}\s*[{{\s]', content)) or \
                                   bool(re.search(rf'\btype\s+{type_name}\s*=', content))
                    if not type_defined:
                        type_defs = self._get_type_definitions()
                        if type_name in type_defs:
                            additions.append(f"// Auto-added for {error.file_path}:{error.line_number}")
                            additions.append(type_defs[type_name])
                            # 自适应学习：记录使用的类型（非占位符）
                            if PREDEFINES_AVAILABLE:
                                get_predefine_manager().learn_type(
                                    type_name, type_defs[type_name], "used_in_fix",
                                    is_placeholder=False
                                )
                        else:
                            # 尝试在 C 源码中搜索原始定义
                            c_result = None
                            if c_searcher:
                                c_result = c_searcher.find_type_definition(type_name)
                            
                            if c_result and c_result.get('found'):
                                # 找到了原始定义（但这里仍然使用占位符，避免引入不完整/错误的结构体字段）
                                placeholder = f"#[repr(C)]\npub struct {type_name} {{ _opaque: [u8; 0] }}"
                                additions.append(
                                    f"// Type found in C source: {c_result.get('file', '')}:{c_result.get('line', 0)}"
                                )
                                additions.append(f"// Original: {c_result.get('definition', '')[:80]}...")
                                additions.append(placeholder)

                                if PREDEFINES_AVAILABLE:
                                    get_predefine_manager().learn_type(
                                        type_name, placeholder, "compilation_error",
                                        is_placeholder=True,
                                        c_source_file=c_result.get('file', ''),
                                        c_source_line=c_result.get('line', 0),
                                        original_c_definition=c_result.get('definition', ''),
                                        failure_reason="找到原始定义但需要手动转换",
                                        diagnostic_notes=c_result.get('notes', [])
                                    )
                            else:
                                # 没找到原始定义
                                placeholder = f"#[repr(C)]\npub struct {type_name} {{ _opaque: [u8; 0] }}"
                                additions.append(f"// ⚠️ Auto-generated placeholder for {type_name}")
                                additions.append(f"// ⚠️ REASON: Cannot find original definition in C source")
                                additions.append(placeholder)

                                if PREDEFINES_AVAILABLE:
                                    get_predefine_manager().learn_type(
                                        type_name, placeholder, "compilation_error",
                                        is_placeholder=True,
                                        failure_reason="未能在 C 源码中找到原始定义",
                                        diagnostic_notes=[
                                            "可能原因：1) 来自未包含的头文件",
                                            "         2) 是宏定义展开后的类型",
                                            "         3) 是条件编译排除的代码",
                                            f"请手动检查类型 '{type_name}' 的真实定义"
                                        ]
                                    )
                        added_types.add(type_name)
                        fixed_count += 1
            
            # E0425: cannot find value
            elif error.error_code == 'E0425':
                patterns = [
                    r"cannot find value `(\w+)`",
                    r"value `(\w+)` not found",
                ]
                const_name = None
                for pattern in patterns:
                    match = re.search(pattern, error.message)
                    if match:
                        const_name = match.group(1)
                        break
                
                if const_name and const_name not in added_consts:
                    # 精确匹配常量名
                    const_defined = bool(re.search(rf'\bconst\s+{const_name}\s*:', content))
                    if not const_defined:
                        const_defs = self._get_const_definitions()
                        if const_name in const_defs:
                            additions.append(f"// Auto-added constant")
                            additions.append(const_defs[const_name])
                            added_consts.add(const_name)
                            fixed_count += 1
                            # 自适应学习：记录使用的常量
                            if PREDEFINES_AVAILABLE:
                                # 从定义中提取类型和值
                                match = re.search(r': (\w+) = (.+);', const_defs[const_name])
                                if match:
                                    get_predefine_manager().learn_constant(
                                        const_name, match.group(1), match.group(2), "used_in_fix",
                                        is_placeholder=False
                                    )
                        else:
                            # 尝试在 C 源码中搜索原始定义
                            c_result = None
                            if c_searcher:
                                c_result = c_searcher.find_constant_definition(const_name)
                            
                            if c_result and c_result.get('found'):
                                # 找到了原始定义 - 尝试提取值
                                definition = c_result.get('definition', '')
                                extracted_value = self._extract_const_value_from_c(const_name, definition)
                                
                                if extracted_value:
                                    rust_type, value = extracted_value
                                    const_def = f"pub const {const_name}: {rust_type} = {value};"
                                    additions.append(f"// Constant found in C source: {c_result.get('file', '')}:{c_result.get('line', 0)}")
                                    additions.append(f"// Original: {definition[:60]}")
                                    additions.append(const_def)
                                    
                                    if PREDEFINES_AVAILABLE:
                                        get_predefine_manager().learn_constant(
                                            const_name, rust_type, value, "compilation_error",
                                            is_placeholder=False,  # 提取到了真实值
                                            c_source_file=c_result.get('file', ''),
                                            c_source_line=c_result.get('line', 0),
                                            original_c_definition=definition,
                                            diagnostic_notes=c_result.get('notes', [])
                                        )
                                else:
                                    # 找到定义但无法提取值
                                    rust_type, default_value = get_predefine_manager().infer_constant_type(const_name) if PREDEFINES_AVAILABLE else ('i32', '0')
                                    const_def = f"pub const {const_name}: {rust_type} = {default_value};"
                                    additions.append(f"// ⚠️ Found in C source but cannot extract value")
                                    additions.append(f"// ⚠️ Original: {definition[:60]}")
                                    additions.append(const_def)
                                    
                                    if PREDEFINES_AVAILABLE:
                                        get_predefine_manager().learn_constant(
                                            const_name, rust_type, default_value, "compilation_error",
                                            is_placeholder=True,
                                            c_source_file=c_result.get('file', ''),
                                            c_source_line=c_result.get('line', 0),
                                            original_c_definition=definition,
                                            failure_reason="找到定义但无法提取值（可能是复杂表达式）",
                                            diagnostic_notes=c_result.get('notes', []) + [
                                                f"原始定义: {definition[:100]}"
                                            ]
                                        )
                            else:
                                # 没找到原始定义
                                if PREDEFINES_AVAILABLE:
                                    rust_type, value = get_predefine_manager().infer_constant_type(const_name)
                                else:
                                    rust_type, value = 'i32', '0'

                                const_def = f"pub const {const_name}: {rust_type} = {value};"
                                additions.append(f"// ⚠️ Auto-generated constant placeholder")
                                additions.append(f"// ⚠️ REASON: Cannot find original definition in C source")
                                additions.append(const_def)

                                if PREDEFINES_AVAILABLE:
                                    get_predefine_manager().learn_constant(
                                        const_name, rust_type, value, "compilation_error",
                                        is_placeholder=True,
                                        failure_reason="未能在 C 源码中找到原始定义",
                                        diagnostic_notes=[
                                            "可能原因：1) 来自未包含的头文件",
                                            "         2) 是 #define 宏定义",
                                            "         3) 是条件编译排除的代码",
                                            f"请手动检查常量 '{const_name}' 的真实值"
                                        ]
                                    )
                            added_consts.add(const_name)
                            fixed_count += 1
        
        if additions:
            with open(types_rs, 'a', encoding='utf-8') as f:
                f.write("\n\n// ============================================================\n")
                f.write("// Auto-added by LLM Precise Fixer (with diagnostic info)\n")
                f.write("// ============================================================\n\n")
                f.write('\n'.join(additions))
                f.write('\n')
        
        return fixed_count
    
    def _extract_const_value_from_c(self, const_name: str, c_definition: str) -> Optional[Tuple[str, str]]:
        """
        从 C 定义中提取常量值
        
        Returns:
            (rust_type, value) 或 None
        """
        # #define NAME value
        define_match = re.search(
            rf'#\s*define\s+{re.escape(const_name)}\s+(-?\d+|0x[0-9a-fA-F]+)',
            c_definition
        )
        if define_match:
            value = define_match.group(1)
            rust_type = 'u32' if value.startswith('0x') else 'i32'
            return (rust_type, value)
        
        # #define NAME (value)
        define_paren_match = re.search(
            rf'#\s*define\s+{re.escape(const_name)}\s+\((-?\d+)\)',
            c_definition
        )
        if define_paren_match:
            return ('i32', define_paren_match.group(1))
        
        # const/enum value = xxx
        const_match = re.search(
            rf'{re.escape(const_name)}\s*=\s*(-?\d+|0x[0-9a-fA-F]+)',
            c_definition
        )
        if const_match:
            value = const_match.group(1)
            rust_type = 'u32' if value.startswith('0x') else 'i32'
            return (rust_type, value)
        
        return None
    
    def try_rule_fix(self, context: FixContext) -> Optional[str]:
        """
        尝试使用规则修复（比 LLM 更快更可靠）
        
        Args:
            context: 修复上下文
            
        Returns:
            修复后的代码行，无法修复返回 None
        """
        error_msg = context.error.message + ' ' + context.error.rendered
        
        # 1. 尝试类型不匹配修复
        for pattern, fix_fn in self.RULE_FIXES.items():
            if re.search(pattern, error_msg, re.IGNORECASE):
                try:
                    fixed_line = fix_fn(context.source_line)
                    if fixed_line != context.source_line:
                        return fixed_line
                except Exception as e:
                    logger.debug(f"规则修复失败: {e}")
        
        # 2. 尝试 E0425: cannot find value 修复
        if context.error.error_code == 'E0425':
            match = re.search(r"cannot find value `(\w+)`", context.error.message)
            if match:
                value_name = match.group(1)
                parts = self._extract_static_decl_parts(context.source_line)
                if parts:
                    _, decl_type, _ = parts
                    # Prefer: if this looks like an enum assignment, use the enum variant (strongly typed).
                    # Example: `AudioFormat = AUDIO_FORMAT_TYPE_PCM_16_BIT;` ->
                    #          `AudioFormat::AUDIO_FORMAT_TYPE_PCM_16_BIT`
                    enum_match = re.match(r"(?:crate::types::)?([A-Za-z_][A-Za-z0-9_]*)\b", (decl_type or "").strip())
                    enum_name = enum_match.group(1) if enum_match else None
                    if enum_name and self._types_enum_has_variant(enum_name, value_name):
                        fixed_line = re.sub(
                            rf"=\s*{re.escape(value_name)}\s*;",
                            f"= crate::types::{enum_name}::{value_name};",
                            context.source_line,
                        )
                        if fixed_line != context.source_line:
                            return fixed_line

                # Prefer: qualify to `crate::types::` if the constant exists there.
                if self._types_has_const(value_name):
                    fixed_line = re.sub(
                        rf"=\s*{re.escape(value_name)}\s*;",
                        f"= crate::types::{value_name};",
                        context.source_line,
                    )
                    if fixed_line != context.source_line:
                        return fixed_line

                # Last resort: replace with a safe placeholder (may lose semantics).
                if value_name in self.VALUE_REPLACEMENT_RULES:
                    replacement = self.VALUE_REPLACEMENT_RULES[value_name]
                    fixed_line = re.sub(
                        rf"=\s*{re.escape(value_name)}\s*;",
                        f"= {replacement};",
                        context.source_line,
                    )
                    if fixed_line != context.source_line:
                        return fixed_line
        
        # 3. 尝试 E0308: mismatched types (自定义类型初始化)
        if context.error.error_code == 'E0308':
            combined_msg = context.error.message + ' ' + context.error.rendered
            parts = self._extract_static_decl_parts(context.source_line)

            # 3.1 `&str` assigned to an array (common from libclang global var extraction)
            if parts and re.search(r"expected `\[[^`]+;\s*\d+\]`, found `&str`", combined_msg):
                _, decl_type, _ = parts
                n = self._extract_array_len_from_type(decl_type)
                if n:
                    fixed_line = re.sub(r"=\s*[^;]+;", f"= [0; {n}];", context.source_line)
                    if fixed_line != context.source_line:
                        return fixed_line

            # 3.2 Enum expected but got an integer constant: try convert CONST -> Enum::CONST if it exists.
            if parts:
                _, decl_type, rhs = parts
                enum_match = re.match(r"(?:crate::types::)?([A-Za-z_][A-Za-z0-9_]*)\b", (decl_type or "").strip())
                enum_name = enum_match.group(1) if enum_match else None
                if enum_name and self._types_enum_has_variant(enum_name, rhs.replace("crate::types::", "")):
                    variant = rhs.replace("crate::types::", "")
                    fixed_line = re.sub(r"=\s*[^;]+;", f"= crate::types::{enum_name}::{variant};", context.source_line)
                    if fixed_line != context.source_line:
                        return fixed_line
                # Special-case: AudioFormat constants in types.rs are `i32` consts derived from enum variants.
                # Map `crate::types::AUDIO_FORMAT_TYPE_*` -> `crate::types::AudioFormat::AUDIO_FORMAT_TYPE_*`.
                if enum_name == "AudioFormat":
                    m = re.search(r"\bAUDIO_FORMAT_TYPE_[A-Za-z0-9_]+\b", rhs)
                    if m:
                        variant = m.group(0)
                        if self._types_enum_has_variant("AudioFormat", variant):
                            fixed_line = re.sub(r"=\s*[^;]+;", f"= crate::types::AudioFormat::{variant};", context.source_line)
                            if fixed_line != context.source_line:
                                return fixed_line

            # 3.3 Custom (non-enum) type expected but got integer/&str/bool -> use zeroed() for compilation.
            if "static" in context.source_line and re.search(r"expected `[^`]+`.*found (integer|`&str`|`bool`)", combined_msg):
                # Avoid `zeroed()` for enums (it can cause E0080). Prefer a real variant if possible.
                if parts:
                    _, decl_type, _ = parts
                    enum_match = re.match(r"(?:crate::types::)?([A-Za-z_][A-Za-z0-9_]*)\b", (decl_type or "").strip())
                    enum_name = enum_match.group(1) if enum_match else None
                    first = self._types_enum_first_variant(enum_name) if enum_name else None
                    if first and enum_name:
                        fixed_line = re.sub(r"=\s*[^;]+;", f"= crate::types::{enum_name}::{first};", context.source_line)
                        if fixed_line != context.source_line:
                            return fixed_line
                fixed_line = re.sub(r"=\s*[^;]+;", "= unsafe { std::mem::zeroed() };", context.source_line)
                if fixed_line != context.source_line:
                    return fixed_line

        # 3.4 E0560: struct has no field named ... (usually from an incorrect struct literal initializer)
        if context.error.error_code == "E0560":
            if "static" in context.source_line and "{" in context.source_line and "}" in context.source_line:
                fixed_line = re.sub(r"=\s*[^;]+;", "= unsafe { std::mem::zeroed() };", context.source_line)
                if fixed_line != context.source_line:
                    return fixed_line

        # 3.5 E0080: attempted to zero-initialize an invalid type (typically enums)
        if context.error.error_code == "E0080":
            # Example: "attempted to zero-initialize type `AudioFormat`, which is invalid"
            m = re.search(r"attempted to zero-initialize type `([^`]+)`", context.error.message + " " + context.error.rendered)
            if m and "zeroed" in context.source_line:
                type_name = m.group(1).split("::")[-1]
                first = self._types_enum_first_variant(type_name)
                if first:
                    fixed_line = re.sub(r"=\s*[^;]+;", f"= crate::types::{type_name}::{first};", context.source_line)
                    if fixed_line != context.source_line:
                        return fixed_line
        
        # 4. 尝试 E0530: function parameters cannot shadow statics
        if context.error.error_code == 'E0530':
            # 参数名与静态变量冲突，重命名参数
            match = re.search(r'function parameters cannot shadow statics', context.error.message)
            if match:
                # 从错误信息或源代码中提取参数名
                param_match = re.search(r'fn\s+\w+\s*\(([^)]+)\)', context.source_line)
                if param_match:
                    params = param_match.group(1)
                    # 给冲突的参数添加下划线前缀
                    # 从 rendered 错误信息中找到冲突的参数名
                    shadow_match = re.search(r'the static `(\w+)` is imported', context.error.rendered)
                    if shadow_match:
                        shadow_name = shadow_match.group(1)
                        # 在参数中找到并重命名
                        new_params = re.sub(rf'\b{shadow_name}\b(?=\s*:)', f'_{shadow_name}', params)
                        if new_params != params:
                            fixed_line = context.source_line.replace(params, new_params)
                            return fixed_line
        
        return None
    
    def fix_single_error(self, context: FixContext) -> Optional[str]:
        """
        使用 LLM 修复单个错误（先尝试规则修复）
        
        Args:
            context: 修复上下文
            
        Returns:
            修复后的代码行，失败返回 None
        """
        # 先尝试规则修复
        rule_fix = self.try_rule_fix(context)
        if rule_fix:
            return rule_fix
        
        if not self.llm_client:
            logger.warning("LLM 客户端未配置，且规则修复失败")
            return None
        
        # 构建类型定义部分
        type_defs = ""
        if context.related_types:
            type_defs = "## Related Type Definitions\n```rust\n"
            for name, definition in context.related_types.items():
                type_defs += f"// {name}\n{definition}\n\n"
            type_defs += "```\n"
        
        # 构建提示词
        prompt = PRECISE_FIX_PROMPT.format(
            error_code=context.error.error_code,
            message=context.error.message,
            file_path=context.error.file_path,
            line_number=context.error.line_number,
            context_before='\n'.join(context.context_before),
            source_line=context.source_line,
            context_after='\n'.join(context.context_after),
            type_definitions=type_defs,
            original_c_code=""  # 暂不提供 C 代码
        )
        
        try:
            response = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a Rust FFI expert. Output ONLY the fixed line of code, nothing else."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=500
            )
            
            fixed_line = response.choices[0].message.content.strip()
            
            # 清理可能的 markdown 标记
            fixed_line = re.sub(r'^```\w*\s*', '', fixed_line)
            fixed_line = re.sub(r'\s*```$', '', fixed_line)
            fixed_line = fixed_line.strip()
            
            # 如果返回了多行，只取第一行
            if '\n' in fixed_line:
                lines = [l.strip() for l in fixed_line.split('\n') if l.strip()]
                # 找到最相关的行
                for line in lines:
                    if any(kw in line for kw in ['fn ', 'let ', 'pub ', 'static ', ':', '=']):
                        fixed_line = line
                        break
                else:
                    fixed_line = lines[0] if lines else fixed_line
            
            return fixed_line
            
        except Exception as e:
            logger.error(f"LLM 修复失败: {e}")
            return None
    
    def fix_file_errors(self, file_path: str, errors: List[CompileError]) -> bool:
        """
        修复单个文件中的所有错误
        
        Args:
            file_path: 文件路径
            errors: 该文件的所有错误
            
        Returns:
            是否成功修复
        """
        if not self.llm_client:
            return False
        
        full_path = self.src_dir.parent / file_path
        if not full_path.exists():
            return False
        
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # 收集相关类型定义
            all_related_types = {}
            for error in errors:
                context = self.build_fix_context(error)
                if context:
                    all_related_types.update(context.related_types)
            
            # 构建错误列表
            error_list = ""
            for i, error in enumerate(errors, 1):
                error_list += f"{i}. Line {error.line_number}: [{error.error_code}] {error.message}\n"
            
            # 构建类型定义
            type_defs = ""
            if all_related_types:
                type_defs = "## Related Type Definitions\n```rust\n"
                for name, definition in all_related_types.items():
                    type_defs += f"// {name}\n{definition}\n\n"
                type_defs += "```\n"
            
            # 读取完整文件内容
            file_content = ''.join(lines)
            
            # 构建提示词
            prompt = PRECISE_FIX_PROMPT_MULTI.format(
                error_list=error_list,
                file_content=file_content,
                type_definitions=type_defs
            )
            
            response = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a Rust FFI expert. Output ONLY the fixed Rust code, no explanations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=8000
            )
            
            fixed_content = response.choices[0].message.content.strip()
            
            # 清理 markdown
            fixed_content = re.sub(r'^```\w*\s*\n?', '', fixed_content)
            fixed_content = re.sub(r'\n?```\s*$', '', fixed_content)
            
            # 验证修复内容看起来像 Rust 代码
            if not self._looks_like_rust_code(fixed_content):
                logger.warning(f"LLM 返回的内容不像 Rust 代码")
                return False
            
            # 写回文件
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(fixed_content)
            
            return True
            
        except Exception as e:
            logger.error(f"修复文件失败: {e}")
            return False
    
    def _looks_like_rust_code(self, content: str) -> bool:
        """检查内容是否看起来像 Rust 代码"""
        rust_indicators = [
            r'\bfn\s+\w+',
            r'\bpub\s+(fn|struct|enum|type|const|static)',
            r'\buse\s+',
            r'\blet\s+',
            r'\b(i32|u32|i64|u64|usize|isize|bool)\b',
            r'\*mut\s+',
            r'\*const\s+',
        ]
        
        for pattern in rust_indicators:
            if re.search(pattern, content):
                return True
        return False
    
    def apply_line_fix(self, file_path: str, line_number: int, fixed_line: str) -> bool:
        """
        应用单行修复
        
        Args:
            file_path: 文件路径
            line_number: 行号 (1-based)
            fixed_line: 修复后的代码行
            
        Returns:
            是否成功
        """
        full_path = self.src_dir.parent / file_path
        if not full_path.exists():
            return False
        
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            line_idx = line_number - 1
            if line_idx < 0 or line_idx >= len(lines):
                return False
            
            # 保持原有缩进
            original_indent = len(lines[line_idx]) - len(lines[line_idx].lstrip())
            new_line = ' ' * original_indent + fixed_line.lstrip() + '\n'
            
            lines[line_idx] = new_line
            
            with open(full_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            
            return True
            
        except Exception as e:
            logger.error(f"应用修复失败: {e}")
            return False

    def _try_fix_globals_pointer_mutex(self, full_path: Path) -> bool:
        """
        规则兜底：修复 globals.rs 中 `static Mutex<*mut/*const T>` 导致的 E0277（指针非 Send）。

        原因（Rust 新版本行为）：
        - `static X: Mutex<*mut T>` 需要 `*mut T: Send` 才能使 `Mutex<*mut T>: Sync`，但 raw pointer 默认不是 Send。

        修复策略（与 skeleton_builder 的新生成逻辑一致）：
        - 将 pointer globals 的存储改为 `Mutex<usize>`，通过 accessor 做 cast。
        """
        try:
            content = full_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return False

        lines = content.splitlines(True)  # keep line endings
        changed = False

        static_pat = re.compile(
            r"^(?P<indent>\s*)pub\s+static\s+(?P<cell>[A-Z0-9_]+)\s*:\s*Mutex<(?P<ptr>\*(?:mut|const)\s+[^>]+)>\s*=\s*Mutex::new\((?P<init>.*)\)\s*;\s*$"
        )
        get_pat = re.compile(
            r"^(?P<indent>\s*)pub\s+fn\s+(?P<name>get_[A-Za-z0-9_]+)\s*\(\)\s*->\s*(?P<ret>\*(?:mut|const)\s+.+?)\s*\{\s*\*(?P<cell>[A-Z0-9_]+)\.lock\(\)\.unwrap\(\)\s*\}\s*$"
        )
        set_pat = re.compile(
            r"^(?P<indent>\s*)pub\s+fn\s+(?P<name>set_[A-Za-z0-9_]+)\s*\(\s*v\s*:\s*(?P<arg>\*(?:mut|const)\s+.+?)\s*\)\s*\{\s*\*(?P<cell>[A-Z0-9_]+)\.lock\(\)\.unwrap\(\)\s*=\s*v\s*;\s*\}\s*$"
        )
        with_pat = re.compile(
            r"^(?P<indent>\s*)pub\s+fn\s+(?P<name>with_[A-Za-z0-9_]+)<R>\(f:\s*impl\s+FnOnce\(&mut\s+(?P<arg>\*(?:mut|const)\s+.+?)\)\s*->\s*R\)\s*->\s*R\s*\{\s*$"
        )

        i = 0
        while i < len(lines):
            m = static_pat.match(lines[i].rstrip("\n"))
            if not m:
                i += 1
                continue
            indent = m.group("indent") or ""
            cell = m.group("cell")
            ptr_ty = (m.group("ptr") or "").strip()

            # Rewrite static to Mutex<usize>
            lines[i] = f"{indent}// NOTE: pointer global stored as usize for `static Mutex` compatibility.\n"
            lines.insert(i + 1, f"{indent}pub static {cell}: Mutex<usize> = Mutex::new(0);\n")
            changed = True

            # Try to rewrite the following accessor trio (get/set/with) if they exist.
            j = i + 2
            # get_*
            if j < len(lines):
                gm = get_pat.match(lines[j].rstrip("\n"))
                if gm and gm.group("cell") == cell:
                    get_name = gm.group("name")
                    lines[j] = f"{indent}pub fn {get_name}() -> {ptr_ty} {{ *{cell}.lock().unwrap() as {ptr_ty} }}\n"
                    changed = True
                    j += 1
            # set_*
            if j < len(lines):
                sm = set_pat.match(lines[j].rstrip("\n"))
                if sm and sm.group("cell") == cell:
                    set_name = sm.group("name")
                    lines[j] = f"{indent}pub fn {set_name}(v: {ptr_ty}) {{ *{cell}.lock().unwrap() = v as usize; }}\n"
                    changed = True
                    j += 1
            # with_*
            if j < len(lines):
                wm = with_pat.match(lines[j].rstrip("\n"))
                if wm:
                    with_name = wm.group("name")
                    # Replace whole function body until matching closing brace.
                    # Find the next line that is just a closing brace at the same indent (best-effort).
                    k = j + 1
                    while k < len(lines):
                        if lines[k].strip() == "}":
                            break
                        k += 1
                    if k < len(lines):
                        new_block = [
                            f"{indent}pub fn {with_name}<R>(f: impl FnOnce(&mut {ptr_ty}) -> R) -> R {{\n",
                            f"{indent}    let mut guard = {cell}.lock().unwrap();\n",
                            f"{indent}    let mut tmp: {ptr_ty} = *guard as {ptr_ty};\n",
                            f"{indent}    let r = f(&mut tmp);\n",
                            f"{indent}    *guard = tmp as usize;\n",
                            f"{indent}    r\n",
                            f"{indent}}}\n",
                        ]
                        lines[j : k + 1] = new_block
                        changed = True
                        j = j + len(new_block)
            i = j

        if not changed:
            return False

        try:
            full_path.write_text("".join(lines), encoding="utf-8")
            return True
        except Exception:
            return False
    
    def fix_all_errors(self, max_rounds: int = 3) -> Tuple[bool, int, int]:
        """
        修复所有编译错误
        
        Args:
            max_rounds: 最大修复轮次
            
        Returns:
            (是否全部修复成功, 修复成功数, 总错误数)
        """
        def _sig(errs: List[CompileError]) -> Tuple[Tuple[str, str, str], ...]:
            # A stable-ish signature to detect no-progress loops.
            # Intentionally ignore column/suggestion to avoid churn.
            return tuple(
                sorted(
                    (
                        (e.file_path or "").strip(),
                        (e.error_code or "").strip(),
                        (e.message or "").strip(),
                    )
                    for e in (errs or [])
                )
            )

        applied_changes = 0  # "attempted" fixes (line/file writes), not necessarily verified.

        # Ensure the framework compat layer exists and is minimally valid.
        # This is idempotent and avoids wasting LLM cycles on pure scaffolding issues.
        try:
            self._ensure_compat_layer_files()
            self._ensure_root_module_decls(["compat", "compatibility"])
        except Exception:
            pass

        # Initial compile
        current_errors = self.get_compile_errors_json()
        if current_errors is None:
            print("  ⚠️ 无法获取编译错误（cargo check json 失败/超时），停止精准修复以避免误判。")
            return False, 0, 1
        if not current_errors:
            print("  ✅ 无编译错误，修复完成！")
            return True, 0, 0

        initial_total = len(current_errors)

        for round_num in range(1, max_rounds + 1):
            print(f"\n🔧 LLM 精准修复 - 第 {round_num}/{max_rounds} 轮")
            print(f"  📋 发现 {len(current_errors)} 个错误")

            before_sig = _sig(current_errors)
            applied_this_round = 0

            # Framework-level structural fix: missing compat layer is not a per-line issue.
            compat_fixed = self.fix_missing_compat_modules(current_errors)
            if compat_fixed > 0:
                print(f"  🧱 自动补全了 compat 模块层（{compat_fixed} 项，待编译验证）")
                applied_this_round += compat_fixed
                applied_changes += applied_this_round

                current_errors = self.get_compile_errors_json()
                if current_errors is None:
                    print("  ⚠️ 编译检查失败（无法获取错误），停止以避免误判")
                    return False, 0, initial_total
                if not current_errors:
                    print("  ✅ 编译错误已清零，修复完成！")
                    fixed_unique = initial_total
                    return True, fixed_unique, initial_total

                after_sig = _sig(current_errors)
                if after_sig == before_sig:
                    print("  ⚠️ 本轮修复后错误集未变化（无进展），停止以避免重复修复")
                    break

                print(f"  📊 本轮已应用修复: {applied_this_round} 个（当前剩余错误: {len(current_errors)}）")
                continue

            # 先尝试修复缺失的类型和常量
            type_const_fixed = self.fix_missing_types_and_consts(current_errors)
            if type_const_fixed > 0:
                print(f"  🛡️ 自动补全了 {type_const_fixed} 个缺失的类型/常量（待编译验证）")
                applied_this_round += type_const_fixed
            
            # 按文件分组
            errors_by_file: Dict[str, List[CompileError]] = {}
            for error in current_errors:
                if error.file_path not in errors_by_file:
                    errors_by_file[error.file_path] = []
                errors_by_file[error.file_path].append(error)
            
            for file_path, file_errors in errors_by_file.items():
                print(f"  🔨 修复 {file_path} ({len(file_errors)} 个错误)...")

                # LLM-first: prefer semantic fixes from the model, fall back to deterministic rule fixes.
                remaining_errors: List[CompileError] = list(file_errors)
                llm_applied_file_level = False

                if self.llm_client:
                    if len(file_errors) == 1:
                        # 单个错误：优先单行 LLM 修复
                        err = file_errors[0]
                        context = self.build_fix_context(err)
                        if context:
                            fixed_line = self.fix_single_error(context)
                            if fixed_line and self.apply_line_fix(file_path, err.line_number, fixed_line):
                                print(f"    ✓ [LLM] 行 {err.line_number}: {err.error_code}")
                                applied_this_round += 1
                                remaining_errors = []
                    else:
                        # 多个错误：优先文件级 LLM 修复（一次覆盖整个文件）
                        if self.fix_file_errors(file_path, file_errors):
                            print(f"    ✓ [LLM] 已应用文件级修复（覆盖 {len(file_errors)} 个错误上下文，待编译验证）")
                            applied_this_round += 1
                            llm_applied_file_level = True
                        else:
                            # 文件级失败，回退到逐行 LLM 修复
                            new_remaining: List[CompileError] = []
                            for err in file_errors:
                                context = self.build_fix_context(err)
                                if context:
                                    fixed_line = self.fix_single_error(context)
                                    if fixed_line and self.apply_line_fix(file_path, err.line_number, fixed_line):
                                        print(f"    ✓ [LLM] 行 {err.line_number}: {err.error_code}")
                                        applied_this_round += 1
                                        continue
                                new_remaining.append(err)
                            remaining_errors = new_remaining

                # If the file-level LLM rewrite succeeded, avoid mixing in line-based rule fixes this round
                # (line numbers/context may have shifted). We'll validate after the round via cargo check.
                if llm_applied_file_level:
                    continue

                # Rule fallback (only for errors the LLM didn't (or couldn't) patch this round).
                if remaining_errors:
                    # File-level deterministic fallback: globals.rs pointer Mutex (E0277) can be fixed structurally.
                    if file_path == "src/globals.rs":
                        combined = "\n".join((e.message or "") + "\n" + (e.rendered or "") for e in remaining_errors)
                        if "cannot be sent between threads safely" in combined:
                            full_path = self.src_dir.parent / file_path
                            if self._try_fix_globals_pointer_mutex(full_path):
                                print("    ✓ [规则] globals.rs: pointer Mutex -> usize（待编译验证）")
                                applied_this_round += 1
                                continue

                    new_remaining: List[CompileError] = []
                    for err in remaining_errors:
                        context = self.build_fix_context(err)
                        if context:
                            rule_fix = self.try_rule_fix(context)
                            if rule_fix and self.apply_line_fix(file_path, err.line_number, rule_fix):
                                print(f"    ✓ [规则] 行 {err.line_number}: {err.error_code}")
                                applied_this_round += 1
                                continue
                        new_remaining.append(err)
                    remaining_errors = new_remaining

                if remaining_errors and not self.llm_client:
                    print(f"    ⚠️ {len(remaining_errors)} 个错误需要 LLM 修复，但 LLM 未配置")

            applied_changes += applied_this_round

            if applied_this_round == 0:
                print("  ⚠️ 本轮未能应用任何修复，停止")
                break

            # Re-compile once to verify progress (avoid "false success" loops).
            current_errors = self.get_compile_errors_json()
            if current_errors is None:
                print("  ⚠️ 编译检查失败（无法获取错误），停止以避免误判")
                return False, 0, initial_total
            if not current_errors:
                print("  ✅ 编译错误已清零，修复完成！")
                fixed_unique = initial_total
                return True, fixed_unique, initial_total

            after_sig = _sig(current_errors)
            if after_sig == before_sig:
                print("  ⚠️ 本轮修复后错误集未变化（无进展），停止以避免重复修复")
                break

            print(f"  📊 本轮已应用修复: {applied_this_round} 个（当前剩余错误: {len(current_errors)}）")
        
        # 最终验证（current_errors 已是最后一次编译结果）
        remaining = len(current_errors)
        success = remaining == 0
        fixed_unique = max(0, initial_total - remaining)

        if not success:
            print(f"  📊 修复汇总: 初始错误 {initial_total}，已消除 {fixed_unique}，剩余 {remaining}，应用修复 {applied_changes}")

        return success, fixed_unique, initial_total


def precise_fix_skeleton(
    skeleton_dir: Path,
    llm_base_url: str = None,
    llm_model: str = None,
    max_rounds: int = 3
) -> Tuple[bool, int, int]:
    """
    便捷函数：精准修复骨架编译错误

    Args:
        skeleton_dir: 骨架目录
        llm_base_url: LLM API 地址（默认从 generation.py 获取）
        llm_model: 模型名称（默认从 generation.py 获取）
        max_rounds: 最大修复轮次

    Returns:
        (是否成功, 修复数, 总错误数)
    """
    # 复用 generation.py 的配置
    try:
        from generate.generation import (
            USE_VLLM, VLLM_BASE_URL, VLLM_API_KEY, VLLM_MODEL_NAME, VLLM_REQUEST_TIMEOUT,
            EXTERNAL_API_BASE_URL, EXTERNAL_API_KEY, EXTERNAL_API_MODEL, EXTERNAL_API_TIMEOUT
        )
        if USE_VLLM:
            default_url = VLLM_BASE_URL
            default_model = VLLM_MODEL_NAME
            api_key = VLLM_API_KEY
            timeout = VLLM_REQUEST_TIMEOUT
        else:
            default_url = EXTERNAL_API_BASE_URL
            default_model = EXTERNAL_API_MODEL
            api_key = EXTERNAL_API_KEY
            timeout = EXTERNAL_API_TIMEOUT
    except ImportError:
        default_url = "http://localhost:8000/v1"
        default_model = "qwen3_coder"
        api_key = "dummy"
        timeout = 600.0

    llm_base_url = llm_base_url or default_url
    llm_model = llm_model or default_model

    try:
        from openai import OpenAI
        llm_client = OpenAI(base_url=llm_base_url, api_key=api_key, timeout=timeout)
    except ImportError:
        print("⚠️ openai 库未安装，无法使用 LLM 修复")
        return False, 0, 0
    except Exception as e:
        print(f"⚠️ LLM 客户端初始化失败: {e}")
        return False, 0, 0

    fixer = LLMPreciseFixer(skeleton_dir, llm_client, llm_model)
    return fixer.fix_all_errors(max_rounds)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLM 精准修复骨架编译错误")
    parser.add_argument("skeleton_dir", type=Path, help="骨架目录")
    parser.add_argument("--llm-url", default=None, help="LLM API 地址（默认从 generation.py 获取）")
    parser.add_argument("--model", default=None, help="模型名称（默认从 generation.py 获取）")
    parser.add_argument("--max-rounds", type=int, default=3, help="最大修复轮次")
    
    args = parser.parse_args()
    
    success, fixed, total = precise_fix_skeleton(
        args.skeleton_dir,
        args.llm_url,
        args.model,
        args.max_rounds
    )
    
    print(f"\n{'='*60}")
    print(f"修复结果: {'✅ 成功' if success else '❌ 部分失败'}")
    print(f"修复统计: {fixed}/{total}")
    print(f"{'='*60}")
