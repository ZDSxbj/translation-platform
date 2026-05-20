#!/usr/bin/env python3
"""
微任务修复器 (Micro-Task Repairer)

基于 Rust 编译器 JSON 输出的确定性错误提取器。
核心理念：
- 不解析人类可读的日志，而是解析编译器的结构化数据
- 直接读取源文件字节获取精确的符号名称
- 消除正则表达式猜测，实现 100% 确定性

这是 "AI 原生自愈架构" 的核心组件。
"""

import subprocess
import json
import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Set, Optional, Dict, Tuple
import logging
import re

logger = logging.getLogger(__name__)


@dataclass
class RepairTask:
    """修复任务"""
    error_code: str       # 错误代码 (E0412, E0425, E0433 等)
    symbol: str           # 缺失的符号名
    file_path: str        # 出错的文件路径
    line: int             # 行号
    column: int           # 列号
    message: str          # 完整错误信息
    error_type: str       # 错误类型 (type, value, module)


@dataclass 
class SelfHealingResult:
    """自愈循环结果"""
    success: bool                   # 是否通过编译
    cycles_used: int                # 使用的循环次数
    symbols_fixed: List[str]        # 修复的符号列表
    remaining_errors: List[str]     # 剩余的错误列表


class MicroTaskRepairer:
    """
    微任务修复器
    
    基于 rustc --error-format=json 输出的确定性错误提取。
    只处理以下确定性错误：
    - E0412: cannot find type
    - E0425: cannot find value
    - E0433: failed to resolve (module/path)
    """
    
    # 支持的确定性错误码
    SUPPORTED_ERRORS = {
        "E0412": "type",      # cannot find type
        "E0425": "value",     # cannot find value (function, constant)
        "E0433": "module",    # failed to resolve (module path)
    }
    
    def __init__(self, project_root: Path, c_source_dir: Path = None):
        """
        初始化微任务修复器
        
        Args:
            project_root: Rust 项目根目录 (包含 Cargo.toml)
            c_source_dir: C 源码目录 (用于查找类型上下文)
        """
        self.project_root = Path(project_root)
        self.c_source_dir = Path(c_source_dir) if c_source_dir else None
        self._c_headers_content = None  # 缓存 C 头文件内容
    
    def extract_tasks(self) -> List[RepairTask]:
        """
        执行 cargo check --message-format=json 并提取确定性修复任务
        
        Returns:
            修复任务列表
        """
        # 使用 cargo check --message-format=json 获取结构化错误
        cmd = ["cargo", "check", "--message-format=json"]
        
        env = os.environ.copy()
        env["RUSTFLAGS"] = "-A unused_imports -A dead_code -A unused_variables -A unused_mut"
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=120,
                env=env
            )
        except subprocess.TimeoutExpired:
            logger.error("cargo check 超时")
            return []
        except Exception as e:
            logger.error(f"cargo check 执行失败: {e}")
            return []
        
        tasks = []
        seen_symbols: Set[str] = set()
        
        for line in result.stdout.splitlines():
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            # 只关注编译器消息
            if data.get("reason") != "compiler-message":
                continue
            
            msg = data.get("message", {})
            if msg.get("level") != "error":
                continue
            
            code_obj = msg.get("code", {})
            if not code_obj:
                continue
            
            error_code = code_obj.get("code", "")
            
            # 🎯 确定性策略：只处理已知的确定性错误
            if error_code not in self.SUPPORTED_ERRORS:
                continue
            
            error_type = self.SUPPORTED_ERRORS[error_code]
            spans = msg.get("spans", [])
            
            if not spans:
                continue
            
            # 找到主要错误位置
            primary = next((s for s in spans if s.get("is_primary")), spans[0])
            
            # 🔑 物理提取：直接读取源文件字节，零概率出错
            symbol = self._read_source_span(primary)
            
            if not symbol:
                # 如果物理提取失败，从错误消息中提取
                symbol = self._extract_symbol_from_message(msg.get("message", ""))
            
            if symbol and symbol not in seen_symbols:
                # 过滤掉明显的非符号名
                if self._is_valid_symbol(symbol):
                    seen_symbols.add(symbol)
                    tasks.append(RepairTask(
                        error_code=error_code,
                        symbol=symbol,
                        file_path=primary.get("file_name", ""),
                        line=primary.get("line_start", 0),
                        column=primary.get("column_start", 0),
                        message=msg.get("message", ""),
                        error_type=error_type
                    ))
        
        logger.info(f"提取了 {len(tasks)} 个确定性修复任务")
        for task in tasks:
            logger.debug(f"  [{task.error_code}] {task.error_type}: {task.symbol}")
        
        return tasks
    
    def _read_source_span(self, span: dict) -> str:
        """
        根据字节偏移量读取精确的符号名称
        
        Args:
            span: rustc span 对象，包含 byte_start, byte_end, file_name
            
        Returns:
            符号名称，失败返回空字符串
        """
        file_name = span.get("file_name", "")
        byte_start = span.get("byte_start", 0)
        byte_end = span.get("byte_end", 0)
        
        if not file_name or byte_start >= byte_end:
            return ""
        
        full_path = self.project_root / file_name
        
        try:
            with open(full_path, "rb") as f:
                f.seek(byte_start)
                byte_len = byte_end - byte_start
                content = f.read(byte_len).decode("utf-8")
                return content.strip()
        except Exception as e:
            logger.debug(f"读取源文件失败 {full_path}: {e}")
            return ""
    
    def _extract_symbol_from_message(self, message: str) -> str:
        """
        从错误消息中提取符号名称（备用方法）
        
        Examples:
            "cannot find type `MyType` in module `crate::types`" -> "MyType"
            "cannot find value `my_func` in this scope" -> "my_func"
        """
        # 尝试从反引号中提取
        match = re.search(r'`([a-zA-Z_][a-zA-Z0-9_]*)`', message)
        if match:
            return match.group(1)
        return ""
    
    def _is_valid_symbol(self, symbol: str) -> bool:
        """
        检查是否是有效的符号名
        
        过滤掉明显的非符号名，如：
        - 空字符串
        - 只包含数字
        - Rust 原生类型
        - 常见关键字
        """
        if not symbol:
            return False
        
        # 必须是有效的标识符格式
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', symbol):
            return False
        
        # 过滤 Rust 原生类型
        rust_primitives = {
            'i8', 'u8', 'i16', 'u16', 'i32', 'u32', 'i64', 'u64', 'i128', 'u128',
            'f32', 'f64', 'bool', 'char', 'str', 'usize', 'isize',
            'Self', 'self', 'super', 'crate',
        }
        if symbol in rust_primitives:
            return False
        
        # 过滤 Rust 关键字
        rust_keywords = {
            'as', 'break', 'const', 'continue', 'else', 'enum', 'extern',
            'false', 'fn', 'for', 'if', 'impl', 'in', 'let', 'loop', 'match',
            'mod', 'move', 'mut', 'pub', 'ref', 'return', 'static', 'struct',
            'trait', 'true', 'type', 'unsafe', 'use', 'where', 'while',
        }
        if symbol in rust_keywords:
            return False
        
        return True
    
    def get_c_context(self, symbol: str) -> str:
        """
        获取符号相关的 C 代码上下文
        
        Args:
            symbol: 缺失的符号名
            
        Returns:
            相关的 C 代码片段
        """
        if self.c_source_dir is None:
            return ""
        
        # 缓存所有 C 头文件内容
        if self._c_headers_content is None:
            self._c_headers_content = self._load_c_headers()
        
        # 在所有头文件中搜索符号
        context_lines = []
        
        # 搜索 typedef, struct, enum, union 定义
        patterns = [
            rf'typedef\s+[^;]*?\b{re.escape(symbol)}\b[^;]*;',
            rf'struct\s+{re.escape(symbol)}\s*\{{[^}}]*\}}',
            rf'enum\s+{re.escape(symbol)}\s*\{{[^}}]*\}}',
            rf'union\s+{re.escape(symbol)}\s*\{{[^}}]*\}}',
            rf'#define\s+{re.escape(symbol)}\b[^\n]*',
            rf'(?:static\s+)?(?:const\s+)?[a-zA-Z_][a-zA-Z0-9_]*\s+{re.escape(symbol)}\s*[=;(]',
        ]
        
        for content in self._c_headers_content:
            for pattern in patterns:
                matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)
                context_lines.extend(matches)
        
        return '\n'.join(context_lines[:5])  # 最多返回 5 个匹配
    
    def _load_c_headers(self) -> List[str]:
        """加载所有 C 头文件内容"""
        contents = []
        
        if not self.c_source_dir or not self.c_source_dir.exists():
            return contents
        
        for pattern in ['*.h', '*.hpp', '*.c', '*.cpp']:
            for file_path in self.c_source_dir.rglob(pattern):
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        contents.append(f.read())
                except Exception as e:
                    logger.debug(f"读取 {file_path} 失败: {e}")
        
        return contents


class LLMFallbackHandler:
    """
    LLM 回退处理器
    
    专门处理微任务的 LLM 代理，使用高度聚焦的 Prompt 确保成功率。
    """
    
    def __init__(self, llm_fn=None):
        """
        初始化 LLM 回退处理器
        
        Args:
            llm_fn: LLM 调用函数，签名: fn(prompt: str) -> str
        """
        self.llm_fn = llm_fn
    
    def repair_symbol(
        self, 
        symbol: str, 
        error_code: str, 
        c_context: str = "",
        error_type: str = "type"
    ) -> str:
        """
        使用 LLM 生成缺失符号的定义
        
        Args:
            symbol: 缺失的符号名
            error_code: 错误代码 (E0412, E0425, E0433)
            c_context: C 源码上下文
            error_type: 错误类型 (type, value, module)
            
        Returns:
            生成的 Rust 代码
        """
        if self.llm_fn is None:
            # 没有 LLM，使用保守的占位符
            return self._generate_fallback(symbol, error_type)
        
        prompt = self._create_prompt(symbol, error_code, c_context, error_type)
        
        try:
            response = self.llm_fn(prompt)
            code = self._extract_code(response)
            if code:
                return code
        except Exception as e:
            logger.warning(f"LLM 调用失败: {e}")
        
        # LLM 失败，使用占位符
        return self._generate_fallback(symbol, error_type)
    
    def _create_prompt(
        self, 
        symbol: str, 
        error_code: str, 
        c_context: str,
        error_type: str
    ) -> str:
        """创建微任务 Prompt"""
        
        if error_type == "type":
            # E0412: cannot find type
            return f'''The Rust compiler cannot find the type `{symbol}` (error {error_code}).
Please find its definition (struct, enum, union, or typedef) in the following C code.

C Context:
```c
{c_context if c_context else "// No C context available"}
```

Task:
1. Generate the valid Rust `#[repr(C)]` definition.
2. If it is a complex C++ type (template, class), generate an opaque struct:
   `#[repr(C)] pub struct {symbol} {{ _private: [u8; 0] }}`
3. If it is a function pointer typedef, generate:
   `pub type {symbol} = Option<unsafe extern "C" fn(...)>;`

Output ONLY the Rust code, no explanation:'''

        elif error_type == "value":
            # E0425: cannot find value
            return f'''The Rust compiler cannot find the value/function `{symbol}` (error {error_code}).
Search for its definition (#define, const, or function declaration) in the C code.

C Context:
```c
{c_context if c_context else "// No C context available"}
```

Task:
1. If it is a constant (#define or const), output: `pub const {symbol}: i32 = ...;` (infer type/value)
2. If it is a function, output: `extern "C" {{ pub fn {symbol}(...); }}`
3. If not found, generate a safe dummy definition.

Output ONLY the Rust code, no explanation:'''

        else:
            # E0433: module resolution
            return f'''The Rust compiler cannot resolve module/path `{symbol}` (error {error_code}).

Generate a minimal module definition or re-export to satisfy the compiler.

Output ONLY the Rust code, no explanation:'''
    
    def _extract_code(self, response: str) -> str:
        """从 LLM 响应中提取代码"""
        # 尝试提取代码块
        code_match = re.search(r'```rust\s*(.*?)\s*```', response, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        
        # 尝试提取裸代码
        code_match = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        
        # 尝试直接返回（如果看起来像 Rust 代码）
        if 'pub ' in response or 'struct ' in response or 'const ' in response:
            # 移除可能的解释文字
            lines = []
            for line in response.split('\n'):
                line = line.strip()
                if line.startswith('//') or line.startswith('#[') or \
                   line.startswith('pub ') or line.startswith('struct ') or \
                   line.startswith('enum ') or line.startswith('type ') or \
                   line.startswith('const ') or line.startswith('extern ') or \
                   line.startswith('}') or line.startswith('{'):
                    lines.append(line)
            return '\n'.join(lines)
        
        return ""
    
    def _generate_fallback(self, symbol: str, error_type: str) -> str:
        """生成保守的占位符"""
        if error_type == "type":
            return f'''/// Opaque placeholder for missing type `{symbol}`
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct {symbol} {{
    _private: [u8; 0],
}}
'''
        elif error_type == "value":
            return f'''/// Placeholder for missing value `{symbol}`
pub const {symbol}: i32 = 0;
'''
        else:
            return f'// Cannot resolve: {symbol}\n'


class SelfHealingLoop:
    """
    自愈循环控制器
    
    在骨架生成末尾引入自愈闭环：
    1. 运行 cargo check --message-format=json
    2. 提取确定性修复任务
    3. 使用 LLM 生成修复代码
    4. 注入到 types.rs 或 globals.rs
    5. 重复直到编译通过或达到最大循环次数
    """
    
    def __init__(
        self, 
        project_root: Path,
        c_source_dir: Path = None,
        llm_fn=None,
        max_cycles: int = 5
    ):
        """
        初始化自愈循环控制器
        
        Args:
            project_root: Rust 项目根目录
            c_source_dir: C 源码目录
            llm_fn: LLM 调用函数
            max_cycles: 最大循环次数
        """
        self.project_root = Path(project_root)
        self.c_source_dir = Path(c_source_dir) if c_source_dir else None
        self.max_cycles = max_cycles
        
        self.repairer = MicroTaskRepairer(project_root, c_source_dir)
        self.llm_handler = LLMFallbackHandler(llm_fn)
    
    def run(self) -> SelfHealingResult:
        """
        运行自愈循环
        
        Returns:
            自愈结果
        """
        truth_mode = os.environ.get("C2R_TRUTH_MODE", "0").strip().lower() in ("1", "true", "yes", "y", "on")
        if truth_mode:
            logger.info("⏭️ Truth-mode: 跳过 Self-Healing（不生成 placeholder/extern/dummy 定义）")
            tasks = self.repairer.extract_tasks()
            return SelfHealingResult(
                success=len(tasks) == 0,
                cycles_used=0,
                symbols_fixed=[],
                remaining_errors=[t.symbol for t in tasks],
            )

        symbols_fixed = []
        all_attempted = set()
        
        for cycle in range(1, self.max_cycles + 1):
            logger.info(f"🔄 Self-Healing Cycle {cycle}/{self.max_cycles}...")
            print(f"🔄 Self-Healing Cycle {cycle}/{self.max_cycles}...")
            
            # 1. 提取确定性修复任务
            tasks = self.repairer.extract_tasks()
            
            # 过滤掉已尝试的符号
            new_tasks = [t for t in tasks if t.symbol not in all_attempted]
            
            if not new_tasks:
                if not tasks:
                    logger.info("✅ No compilation errors found!")
                    print("✅ No compilation errors found!")
                    return SelfHealingResult(
                        success=True,
                        cycles_used=cycle,
                        symbols_fixed=symbols_fixed,
                        remaining_errors=[]
                    )
                else:
                    # 有错误但都已尝试过，无法继续修复
                    logger.warning(f"⚠ {len(tasks)} errors remain but all have been attempted")
                    break
            
            logger.info(f"🔧 Found {len(new_tasks)} new missing symbols: {[t.symbol for t in new_tasks]}")
            print(f"🔧 Found {len(new_tasks)} new missing symbols: {[t.symbol for t in new_tasks]}")
            
            # 2. 为每个任务生成修复代码
            definitions = []
            for task in new_tasks:
                all_attempted.add(task.symbol)
                
                # 获取 C 代码上下文
                c_context = self.repairer.get_c_context(task.symbol)
                
                # 生成修复代码
                definition = self.llm_handler.repair_symbol(
                    symbol=task.symbol,
                    error_code=task.error_code,
                    c_context=c_context,
                    error_type=task.error_type
                )
                
                if definition:
                    definitions.append(definition)
                    symbols_fixed.append(task.symbol)
                    logger.debug(f"  Generated fix for {task.symbol}")
            
            # 3. 注入修复代码
            if definitions:
                self._inject_definitions(definitions)
            
            # 短暂等待文件系统同步
            import time
            time.sleep(0.1)
        
        # 达到最大循环次数
        remaining_tasks = self.repairer.extract_tasks()
        return SelfHealingResult(
            success=len(remaining_tasks) == 0,
            cycles_used=self.max_cycles,
            symbols_fixed=symbols_fixed,
            remaining_errors=[t.symbol for t in remaining_tasks]
        )
    
    def _inject_definitions(self, definitions: List[str]):
        """
        将生成的定义注入到 types.rs
        
        Args:
            definitions: 要注入的定义列表
        """
        types_rs = self.project_root / "src" / "types.rs"
        
        if not types_rs.exists():
            logger.warning("types.rs 不存在，跳过注入")
            return
        
        # 读取现有内容
        with open(types_rs, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查是否已有自愈标记
        marker = "// ============== Self-Healing Fixes =============="
        if marker not in content:
            content += f"\n\n{marker}\n"
        
        # 追加新定义
        for definition in definitions:
            # 检查是否已存在（避免重复）
            # 简单检查：如果定义的第一行已存在，就跳过
            first_line = definition.strip().split('\n')[0]
            if first_line in content:
                continue
            
            content += f"\n{definition}\n"
        
        # 写回
        with open(types_rs, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"已注入 {len(definitions)} 个修复定义到 types.rs")


# =========================================================================
# 便捷函数
# =========================================================================

def run_self_healing(
    rust_project_dir: str,
    c_source_dir: str = None,
    llm_fn=None,
    max_cycles: int = 5
) -> SelfHealingResult:
    """
    运行自愈循环的便捷函数
    
    Args:
        rust_project_dir: Rust 项目目录
        c_source_dir: C 源码目录（可选）
        llm_fn: LLM 调用函数（可选）
        max_cycles: 最大循环次数
        
    Returns:
        自愈结果
    """
    loop = SelfHealingLoop(
        project_root=Path(rust_project_dir),
        c_source_dir=Path(c_source_dir) if c_source_dir else None,
        llm_fn=llm_fn,
        max_cycles=max_cycles
    )
    return loop.run()


if __name__ == "__main__":
    # 测试代码
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python micro_task_repairer.py <rust_project_dir> [c_source_dir]")
        sys.exit(1)
    
    rust_dir = sys.argv[1]
    c_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    logging.basicConfig(level=logging.DEBUG)
    
    result = run_self_healing(rust_dir, c_dir)
    
    print(f"\n{'='*60}")
    print(f"Self-Healing Result:")
    print(f"  Success: {result.success}")
    print(f"  Cycles Used: {result.cycles_used}")
    print(f"  Symbols Fixed: {result.symbols_fixed}")
    print(f"  Remaining Errors: {result.remaining_errors}")





























