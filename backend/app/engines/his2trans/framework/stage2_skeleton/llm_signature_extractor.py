#!/usr/bin/env python3
"""
LLM 函数签名提取器 (LLM Signature Extractor)

当 Tree-sitter 解析失败或效果不佳时，使用 LLM 作为兜底方案。

设计原则：
1. Tree-sitter 优先：快速、确定性
2. LLM 兜底：处理复杂宏、非标准语法等边缘情况
3. 结果合并：取两者的并集，优先使用 Tree-sitter 的结果
"""

import json
import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ExtractedSignature:
    """提取的函数签名"""
    name: str
    return_type: str
    parameters: List[Tuple[str, str]]  # [(name, type), ...]
    is_static: bool = False
    raw_signature: str = ""


# LLM 提示词模板
EXTRACT_SIGNATURES_PROMPT = '''你是一个专业的 C/C++ 代码分析专家。请从以下 C/C++ 源代码中提取所有函数定义的签名。

注意事项：
1. 只提取**函数定义**（有函数体的），不要提取函数声明（只有 extern 或分号结尾的）
2. 处理各种宏修饰符（如 STATIC, INLINE, ATTRIBUTE_ISR, __attribute__((...)) 等）
3. 正确识别返回类型（包括指针、const 修饰等）
4. 正确解析参数列表

源代码：
```c
{source_code}
```

请以 JSON 格式返回，格式如下：
```json
{{
  "functions": [
    {{
      "name": "函数名",
      "return_type": "返回类型",
      "parameters": [
        {{"name": "参数名", "type": "参数类型"}},
        ...
      ],
      "is_static": true/false,
      "raw_signature": "原始签名（不含函数体）"
    }},
    ...
  ]
}}
```

只返回 JSON，不要有其他解释。'''


class LLMSignatureExtractor:
    """
    LLM 函数签名提取器
    
    用法：
        extractor = LLMSignatureExtractor(llm_client)
        signatures = extractor.extract(source_code)
    """
    
    def __init__(self, llm_client=None, model_name: str = "qwen3_coder"):
        """
        初始化提取器
        
        Args:
            llm_client: LLM 客户端（支持 chat 方法）
            model_name: 模型名称
        """
        self.llm_client = llm_client
        self.model_name = model_name
    
    def extract(self, source_code: str, max_retries: int = 2) -> List[ExtractedSignature]:
        """
        使用 LLM 提取函数签名
        
        Args:
            source_code: C/C++ 源代码
            max_retries: 最大重试次数
        
        Returns:
            提取的函数签名列表
        """
        if not self.llm_client:
            logger.warning("LLM 客户端未配置，跳过 LLM 提取")
            return []
        
        # 如果代码太长，截断或分块处理
        if len(source_code) > 15000:
            logger.warning(f"源代码过长 ({len(source_code)} 字符)，截断到 15000 字符")
            source_code = source_code[:15000] + "\n// ... (truncated)"
        
        prompt = EXTRACT_SIGNATURES_PROMPT.format(source_code=source_code)
        
        for attempt in range(max_retries):
            try:
                response = self._call_llm(prompt)
                signatures = self._parse_response(response)
                
                if signatures:
                    logger.info(f"LLM 提取了 {len(signatures)} 个函数签名")
                    return signatures
                    
            except Exception as e:
                logger.warning(f"LLM 提取尝试 {attempt + 1}/{max_retries} 失败: {e}")
        
        return []
    
    def _call_llm(self, prompt: str) -> str:
        """调用 LLM"""
        if hasattr(self.llm_client, 'chat'):
            # OpenAI 兼容接口
            response = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=4096
            )
            return response.choices[0].message.content
        elif hasattr(self.llm_client, 'generate'):
            # 简单生成接口
            return self.llm_client.generate(prompt)
        else:
            raise ValueError("不支持的 LLM 客户端类型")
    
    def _parse_response(self, response: str) -> List[ExtractedSignature]:
        """解析 LLM 响应"""
        # 提取 JSON 块
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # 尝试直接解析
            json_str = response
        
        try:
            data = json.loads(json_str)
            functions = data.get('functions', [])
            
            signatures = []
            for func in functions:
                params = [(p.get('name', ''), p.get('type', '')) for p in func.get('parameters', [])]
                sig = ExtractedSignature(
                    name=func.get('name', ''),
                    return_type=func.get('return_type', 'void'),
                    parameters=params,
                    is_static=func.get('is_static', False),
                    raw_signature=func.get('raw_signature', '')
                )
                if sig.name:  # 确保函数名不为空
                    signatures.append(sig)
            
            return signatures
            
        except json.JSONDecodeError as e:
            logger.warning(f"JSON 解析失败: {e}")
            return []


def merge_signatures(
    treesitter_sigs: List,  # FunctionSignature from skeleton_builder
    llm_sigs: List[ExtractedSignature],
    source_code: str = ""
) -> List:
    """
    合并 Tree-sitter 和 LLM 提取的签名
    
    策略：
    1. 以 Tree-sitter 结果为基础（更可靠）
    2. 添加 LLM 发现但 Tree-sitter 遗漏的函数
    3. 使用函数名作为去重 key
    
    Args:
        treesitter_sigs: Tree-sitter 提取的签名
        llm_sigs: LLM 提取的签名
        source_code: 原始源代码（用于验证）
    
    Returns:
        合并后的签名列表
    """
    # 以函数名为 key 建立索引
    merged = {sig.name: sig for sig in treesitter_sigs}
    
    # 添加 LLM 发现的新函数
    added_count = 0
    for llm_sig in llm_sigs:
        if llm_sig.name not in merged:
            # 验证函数名确实存在于源码中
            if source_code and llm_sig.name not in source_code:
                logger.debug(f"跳过 LLM 提取的不存在函数: {llm_sig.name}")
                continue
            
            # 转换为 FunctionSignature 格式（需要 skeleton_builder 中的类）
            # 这里返回 ExtractedSignature，由调用方转换
            merged[llm_sig.name] = llm_sig
            added_count += 1
    
    if added_count > 0:
        logger.info(f"LLM 补充了 {added_count} 个 Tree-sitter 遗漏的函数")
    
    return list(merged.values())


def should_use_llm_fallback(
    treesitter_sigs: List,
    ast_health_report: dict,
    expected_function_count: int = 0
) -> bool:
    """
    判断是否需要使用 LLM 兜底
    
    条件（满足任一）：
    1. AST 错误率 > 10%
    2. Tree-sitter 提取的函数数明显少于预期
    3. 提取到 0 个函数但源码明显有函数定义
    
    Args:
        treesitter_sigs: Tree-sitter 提取的签名
        ast_health_report: AST 健康检查报告
        expected_function_count: 预期函数数（0 表示未知）
    
    Returns:
        是否需要 LLM 兜底
    """
    # 条件 1: AST 错误率高
    if ast_health_report.get('error_rate', 0) > 0.1:
        logger.info(f"AST 错误率 {ast_health_report['error_rate']:.1%} > 10%，触发 LLM 兜底")
        return True
    
    # 条件 2: 函数数明显不足
    actual_count = len(treesitter_sigs)
    if expected_function_count > 0 and actual_count < expected_function_count * 0.7:
        logger.info(f"函数数 {actual_count} < 预期 {expected_function_count} 的 70%，触发 LLM 兜底")
        return True
    
    # 条件 3: 提取到 0 个函数
    if actual_count == 0:
        logger.info("提取到 0 个函数，触发 LLM 兜底")
        return True
    
    return False


# =========================================================================
# 便捷函数
# =========================================================================

def extract_with_llm_fallback(
    source_code: str,
    treesitter_extractor,  # skeleton_builder.SkeletonBuilder
    llm_client=None,
    expected_function_count: int = 0
) -> List:
    """
    使用 Tree-sitter + LLM 兜底提取函数签名
    
    Args:
        source_code: C/C++ 源代码
        treesitter_extractor: Tree-sitter 提取器（SkeletonBuilder 实例）
        llm_client: LLM 客户端（可选）
        expected_function_count: 预期函数数（可选）
    
    Returns:
        函数签名列表
    """
    # 1. 先用 Tree-sitter 提取
    treesitter_sigs = treesitter_extractor.extract_function_signatures(source_code)
    
    # 2. 检查是否需要 LLM 兜底
    # 获取 AST 健康报告（如果可用）
    ast_health = getattr(treesitter_extractor, '_last_ast_health', {'error_rate': 0})
    
    if not llm_client:
        return treesitter_sigs
    
    if should_use_llm_fallback(treesitter_sigs, ast_health, expected_function_count):
        # 3. 使用 LLM 提取
        llm_extractor = LLMSignatureExtractor(llm_client)
        llm_sigs = llm_extractor.extract(source_code)
        
        # 4. 合并结果
        return merge_signatures(treesitter_sigs, llm_sigs, source_code)
    
    return treesitter_sigs


# =========================================================================
# Bindgen 失败时的 LLM 兜底：类型定义生成
# =========================================================================

BINDGEN_FALLBACK_PROMPT = '''You are a strict C-to-Rust FFI compiler. Generate ONLY Rust type definitions.

## CRITICAL OUTPUT RULES (VIOLATION = IMMEDIATE FAILURE)

1. **OUTPUT RAW RUST CODE ONLY**
   - ❌ FORBIDDEN: ```rust, ```, "Here is", "I'll", explanations, comments about your approach
   - ❌ FORBIDDEN: Anything that is not valid Rust code
   - ✅ REQUIRED: Start DIRECTLY with `use std::os::raw::*;`
   - ✅ REQUIRED: Only output valid Rust type definitions

2. **NO FUNCTIONS - TYPES ONLY**
   - ❌ FORBIDDEN: `extern "C" {{ }}`, `fn foo()`, function declarations
   - ✅ REQUIRED: Only `pub struct`, `pub enum`, `pub union`, `pub type`, `pub const`

3. **FFI TYPE MAPPING RULES**
   | C Type | Rust Type |
   |--------|-----------|
   | int | c_int |
   | unsigned int | c_uint |
   | long | c_long |
   | char* | *mut c_char |
   | const char* | *const c_char |
   | void* | *mut c_void |
   | size_t | usize |
   | uint8_t, uint16_t, uint32_t, uint64_t | u8, u16, u32, u64 |
   | int8_t, int16_t, int32_t, int64_t | i8, i16, i32, i64 |
   | void (*)(T) | Option<unsafe extern "C" fn(T)> |
   | T[N] | [T; N] |

4. **STRUCT/ENUM RULES**
   - ALL structs/enums MUST have `#[repr(C)]`
   - Forward declarations (struct Foo;) -> `pub struct Foo {{ _unused: [u8; 0] }}`
   - Unknown types -> `pub struct UnknownType {{ _opaque: [u8; 128] }}`
   - Complex nested structs: simplify to opaque if needed

5. **ANTI-HALLUCINATION**
   - ONLY generate types that EXIST in the input C code
   - Do NOT invent types like `sockaddr_v1`, `sockaddr_v2`, etc.
   - If unsure about a type, use opaque struct with `_opaque: [u8; 128]`

## C HEADER CONTENT
{header_content}

## OUTPUT NOW (RAW RUST CODE, NO MARKDOWN):
use std::os::raw::*;'''


class LLMBindgenFallback:
    """
    Bindgen 失败时的 LLM 兜底类型生成器
    
    用法：
        fallback = LLMBindgenFallback(llm_client, "qwen3_coder")
        rust_types = fallback.generate_types(header_content)
    """
    
    def __init__(self, llm_client=None, model_name: str = "qwen3_coder"):
        self.llm_client = llm_client
        self.model_name = model_name
    
    def generate_types(self, header_content: str, source_files: List[str] = None) -> Optional[str]:
        """
        从 C 头文件生成 Rust 类型定义
        
        Args:
            header_content: C 头文件内容（可以是多个头文件拼接）
            source_files: 可选的源文件列表（用于提取更多上下文）
        
        Returns:
            Rust 类型定义代码，失败返回 None
        """
        if not self.llm_client:
            logger.warning("LLM client 未配置，无法进行 bindgen 兜底")
            return None
        
        # 如果内容太长，截取重要部分
        if len(header_content) > 15000:
            header_content = self._truncate_header(header_content)
        
        prompt = BINDGEN_FALLBACK_PROMPT.format(header_content=header_content)
        
        try:
            response = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a C to Rust FFI expert. Output ONLY raw Rust code without any markdown formatting, code fences, or explanations. Never use ``` in your output."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,  # 低温度，确保输出稳定
                max_tokens=16000
            )
            
            result = response.choices[0].message.content
            rust_code = self._extract_rust_code(result)
            
            if rust_code:
                # 添加必要的头部
                rust_code = self._add_header(rust_code)
                logger.info(f"LLM bindgen 兜底成功，生成了 {len(rust_code)} 字节的类型定义")
                return rust_code
            else:
                logger.warning("LLM 返回内容中未找到有效的 Rust 代码")
                return None
                
        except Exception as e:
            logger.error(f"LLM bindgen 兜底失败: {e}")
            return None
    
    def _truncate_header(self, content: str) -> str:
        """
        保守型预处理：只移除对类型生成绝对无用的内容，保留所有可能的上下文。
        
        保留内容：
        1. #define (用于数组大小常量)
        2. typedef (用于类型别名)
        3. struct/enum/union 定义
        4. 大括号 {} (用于确定作用域)
        
        移除内容：
        1. #include (LLM 无法读取外部文件)
        2. 纯函数声明 (void foo(); 对生成 struct 无用)
        3. 注释 (节省 Token)
        """
        # 1. 移除注释 (使用正则，包括 // 和 /* ... */)
        # 注意：这一步最先做，防止注释里包含代码关键字干扰判断
        content = re.sub(r'//.*', '', content)
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        
        lines = content.split('\n')
        kept_lines = []
        
        # 预编译正则：匹配看起来像函数声明的行
        # 逻辑：包含括号()，以分号;结尾，且行首不是 typedef/struct/return/else 等
        # 这是一个保守的正则，避免误伤 typedef void (*ptr)();
        # 使用 try-except 包装，防止正则表达式匹配异常
        try:
            func_decl_pattern = re.compile(r'^\s*(?!typedef|struct|enum|union|#)[a-zA-Z_]\w*[\w\s\*]+\([^)]*\)\s*;')
        except re.error as e:
            logger.warning(f"正则表达式编译失败: {e}，跳过函数声明过滤")
            func_decl_pattern = None
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            
            # --- 必须移除的垃圾 ---
            
            # 1. 移除 #include
            if stripped.startswith('#include') or stripped.startswith('# include'):
                continue
                
            # 2. 移除 #pragma (通常无关，且有些 pragma 很长)
            if stripped.startswith('#pragma'):
                continue
            
            # 3. 移除纯函数声明 (节省大量 Token)
            # 例如: void DoSomething(int a); 
            # 但保留: typedef void (*Cb)(int); (因为有 typedef)
            if func_decl_pattern is not None:
                try:
                    if func_decl_pattern.match(stripped):
                        continue
                except Exception:
                    pass  # 匹配失败时不跳过该行
            
            # --- 必须保留的精华 ---
            
            # 1. 保留 #define (关键！用于数组长度常量，如 char name[MAX_LEN])
            if stripped.startswith('#define'):
                kept_lines.append(line)
                continue
            
            # 2. 保留所有包含类型定义的行
            if any(k in stripped for k in ['struct', 'enum', 'union', 'typedef', 'class']):
                kept_lines.append(line)
                continue
            
            # 3. 保留大括号 (结构体可能跨多行)
            if '{' in stripped or '}' in stripped:
                kept_lines.append(line)
                continue
            
            # 4. 保留分号结尾的行 (可能是结构体字段 int a;)
            # 前面已经过滤了函数声明，这里剩下的分号行通常是字段定义或全局变量
            if stripped.endswith(';'):
                kept_lines.append(line)
                continue
            
            # 5. 保留预处理条件 (防止破坏结构体定义的完整性)
            if stripped.startswith('#if') or stripped.startswith('#else') or stripped.startswith('#endif'):
                kept_lines.append(line)
                continue
        
        result = '\n'.join(kept_lines)
        
        # 最后的兜底截断：如果清理后依然超过 20k 字符，为了防止 API 报错，必须硬截断
        # 但我们尽量保留头部，因为核心定义通常在前面
        if len(result) > 20000:
            logger.warning(f"Header content massive after sanitization ({len(result)} chars). Hard truncating.")
            result = result[:20000] + '\n\n// ... (Content truncated for LLM context limit) ...'
            
        return result
    
    def _extract_rust_code(self, response: str) -> Optional[str]:
        """
        从 LLM 响应中提取 Rust 代码
        
        ★★★ 2025-12-11 增强：更强的清洗策略和验证 ★★★
        """
        if not response:
            return None
        
        # 步骤 0: 预处理 - 移除常见的 LLM "答非所问" 前缀
        response = self._remove_llm_preamble(response)
        
        # 步骤 1: 尝试提取 ```rust ... ``` 块
        rust_pattern = r'```rust\s*(.*?)\s*```'
        matches = re.findall(rust_pattern, response, re.DOTALL)
        if matches:
            code = matches[0].strip()
        else:
            # 步骤 2: 尝试提取 ``` ... ``` 块
            code_pattern = r'```\s*(.*?)\s*```'
            matches = re.findall(code_pattern, response, re.DOTALL)
            if matches:
                code = matches[0].strip()
            else:
                # 步骤 3: 直接使用响应内容
                code = response.strip()
        
        # 步骤 4: 暴力清理所有 Markdown 和无效内容
        code = self._clean_markdown(code)
        
        # 步骤 5: 验证并修复常见问题
        code = self._fix_common_issues(code)
        
        # 验证是否包含有效的 Rust 代码
        if 'pub struct' in code or 'pub type' in code or 'pub enum' in code or 'pub const' in code:
            return code
        
        # 如果没有找到有效代码，尝试从响应中找到第一个 "use" 开始的内容
        use_match = re.search(r'(use\s+std::.*)', response, re.DOTALL)
        if use_match:
            code = use_match.group(1)
            code = self._clean_markdown(code)
            code = self._fix_common_issues(code)
            if 'pub struct' in code or 'pub type' in code:
                return code
        
        return None
    
    def _remove_llm_preamble(self, response: str) -> str:
        """
        移除 LLM 常见的"答非所问"前缀
        
        例如:
        - "Here is the Rust code:"
        - "I'll convert the C types to Rust FFI types."
        - "Based on the C header, here are the type definitions:"
        """
        # 常见的无用前缀模式
        preamble_patterns = [
            r'^[Hh]ere\s+(is|are)\s+.*?:\s*\n',
            r'^I\'ll\s+.*?\.\s*\n',
            r'^I will\s+.*?\.\s*\n',
            r'^Based on\s+.*?:\s*\n',
            r'^The following\s+.*?:\s*\n',
            r'^Let me\s+.*?\.\s*\n',
            r'^Sure[,!]?\s*.*?\.\s*\n',
            r'^Okay[,!]?\s*.*?\.\s*\n',
            r'^Below is\s+.*?:\s*\n',
        ]
        
        for pattern in preamble_patterns:
            response = re.sub(pattern, '', response, flags=re.IGNORECASE)
        
        return response.strip()
    
    def _fix_common_issues(self, code: str) -> str:
        """
        修复 LLM 生成代码中的常见问题
        """
        if not code:
            return code
        
        lines = code.split('\n')
        fixed_lines = []
        
        for line in lines:
            # 移除行末多余的分号（如 pub struct Foo {}; -> pub struct Foo {}）
            if line.strip().endswith('};') and 'pub struct' in line:
                line = line.rstrip(';').rstrip()
            
            # 移除 extern "C" 块（我们只要类型）
            if 'extern "C"' in line:
                continue
            
            # 移除独立的 fn 声明行
            if re.match(r'^\s*fn\s+\w+\s*\(', line) or re.match(r'^\s*pub\s+fn\s+', line):
                continue
            
            fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
    def _clean_markdown(self, code: str) -> str:
        """
        暴力移除所有 Markdown 标记和 LLM 解释性文本
        
        ★★★ 2025-12-11 增强：更全面的清洗 ★★★
        """
        if not code:
            return code
        
        lines = code.split('\n')
        cleaned_lines = []
        in_explanation = False  # 跟踪是否在解释性文本中
        
        for line in lines:
            stripped = line.strip()
            
            # 跳过 Markdown 代码块标记
            if stripped.startswith('```'):
                continue
            # 跳过纯 Markdown 行
            if stripped in ('```rust', '```c', '```', '```rs', '```c++', '```cpp'):
                continue
            # 移除行内的反引号（如果整行只是反引号）
            if re.match(r'^`+$', stripped):
                continue
            
            # ★ 新增：跳过解释性行 ★
            # 常见模式：纯英文句子，以句号结尾，不含 Rust 关键字
            if (stripped.endswith('.') and 
                not any(kw in stripped for kw in ['pub struct', 'pub type', 'pub enum', 'pub const', '#[repr']) and
                re.match(r'^[A-Z][a-zA-Z\s,\'"]+\.$', stripped)):
                continue
            
            # ★ 新增：跳过以 Note:, Warning:, Important: 开头的行 ★
            if re.match(r'^(Note|Warning|Important|Tip|Info|Please)[:!]', stripped, re.IGNORECASE):
                continue
            
            # ★ 新增：跳过纯数字序号行（如 "1.", "2." ）★
            if re.match(r'^\d+\.\s*$', stripped):
                continue
            
            cleaned_lines.append(line)
        
        result = '\n'.join(cleaned_lines)
        
        # 额外清理：移除可能残留的 ```rust 或 ``` 标记
        result = re.sub(r'^```rust\s*\n?', '', result, flags=re.MULTILINE)
        result = re.sub(r'^```\s*\n?', '', result, flags=re.MULTILINE)
        result = re.sub(r'\n```\s*$', '', result)
        result = re.sub(r'```\s*$', '', result)
        
        return result.strip()
    
    def _add_header(self, rust_code: str) -> str:
        """添加必要的 Rust 头部"""
        # 先清理任何残留的 Markdown 标记
        rust_code = self._clean_markdown(rust_code)
        
        header = '''//! Auto-generated type definitions (LLM Fallback)
//! 
//! Note: Generated by LLM because bindgen failed.
//! Please review for correctness.

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
#![allow(dead_code)]
#![allow(unused)]

use std::os::raw::*;
use std::ffi::c_void;

// Common type aliases
pub type size_t = usize;
pub type ssize_t = isize;

'''
        # 检查是否已有这些头部
        if '#![allow' not in rust_code:
            return header + rust_code
        return rust_code


def generate_types_with_llm_fallback(
    header_files: List[Path],
    llm_client=None,
    model_name: str = "qwen3_coder"
) -> Optional[str]:
    """
    便捷函数：从头文件生成类型定义
    
    Args:
        header_files: 头文件路径列表
        llm_client: LLM 客户端
        model_name: 模型名称
    
    Returns:
        Rust 类型定义代码
    """
    if not llm_client:
        return None
    
    # 读取所有头文件内容
    header_content = []
    for hf in header_files:
        try:
            with open(hf, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                header_content.append(f"// ========== {hf.name} ==========\n{content}")
        except Exception as e:
            logger.warning(f"无法读取头文件 {hf}: {e}")
    
    if not header_content:
        return None
    
    combined = '\n\n'.join(header_content)
    
    fallback = LLMBindgenFallback(llm_client, model_name)
    return fallback.generate_types(combined)
