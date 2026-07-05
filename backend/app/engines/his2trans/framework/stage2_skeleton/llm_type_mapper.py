#!/usr/bin/env python3
"""
LLM 驱动的类型映射器

设计理念：
1. TypeMapper 作为参考信息，提供规则映射结果
2. 收集完整上下文：原始 C 代码、types.rs 定义、规则映射参考
3. LLM 直接输出最终结果（不是验证，是决策）

工作流程：
- 规则映射结果只是参考信息
- LLM 综合所有信息直接输出最终的 Rust 类型/签名
- 强制输出格式，不输出无关内容

使用场景：
- 函数签名翻译
- 全局变量声明翻译
- 复杂类型映射
"""

import os
import re
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

# 确保可以导入项目模块
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# 导入现有的 TypeMapper
TYPE_MAPPER_AVAILABLE = False
TypeMapper = None

try:
    from type_mapper import TypeMapper as _TypeMapper
    TypeMapper = _TypeMapper
    TYPE_MAPPER_AVAILABLE = True
except ImportError as e:
    # 尝试直接导入
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("type_mapper", _project_root / "type_mapper.py")
        if spec and spec.loader:
            type_mapper_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(type_mapper_module)
            TypeMapper = type_mapper_module.TypeMapper
            TYPE_MAPPER_AVAILABLE = True
    except Exception as e2:
        print(f"警告: TypeMapper 不可用 ({e}, {e2})")

# 导入提示词
try:
    from config.type_mapping_prompts import (
        TYPE_MAPPING_SYSTEM_PROMPT,
        TYPE_MAPPING_USER_PROMPT,
        SIGNATURE_TRANSLATION_SYSTEM_PROMPT,
        SIGNATURE_TRANSLATION_USER_PROMPT,
        GLOBAL_VAR_TRANSLATION_SYSTEM_PROMPT,
        GLOBAL_VAR_TRANSLATION_USER_PROMPT,
        BATCH_TYPE_MAPPING_SYSTEM_PROMPT,
        BATCH_TYPE_MAPPING_USER_PROMPT,
        extract_available_types,
        format_batch_type_mappings,
    )
    PROMPTS_AVAILABLE = True
except ImportError:
    PROMPTS_AVAILABLE = False
    print("警告: 类型映射提示词不可用")

logger = logging.getLogger(__name__)


@dataclass
class TypeMappingResult:
    """类型映射结果"""
    c_type: str                          # 原始 C 类型
    preprocessed_rust_type: str          # TypeMapper 预处理结果
    final_rust_type: str                 # 最终 Rust 类型
    needs_correction: bool = False       # 是否需要修正
    confidence: float = 1.0              # 置信度
    reasoning: str = ""                  # 修正原因
    warnings: List[str] = field(default_factory=list)  # 警告信息


@dataclass
class SignatureTranslationResult:
    """函数签名翻译结果"""
    c_signature: str                     # 原始 C 签名
    preprocessed_signature: str          # TypeMapper 预处理结果
    final_signature: str                 # 最终 Rust 签名
    function_modifier: str               # fn 或 pub extern "C" fn
    parameters: List[Dict]               # 参数列表
    return_type: Dict                    # 返回类型
    needs_correction: bool = False       # 是否需要修正
    confidence: float = 1.0              # 置信度
    warnings: List[str] = field(default_factory=list)


class LLMTypeMapper:
    """
    LLM 驱动的类型映射器
    
    设计理念：
    - TypeMapper 只作为预处理，提供参考结果
    - 所有类型都经过 LLM 处理，确保正确性
    - LLM 拥有最终决策权
    
    工作流程：
    1. 使用 TypeMapper 进行预处理，得到参考结果
    2. 收集完整上下文（原始 C 代码、types.rs 定义、预处理结果）
    3. 调用 LLM 进行最终判断
    4. 返回 LLM 的结果
    """
    
    def __init__(
        self,
        types_rs_path: Optional[Path] = None,
        llm_client = None,
        model_name: str = "qwen3_coder",
        enable_llm: bool = True,
        cache_enabled: bool = True,
    ):
        """
        初始化 LLM 类型映射器
        
        Args:
            types_rs_path: types.rs 文件路径
            llm_client: OpenAI 兼容的 LLM 客户端
            model_name: 模型名称
            enable_llm: 是否启用 LLM 验证
            cache_enabled: 是否启用缓存
        """
        self.types_rs_path = types_rs_path
        self.llm_client = llm_client
        self.model_name = model_name
        self.enable_llm = enable_llm and llm_client is not None
        self.cache_enabled = cache_enabled
        
        # 缓存
        self._types_rs_content: Optional[str] = None
        self._available_types: Optional[str] = None
        self._type_mapping_cache: Dict[str, TypeMappingResult] = {}
        
        # 加载 types.rs
        if types_rs_path:
            self._load_types_rs()
    
    def _load_types_rs(self):
        """加载 types.rs 内容"""
        if self.types_rs_path and self.types_rs_path.exists():
            try:
                self._types_rs_content = self.types_rs_path.read_text(encoding='utf-8')
                if PROMPTS_AVAILABLE:
                    self._available_types = extract_available_types(self._types_rs_content)
                else:
                    self._available_types = self._extract_types_simple()
                logger.info(f"已加载 types.rs: {len(self._types_rs_content)} 字符")
            except Exception as e:
                logger.warning(f"无法加载 types.rs: {e}")
    
    def _extract_types_simple(self) -> str:
        """简单的类型提取（不依赖提示词模块）"""
        if not self._types_rs_content:
            return "// 没有可用类型"
        
        types = []
        for match in re.finditer(r'pub\s+(struct|enum|type|union)\s+(\w+)', self._types_rs_content):
            types.append(f"// {match.group(1)} {match.group(2)}")
        
        return '\n'.join(types) if types else "// 没有找到类型定义"
    
    def _preprocess_with_type_mapper(
        self, 
        c_type: str, 
        is_pointer: bool = False, 
        is_const: bool = False
    ) -> str:
        """使用 TypeMapper 进行预处理"""
        if TYPE_MAPPER_AVAILABLE:
            return TypeMapper.map_c_type(c_type, is_pointer, is_const)
        else:
            # 基础回退
            return c_type.replace('struct ', '').replace('enum ', '').strip()
    
    def _call_llm(self, system_prompt: str, user_prompt: str, force_json: bool = True) -> Optional[str]:
        """
        调用 LLM
        
        Args:
            system_prompt: 系统提示
            user_prompt: 用户提示
            force_json: 是否强制 JSON 输出格式
        """
        if not self.llm_client:
            return None
        
        try:
            # 构建消息
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # 如果强制 JSON，添加格式约束
            if force_json:
                messages.append({
                    "role": "assistant", 
                    "content": "{"  # 强制以 JSON 开始
                })
            
            api_kwargs = dict(
                model=self.model_name,
                messages=messages,
                temperature=0.05,
                max_tokens=1500,
            )
            # Disable reasoning/thinking for DeepSeek v4 models
            if "deepseek-v4" in str(self.model_name).lower():
                api_kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
            response = self.llm_client.chat.completions.create(**api_kwargs)
            
            result = response.choices[0].message.content
            
            # 如果强制 JSON，补全开头的 {
            if force_json and result and not result.strip().startswith('{'):
                result = '{' + result
            
            return result
        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            return None
    
    def _parse_json_response(self, response: str) -> Optional[Dict]:
        """解析 LLM 的 JSON 响应"""
        if not response:
            return None
        
        try:
            # 尝试提取 JSON 块
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            
            # 尝试直接解析
            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON 解析失败: {e}")
            return None
    
    def map_type(
        self,
        c_type: str,
        is_pointer: bool = False,
        is_const: bool = False,
        is_array: bool = False,
        array_size: Optional[str] = None,
        context_location: str = "unknown",
        c_code_snippet: str = "",
    ) -> TypeMappingResult:
        """
        映射单个 C 类型到 Rust 类型
        
        Args:
            c_type: 原始 C 类型
            is_pointer: 是否为指针
            is_const: 是否为 const
            is_array: 是否为数组
            array_size: 数组大小
            context_location: 使用上下文（参数/返回值/全局变量等）
            c_code_snippet: 原始 C 代码片段（提供上下文）
        
        Returns:
            TypeMappingResult 对象
        """
        # 检查缓存
        cache_key = f"{c_type}:{is_pointer}:{is_const}:{is_array}:{array_size}:{context_location}"
        if self.cache_enabled and cache_key in self._type_mapping_cache:
            return self._type_mapping_cache[cache_key]
        
        # Step 1: TypeMapper 预处理（作为参考信息）
        preprocessed = self._preprocess_with_type_mapper(c_type, is_pointer, is_const)
        
        # Step 2: 如果 LLM 未启用，使用预处理结果
        if not self.enable_llm:
            result = TypeMappingResult(
                c_type=c_type,
                preprocessed_rust_type=preprocessed,
                final_rust_type=preprocessed,
                needs_correction=False,
                confidence=0.7,
                warnings=["LLM 未启用，使用 TypeMapper 预处理结果"]
            )
            if self.cache_enabled:
                self._type_mapping_cache[cache_key] = result
            return result
        
        # Step 3: 调用 LLM 进行最终判断
        if not PROMPTS_AVAILABLE:
            # 无提示词模块，使用预处理结果
            result = TypeMappingResult(
                c_type=c_type,
                preprocessed_rust_type=preprocessed,
                final_rust_type=preprocessed,
                needs_correction=False,
                confidence=0.7,
                warnings=["提示词模块不可用，未进行 LLM 验证"]
            )
            if self.cache_enabled:
                self._type_mapping_cache[cache_key] = result
            return result
        
        user_prompt = TYPE_MAPPING_USER_PROMPT.format(
            c_type=c_type,
            preprocessed_rust_type=preprocessed,
            context_location=context_location,
            is_pointer=is_pointer,
            is_const=is_const,
            is_array=is_array,
            array_size=array_size or "N/A",
            available_types=self._available_types or "// 没有可用类型信息",
            c_code_snippet=c_code_snippet or "// 没有代码片段",
        )
        
        llm_response = self._call_llm(TYPE_MAPPING_SYSTEM_PROMPT, user_prompt, force_json=True)
        parsed = self._parse_json_response(llm_response)
        
        if parsed:
            # LLM 直接输出最终结果
            final_type = parsed.get('rust_type', preprocessed)
            result = TypeMappingResult(
                c_type=c_type,
                preprocessed_rust_type=preprocessed,
                final_rust_type=final_type,
                needs_correction=(final_type != preprocessed),
                confidence=parsed.get('confidence', 0.9),
                reasoning=parsed.get('reasoning', ''),
                warnings=parsed.get('warnings', []),
            )
        else:
            # LLM 调用失败，回退到规则映射结果
            result = TypeMappingResult(
                c_type=c_type,
                preprocessed_rust_type=preprocessed,
                final_rust_type=preprocessed,
                needs_correction=False,
                confidence=0.6,
                warnings=["LLM 调用失败，回退到规则映射结果"]
            )
        
        if self.cache_enabled:
            self._type_mapping_cache[cache_key] = result
        return result
    
    def translate_signature(
        self,
        c_signature: str,
        c_function_code: str = "",
        function_name: str = "",
        is_static: bool = False,
        source_file: str = "",
        other_signatures: str = "",
    ) -> SignatureTranslationResult:
        """
        翻译 C 函数签名为 Rust
        
        Args:
            c_signature: 原始 C 函数签名
            c_function_code: 完整的 C 函数代码
            function_name: 函数名
            is_static: 是否为 static 函数
            source_file: 源文件名
            other_signatures: 同文件的其他函数签名
        
        Returns:
            SignatureTranslationResult 对象
        """
        # Step 1: 使用 TypeMapper 预处理（作为参考信息）
        func_mod = "fn"
        if TYPE_MAPPER_AVAILABLE:
            # 解析 C 签名
            params, ret_type = self._parse_c_signature(c_signature)
            func_mod, params_str, ret_str = TypeMapper.process_function_signature(
                ret_type, params, is_static
            )
            preprocessed = f"{func_mod} {function_name}({params_str}){ret_str}"
        else:
            preprocessed = c_signature  # 无法预处理
        
        # Step 2: 如果 LLM 未启用，使用预处理结果
        if not self.enable_llm:
            return SignatureTranslationResult(
                c_signature=c_signature,
                preprocessed_signature=preprocessed,
                final_signature=preprocessed,
                function_modifier=func_mod,
                parameters=[],
                return_type={},
                needs_correction=False,
                confidence=0.7,
                warnings=["LLM 未启用，使用 TypeMapper 预处理结果"]
            )
        
        # Step 3: 调用 LLM 进行最终判断
        if not PROMPTS_AVAILABLE:
            return SignatureTranslationResult(
                c_signature=c_signature,
                preprocessed_signature=preprocessed,
                final_signature=preprocessed,
                function_modifier=func_mod if TYPE_MAPPER_AVAILABLE else "fn",
                parameters=[],
                return_type={},
                needs_correction=False,
                confidence=0.7,
                warnings=["提示词模块不可用"]
            )
        
        user_prompt = SIGNATURE_TRANSLATION_USER_PROMPT.format(
            c_signature=c_signature,
            c_function_code=c_function_code or c_signature,
            preprocessed_signature=preprocessed,
            available_types=self._available_types or "// 没有可用类型信息",
            other_signatures=other_signatures or "// 没有其他签名",
            function_name=function_name,
            is_static=is_static,
            source_file=source_file,
        )
        
        llm_response = self._call_llm(SIGNATURE_TRANSLATION_SYSTEM_PROMPT, user_prompt, force_json=True)
        parsed = self._parse_json_response(llm_response)
        
        if parsed:
            # LLM 直接输出最终的 Rust 签名
            final_sig = parsed.get('rust_signature', preprocessed)
            return SignatureTranslationResult(
                c_signature=c_signature,
                preprocessed_signature=preprocessed,
                final_signature=final_sig,
                function_modifier=parsed.get('function_modifier', 'pub extern "C" fn'),
                parameters=parsed.get('parameters', []),
                return_type=parsed.get('return_type', {}),
                needs_correction=(final_sig != preprocessed),
                confidence=parsed.get('confidence', 0.9),
                warnings=parsed.get('warnings', []),
            )
        else:
            # LLM 调用失败，回退到规则映射结果
            return SignatureTranslationResult(
                c_signature=c_signature,
                preprocessed_signature=preprocessed,
                final_signature=preprocessed,
                function_modifier=func_mod if TYPE_MAPPER_AVAILABLE else "fn",
                parameters=[],
                return_type={},
                needs_correction=False,
                confidence=0.6,
                warnings=["LLM 调用失败，回退到规则映射结果"]
            )
    
    def _parse_c_signature(self, c_signature: str) -> Tuple[List[Tuple[str, str]], str]:
        """解析 C 函数签名，提取参数列表和返回类型"""
        # 简化的解析器
        params = []
        ret_type = "void"
        
        # 匹配函数签名
        match = re.match(r'(\w[\w\s\*]+?)\s+(\w+)\s*\((.*?)\)', c_signature, re.DOTALL)
        if match:
            ret_type = match.group(1).strip()
            # func_name = match.group(2)
            params_str = match.group(3).strip()
            
            if params_str and params_str != 'void':
                for param in params_str.split(','):
                    param = param.strip()
                    if param:
                        # 简单分割：最后一个空格/星号后是参数名
                        parts = re.split(r'\s+', param)
                        if len(parts) >= 2:
                            param_name = parts[-1].lstrip('*')
                            param_type = ' '.join(parts[:-1])
                            if '*' in parts[-1]:
                                param_type += '*'
                            params.append((param_name, param_type))
                        else:
                            params.append((param, "int"))
        
        return params, ret_type
    
    def batch_map_types(
        self,
        type_mappings: List[Dict],
    ) -> List[TypeMappingResult]:
        """
        批量映射类型（所有类型都经过 LLM 处理）
        
        Args:
            type_mappings: 类型映射列表，每个元素是 dict:
                {
                    "c_type": "C 类型",
                    "is_pointer": bool,
                    "is_const": bool,
                    "context": "使用上下文"
                }
        
        Returns:
            TypeMappingResult 列表
        """
        results = []
        llm_batch = []
        
        # Step 1: 预处理所有类型，准备 LLM 批处理
        for i, mapping in enumerate(type_mappings):
            c_type = mapping.get('c_type', '')
            is_pointer = mapping.get('is_pointer', False)
            is_const = mapping.get('is_const', False)
            
            preprocessed = self._preprocess_with_type_mapper(c_type, is_pointer, is_const)
            
            # 所有类型都加入 LLM 批处理队列
            llm_batch.append({
                "id": str(i),
                "c_type": c_type,
                "preprocessed": preprocessed,
                "context": mapping.get('context', 'unknown'),
            })
        
        # Step 2: 批量调用 LLM
        if llm_batch and self.enable_llm and PROMPTS_AVAILABLE:
            user_prompt = BATCH_TYPE_MAPPING_USER_PROMPT.format(
                available_types=self._available_types or "// 没有可用类型",
                type_mappings_json=format_batch_type_mappings(llm_batch),
            )
            
            llm_response = self._call_llm(BATCH_TYPE_MAPPING_SYSTEM_PROMPT, user_prompt)
            parsed = self._parse_json_response(llm_response)
            
            if parsed and isinstance(parsed, list):
                for item in parsed:
                    idx = int(item.get('id', -1))
                    if idx >= 0:
                        corrected = item.get('corrected_rust_type', item.get('preprocessed_rust_type', ''))
                        preprocessed = item.get('preprocessed_rust_type', '')
                        results.append((idx, TypeMappingResult(
                            c_type=item.get('original_c_type', ''),
                            preprocessed_rust_type=preprocessed,
                            final_rust_type=corrected,
                            needs_correction=item.get('needs_correction', False),
                            confidence=item.get('confidence', 0.8),
                        )))
            else:
                # LLM 调用失败，使用预处理结果作为回退
                for item in llm_batch:
                    idx = int(item['id'])
                    results.append((idx, TypeMappingResult(
                        c_type=item['c_type'],
                        preprocessed_rust_type=item['preprocessed'],
                        final_rust_type=item['preprocessed'],
                        needs_correction=False,
                        confidence=0.6,
                        warnings=["LLM 批量处理失败，使用 TypeMapper 预处理结果作为回退"]
                    )))
        elif llm_batch:
            # LLM 未启用或提示词不可用，使用预处理结果作为回退
            for item in llm_batch:
                idx = int(item['id'])
                results.append((idx, TypeMappingResult(
                    c_type=item['c_type'],
                    preprocessed_rust_type=item['preprocessed'],
                    final_rust_type=item['preprocessed'],
                    needs_correction=False,
                    confidence=0.6,
                    warnings=["LLM 未启用，使用 TypeMapper 预处理结果作为回退"]
                )))
        
        # Step 3: 按原始顺序排序返回
        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]


# ============================================================
# 便捷函数
# ============================================================

def create_llm_type_mapper(
    types_rs_path: Optional[Path] = None,
    vllm_url: str = None,
    model_name: str = None,
    enable_llm: bool = True,
) -> LLMTypeMapper:
    """
    创建 LLM 类型映射器的便捷函数

    Args:
        types_rs_path: types.rs 文件路径
        vllm_url: vLLM 服务地址（默认从 generation.py 获取）
        model_name: 模型名称（默认从 generation.py 获取）
        enable_llm: 是否启用 LLM 验证

    Returns:
        LLMTypeMapper 实例
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
        timeout = 300.0

    vllm_url = vllm_url or default_url
    model_name = model_name or default_model

    llm_client = None

    if enable_llm:
        try:
            from openai import OpenAI
            llm_client = OpenAI(
                base_url=vllm_url,
                api_key=api_key,
                timeout=timeout,
            )
            # 测试连接（忽略不支持的 providers 如 DeepSeek）
            try:
                llm_client.models.list()
                logger.info(f"已连接到 LLM: {vllm_url}")
            except Exception:
                logger.info(f"LLM models.list() 不可用（部分 provider 不支持），已跳过连接测试")
        except Exception as e:
            logger.warning(f"无法连接到 LLM: {e}")
            llm_client = None

    return LLMTypeMapper(
        types_rs_path=types_rs_path,
        llm_client=llm_client,
        model_name=model_name,
        enable_llm=enable_llm and llm_client is not None,
    )


# ============================================================
# 测试代码
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("LLM 驱动类型映射器测试")
    print("=" * 60)
    
    # 测试类型映射
    test_types = [
        ("int", False, False),
        ("unsigned int", False, False),
        ("char*", True, False),
        ("const char*", True, True),
        ("struct my_struct", False, False),
        ("usart_pin_map_t[]", False, False),
        ("int[10]", False, False),
        ("void (*callback)(int)", False, False),
    ]
    
    print("\n1. 测试模式 A：LLM 未启用（使用 TypeMapper 预处理结果作为回退）")
    print("-" * 50)
    mapper_no_llm = LLMTypeMapper(enable_llm=False)
    for c_type, is_ptr, is_const in test_types:
        result = mapper_no_llm.map_type(c_type, is_ptr, is_const)
        print(f"  {c_type:30s} -> {result.final_rust_type}")
        if result.warnings:
            print(f"    ⚠️ {result.warnings[0]}")
    
    print("\n2. 测试模式 B：连接 vLLM（所有类型都经过 LLM 处理）")
    print("-" * 50)
    try:
        mapper_with_llm = create_llm_type_mapper(
            enable_llm=True,
        )
        
        if mapper_with_llm.enable_llm:
            print("  ✓ 已连接到 vLLM")
            # 测试一个类型
            test_type = "const usart_pin_map_t*"
            result = mapper_with_llm.map_type(
                test_type, 
                is_pointer=True, 
                is_const=True,
                context_location="函数参数",
                c_code_snippet="void usart_init(const usart_pin_map_t* pin_map) { ... }"
            )
            print(f"  {test_type} -> {result.final_rust_type}")
            print(f"    置信度: {result.confidence}")
            print(f"    是否修正: {result.needs_correction}")
            if result.reasoning:
                print(f"    修正原因: {result.reasoning}")
        else:
            print("  ⚠️ vLLM 不可用，跳过 LLM 测试")
            
    except Exception as e:
        print(f"  ⚠️ 无法连接 vLLM: {e}")
    
    print("\n" + "=" * 60)
    print("设计说明：")
    print("-" * 60)
    print("• TypeMapper 结果只作为参考信息传递给 LLM")
    print("• 所有类型都经过 LLM 处理，LLM 拥有最终决策权")
    print("• 当 LLM 不可用时，才回退到 TypeMapper 预处理结果")
    print("=" * 60)
    print("\n✓ 测试完成")
