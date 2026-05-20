import os
import logging
import time
import sys
import re
import json
from pathlib import Path
from threading import Lock, Semaphore
from concurrent.futures import ThreadPoolExecutor, as_completed
from generate.generation import generation, generation_in_parallel

from itertools import islice

# 设置日志配置
logging.basicConfig(filename=f"translate_throughLLM.log", level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def extract_opaque_types(dependencies_path: str) -> list:
    """
    从 skeleton 目录中提取 opaque 类型列表。

    提取来源（按优先级）：
    1. compat.rs 中的 accessor shim 函数: c2r_field_ptr_TypeName__field_name
    2. types.rs 中定义的 opaque 结构体: pub struct TypeName { _opaque: ... }
    3. types.rs 中的 c_void 类型别名: pub type TypeName = c_void

    Args:
        dependencies_path: skeleton 目录路径

    Returns:
        list: 去重后的 opaque 类型名称列表
    """
    opaque_types = set()
    deps_path = Path(dependencies_path)

    # 常见的系统类型（这些类型 LLM 不会直接访问字段）
    system_types = {
        'FILE', 'pthread_mutex_t', 'pthread_cond_t',
        'pthread_attr_t', 'pthread_rwlock_t', 'pthread_t',
        '_IO_FILE', '_IO_marker', '_IO_codecvt', '_IO_wide_data'
    }

    # === 来源 1: 从 compat.rs 中提取 accessor shim 函数的类型名 ===
    # 这是最可靠的来源，因为只有需要字段访问的 opaque 类型才会有 accessor shim
    compat_file = deps_path / "src" / "compat.rs"
    if compat_file.exists():
        try:
            content = compat_file.read_text(encoding='utf-8', errors='ignore')
            # 匹配 c2r_field_ptr_TypeName__field_name 模式
            # 函数名格式: c2r_field_ptr_<StructName>__<field_name>
            accessor_pattern = r'pub fn c2r_field_ptr_([A-Za-z_][A-Za-z0-9_]*)__([A-Za-z_][A-Za-z0-9_]*)\s*\('
            for match in re.finditer(accessor_pattern, content):
                type_name = match.group(1)
                if type_name not in system_types:
                    opaque_types.add(type_name)
        except Exception as e:
            logging.debug(f"读取 compat.rs 提取 opaque 类型失败: {e}")

    # === 来源 2: 从 types.rs 中提取 opaque 结构体定义 ===
    types_file = deps_path / "src" / "types.rs"
    if types_file.exists():
        try:
            content = types_file.read_text(encoding='utf-8', errors='ignore')

            # 2a: 匹配 pub struct TypeName { _opaque: [u8; N] } 模式
            opaque_patterns = [
                # 单行格式: pub struct TypeName { _opaque: [u8; 0] }
                r'pub struct ([A-Za-z_][A-Za-z0-9_]*)\s*\{\s*_opaque:\s*\[u8;\s*\d+\]\s*\}',
                # 多行格式（处理可能的换行）
                r'pub struct ([A-Za-z_][A-Za-z0-9_]*)\s*\{[^}]*_opaque:\s*\[u8;\s*\d+\][^}]*\}',
            ]
            for pattern in opaque_patterns:
                for match in re.finditer(pattern, content, re.DOTALL):
                    type_name = match.group(1)
                    if type_name not in system_types:
                        opaque_types.add(type_name)

            # 2b: 查找 "/// Opaque placeholder" 注释后的结构体
            opaque_comment_pattern = r'///\s*Opaque placeholder[^\n]*\n(?:[^\n]*\n)*?pub struct ([A-Za-z_][A-Za-z0-9_]*)'
            for match in re.finditer(opaque_comment_pattern, content):
                type_name = match.group(1)
                if type_name not in system_types:
                    opaque_types.add(type_name)

            # 2c: 匹配 pub type TypeName = c_void; 模式（通常是 opaque 句柄类型）
            void_alias_pattern = r'pub type ([A-Za-z_][A-Za-z0-9_]*)\s*=\s*c_void\s*;'
            for match in re.finditer(void_alias_pattern, content):
                type_name = match.group(1)
                if type_name not in system_types:
                    opaque_types.add(type_name)

        except Exception as e:
            logging.debug(f"读取 types.rs 提取 opaque 类型失败: {e}")

    # 返回排序后的列表
    return sorted(list(opaque_types))


def extract_code(content):


    # pattern = r'\[translated result\](.*?)'
    # translated_result = re.findall(pattern, content, re.DOTALL)[0].strip()
    translated_result = content
    # translated_result = content.split("[translated result]")[1]
    # print("---")
    # print(translated_result)
    # translated_result = translated_result.split("[#END]")[0]
    # print("---")
    # print(translated_result)

    translated_code = None

    patterns = [r'```rust(.*?)```',r'```Rust(.*?)```']

    for pattern in patterns:
        if translated_code is None:
            try:
                match = re.findall(pattern, translated_result, re.DOTALL)
                if match:
                    candidate = match[0].strip()
                    if candidate:  # reject empty code blocks
                        translated_code = candidate
            except:
                translated_code = None
        if translated_code is not None and translated_code.strip():
            break

    if translated_code is None or not translated_code.strip():
        translated_code = translated_result


    return translated_code

# 用于线程安全的打印锁
print_lock = Lock()

# vLLM 并发控制信号量 - 限制同时最多 120 个请求
# 从环境变量获取配置，默认 120
VLLM_CONCURRENT_LIMIT = int(os.environ.get("VLLM_CONCURRENT_LIMIT", "120"))
vllm_semaphore = Semaphore(VLLM_CONCURRENT_LIMIT)

def process_single_function(args):
    """
    处理单个函数的翻译任务（用于并行处理）
    """
    question_path, source_dir, target_dir, llm, dependencies_path, rag_path_function, project = args
    use_rag_context = os.environ.get("C2R_USE_RAG_CONTEXT", "true").strip().lower() in (
        "1",
        "true",
        "yes",
        "y",
        "on",
    )

    # RQ3.3 knowledge-channel ablation (C0-C6). Defaults MUST keep old behavior.
    def _norm_mode(raw: str) -> str:
        v = (raw or "").strip().lower()
        if v in ("none", "n", "no", "off", "false", "0"):
            return "none"
        if v in ("oracle", "o"):
            return "oracle"
        return "predicted"

    rq3_api_mode = _norm_mode(os.environ.get("C2R_RQ3_API_MAPPING_MODE", ""))
    rq3_partial_mode = _norm_mode(os.environ.get("C2R_RQ3_PARTIAL_IDIOM_MODE", ""))
    oracle_root = (os.environ.get("C2R_ORACLE_KNOWLEDGE_ROOT") or "").strip()
    if not oracle_root:
        oracle_root = str(Path(__file__).resolve().parent.parent / "oracle_rust")
    oracle_auto_extract = (os.environ.get("C2R_ORACLE_AUTO_EXTRACT", "0").strip().lower() in ("1", "true", "yes", "y", "on"))
    
    from workspace_config import (
        get_signature_match_path, get_dependencies_not_in_file_rs_path,
        get_translated_path, WORKSPACE_ROOT
    )
    
    target_dir = get_translated_path(project, llm)
    target_dir.mkdir(parents=True, exist_ok=True)
    
    rs_path = "_".join(question_path.split("_")[:-1]) + ".rs"
    output_path = target_dir / question_path
    
    # 检查是否已存在
    if output_path.exists():
        return {"status": "skipped", "file": question_path}
    
    try:
        # 读取问题和依赖信息
        with open(os.path.join(source_dir, question_path), 'r', encoding='utf-8', errors='ignore') as input_file:
            corpus_func = input_file.read()

        try:
            signature_file = get_signature_match_path(project, llm) / question_path
            if signature_file.exists():
                with open(signature_file, 'r', encoding='utf-8', errors='ignore') as input_file:
                    query_func_signature = input_file.read()
            else:
                query_func_signature = ""
        except:
            query_func_signature = ""
        
        skeleton_file = Path(dependencies_path) / "src" / rs_path
        if skeleton_file.exists():
            with open(skeleton_file, 'r', encoding='utf-8', errors='ignore') as input_file:
                related_function_and_datatype = input_file.read()
        else:
            related_function_and_datatype = ""
        
        dependency_other_file = Path(dependencies_path) / "src" / "dependency_from_other_project.rs"
        if dependency_other_file.exists():
            with open(dependency_other_file, 'r', encoding='utf-8', errors='ignore') as input_file:
                related_function_and_datatype += "\n" + input_file.read()
        
        deps_not_in_file = get_dependencies_not_in_file_rs_path(project) / question_path
        if deps_not_in_file.exists():
            with open(deps_not_in_file, 'r', encoding='utf-8', errors='ignore') as input_file:
                related_function_and_datatype += "\n\n" + input_file.read()

        translation_examples_func = ""
        extracted_knowledge_text = ""

        # Fast path: keep legacy behavior unchanged unless ablation modes are enabled.
        if rq3_api_mode == "predicted" and rq3_partial_mode == "predicted":
            if use_rag_context:
                try:
                    rag_file = Path(rag_path_function) / question_path
                    if rag_file.exists():
                        with open(rag_file, 'r', encoding='utf-8', errors='ignore') as input_file:
                            rag_content = input_file.read()
                            # 每个 block 对应一个检索到的候选代码对（按 reranker 排序）
                            rag_blocks = [b for b in rag_content.split("-" * 50) if b.strip()]

                            # 例子（Rust code examples）保持旧行为：只取前 2 个 code pair，避免 prompt 变长造成混杂变量
                            example_blocks = rag_blocks[:2]

                            # 知识注入的 top-k：限制“知识条目数”，而不是“代码对数”
                            rag_topk = None
                            _rag_topk_raw = os.environ.get("C2R_RAG_TOPK", "").strip()
                            if _rag_topk_raw:
                                try:
                                    rag_topk = int(_rag_topk_raw)
                                except Exception:
                                    rag_topk = None

                            if rag_topk is not None and rag_topk > 0:
                                top_k_api = rag_topk
                                top_k_partial = rag_topk
                            else:
                                top_k_api = 10
                                top_k_partial = 2

                            # --- Rust code examples ---
                            translation_examples = []
                            for i, block in enumerate(example_blocks, start=1):
                                if "Function:" in block and "Unixcoder Score:" in block:
                                    rust_example = block.split("Function:")[1].split("Unixcoder Score:")[0].strip()
                                    translation_examples.append(f"[rust code {i}]\n{rust_example}")
                            translation_examples_func = "\n".join(translation_examples)

                            # --- Extracted Knowledge (API_Mapping / Partial) ---
                            api_mapping_hints = []
                            partial_hints = []

                            for block in rag_blocks:
                                if len(api_mapping_hints) >= top_k_api and len(partial_hints) >= top_k_partial:
                                    break
                                if "Extracted_Knowledge:" not in block:
                                    continue
                                try:
                                    knowledge_str = block.split("Extracted_Knowledge:")[1].split("Unixcoder Score:")[0].strip()
                                    if not knowledge_str:
                                        continue
                                    knowledge_list = json.loads(knowledge_str)
                                    if isinstance(knowledge_list, dict):
                                        knowledge_list = [knowledge_list]
                                    if not isinstance(knowledge_list, list):
                                        continue
                                except (json.JSONDecodeError, IndexError):
                                    continue

                                for k in knowledge_list:
                                    k_type = k.get("knowledge_type", "Unknown")
                                    if k_type == "API_Mapping" and len(api_mapping_hints) < top_k_api:
                                        c_api = k.get("c_api", "")
                                        rust_api = k.get("rust_api", "")
                                        desc = k.get("description", "")
                                        if c_api and rust_api:
                                            api_mapping_hints.append(f"- API Mapping: `{c_api}` → `{rust_api}` ({desc})")
                                    elif k_type == "Partial" and len(partial_hints) < top_k_partial:
                                        c_frag = k.get("c_fragment", "")[:50]  # 截断避免过长
                                        rust_frag = k.get("rust_fragment", "")[:50]
                                        desc = k.get("description", "")
                                        if c_frag and rust_frag:
                                            partial_hints.append(f"- Pattern: `{c_frag}...` → `{rust_frag}...` ({desc})")
                                    # Full 类型不再添加到提示词中

                            # 合并：先 Partial（代码模式），后 API_Mapping（API 映射）
                            knowledge_hints = partial_hints + api_mapping_hints
                            extracted_knowledge_text = "\n".join(knowledge_hints) if knowledge_hints else ""
                except Exception as e:
                    logging.info(f"Error reading RAG file {question_path}: {e}")
        else:
            # Controlled ablation path (C0-C6): per-channel = none/predicted/oracle.
            func_key = Path(question_path).stem  # e.g. <file_name>_<index>

            # top-k: cap by knowledge items (same semantics as incremental_translate.py)
            rag_topk = None
            _rag_topk_raw = os.environ.get("C2R_RAG_TOPK", "").strip()
            if _rag_topk_raw:
                try:
                    rag_topk = int(_rag_topk_raw)
                except Exception:
                    rag_topk = None

            if rag_topk is not None and rag_topk > 0:
                top_k_api = rag_topk
                top_k_partial = rag_topk
            else:
                top_k_api = 10
                top_k_partial = 2

            api_mapping_hints = []
            partial_hints = []

            def _clean_json_text(s: str) -> str:
                s = (s or "").strip()
                if s.startswith("```json"):
                    s = s[len("```json") :]
                if s.startswith("```"):
                    s = s[len("```") :]
                if s.endswith("```"):
                    s = s[: -len("```")]
                return s.strip()

            def _extract_oracle_knowledge_via_llm(c_code: str, rust_code: str) -> list:
                from vllm_parallel_knowledge_extractor import PROMPT_TEMPLATE, convert_llm_record_to_extracted_knowledge
                prompt = PROMPT_TEMPLATE.replace("{c_code}", c_code or "").replace("{rust_code}", rust_code or "")
                content = generation([{"role": "user", "content": prompt}])
                cleaned = _clean_json_text(content)
                try:
                    obj = json.loads(cleaned)
                except Exception:
                    start = cleaned.find("{")
                    end = cleaned.rfind("}")
                    obj = json.loads(cleaned[start : end + 1]) if start >= 0 and end > start else {}
                if not isinstance(obj, dict):
                    return []
                extracted = convert_llm_record_to_extracted_knowledge(obj, source="oracle_vllm")
                return [k for k in extracted if isinstance(k, dict) and k.get("knowledge_type") in ("API_Mapping", "Partial")]

            def _load_oracle_extracted_knowledge() -> list:
                proj_dir = Path(oracle_root) / str(project)
                rec_path = proj_dir / f"{func_key}.json"

                cache_dir = WORKSPACE_ROOT / "rag" / "oracle_extracted_cache" / str(project)
                cache_path = cache_dir / f"{func_key}.json"
                if cache_path.exists():
                    try:
                        data = json.loads(cache_path.read_text(encoding="utf-8", errors="ignore") or "[]")
                        return data if isinstance(data, list) else []
                    except Exception:
                        pass

                rec = None
                if rec_path.exists():
                    try:
                        rec = json.loads(rec_path.read_text(encoding="utf-8", errors="ignore") or "{}")
                    except Exception:
                        rec = None
                if not isinstance(rec, dict):
                    return []

                ek = rec.get("extracted_knowledge")
                if isinstance(ek, list) and ek:
                    try:
                        cache_dir.mkdir(parents=True, exist_ok=True)
                        cache_path.write_text(json.dumps(ek, ensure_ascii=False, indent=2), encoding="utf-8")
                    except Exception:
                        pass
                    return ek

                if not oracle_auto_extract:
                    return []

                rust_code = (rec.get("rust_code") or "").strip()
                if not rust_code:
                    return []

                extracted = _extract_oracle_knowledge_via_llm(corpus_func, rust_code)
                if extracted:
                    try:
                        cache_dir.mkdir(parents=True, exist_ok=True)
                        cache_path.write_text(json.dumps(extracted, ensure_ascii=False, indent=2), encoding="utf-8")
                    except Exception:
                        pass
                return extracted

            # -------- predicted knowledge (RAG) --------
            rag_blocks = []
            if use_rag_context and (rq3_api_mode == "predicted" or rq3_partial_mode == "predicted"):
                try:
                    rag_file = Path(rag_path_function) / question_path
                    if rag_file.exists():
                        rag_content = rag_file.read_text(encoding="utf-8", errors="ignore")
                        rag_blocks = [b for b in rag_content.split("-" * 50) if b.strip()]
                except Exception as e:
                    logging.info(f"Error reading RAG file {question_path}: {e}")

            # Examples are treated as part of the Partial/Idiom channel to avoid leakage in oracle/none settings.
            if rag_blocks and rq3_partial_mode == "predicted":
                example_blocks = rag_blocks[:2]
                translation_examples = []
                for i, block in enumerate(example_blocks, start=1):
                    if "Function:" in block and "Unixcoder Score:" in block:
                        rust_example = block.split("Function:")[1].split("Unixcoder Score:")[0].strip()
                        translation_examples.append(f"[rust code {i}]\n{rust_example}")
                translation_examples_func = "\n".join(translation_examples)

            if rag_blocks:
                for block in rag_blocks:
                    if len(api_mapping_hints) >= top_k_api and len(partial_hints) >= top_k_partial:
                        break
                    if "Extracted_Knowledge:" not in block:
                        continue
                    try:
                        knowledge_str = block.split("Extracted_Knowledge:")[1].split("Unixcoder Score:")[0].strip()
                        if not knowledge_str:
                            continue
                        knowledge_list = json.loads(knowledge_str)
                        if isinstance(knowledge_list, dict):
                            knowledge_list = [knowledge_list]
                        if not isinstance(knowledge_list, list):
                            continue
                    except Exception:
                        continue

                    for k in knowledge_list:
                        k_type = k.get("knowledge_type", "Unknown")
                        if k_type == "API_Mapping" and rq3_api_mode == "predicted" and len(api_mapping_hints) < top_k_api:
                            c_api = (k.get("c_api") or "").strip()
                            rust_api = (k.get("rust_api") or "").strip()
                            desc = (k.get("description") or "").strip()
                            if c_api and rust_api:
                                api_mapping_hints.append(f"- API Mapping: `{c_api}` → `{rust_api}` ({desc})")
                        elif k_type == "Partial" and rq3_partial_mode == "predicted" and len(partial_hints) < top_k_partial:
                            c_frag = (k.get("c_fragment") or "")[:50]
                            rust_frag = (k.get("rust_fragment") or "")[:50]
                            desc = (k.get("description") or "").strip()
                            if c_frag and rust_frag:
                                partial_hints.append(f"- Pattern: `{c_frag}...` → `{rust_frag}...` ({desc})")

            # -------- oracle knowledge --------
            if rq3_api_mode == "oracle" or rq3_partial_mode == "oracle":
                oracle_knowledge = _load_oracle_extracted_knowledge()
                if oracle_knowledge:
                    if rq3_api_mode == "oracle":
                        for k in oracle_knowledge:
                            if len(api_mapping_hints) >= top_k_api:
                                break
                            if not isinstance(k, dict):
                                continue
                            if k.get("knowledge_type") != "API_Mapping":
                                continue
                            c_api = (k.get("c_api") or "").strip()
                            rust_api = (k.get("rust_api") or "").strip()
                            desc = (k.get("description") or "").strip()
                            if c_api and rust_api:
                                api_mapping_hints.append(f"- API Mapping: `{c_api}` → `{rust_api}` ({desc})")
                    if rq3_partial_mode == "oracle":
                        for k in oracle_knowledge:
                            if len(partial_hints) >= top_k_partial:
                                break
                            if not isinstance(k, dict):
                                continue
                            if k.get("knowledge_type") != "Partial":
                                continue
                            c_frag = (k.get("c_fragment") or "")[:50]
                            rust_frag = (k.get("rust_fragment") or "")[:50]
                            desc = (k.get("description") or "").strip()
                            if c_frag and rust_frag:
                                partial_hints.append(f"- Pattern: `{c_frag}...` → `{rust_frag}...` ({desc})")

            knowledge_hints = partial_hints + api_mapping_hints
            extracted_knowledge_text = "\n".join(knowledge_hints) if knowledge_hints else ""

        # [V10 新增] 构建知识增强的翻译消息
        translation_message = f"""
## Rust code examples
Here are some rust codes for reference.
"""
        translation_message += translation_examples_func
        
        # [V10 新增] 如果有提取的知识，添加到消息中
        if extracted_knowledge_text:
            translation_message += f"""

## Extracted Translation Knowledge
The following patterns and API mappings were extracted from similar C-to-Rust translations:
{extracted_knowledge_text}
"""
        
        corpus_lang = "C++"
        query_lang = "Rust"

        # 提取 opaque 类型列表
        opaque_types_list = extract_opaque_types(dependencies_path)
        if opaque_types_list:
            opaque_types_str = ", ".join(f"`{t}`" for t in opaque_types_list)
            opaque_types_section = f"""## Opaque Type Field Access (CRITICAL - MUST READ)
**The following types are OPAQUE in this project and require special handling:**
{opaque_types_str}

**RULE: Direct field access `(*ptr).field_name` will NOT compile for opaque types!**

For ALL field accesses on these opaque types, you MUST use accessor shim functions:
- Read pattern: `(*s).field_name` → `*(crate::compat::c2r_field_ptr_TypeName__field_name(s as *mut _) as *mut FieldType)`
- Write pattern: `*(crate::compat::c2r_field_ptr_TypeName__field_name(ptr as *mut _) as *mut FieldType) = value;`
- The accessor function follows naming: `c2r_field_ptr_<StructName>__<field_name>`

Example for `EState`:
- C code: `s->bsBuff = 0;`
- Rust code: `*(crate::compat::c2r_field_ptr_EState__bsBuff(s as *mut _) as *mut UInt32) = 0;`
"""
        else:
            opaque_types_section = """## Opaque Type Field Access (IMPORTANT)
Some struct types in the Rust codebase are defined as **opaque types** without internal field definitions:
- Opaque types have the form: `pub struct TypeName {{ _opaque: [u8; 0] }}`
- **Direct field access like `(*ptr).field_name` will NOT compile for opaque types!**

For opaque types, you MUST use **accessor shim functions** instead of direct field access:
- Pattern: `(*s).field_name` → `*(crate::compat::c2r_field_ptr_TypeName__field_name(s as *mut _) as *mut FieldType)`
- The accessor function name follows: `c2r_field_ptr_<StructName>__<field_name>`
"""

        message = f"""
# Code Translation
## Task Description
Translate the focal {corpus_lang} function to {query_lang}

## Basic Details about the function
Here are the basic details about the function under translation
<focal {corpus_lang} function>
[source code of function]
{corpus_func}
</focal {corpus_lang} function>
<{query_lang} function signature>
{query_func_signature}
</{query_lang} function signature>
<{query_lang} function dependencies, and data type declarations>\n{related_function_and_datatype}\n</{query_lang} function dependencies and data type declarations>

## Steps to Translate
Please translate the function following the given steps
1. Confirm the functionality to be implemented by the current function
2. Distinguish the dependencies differences between the source and target programming languages
2.1 The focal function may have differences in dependencies between the source language and the programming language. For example:
- Variable Differences: For example, variable names may differ, or the structure of the data types corresponding to the variables may vary.
- Function Differences: For instance, function names might be different, or a function that exists in the source language might not be available in the target language.
- Data Type Differences: There may be differences in custom data types between the source language and the target language.
2.2 Enumerate all used dependencies including function, data type and variable within this function in target programming languages(if any)
3. Enumerate all used local variables and their mut status and data types in previous translated code snippets(if any)
4. Distinguish the syntax differences between the source and target programming languages(if any). For example:
- In C, Java, and Python, variables declared are mutable by default, while in Rust, variables are immutable by default. To make a variable mutable in Rust, the `mut` keyword must be explicitly used.
- In C and Java, null pointer checks are performed at runtime for variables, whereas in Rust, such checks are not required for variables that are not of the `Option` type.
- Memory safety checks required in C are not required in Rust.
- Data type checking required in Python is not required in Rust.
5. Translate the focal function based on the functionality implemented by the function, the dependencies used, the translated code snippets and the syntax differences.

## Note
The translation process must adhere to the following rules:
- Do not perform a simple one-to-one translation; instead, consider the functional consistency of the code. The translated result only needs to achieve the same functionality as the original language's function.
- ** Ensure that the local variables used genuinely exist in the <translated code snippets>, the dependencies used are actually present in the provided <{query_lang} function dependencies, and data type declarations>, and the syntax used is valid in the target language. **
- For an `if` code block, if the condition being checked is unnecessary in Rust according to the syntax differences, the `if` block can be omitted entirely, and the translated result should be an empty line. Examples include null pointer checks in C and Java, type checks in Python, or memory safety checks for structs in C.

{opaque_types_section}

## Array Index Type Casting
In Rust, array indices must be of type `usize`. When the index variable is `i32`, `u32`, or other integer types:
- Cast the index: `arr[idx as usize]` instead of `arr[idx]`
- Example: `block[i]` where `i: Int32` → `block[i as usize]`

{translation_message}

## Output Format
```{query_lang}
** (only reply with the translated result of the focal function) **
```
"""

        system_prompt = f"""
You, a professional {corpus_lang} programmer & Rust programmer, the co-worker with the user in the pair-programming, are going to translate a method from {corpus_lang} to Rust following the user's instructions. You'll be provided with:

1. The source code of the focal method in {corpus_lang} version
2. The signature of the focal method in Rust version
3. The source code of dependencies including function dependencies, variable dependencies and data type dependencies of the focal method in Rust version .

The instructions will be in detail in each phase's user prompt.

The basic information of your workarounds are:
1. The Programming Langauge: {corpus_lang}, Rust
2. The Language Style: Rust 2021

The basic requirements for your responses are:

1. Complete all required tasks as outlined in the user's message in a SINGLE response.
2. Adhere meticulously to all instructions provided by the user.
3. Deliver precise and accurate responses, getting straight to the point.


Now you're going to be shown to the user. You're going to follow the user's instructions on executing the plan. We expect your excellent performance.                
"""
        
        # 构建消息列表
        messages = [
            {
                "role" : "system", 
                "content" : system_prompt
            },
            {
                "role" : "user",
                "content" : message
            }
        ]
        
        # 保存提示词
        try:
            from save_llm_prompts import save_llm_prompt, save_llm_prompt_text
            # 提取函数名（从 question_path 中提取，例如 "camera_device_client_0.txt" -> "camera_device_client")
            func_name = "_".join(question_path.split("_")[:-1]) if "_" in question_path else question_path.replace(".txt", "")
            metadata = {
                "file_name": question_path,
                "signature": query_func_signature[:200] if query_func_signature else "",
                "has_rag": bool(translation_examples_func or extracted_knowledge_text),
                "corpus_lang": corpus_lang,
                "query_lang": query_lang
            }
            save_llm_prompt(
                messages=messages,
                project_name=project,
                llm_name=llm,
                task_type="function_translate",
                function_name=func_name,
                metadata=metadata
            )
            save_llm_prompt_text(
                messages=messages,
                project_name=project,
                llm_name=llm,
                task_type="function_translate",
                function_name=func_name,
                metadata=metadata
            )
        except Exception as e:
            logging.warning(f"保存提示词失败: {e}")
        
        # 调用generation并获取token使用信息
        # 使用信号量控制并发：同时最多 VLLM_CONCURRENT_LIMIT 个请求
        vllm_semaphore.acquire()
        try:
            result = generation(messages, return_usage=True)
        finally:
            vllm_semaphore.release()

        # 处理返回结果（可能是字典或字符串，保持向后兼容）
        if isinstance(result, dict):
            response = result["content"]
            usage = result.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)
        else:
            # 向后兼容：如果返回的是字符串
            response = result
            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0
        
        translated_code = extract_code(response)
        
        with open(output_path, 'w') as f:
            f.write(f"<message>\n{message}\n</message>\n")
            f.write(f"<function>\n{query_func_signature}\n</function>\n<translated function>\n{translated_code}</translated function>")
        
        return {
            "status": "success", 
            "file": question_path,
            "tokens": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            }
        }
        
    except Exception as e:
        error_msg = str(e)
        with print_lock:
            print(f"  ✗ 翻译失败: {question_path}")
            print(f"    错误: {error_msg}")
        logging.error(f"Error processing {question_path}: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {"status": "error", "file": question_path, "error": error_msg}


def read_message(source_dir, target_dir, llm, dependencies_path, rag_path_function, max_workers=48):
    """
    函数体翻译，支持并行处理
    
    Args:
        source_dir: 源函数目录
        target_dir: 目标目录（未使用，由内部函数动态获取）
        llm: LLM名称
        dependencies_path: 依赖路径
        rag_path_function: RAG路径
        max_workers: 最大并行工作线程数，默认48
    """
    # 使用新的工作空间路径
    from workspace_config import (
        get_signature_match_path, get_dependencies_not_in_file_rs_path,
        get_translated_path
    )
    from project_config import PROJECT_NAME
    
    project = PROJECT_NAME
    target_dir = get_translated_path(project, llm)
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # source_dir 现在直接是 functions 目录，不再有项目子目录
    if not os.path.exists(source_dir):
        print(f"错误: 源目录不存在: {source_dir}")
        return
    
    questions_path = [f for f in os.listdir(source_dir) if f.endswith('.txt')]
    
    print(f"\n{'='*60}")
    print(f"函数体翻译开始")
    print(f"{'='*60}")
    print(f"项目: {project}")
    print(f"LLM: {llm}")
    print(f"源目录: {source_dir}")
    print(f"目标目录: {target_dir}")
    print(f"依赖路径: {dependencies_path}")
    print(f"RAG 路径: {rag_path_function}")
    print(f"找到 {len(questions_path)} 个函数文件")
    print(f"使用 {max_workers} 个并行工作线程")
    print(f"vLLM 并发限制: {VLLM_CONCURRENT_LIMIT} 个同时请求")
    print(f"{'='*60}\n")
    
    if len(questions_path) == 0:
        print("没有找到需要翻译的函数文件，跳过")
        return
    
    # 准备任务参数
    tasks = [
        (question_path, source_dir, target_dir, llm, dependencies_path, rag_path_function, project)
        for question_path in questions_path
    ]
    
    processed_cnt = 0
    skip_cnt = 0
    error_cnt = 0
    
    # Token统计
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    
    # 使用并行处理
    from tqdm import tqdm
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_task = {
            executor.submit(process_single_function, task): task[0] 
            for task in tasks
        }
        
        # 使用tqdm显示进度
        with tqdm(total=len(tasks), desc="函数体翻译") as pbar:
            for future in as_completed(future_to_task):
                question_path = future_to_task[future]
                try:
                    result = future.result()
                    if result["status"] == "success":
                        processed_cnt += 1
                        # 累计token使用量
                        if "tokens" in result:
                            tokens = result["tokens"]
                            total_prompt_tokens += tokens.get("prompt_tokens", 0)
                            total_completion_tokens += tokens.get("completion_tokens", 0)
                            total_tokens += tokens.get("total_tokens", 0)
                    elif result["status"] == "skipped":
                        skip_cnt += 1
                    elif result["status"] == "error":
                        error_cnt += 1
                except Exception as e:
                    error_cnt += 1
                    with print_lock:
                        print(f"  ✗ 处理异常: {question_path}")
                        print(f"    错误: {str(e)}")
                    logging.error(f"Exception processing {question_path}: {e}")
                    import traceback
                    logging.error(traceback.format_exc())
                finally:
                    pbar.update(1)
    
    print(f"\n{'='*60}")
    print(f"函数体翻译完成")
    print(f"{'='*60}")
    print(f"总文件数: {len(questions_path)}")
    print(f"已处理: {processed_cnt}")
    print(f"已跳过: {skip_cnt}")
    print(f"失败: {error_cnt}")
    if processed_cnt > 0:
        success_rate = float(processed_cnt) / (processed_cnt + error_cnt) * 100 if (processed_cnt + error_cnt) > 0 else 0
        print(f"成功率: {success_rate:.2f}%")
    print(f"\nToken 使用统计:")
    print(f"  输入 Token (Prompt): {total_prompt_tokens:,}")
    print(f"  输出 Token (Completion): {total_completion_tokens:,}")
    print(f"  总计 Token: {total_tokens:,}")
    if processed_cnt > 0:
        avg_prompt_tokens = total_prompt_tokens / processed_cnt
        avg_completion_tokens = total_completion_tokens / processed_cnt
        avg_total_tokens = total_tokens / processed_cnt
        print(f"  平均每函数:")
        print(f"    输入: {avg_prompt_tokens:.1f} tokens")
        print(f"    输出: {avg_completion_tokens:.1f} tokens")
        print(f"    总计: {avg_total_tokens:.1f} tokens")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    source_dir = sys.argv[1]
    target_dir = sys.argv[2]
    llm = sys.argv[3]
    dependencies_path = sys.argv[4]
    rag_path_function = sys.argv[5]
    # 从环境变量获取并行数量，默认48
    max_workers = int(os.environ.get("TRANSLATE_FUNCTION_MAX_WORKERS", "48"))
    read_message(source_dir, target_dir, llm, dependencies_path, rag_path_function, max_workers=max_workers)
