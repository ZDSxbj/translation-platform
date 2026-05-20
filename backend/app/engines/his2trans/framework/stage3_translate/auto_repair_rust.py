import os
import logging
import time
import sys
import re
import json
from generate.generation import generation, generation_in_parallel
from auto_test_rust import run
from itertools import islice
from pathlib import Path

# 设置日志配置
logging.basicConfig(filename=f"repair_throughLLM.log", level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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
        if translated_code == None:
            try:
                translated_code = re.findall(pattern, translated_result, re.DOTALL)[0].strip()
                
            except:
                translated_code = None
        else:
            break
    
    if translated_code == None:
        translated_code = translated_result
    

    return translated_code

# ---------------- 新增：整仓构建与按文件聚合修复 ----------------
import subprocess

def _cargo_build(project_dir: Path, timeout: int = 900):
    """
    在 project_dir 下运行 cargo build，返回 (stdout, stderr, success)
    
    使用 RUSTFLAGS 抑制无害警告，专注于真正的编译错误：
    - unused_imports: LLM 倾向于添加防御性 use 语句
    - dead_code: 增量翻译中间状态
    - unused_variables: 占位参数
    - unused_mut: 防御性可变声明
    """
    try:
        import os
        env = os.environ.copy()
        env["RUSTFLAGS"] = "-A unused_imports -A dead_code -A unused_variables -A unused_mut"
        
        cmd = ["cargo", "build"]
        try:
            from cargo_utils import with_cargo_jobs
            cmd = with_cargo_jobs(cmd)
        except Exception:
            pass

        result = subprocess.run(
            cmd,
            cwd=str(project_dir), 
            timeout=timeout, 
            capture_output=True, 
            text=True,
            env=env
        )
        output = result.stdout
        error = result.stderr
        success = (result.returncode == 0) or ("Finished" in output)
        return output, error, success
    except subprocess.TimeoutExpired as e:
        return "", f"cargo build timeout: {e}", False
    except Exception as e:
        return "", f"cargo build exception: {e}", False


def _parse_errors_by_file(stderr_text: str):
    """粗略解析 cargo stderr，将以 error 开头的块按文件聚合，返回 {filename: [error_block, ...]}"""
    if not stderr_text:
        return {}
    # 取从第一个 "error:" 或 "error[E" 起的内容
    start1 = stderr_text.find("error:")
    start2 = stderr_text.find("error[E")
    if start1 == -1 and start2 == -1:
        body = stderr_text
    else:
        starts = [i for i in [start1, start2] if i != -1]
        body = stderr_text[min(starts):]
    blocks = body.split("\n\n")
    file_to_blocks = {}
    for b in blocks:
        bb = b.strip()
        if not (bb.startswith("error[") or bb.startswith("error:")):
            continue
        lines = bb.splitlines()
        # 经验：第二行通常含有形如 "--> path/to/file.rs:line:col"
        if len(lines) >= 2 and "-->" in lines[1]:
            loc = lines[1]
            # 提取文件名
            try:
                fn = loc.split("-->")[1].strip().split(":")[0]
                filename = os.path.basename(fn)
            except Exception:
                filename = None
        else:
            filename = None
        if filename:
            file_to_blocks.setdefault(filename, []).append(bb)
    return file_to_blocks


def _copy_tree_replace(src: Path, dst: Path):
    if dst.exists():
        import shutil
        shutil.rmtree(dst)
    import shutil
    shutil.copytree(src, dst)


def _read_text_safe(p: Path) -> str:
    try:
        return p.read_text(encoding='utf-8')
    except Exception:
        return ""


def _build_repair_message(previous_response: str, combined_error_messages: str, corpus_func: str,
                          query_func_signature: str, related_function_and_datatype: str) -> str:
    corpus_lang = "C++"
    query_lang = "Rust"
    return f"""
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


def _system_prompt() -> str:
    return (
        "Hello. You are a talented Rust programmer. Here you're going to help the user fix the translated Rust file from C++ with error.\n\n"
        "You have known enough for understanding and using the Rust compilation rule and syntax. Please follow the user's instructions and requirements to fix the Rust code provided by the user\n"
    )

# ---------------- 隔离修复产物生成（逐文件，不整仓构建） ----------------
def generate_isolate_repairs(source_funtion_dir_path: str,
                             translated_result_dir_path: str,
                             test_result_dir_path: str,
                             repair_result_dir_path: str,
                             llm: str,
                             dependencies_path: str):
    from workspace_config import (
        get_functions_path, get_signature_match_path, get_skeleton_path,
        get_dependencies_not_in_file_rs_path
    )
    from project_config import PROJECT_NAME
    project = PROJECT_NAME
    llm_name = llm.replace("translate_by_", "") if llm.startswith("translate_by_") else llm
    translated_dir = Path(translated_result_dir_path)
    test_results_dir = Path(test_result_dir_path)
    target_dir = Path(repair_result_dir_path)
    target_dir.mkdir(parents=True, exist_ok=True)
    functions_dir = get_functions_path(project)
    signature_dir = get_signature_match_path(project, llm_name)
    skeleton_dir = get_skeleton_path(project)
    deps_not_in_file_rs_dir = get_dependencies_not_in_file_rs_path(project)

    if not translated_dir.exists():
        print(f"错误: 翻译结果目录不存在: {translated_dir}")
        return

    files = [f for f in os.listdir(translated_dir) if f.endswith(".txt")]
    print(f"隔离修复：发现 {len(files)} 个待处理文件 → {target_dir}")
    for question_path in files:
        try:
            output_path = target_dir / question_path
            if output_path.exists():
                continue
            # 读取逐文件测试结果
            test_file = test_results_dir / question_path
            error = test_file.read_text(encoding="utf-8", errors="ignore") if test_file.exists() else ""
            if error.startswith("Success"):
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write("Success\n")
                    translated_file = translated_dir / question_path
                    if translated_file.exists():
                        f.write(translated_file.read_text(encoding="utf-8", errors="ignore"))
                continue

            # 上下文收集
            rs_path = "_".join(question_path.split("_")[:-1]) + ".rs"
            func_file = functions_dir / question_path
            corpus_func = func_file.read_text(encoding="utf-8", errors="ignore") if func_file.exists() else ""
            translated_file = translated_dir / question_path
            if translated_file.exists():
                prev_text = translated_file.read_text(encoding="utf-8", errors="ignore")
                m = re.findall(r'<translated function>(.*?)</translated function>', prev_text, re.DOTALL)
                previous_response = m[0].strip() if m else ""
            else:
                previous_response = ""
            signature_file = signature_dir / question_path
            query_func_signature = signature_file.read_text(encoding="utf-8", errors="ignore") if signature_file.exists() else ""
            related_function_and_datatype = ""
            sk_rs = skeleton_dir / "src" / rs_path
            if sk_rs.exists():
                related_function_and_datatype += sk_rs.read_text(encoding="utf-8", errors="ignore")
            dep_other = skeleton_dir / "src" / "dependency_from_other_project.rs"
            if dep_other.exists():
                related_function_and_datatype += "\n" + dep_other.read_text(encoding="utf-8", errors="ignore")
            dep_not_file = deps_not_in_file_rs_dir / question_path
            if dep_not_file.exists():
                related_function_and_datatype += "\n\n" + dep_not_file.read_text(encoding="utf-8", errors="ignore")

            # 错误片段提取
            start1 = error.find("error:")
            start2 = error.find("error[E")
            if start2 == -1:
                start2 = start1 + 1
            starts = [x for x in [start1, start2] if x != -1]
            start = min(starts) if starts else -1
            if start >= 0:
                error = error[start:]
            parts = error.split("\n\n")
            parts = [x for x in parts if x.startswith("error[E") or x.startswith("error:")]
            error_messages = "\n\n".join(parts)

            # 组 prompt
            message = f"""
# Code Repair
## Task Description
you were asked to translate the given C++ function to Rust. Some errors occurred when executing your code. Fix the error in the Rust code from previous response.

## Basic Details about the function
Here are the basic details about the function under repair
<previous response>
{previous_response}
</previous response>
<error message>
{error_messages}
</error message>
<C++ function>
{corpus_func}
</C++ function>
<Rust function signature>
{query_func_signature}
</Rust function signature>
<Rust function dependencies, and data type declarations>
{related_function_and_datatype}
</Rust function dependencies and data type declarations>

## Note
The Repair process must adhere to the following rules:
    - ** ONLY fix the error code, repeat completely the rest of the code. **
    - ** Please fix all the errors mentioned in <error message>**

## Output Format
```rust
** (only reply with the translated result of the focal code) **
```
"""
            sys_prompt = _system_prompt()
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": message},
            ]
            
            # 保存提示词
            try:
                from save_llm_prompts import save_llm_prompt, save_llm_prompt_text
                from project_config import PROJECT_NAME
                metadata = {
                    "file_name": question_path,
                    "has_signature": bool(query_func_signature),
                }
                save_llm_prompt(
                    messages=messages,
                    project_name=PROJECT_NAME,
                    llm_name=llm,
                    task_type="isolate_repair",
                    function_name=question_path.replace(".txt", ""),
                    metadata=metadata
                )
                save_llm_prompt_text(
                    messages=messages,
                    project_name=PROJECT_NAME,
                    llm_name=llm,
                    task_type="isolate_repair",
                    function_name=question_path.replace(".txt", ""),
                    metadata=metadata
                )
            except Exception as e:
                logging.warning(f"保存提示词失败: {e}")
            
            response = generation(messages)
            repaired = extract_code(response)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(f"<message>\n{message}\n</message>\n")
                f.write(f"<function>\n{query_func_signature}\n</function>\n<translated function>\n{repaired}</translated function>")
        except Exception as e:
            print(f"  ✗ 隔离修复生成失败: {question_path}: {e}")
            logging.error(f"isolate gen failed for {question_path}: {e}")

# ---------------- 主流程：整仓多轮修复（完全按照原版逻辑） ----------------

def _write_functions_to_skeleton(skeleton_dir: Path, translated_dir: Path, signature_dir: Path, target_project_dir: Path):
    """
    将翻译的函数体写回骨架项目，形成完整的项目目录（用于整仓编译）
    使用改进的签名匹配逻辑
    """
    from auto_test_rust import (
        read_translated_function, add_import_to_translated_result, 
        get_frist_function_position_start, find_and_replace_function_signature
    )
    
    translated_files = list(translated_dir.glob("*.txt"))
    for translated_file in translated_files:
        try:
            translated_content = translated_file.read_text(encoding="utf-8", errors="ignore")
            if translated_content.startswith("Success"):
                continue

            source_code, translated_function, translated_code_import = read_translated_function(translated_content)
            translated_function = "\n".join(translated_function)

            function_path = "_".join(translated_file.name.split("_")[:-1]) + ".rs"
            target_rs_file = target_project_dir / "src" / function_path
            if not target_rs_file.exists():
                continue

            skeleton_result = target_rs_file.read_text(encoding="utf-8", errors="ignore")

            # 尝试读取签名文件（若存在则优先按签名替换，否则回退为“插入到第一个函数位置”）
            translated_function_signature = ""
            signature_file = signature_dir / translated_file.name
            if signature_file.exists():
                translated_function_signature = signature_file.read_text(encoding="utf-8", errors="ignore").strip()

            translated_result = None
            if translated_function_signature:
                translated_result, success = find_and_replace_function_signature(
                    skeleton_result, translated_function_signature, translated_function
                )
                if not success:
                    # 兼容旧逻辑：字符串 split 替换（对部分项目/签名格式更宽松）
                    try:
                        left, right = skeleton_result.split(translated_function_signature, 1)
                        translated_result = None
                        for line in skeleton_result.splitlines():
                            if translated_function_signature in line:
                                if line.strip().endswith(";"):
                                    translated_result = left + translated_function + right[right.find(";") + 1 :]
                                else:
                                    translated_result = left + translated_function + right[right.find("}") + 1 :]
                                break
                    except Exception:
                        translated_result = None

            if not translated_result:
                # 没有签名文件或替换失败：在第一个函数位置插入
                frist_function_position_start = get_frist_function_position_start(skeleton_result)
                translated_result = (
                    skeleton_result[:frist_function_position_start]
                    + translated_function
                    + skeleton_result[frist_function_position_start:]
                )

            # 添加导入并写回
            translated_result = add_import_to_translated_result(translated_result, translated_code_import)
            target_rs_file.write_text(translated_result, encoding="utf-8")
        except Exception as e:
            logging.error(f"写入函数到骨架失败 {translated_file.name}: {e}")
            continue


def whole_crate_repair_flow(repair_result_dir_path: str, llm: str):
    """
    完全按照原版 translate.py::repair() 的逻辑：
    整仓编译→解析错误→按 .rs 文件修复→多轮
    """
    from workspace_config import (
        get_skeleton_path, get_source_skeleton_path,
        get_signature_match_path, get_translated_path
    )
    from project_config import PROJECT_NAME
    from auto_test_rust import run_tests

    project = PROJECT_NAME
    llm_name = llm.replace("translate_by_", "") if llm.startswith("translate_by_") else llm

    skeleton_dir = get_skeleton_path(project)
    source_skeleton_dir = get_source_skeleton_path(project)
    signature_dir = get_signature_match_path(project, llm_name)
    translated_dir = get_translated_path(project, llm_name)
    
    # 判断是否为外层指定的 round 目录（只跑一轮）
    single_round = False
    base_name = os.path.basename(str(repair_result_dir_path))
    if "round_" in base_name:
        single_round = True
    total_rounds = 1 if single_round else 3

    # 初始项目目录为骨架（用于最后写回）
    skeleton_dir_path = get_skeleton_path(project)
    target_project_path = skeleton_dir_path
    previous_project_path = skeleton_dir_path
    
    # 备份原始骨架（按照原版逻辑）
    backup_path = skeleton_dir_path.parent / f"{project}_before_repair"
    if backup_path.exists():
        import shutil
        shutil.rmtree(backup_path)
    import shutil
    _copy_tree_replace(previous_project_path, backup_path)
    previous_project_path = backup_path

    for i in range(total_rounds):
        round_num = i + 1
        print(f"\n{'='*60}")
        print(f"整仓修复 轮次 {round_num}/{total_rounds}")
        print(f"项目: {project}")
        print(f"LLM: {llm_name}")
        print(f"基线工程: {previous_project_path}")
        print(f"{'='*60}\n")
        
        # 1) 准备完整项目
        current_repair_dir = Path(repair_result_dir_path)
        if not single_round:
            current_repair_dir = current_repair_dir / f"round_{round_num}"
        current_repair_project_dir = current_repair_dir / project
        current_repair_project_dir.mkdir(parents=True, exist_ok=True)
        
        # 检查是否已存在修复结果
        if (current_repair_project_dir / "src").exists() and len(list((current_repair_project_dir / "src").glob("*.rs"))) > 0:
            print(f"已存在修复结果: {current_repair_project_dir}，跳过")
            previous_project_path = current_repair_project_dir
            continue
        
        if round_num == 1:
            # 第一轮：从骨架开始，将函数体写回
            _copy_tree_replace(previous_project_path, current_repair_project_dir)
            print("将函数体写回骨架项目...")
            _write_functions_to_skeleton(skeleton_dir, translated_dir, signature_dir, current_repair_project_dir)
        else:
            # 后续轮次：直接使用上一轮修复后的项目目录（已包含所有函数体）
            _copy_tree_replace(previous_project_path, current_repair_project_dir)
        
        # 2) 整仓构建
        print("运行整仓编译...")
        output, error, ok = _cargo_build(current_repair_project_dir)
        
        if ok:
            print("✓ 整仓构建成功，无需继续修复")
            try:
                _copy_tree_replace(current_repair_project_dir, target_project_path)
            except Exception as e:
                print(f"拷贝回骨架失败: {e}")
            return True

        # 3) 解析错误，按文件聚合
        start1 = error.find("error:")
        start2 = error.find("error[E")
        if start1 == -1 and start2 == -1:
            error_body = error
        else:
            starts = [i for i in [start1, start2] if i != -1]
            error_body = error[min(starts):]
        
        error_messages = error_body.split("\n\n")
        error_files_to_messages = {}
        for error_message in error_messages:
            if error_message.strip().startswith("error[E"):
                error_lines = error_message.splitlines()
                if len(error_lines) >= 2:
                    # 从第二行提取文件名：--> path/to/file.rs:line:col
                    loc_line = error_lines[1]
                    if "-->" in loc_line:
                        try:
                            error_file = loc_line.split("-->")[1].strip().split(":")[0]
                            error_file = os.path.basename(error_file)
                            if error_file not in error_files_to_messages:
                                error_files_to_messages[error_file] = []
                            error_files_to_messages[error_file].append(error_message)
                        except:
                            pass
        
        if not error_files_to_messages:
            print("⚠️ 未解析到错误文件，结束本轮")
            previous_project_path = current_repair_project_dir
            continue
        
        print(f"发现 {len(error_files_to_messages)} 个报错的 .rs 文件，开始修复...")
        
        # 4) 逐文件修复（按 .rs 文件，不是按函数文件）
        repaired_count = 0
        failed_count = 0
        for error_file, error_messages_list in error_files_to_messages.items():
            rs_file_path = current_repair_project_dir / "src" / error_file
            if not rs_file_path.exists():
                print(f"  ⚠️ 警告: 文件不存在: {rs_file_path}")
                continue
            
            # 读取整个 .rs 文件
            with open(rs_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                previous_response = f.read()
            
            # 读取对应的 C++ 源文件（.txt）
            txt_file_path = source_skeleton_dir / "src" / error_file.replace(".rs", ".txt")
            corpus_func = ""
            if txt_file_path.exists():
                with open(txt_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    corpus_func = f.read()
            
            combined_error_messages = "\n\n".join(error_messages_list)
            corpus_lang = "C++"
            query_lang = "Rust"

            system_prompt = _system_prompt()
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
            try:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                
                # 保存提示词
                try:
                    from save_llm_prompts import save_llm_prompt, save_llm_prompt_text
                    from project_config import PROJECT_NAME
                    metadata = {
                        "file_name": error_file,
                        "round": round_num,
                        "error_type": "whole_crate",
                    }
                    save_llm_prompt(
                        messages=messages,
                        project_name=PROJECT_NAME,
                        llm_name=llm,
                        task_type="whole_crate_repair",
                        function_name=f"{error_file}_round_{round_num}",
                        metadata=metadata
                    )
                    save_llm_prompt_text(
                        messages=messages,
                        project_name=PROJECT_NAME,
                        llm_name=llm,
                        task_type="whole_crate_repair",
                        function_name=f"{error_file}_round_{round_num}",
                        metadata=metadata
                    )
                except Exception as e:
                    logging.warning(f"保存提示词失败: {e}")
                
                response = generation(messages)
                repair_result = extract_code(response)
                with open(rs_file_path, 'w', encoding='utf-8') as f:
                    f.write(repair_result)
                print(f"  ✓ 修复完成: {error_file}")
                repaired_count += 1
            except Exception as e:
                print(f"  ✗ 修复失败: {error_file} - {e}")
                logging.error(f"repair failed for {error_file}: {e}")
                failed_count += 1
        
        print(f"  修复统计: 成功 {repaired_count}, 失败 {failed_count}, 总计 {len(error_files_to_messages)}")
        
        # 5) 下一轮使用本轮修复后的项目目录
        previous_project_path = current_repair_project_dir
    
    # 最后将上一轮结果复制回骨架
    try:
        _copy_tree_replace(previous_project_path, target_project_path)
    except Exception as e:
        print(f"拷贝回骨架失败: {e}")
    
    # 返回修复的文件总数（如果存在）
    if 'error_files_to_messages' in locals():
        return len(error_files_to_messages)
    return 0


# ---------------- 逐函数修复（按照原版逻辑） ----------------
def read_message_function_repair(source_funtion_dir_path: str, 
                                  translated_result_dir_path: str, 
                                  test_result_dir_path: str, 
                                  repair_result_dir_path: str, 
                                  llm: str, 
                                  dependencies_path: str) -> int:
    """
    按照原版逻辑：逐函数修复
    读取测试结果，为每个失败的函数生成修复结果
    
    返回: 修复的文件数量
    """
    from workspace_config import (
        get_functions_path, get_signature_match_path, get_skeleton_path,
        get_dependencies_not_in_file_rs_path, get_translated_path
    )
    from project_config import PROJECT_NAME
    
    project = PROJECT_NAME
    llm_name = llm.replace("translate_by_", "") if llm.startswith("translate_by_") else llm
    
    target_dir = Path(repair_result_dir_path) / f"translate_by_{llm_name}"
    translated_result_dir_path_obj = Path(translated_result_dir_path) / f"translate_by_{llm_name}"
    test_result_dir_path_obj = Path(test_result_dir_path) / f"translate_by_{llm_name}"
    
    target_dir.mkdir(parents=True, exist_ok=True)
    project_dir = target_dir / project
    project_dir.mkdir(parents=True, exist_ok=True)
    
    functions_dir = get_functions_path(project)
    signature_dir = get_signature_match_path(project, llm_name)
    skeleton_dir = get_skeleton_path(project)
    deps_not_in_file_rs_dir = get_dependencies_not_in_file_rs_path(project)
    
    if not translated_result_dir_path_obj.exists():
        print(f"错误: 翻译结果目录不存在: {translated_result_dir_path_obj}")
        return 0
    
    questions_path = list(translated_result_dir_path_obj.glob("*.txt"))
    print(f"逐函数修复：发现 {len(questions_path)} 个待处理函数")
    
    repair_count = 0  # 计数修复的文件数量
    
    for question_path in questions_path:
        question_file = question_path.name
        output_path = project_dir / question_file
        
        if output_path.exists():
            logging.info(f" {question_file} already exists")
            print(f"{question_file} already exists")
            continue
        
        # 读取测试结果
        test_file = test_result_dir_path_obj / question_file
        if not test_file.exists():
            continue
        
        error = test_file.read_text(encoding='utf-8', errors='ignore')
        
        if error.startswith("Success"):
            print(f"already success {question_file}")
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("Success\n")
                with open(question_path, 'r', encoding='utf-8', errors='ignore') as input_file:
                    previous_response = input_file.read()
                f.write(previous_response)
            continue
        
        try:
            # 读取 C++ 函数代码
            func_file = functions_dir / question_file
            if not func_file.exists():
                continue
            corpus_func = func_file.read_text(encoding='utf-8', errors='ignore')
            
            # 读取翻译结果
            with open(question_path, 'r', encoding='utf-8', errors='ignore') as input_file:
                content = input_file.read()
                previous_response_match = re.findall(r'<translated function>(.*?)</translated function>', content, re.DOTALL)
                previous_response = previous_response_match[0].strip() if previous_response_match else ""
            
            # 读取函数签名
            signature_file = signature_dir / question_file
            query_func_signature = signature_file.read_text(encoding='utf-8', errors='ignore') if signature_file.exists() else ""
            
            # 读取依赖和类型声明
            rs_path = "_".join(question_file.split("_")[:-1]) + ".rs"
            related_function_and_datatype = ""
            sk_rs = skeleton_dir / "src" / rs_path
            if sk_rs.exists():
                related_function_and_datatype = sk_rs.read_text(encoding='utf-8', errors='ignore')
            dep_other = skeleton_dir / "src" / "dependency_from_other_project.rs"
            if dep_other.exists():
                related_function_and_datatype += "\n" + dep_other.read_text(encoding='utf-8', errors='ignore')
            dep_not_file = deps_not_in_file_rs_dir / question_file
            if dep_not_file.exists():
                related_function_and_datatype += "\n\n" + dep_not_file.read_text(encoding='utf-8', errors='ignore')
            
            # 提取错误信息
            start1 = error.find("error:")
            start2 = error.find("error[E")
            if start2 == -1:
                start2 = start1 + 1 if start1 != -1 else -1
            starts = [x for x in [start1, start2] if x != -1]
            if not starts:
                continue
            start = min(starts)
            error = error[start:]
            
            error_messages = error.split("\n\n")
            error_messages = [x for x in error_messages if x.startswith("error[E") or x.startswith("error:")]
            error_messages = "\n\n".join(error_messages)
            
            corpus_lang = "C++"
            query_lang = "Rust"
            
            message = f"""
# Code Repair
## Task Description
you were asked to translate the given {corpus_lang} function to {query_lang}. Some errors occurred when executing your code. Fix the error in the Rust code from previous response.

## Basic Details about the function
Here are the basic details about the function under repair
<previous response>\n{previous_response}\n</previous response>\n<error message>\n{error_messages}\n</error message>\n<{corpus_lang} function>\n{corpus_func}\n</{corpus_lang} function>\n<{query_lang} function signature>\n{query_func_signature}\n</{query_lang} function signature>\n<{query_lang} function dependencies, and data type declarations>\n{related_function_and_datatype}\n</{query_lang} function dependencies and data type declarations>\n

## Note
The Repair process must adhere to the following rules:
    - ** ONLY fix the error code, repeat completely the rest of the code. **
    - ** Please fix all the errors mentioned in <error message>**

## Output Format
```rust
** (only reply with the translated result of the focal code) **
```
"""
            
            system_prompt = _system_prompt()
            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": message
                }
            ]
            
            # 保存提示词
            try:
                from save_llm_prompts import save_llm_prompt, save_llm_prompt_text
                from project_config import PROJECT_NAME
                metadata = {
                    "file_name": question_path,
                    "has_signature": bool(query_func_signature),
                    "has_previous_response": bool(previous_response),
                }
                save_llm_prompt(
                    messages=messages,
                    project_name=PROJECT_NAME,
                    llm_name=llm_name,
                    task_type="function_repair",
                    function_name=question_path.replace(".txt", ""),
                    metadata=metadata
                )
                save_llm_prompt_text(
                    messages=messages,
                    project_name=PROJECT_NAME,
                    llm_name=llm_name,
                    task_type="function_repair",
                    function_name=question_path.replace(".txt", ""),
                    metadata=metadata
                )
            except Exception as e:
                logging.warning(f"保存提示词失败: {e}")
            
            response = generation(messages)
            translated_code = extract_code(response)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"<message>\n{message}\n</message>\n")
                f.write(f"<function>\n{query_func_signature}\n</function>\n<translated function>\n{translated_code}</translated function>")
            
            print(f"successfully get {output_path}")
            repair_count += 1
            
        except Exception as e:
            logging.info(f"Error processing {question_file}: {e}")
            print(f"Error processing {question_file}: {e}")
            continue
    
    return repair_count


# ---------------- 兼容旧 CLI，支持逐函数修复（原版逻辑）和整仓修复 ----------------
if __name__ == "__main__":
    from project_config import PROJECT_NAME
    
    source_funtion_dir_path = sys.argv[1]
    translated_result_dir_path = sys.argv[2]
    test_result_dir_path = sys.argv[3]
    repair_result_dir_path = sys.argv[4]
    llm = sys.argv[5]
    dependencies_path = sys.argv[6]
    
    print(f"\n{'='*60}")
    print(f"自动修复流程开始")
    print(f"{'='*60}")
    print(f"修复结果目录: {repair_result_dir_path}")
    print(f"LLM: {llm}")
    print(f"{'='*60}\n")

    # 如果是"隔离修复"阶段（目录名中包含 isolate），则执行逐文件修复产物生成
    path_concat = " ".join([translated_result_dir_path, test_result_dir_path, repair_result_dir_path]).lower()
    if "isolate" in path_concat:
        print("检测到隔离修复模式（路径包含 'isolate'），执行逐文件修复产物生成，不进行整仓构建。")
        try:
            generate_isolate_repairs(
                source_funtion_dir_path,
                translated_result_dir_path,
                test_result_dir_path,
                repair_result_dir_path,
                llm,
                dependencies_path
            )
            print("隔离修复产物已生成。")
        except Exception as e:
            print(f"✗ 隔离修复阶段失败: {e}")
            import traceback
            traceback.print_exc()
        sys.exit(0)

    # 按照原版逻辑：逐函数修复（最多5轮）
    from auto_test_rust import run
    
    previous_translate_result_path = translated_result_dir_path
    previous_test_result_path = test_result_dir_path
    
    import time
    repair_start_time = time.time()
    
    for i in range(5):  # 原版是5轮
        round_start = time.time()
        print(f"\n{'='*60}")
        print(f"逐函数修复 第 {i+1}/5 轮")
        print(f"{'='*60}\n")
        
        # 生成修复结果
        repair_count = read_message_function_repair(
            source_funtion_dir_path,
            previous_translate_result_path,
            previous_test_result_path,
            repair_result_dir_path + f"_{i+1}",
            llm,
            dependencies_path
        )
        
        # 测试修复结果（按照原版逻辑）
        # 原版的 run 函数期望路径结构：translate_result_path/target_llm/target_project/
        # 当前代码的 read_message_function_repair 生成的路径结构：repair_result_dir_path/translate_by_{llm_name}/{project}/
        # 所以需要传入正确的路径
        llm_name = llm.replace("translate_by_", "") if llm.startswith("translate_by_") else llm
        from auto_test_rust import run
        # 传入修复结果目录（包含 translate_by_{llm_name}/{project}/ 结构）
        repair_result_full_path = repair_result_dir_path + f"_{i+1}"
        test_result = run(
            repair_result_full_path,
            test_result_dir_path + f"_repair_{i+1}",
            f"translate_by_{llm_name}",
            PROJECT_NAME
        )
        
        previous_translate_result_path = repair_result_dir_path + f"_{i+1}"
        previous_test_result_path = test_result_dir_path + f"_repair_{i+1}"
        round_time = time.time() - round_start
        print(f"完成第 {i+1} 轮修复 (耗时: {round_time:.1f}秒)")
        if repair_count is not None and repair_count > 0:
            print(f"  - 修复了 {repair_count} 个文件")
        
        # 如果没有修复任何文件，提前退出
        if repair_count == 0:
            print("  - 没有需要修复的文件，提前结束修复流程")
            break
    
    total_repair_time = time.time() - repair_start_time
    print(f"\n{'='*60}")
    print(f"自动修复流程完成！总耗时: {total_repair_time:.1f}秒 ({total_repair_time/60:.1f}分钟)")
    print(f"{'='*60}\n")
