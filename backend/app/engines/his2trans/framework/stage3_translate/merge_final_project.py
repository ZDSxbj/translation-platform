#!/usr/bin/env python3
"""
最终项目合并脚本
将所有编译通过的函数合并到骨架中，进行完整项目编译测试，并保存最终结果
"""
import os
import sys
import shutil
import subprocess
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

from tree_sitter import Language, Parser, QueryCursor

def _query_captures(query, root_node):
    """Compat wrapper: tree_sitter >=0.25 removed Query.captures()."""
    cursor = QueryCursor(query)
    captures = cursor.captures(root_node)
    if isinstance(captures, dict):
        out = []
        for cap_name, nodes in captures.items():
            for node in nodes:
                out.append((node, cap_name))
        return out
    return captures or []
try:
    import tree_sitter_rust  # type: ignore
    _HAS_TREE_SITTER_RUST = True
except Exception:
    tree_sitter_rust = None  # type: ignore
    _HAS_TREE_SITTER_RUST = False

from workspace_config import (
    get_skeleton_path,
    get_translated_path,
    get_test_results_path,
    get_repair_results_path,
    get_signature_match_path,
    get_final_project_path,
    TRANSLATED_DIR,
)
from auto_test_rust import (
    read_translated_function,
    add_import_to_translated_result,
    find_and_replace_function_signature,
    get_frist_function_position_start,
    run_tests
)

# 初始化 tree-sitter
if _HAS_TREE_SITTER_RUST:
    RUST_LANGUAGE = Language(tree_sitter_rust.language())
    parser = Parser(RUST_LANGUAGE)
else:
    RUST_LANGUAGE = None
    parser = None


def find_last_round_repair_results(project_name: str, llm_name: str, repair_results_base_dir: Path) -> Optional[Path]:
    """
    查找最后一轮修复结果目录
    
    修复结果的目录结构（根据 auto_repair_rust.py 的实际实现）：
    - repair_results/{project_name}_1/translate_by_{llm_name}/{project}/ (第1轮)
    - repair_results/{project_name}_2/translate_by_{llm_name}/{project}/ (第2轮)
    - ...
    
    返回: 最后一轮修复结果的路径，如果没有修复则返回 None
    """
    # 查找所有修复轮次目录（格式：{project_name}_N）
    repair_rounds = []
    
    # repair_results_base_dir 应该是 repair_results/ 目录
    if not repair_results_base_dir.exists():
        return None
    
    for item in repair_results_base_dir.iterdir():
        if not item.is_dir():
            continue
        
        # 检查是否是修复轮次目录（格式：{project_name}_N）
        name = item.name
        if name.startswith(f"{project_name}_") and name != project_name:
            # 提取轮次号
            try:
                round_num = int(name.split("_")[-1])
                translated_dir = item / f"translate_by_{llm_name}" / project_name
                if translated_dir.exists() and any(translated_dir.glob("*.txt")):
                    repair_rounds.append((round_num, translated_dir))
            except (ValueError, IndexError):
                pass
    
    if not repair_rounds:
        return None
    
    # 返回最后一轮的目录
    repair_rounds.sort(key=lambda x: x[0], reverse=True)
    return repair_rounds[0][1]


def find_passed_functions(project_name: str, llm_name: str) -> Dict[str, Tuple[Path, Path]]:
    """
    找出所有编译通过的函数及其对应的翻译结果和测试结果
    
    返回: Dict[function_filename, (translated_result_path, test_result_path)]
    """
    from workspace_config import REPAIR_RESULTS_DIR, TEST_RESULTS_DIR, TRANSLATED_DIR
    
    passed_functions = {}
    
    # 1. 查找修复结果（多种可能的目录结构）
    repair_results_base_dir = REPAIR_RESULTS_DIR
    
    # 测试结果可能的路径：
    # - test_results/{project_name}/translate_by_{llm_name}/
    # - test_results/{project_name}/translate_by_{llm_name}/{project_name}/
    test_results_base = TEST_RESULTS_DIR / project_name / f"translate_by_{llm_name}"
    
    repair_test_results = None
    last_repair_translated = None
    repair_round = 0
    
    # 查找修复测试结果目录（格式：translate_by_{llm_name}_repair_N）
    repair_test_dirs = []
    test_results_parent = TEST_RESULTS_DIR / project_name
    if test_results_parent.exists():
        for item in test_results_parent.iterdir():
            if not item.is_dir():
                continue
            
            name = item.name
            if "_repair_" in name and name.startswith(f"translate_by_{llm_name}"):
                # 提取轮次号
                parts = name.split("_repair_")
                if len(parts) == 2:
                    try:
                        round_num = int(parts[1])
                        # 测试结果文件可能直接在目录下，也可能在 project_name 子目录
                        test_dir = item
                        if (item / project_name).exists():
                            test_dir = item / project_name
                        if any(test_dir.glob("*.txt")):
                            repair_test_dirs.append((round_num, test_dir))
                    except ValueError:
                        pass
    
    if repair_test_dirs:
        repair_test_dirs.sort(key=lambda x: x[0], reverse=True)
        repair_test_results = repair_test_dirs[0][1]
        repair_round = repair_test_dirs[0][0]
        last_repair_translated = find_last_round_repair_results(project_name, llm_name, repair_results_base_dir)
    
    # 2. 查找原始测试结果（处理多种路径结构）
    original_test_results = None
    for candidate in [
        test_results_base,
        test_results_base / project_name,
        TEST_RESULTS_DIR / project_name,
    ]:
        if candidate.exists() and any(candidate.glob("*.txt")):
            original_test_results = candidate
            break
    
    # 原始翻译结果
    original_translated = None
    translated_base = TRANSLATED_DIR / project_name / f"translate_by_{llm_name}"
    for candidate in [
        translated_base,
        translated_base / project_name,
        get_translated_path(project_name, llm_name),
    ]:
        if candidate.exists() and any(candidate.glob("*.txt")):
            original_translated = candidate
            break
    
    # 3. 优先使用修复结果（如果存在）
    if repair_test_results and repair_test_results.exists() and last_repair_translated and last_repair_translated.exists():
        print(f"  使用修复结果（第 {repair_round} 轮）: {repair_test_results}")
        for test_file in repair_test_results.glob("*.txt"):
            function_name = test_file.name
            try:
                with open(test_file, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                # 支持多种成功标记
                if content.strip().startswith("Success") or content.strip().lower().startswith("success"):
                    translated_file = last_repair_translated / function_name
                    if translated_file.exists():
                        passed_functions[function_name] = (translated_file, test_file)
            except Exception as e:
                print(f"  警告: 读取测试结果 {test_file} 失败: {e}")
    
    # 4. 补充原始测试结果中通过但在修复结果中没有的函数
    if original_test_results and original_test_results.exists():
        if original_translated and original_translated.exists():
            print(f"  补充原始测试结果: {original_test_results}")
        for test_file in original_test_results.glob("*.txt"):
            function_name = test_file.name
            if function_name in passed_functions:
                continue  # 已经使用修复结果
            
            try:
                with open(test_file, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                # 支持多种成功标记
                if content.strip().startswith("Success") or content.strip().lower().startswith("success"):
                    translated_file = original_translated / function_name
                    if translated_file.exists():
                        passed_functions[function_name] = (translated_file, test_file)
            except Exception as e:
                print(f"  警告: 读取测试结果 {test_file} 失败: {e}")
        else:
            print(f"  警告: 原始翻译结果目录不存在或为空")
            print(f"    尝试过的路径: {translated_base}, {get_translated_path(project_name, llm_name)}")
    else:
        print(f"  警告: 原始测试结果目录不存在或为空")
        print(f"    尝试过的路径: {test_results_base}")
    
    # 调试信息
    if len(passed_functions) == 0:
        print(f"  调试信息:")
        print(f"    repair_results_base_dir: {repair_results_base_dir} (exists: {repair_results_base_dir.exists() if repair_results_base_dir else 'N/A'})")
        print(f"    test_results_base: {test_results_base} (exists: {test_results_base.exists()})")
        print(f"    original_test_results: {original_test_results} (exists: {original_test_results.exists() if original_test_results else 'N/A'})")
        print(f"    original_translated: {original_translated} (exists: {original_translated.exists() if original_translated else 'N/A'})")
    
    return passed_functions


def remove_function_by_name(code: str, func_name: str, remove_all: bool = True) -> str:
    """
    通用辅助函数：使用 tree-sitter 按函数名移除函数定义
    
    Args:
        code: 源代码
        func_name: 函数名
        remove_all: 是否删除所有同名函数（包括 impl 块中的），默认为 True
        
    Returns:
        移除指定函数后的代码
    """
    # tree-sitter-rust 不可用时：退化为正则/lexer 定位函数块并删除
    if parser is None or RUST_LANGUAGE is None:
        try:
            from auto_test_rust import extract_function_block
        except Exception:
            extract_function_block = None

        if not extract_function_block:
            return code

        cur = code
        while True:
            block = extract_function_block(cur, f"fn {func_name}", func_name)
            if not block:
                break
            start = cur.find(block)
            if start == -1:
                break
            cur = cur[:start] + cur[start + len(block):]
            if not remove_all:
                break
        return cur

    try:
        tree = parser.parse(bytes(code, "utf8"))
    except Exception as e:
        print(f"  [Debug] 解析代码失败: {e}")
        return code

    # 查询函数定义
    query_scm = """
    (function_item
        name: (identifier) @func_name
    ) @func_def
    """
    from tree_sitter import Query
    query = Query(RUST_LANGUAGE, query_scm)
    captures = _query_captures(query, tree.root_node)
    
    ranges_to_remove = []
    
    for node, capture_name in captures:
        if capture_name == 'func_name':
            # 检查函数名是否匹配
            node_text = code[node.start_byte:node.end_byte]
            if node_text == func_name:
                # 找到对应的 function_item 节点 (父节点)
                func_def_node = node.parent
                while func_def_node and func_def_node.type != 'function_item':
                    func_def_node = func_def_node.parent
                
                if func_def_node:
                    # 检查父节点类型
                    parent = func_def_node.parent
                    
                    # 如果不删除所有，则跳过 impl 块中的函数
                    if not remove_all and parent and parent.type == 'impl_item':
                        continue
                        
                    # 避免重复添加同一范围
                    range_tuple = (func_def_node.start_byte, func_def_node.end_byte)
                    if range_tuple not in ranges_to_remove:
                        ranges_to_remove.append(range_tuple)

    if not ranges_to_remove:
        return code

    # 从后往前删除，保持索引有效
    ranges_to_remove.sort(key=lambda x: x[0], reverse=True)
    
    # 转换为 bytearray 以便操作
    code_bytes = bytearray(code, "utf8")
    
    for start, end in ranges_to_remove:
        # 同时尝试删除前面的空白行
        # 向前扫描找到换行符
        idx = start - 1
        while idx >= 0:
            if code_bytes[idx] == 10: # \n
                start = idx + 1 # 保留换行符
                break
            elif code_bytes[idx] in (32, 9): # space or tab
                idx -= 1
            else:
                # 遇到非空白字符，停止
                break
        
        # 执行删除
        del code_bytes[start:end]
        
    return code_bytes.decode("utf8")


def extract_function_name_from_code(code: str) -> Optional[str]:
    """
    从函数代码中提取函数名
    
    Args:
        code: 函数代码
        
    Returns:
        函数名，如果提取失败返回 None
    """
    match = re.search(r'\bfn\s+(\w+)\s*[<(]', code)
    if match:
        return match.group(1)
    return None


def validate_function_code(code: str) -> bool:
    """
    验证函数代码是否完整（基本语法检查）
    
    Args:
        code: 函数代码
        
    Returns:
        是否有效
    """
    if not code or not code.strip():
        return False
    
    # 检查是否包含 fn 关键字
    if 'fn ' not in code:
        return False
    
    # 检查大括号是否匹配
    open_braces = code.count('{')
    close_braces = code.count('}')
    if open_braces != close_braces:
        return False
    
    # 至少要有一对大括号（函数体）
    if open_braces < 1:
        return False
    
    return True

def infer_rs_filename(function_filename: str, translated_file: Path) -> str:
    """
    从函数文件名推导出对应的 .rs 文件名
    
    支持多种命名格式：
    - cJsonMock_1.txt -> cJsonMock.rs
    - cJsonMock_cJSON_CreateArray.txt -> cJsonMock.rs
    - module_func_name.txt -> module.rs
    
    Args:
        function_filename: 函数文件名 (例如 "cJsonMock_1.txt")
        translated_file: 翻译文件的完整路径（用于备用推导）
        
    Returns:
        推导出的 .rs 文件名
    """
    # 移除 .txt 后缀
    base_name = function_filename
    if base_name.endswith('.txt'):
        base_name = base_name[:-4]
    
    # 策略1: 如果最后一部分是纯数字，去掉它
    # 例如 cJsonMock_1 -> cJsonMock
    parts = base_name.split('_')
    if len(parts) > 1 and parts[-1].isdigit():
        return '_'.join(parts[:-1]) + '.rs'
    
    # 策略2: 尝试找到第一个大写字母开头的部分之前的内容作为模块名
    # 例如 cJsonMock_cJSON_CreateArray -> cJsonMock
    for i, part in enumerate(parts):
        if i > 0 and part and part[0].isupper() and len(part) > 1:
            # 可能是函数名的开始
            return '_'.join(parts[:i]) + '.rs'
    
    # 策略3: 默认取除了最后一个部分之外的所有部分
    if len(parts) > 1:
        return '_'.join(parts[:-1]) + '.rs'
    
    # 策略4: 直接使用基础名
    return base_name + '.rs'


def merge_functions_to_skeleton(
    skeleton_dir: Path,
    passed_functions: Dict[str, Tuple[Path, Path]],
    signature_dir: Path,
    target_project_dir: Path
) -> int:
    """
    将所有编译通过的函数合并到骨架中
    
    返回: 成功合并的函数数量
    """
    # 复制骨架到目标目录
    if target_project_dir.exists():
        shutil.rmtree(target_project_dir)
    shutil.copytree(skeleton_dir, target_project_dir)
    
    merged_count = 0
    
    # 跟踪已合并的函数名，避免重复
    merged_function_names: Dict[str, set] = {}  # rs_filename -> set of func_names
    
    # 按文件分组（同一个 .rs 文件可能包含多个函数）
    file_groups: Dict[str, list] = {}
    for function_name, (translated_file, _) in passed_functions.items():
        # 使用改进的文件名推导逻辑
        rs_filename = infer_rs_filename(function_name, translated_file)
        if rs_filename not in file_groups:
            file_groups[rs_filename] = []
        file_groups[rs_filename].append((function_name, translated_file))
    
    # 对每个 .rs 文件，合并其中的所有函数
    for rs_filename, functions in file_groups.items():
        target_rs_file = target_project_dir / "src" / rs_filename
        if not target_rs_file.exists():
            # 尝试查找相似的文件名
            src_dir = target_project_dir / "src"
            if src_dir.exists():
                available_files = [f.name for f in src_dir.glob("*.rs")]
                print(f"警告: 骨架中不存在 {rs_filename}，可用文件: {available_files}")
            else:
                print(f"警告: 骨架中不存在 {rs_filename}，跳过合并")
            continue
        
        with open(target_rs_file, 'r', encoding='utf-8', errors='ignore') as f:
            skeleton_code = f.read()
        
        # 初始化该文件的已合并函数集合
        if rs_filename not in merged_function_names:
            merged_function_names[rs_filename] = set()
        
        # 收集所有需要的导入（统一处理）
        all_imports = []
        
        # 依次合并该文件中的所有函数
        for function_name, translated_file in functions:
            try:
                with open(translated_file, 'r', encoding='utf-8', errors='ignore') as f:
                    translated_content = f.read()
                
                if translated_content.startswith("Success"):
                    continue
                
                source_code, translated_function_list, translated_code_import = read_translated_function(translated_content)
                
                # 如果提取失败或为空，跳过
                if not translated_function_list:
                    print(f"警告: 无法从 {function_name} 提取函数体，跳过")
                    continue
                
                translated_function = "\n".join(translated_function_list)
                
                # 验证函数代码完整性
                if not validate_function_code(translated_function):
                    print(f"警告: {function_name} 的函数代码不完整，跳过")
                    continue
                
                # 收集导入（后续统一添加）
                if translated_code_import:
                    all_imports.extend(translated_code_import)
                
                # 尝试读取签名文件
                signature_file = signature_dir / function_name
                translated_function_signature = ""
                if signature_file.exists():
                    with open(signature_file, 'r', encoding='utf-8', errors='ignore') as f:
                        translated_function_signature = f.read().strip()
                
                # 处理 &self 问题
                if "self" in translated_function:
                    # 移除 self 参数: ( &self, ... ) -> (...)
                    translated_function = re.sub(r'\(\s*&?mut?\s*self\s*,\s*', '(', translated_function)
                    # 移除 self 参数: ( &self ) -> ()
                    translated_function = re.sub(r'\(\s*&?mut?\s*self\s*\)', '()', translated_function)

                    if translated_function_signature:
                        translated_function_signature = re.sub(r'\(\s*&?mut?\s*self\s*,\s*', '(', translated_function_signature)
                        translated_function_signature = re.sub(r'\(\s*&?mut?\s*self\s*\)', '()', translated_function_signature)

                # 提取真实函数名（用于去重和删除）
                extracted_name = extract_function_name_from_code(translated_function)
                if not extracted_name and translated_function_signature:
                    extracted_name = extract_function_name_from_code(translated_function_signature)
                
                if not extracted_name:
                    print(f"警告: 无法从 {function_name} 提取函数名，跳过")
                    continue
                
                # 检查是否已经合并过同名函数
                if extracted_name in merged_function_names[rs_filename]:
                    # 已合并过，跳过（避免重复）
                    continue
                
                # 1. 先删除骨架中所有同名函数定义（避免重复）
                skeleton_code = remove_function_by_name(skeleton_code, extracted_name, remove_all=True)
                
                # 2. 尝试使用签名匹配替换
                success = False
                if translated_function_signature:
                    skeleton_code, success = find_and_replace_function_signature(
                        skeleton_code, translated_function_signature, translated_function
                    )
                
                # 3. 如果签名匹配失败，在适当位置插入
                if not success:
                    # 在第一个函数位置插入
                    try:
                        first_pos = get_frist_function_position_start(skeleton_code)
                    except (IndexError, Exception):
                        # 如果找不到函数位置，在文件末尾添加
                        first_pos = len(skeleton_code)
                    
                    skeleton_code = skeleton_code[:first_pos] + translated_function + "\n\n" + skeleton_code[first_pos:]
                
                # 记录已合并的函数名
                merged_function_names[rs_filename].add(extracted_name)
                merged_count += 1
                
            except Exception as e:
                print(f"警告: 合并函数 {function_name} 时出错: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # 统一添加所有导入
        if all_imports:
            # 去重（保持顺序）
            seen = set()
            unique_imports = []
            for imp in all_imports:
                if imp not in seen:
                    seen.add(imp)
                    unique_imports.append(imp)
            skeleton_code = add_import_to_translated_result(skeleton_code, unique_imports)
        
        # 写回文件
        with open(target_rs_file, 'w', encoding='utf-8', errors='ignore') as f:
            f.write(skeleton_code)
    
    return merged_count


def test_final_project(project_dir: Path, test_cmd=None) -> Tuple[bool, str, str]:
    """
    测试最终合并后的完整项目
    
    返回: (是否成功, stdout, stderr)
    """
    if test_cmd is None:
        test_cmd = ["cargo", "build", "--release"]
        try:
            from cargo_utils import with_cargo_jobs
            test_cmd = with_cargo_jobs(test_cmd)
        except Exception:
            pass
    
    output, error, result = run_tests(project_dir, test_cmd)
    return result, output, error


def main():
    """主函数"""
    if len(sys.argv) < 3:
        print("用法: python3 merge_final_project.py <project_name> <llm_name>")
        sys.exit(1)
    
    project_name = sys.argv[1]
    llm_name = sys.argv[2]
    
    # 确保工作空间路径正确（从环境变量读取，如果存在）
    # workspace_config 会在导入时自动读取 C2R_WORKSPACE_ROOT
    # 这里只是确保路径配置正确
    
    print(f"\n{'='*60}")
    print(f"开始最终项目合并: {project_name} (LLM: {llm_name})")
    
    # 显示当前使用的工作空间路径
    from workspace_config import WORKSPACE_ROOT
    print(f"工作空间根目录: {WORKSPACE_ROOT}")
    print(f"{'='*60}\n")
    
    # 1. 查找所有编译通过的函数
    print("步骤 1: 查找所有编译通过的函数...")
    passed_functions = find_passed_functions(project_name, llm_name)
    print(f"  找到 {len(passed_functions)} 个编译通过的函数")
    
    if len(passed_functions) == 0:
        print("错误: 没有找到编译通过的函数，无法合并")
        sys.exit(1)
    
    # 2. 获取必要的路径
    skeleton_dir = get_skeleton_path(project_name)
    signature_dir = get_signature_match_path(project_name, llm_name)
    final_project_dir = get_final_project_path(project_name, llm_name)
    
    if not skeleton_dir.exists():
        print(f"错误: 骨架目录不存在: {skeleton_dir}")
        sys.exit(1)
    
    # 3. 合并函数到骨架
    print("\n步骤 2: 合并所有函数到骨架中...")
    merged_count = merge_functions_to_skeleton(
        skeleton_dir,
        passed_functions,
        signature_dir,
        final_project_dir
    )
    print(f"  成功合并 {merged_count} 个函数")
    
    # 4. 进行完整项目编译测试
    print("\n步骤 3: 进行完整项目编译测试...")
    success, output, error = test_final_project(final_project_dir)
    
    if success:
        print("  ✓ 完整项目编译成功！")
    else:
        print("  ✗ 完整项目编译失败")
        print(f"  错误信息:\n{error[:1000]}")  # 只打印前1000字符
    
    # 5. 保存测试结果
    result_file = final_project_dir / "build_result.txt"
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write("Success\n" if success else "Fail\n")
        f.write(f"\n合并的函数数量: {merged_count}\n")
        f.write(f"编译成功的函数数量: {len(passed_functions)}\n")
        f.write(f"\n=== 编译输出 ===\n{output}\n")
        f.write(f"\n=== 编译错误 ===\n{error}\n")
    
    print(f"\n{'='*60}")
    print("最终项目合并完成！")
    print(f"{'='*60}\n")
    print(f"最终项目目录: {final_project_dir}")
    print(f"合并函数数量: {merged_count}")
    print(f"完整项目编译: {'成功' if success else '失败'}")
    print(f"\n测试结果已保存到: {result_file}")
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
