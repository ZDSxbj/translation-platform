import os

from translate import get_function, get_struct_code,get_function_mapping

from rank_bm25 import BM25Plus
from project_config import PROJECT_NAME
import json
from workspace_config import (
    get_skeleton_path, get_source_skeleton_path, get_functions_path,
    get_signature_match_path, get_dependencies_not_in_file_path,
    get_dependencies_not_in_file_rs_path, ensure_dirs,
    safe_module_name, safe_module_name_legacy
)
import re
import traceback

try:
    # Optional dependency: only used when falling back to BM25 matching.
    # Keep it optional to make the open-sourced repo easier to run.
    from nltk.corpus import stopwords  # type: ignore
    from nltk.stem import PorterStemmer  # type: ignore

    _NLTK_AVAILABLE = True
except Exception:
    stopwords = None  # type: ignore
    PorterStemmer = None  # type: ignore
    _NLTK_AVAILABLE = False

# Fallback stopwords if NLTK (or its corpora) is unavailable.
_FALLBACK_STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "for", "while", "do", "in", "on", "of", "to",
    "from", "with", "by", "as", "is", "are", "was", "were", "be", "been", "being", "this", "that", "these",
    "those", "it", "its", "we", "you", "they", "he", "she", "i", "me", "my", "our", "your", "their",
}


def read_corpus(corpus_files_path):
    corpus = {}

    corpus_files = os.listdir(corpus_files_path)
    for corpus_file in corpus_files:
        with open(os.path.join(corpus_files_path, corpus_file), 'r') as input_file:
            tmp = input_file.read()

            corpus[tmp.strip()] = corpus_file
    
    return corpus

def normalize_text(text):
    # 大小写归一化
    # text = text.lower()
    
    # 分词
    words = re.findall(r'[a-zA-Z0-9]+|[^\s\w]+', text)
    # print(words)
    # 去除停用词
    if _NLTK_AVAILABLE:
        try:
            stop_words = set(stopwords.words("english"))  # type: ignore[union-attr]
        except Exception:
            stop_words = _FALLBACK_STOP_WORDS
    else:
        stop_words = _FALLBACK_STOP_WORDS
    words = [word for word in words if word not in stop_words]
    pattern = r'[A-Z][a-z]+|[a-z]+'
    new_words = []
    for word in words:
        # print(word)
        tmp = re.findall(pattern, word)
        tmp = [word.lower() for word in tmp]
        new_words.extend(tmp)
    
    if len(new_words) != 0:
        words = new_words
    
    # 词干提取（启用以保持与 build_knowledge_base.py 一致；NLTK 不可用时跳过）
    if _NLTK_AVAILABLE and PorterStemmer is not None:
        try:
            stemmer = PorterStemmer()
            words = [stemmer.stem(word) for word in words]
        except Exception:
            pass
    
    # 过滤掉仅由符号组成的元素
    symbol_pattern = re.compile(r'^[^\w\s]+$')
    words = [word for word in words if not symbol_pattern.match(word)]

    return words


# 使用正则表达式进行分词
def tokenize_code(code):
    # 使用归一化
    return normalize_text(code)

    # 将代码按单词、标点符号和特殊字符分割
    tokens = re.findall(r'\w+|[^\s\w]+', code)
    return tokens

def get_mapping(llm=None):
    project_path = PROJECT_NAME

    # ============================================================
    # 优先使用阶段1生成的确定性映射（彻底绕过 BM25 猜测）
    # ============================================================
    # mapping: func_file -> rust_signature
    # - 阶段1(translate_skeleton_layered) 会在 source_skeletons/<proj>/func_file_to_rust_sig.json 写出
    # - 阶段3(incremental_translate) 也会优先读取该映射
    try:
        mapping_path = get_source_skeleton_path(project_path) / "func_file_to_rust_sig.json"
        if mapping_path.exists():
            with open(mapping_path, 'r', encoding='utf-8') as f:
                funcfile_to_sig = json.load(f)

            signature_output_path = get_signature_match_path(project_path)
            signature_output_path.mkdir(parents=True, exist_ok=True)
            signature_output_path_llm = None
            if llm:
                signature_output_path_llm = get_signature_match_path(project_path, llm)
                signature_output_path_llm.mkdir(parents=True, exist_ok=True)

            written = 0
            for func_file, rust_sig in (funcfile_to_sig or {}).items():
                if not rust_sig:
                    continue
                # signature_matches 目录中文件名需要与 extracted/<proj>/functions/<func_file>.txt 同名（带 .txt）
                out = signature_output_path / f"{func_file}.txt"
                with open(out, 'w', encoding='utf-8') as wf:
                    wf.write(str(rust_sig).strip())
                if signature_output_path_llm is not None:
                    out_llm = signature_output_path_llm / f"{func_file}.txt"
                    with open(out_llm, 'w', encoding='utf-8') as wf:
                        wf.write(str(rust_sig).strip())
                written += 1

            print(f"使用确定性签名映射写入 signature_matches: {written} 个函数 ({mapping_path})")
            return
    except Exception as e:
        print(f"警告: 读取确定性签名映射失败，将回退到 BM25 匹配: {e}")

    # 使用新的工作空间路径
    skeleton_dir = get_skeleton_path(project_path) / "src"
    source_skeleton_dir = get_source_skeleton_path(project_path) / "src"

    # 向后兼容：跟踪是否已经输出过旧命名规则警告
    _legacy_warning_shown = False

    def find_source_skeleton_file(module_name):
        """查找源代码骨架文件，支持向后兼容旧命名规则"""
        nonlocal _legacy_warning_shown

        # 尝试新规则命名
        new_path = source_skeleton_dir / f"{module_name}.txt"
        if new_path.exists():
            return new_path

        # 尝试旧规则命名（向后兼容）
        # 旧规则可能包含非法字符或数字开头
        # 我们需要反推可能的旧名字
        legacy_candidates = []

        # 如果模块名以下划线开头（新规则处理数字开头），尝试去掉前缀
        if module_name.startswith('_') and len(module_name) > 1 and module_name[1].isdigit():
            legacy_candidates.append(module_name[1:])

        # 旧规则不会清理非法字符，但我们无法反推原始字符
        # 只能尝试最常见的情况

        for legacy_name in legacy_candidates:
            legacy_path = source_skeleton_dir / f"{legacy_name}.txt"
            if legacy_path.exists():
                if not _legacy_warning_shown:
                    print(f"[警告] 使用旧命名规则的输出文件: {legacy_path}")
                    print(f"[警告] 旧命名规则已废弃，建议清理输出目录并重新运行 get_dependencies.py")
                    _legacy_warning_shown = True
                return legacy_path

        return None

    def find_function_files(module_name, functions_dir):
        """查找函数文件，支持向后兼容旧命名规则

        注意：只匹配精确的 {module_name}_{数字}.txt 格式，避免匹配到
        类似 {module_name}_virtual_{数字}.txt 的其他模块文件。
        """
        nonlocal _legacy_warning_shown
        import re

        # 尝试新规则命名
        # 使用精确匹配：module_name_数字.txt，避免匹配到 module_name_virtual_数字.txt
        all_candidates = list(functions_dir.glob(f"{module_name}_*.txt"))

        # 过滤：只保留 {module_name}_{纯数字}.txt 格式的文件
        # 例如 src_lnn_time_sync_manager_1.txt 保留
        # 但 src_lnn_time_sync_manager_virtual_1.txt 不保留
        pattern = re.compile(rf"^{re.escape(module_name)}_(\d+)\.txt$")
        function_files = [f for f in all_candidates if pattern.match(f.name)]

        if function_files:
            return function_files

        # 尝试旧规则命名（向后兼容）
        legacy_candidates = []

        # 如果模块名以下划线开头（新规则处理数字开头），尝试去掉前缀
        if module_name.startswith('_') and len(module_name) > 1 and module_name[1].isdigit():
            legacy_candidates.append(module_name[1:])

        for legacy_name in legacy_candidates:
            all_legacy = list(functions_dir.glob(f"{legacy_name}_*.txt"))
            # 同样使用精确匹配
            legacy_pattern = re.compile(rf"^{re.escape(legacy_name)}_(\d+)\.txt$")
            legacy_files = [f for f in all_legacy if legacy_pattern.match(f.name)]
            if legacy_files:
                if not _legacy_warning_shown:
                    print(f"[警告] 使用旧命名规则的函数文件: {legacy_name}_*.txt")
                    print(f"[警告] 旧命名规则已废弃，建议清理输出目录并重新运行 get_dependencies.py")
                    _legacy_warning_shown = True
                return legacy_files

        return []
    
    if not skeleton_dir.exists():
        print(f"错误: 骨架目录不存在: {skeleton_dir}")
        return
    
    src_files = list(skeleton_dir.glob("*.rs"))

    # 确保目录存在
    ensure_dirs(project_path, llm)
    
    # 精简输出：只在开始时输出一次
    print(f"开始函数签名匹配，共 {len(src_files)} 个文件...")
    
    for src_file in src_files:
        current_file_name = src_file.stem  # 不含扩展名

        if "dependency_from_other_project" in current_file_name:
            continue
        
        # 跳过自动生成的文件（这些文件没有对应的 C++ 源文件）
        # types.rs: bindgen 生成的类型定义
        # globals.rs: tree-sitter 提取的全局变量
        if current_file_name in ['types', 'globals']:
            continue
        
        skeleton_file = skeleton_dir / f"{current_file_name}.rs"
        # 使用向后兼容的查找函数
        source_skeleton_file = find_source_skeleton_file(current_file_name)

        if not skeleton_file.exists():
            print(f"警告: 骨架文件不存在: {skeleton_file}")
            continue

        with open(skeleton_file, 'r', encoding='utf-8') as f:
            translated_result = f.read()

        if source_skeleton_file is None:
            # 常见入口文件名，通常没有对应的 C++ 源文件，这是正常的
            common_entry_points = ['main', 'lib', 'mod']
            if current_file_name.lower() in common_entry_points:
                # 这是正常的入口点文件，跳过而不输出警告
                continue
            
            # 检查骨架文件是否只是一个空的入口点文件（只有基本的模块声明和空函数）
            # 如果是这种情况，可以安全跳过签名匹配
            lines = [l.strip() for l in translated_result.splitlines() 
                    if l.strip() and not l.strip().startswith('//') and not l.strip().startswith('#')]
            
            # 如果文件很小且只包含基本的模块声明和空函数，跳过匹配（这是正常的）
            if len(translated_result) < 500 and len(lines) <= 10:
                # 检查是否主要是模块声明和空函数
                has_only_basic_structure = all(
                    any(keyword in line for keyword in ['pub mod', 'mod ', 'fn main()', 'fn ', '{}', ';', 'use ', 'extern']) 
                    for line in lines
                ) if lines else True
                if has_only_basic_structure:
                    # 这是正常的空入口点文件，跳过而不输出警告
                    continue
            
            # 检查是否包含函数定义（如果有函数定义，应该有源代码骨架文件）
            # 简单检查：如果文件包含函数定义（fn 关键字 + 函数体），则应该警告
            has_function_definitions = False
            fn_positions = []
            for i, char in enumerate(translated_result):
                if translated_result[i:i+3] == 'fn ' and (i == 0 or translated_result[i-1] in ' \n\t{('):
                    fn_positions.append(i)
            
            # 检查每个 fn 后面是否有函数体（找到对应的 {）
            for fn_pos in fn_positions:
                # 查找下一个 {，但跳过注释和字符串
                search_start = fn_pos + 3
                search_end = min(fn_pos + 500, len(translated_result))  # 限制搜索范围
                for i in range(search_start, search_end):
                    if translated_result[i] == '{':
                        has_function_definitions = True
                        break
                    elif translated_result[i] == '\n' and i > search_start + 100:
                        # 如果超过100个字符还没找到 {，可能只是函数签名
                        break
                if has_function_definitions:
                    break
            
            # 只有当文件包含实际的函数定义时才输出警告
            if has_function_definitions:
                expected_path = source_skeleton_dir / f"{current_file_name}.txt"
                print(f"警告: 源代码骨架文件不存在: {expected_path}")
            # 否则静默跳过（可能是自动生成的骨架文件或只有函数签名的文件）
            continue

        with open(source_skeleton_file, 'r', encoding='utf-8') as f:
            code_under_translation = f.read()

        source_function_signature, translated_function_signatures = get_function_mapping(code_under_translation, translated_result, current_file_name)

        # print(len(source_function_signature), " ", len(translated_function_signatures))
        # continue
        function_file2code = {}
        functions_dir = get_functions_path(project_path)
        if functions_dir.exists():
            # 使用向后兼容的查找函数
            function_files = find_function_files(current_file_name, functions_dir)
            for function_file in function_files:
                # 精简输出：移除详细文件读取信息
                # print("start to get " + function_file.name)
                with open(function_file, 'r', encoding='utf-8') as f:
                    function_code = f.read()
                function_file2code[function_code] = function_file.name
        
        translated_function_file2code = {}
        
        function_codes = list(function_file2code.keys())
        # 精简输出：已移除详细输出
        # print(f"找到 {len(function_codes)} 个 C++ 函数定义")
        # print(f"需要匹配 {len(translated_function_signatures)} 个 Rust 函数签名")
        
        # 检查语料库是否为空
        if not function_codes:
            print(f"警告: 文件 {current_file_name} 没有找到对应的 C++ 函数定义文件")
            print(f"  跳过 BM25 匹配，将使用空匹配结果")
            # 创建空的匹配结果文件
            signature_output_path = get_signature_match_path(project_path)
            signature_output_path.mkdir(parents=True, exist_ok=True)
            if llm:
                signature_output_path_llm = get_signature_match_path(project_path, llm)
                signature_output_path_llm.mkdir(parents=True, exist_ok=True)
            
            # 为每个翻译的函数签名创建空的匹配文件
            for query in translated_function_signatures:
                query_name = query.split("(")[0].split("fn")[-1].strip()
                if query_name.startswith("pub"):
                    query_name = query_name.replace("pub", "").strip()
                question_path = f"{current_file_name}_{query_name}.txt"
                if llm:
                    output_path = get_signature_match_path(project_path, llm) / question_path
                else:
                    output_path = signature_output_path / question_path
                
                # 创建空的匹配结果
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write("")  # 空文件，表示没有匹配
            continue
        
        tokenized_corpus = [tokenize_code(doc.split("(")[0].split("::")[-1].split(" ")[-1]) for doc in function_codes]
        # 精简输出：移除详细进度
        # print(f"语料库 tokenized 完成，共 {len(tokenized_corpus)} 个文档")
        
        # 检查 tokenized_corpus 是否为空（所有文档都为空字符串的情况）
        if not tokenized_corpus or all(not tokens for tokens in tokenized_corpus):
            print(f"警告: 文件 {current_file_name} 的 tokenized 语料库为空")
            print(f"  跳过 BM25 匹配，将使用空匹配结果")
            # 创建空的匹配结果文件（同上）
            signature_output_path = get_signature_match_path(project_path)
            signature_output_path.mkdir(parents=True, exist_ok=True)
            if llm:
                signature_output_path_llm = get_signature_match_path(project_path, llm)
                signature_output_path_llm.mkdir(parents=True, exist_ok=True)
            
            for query in translated_function_signatures:
                query_name = query.split("(")[0].split("fn")[-1].strip()
                if query_name.startswith("pub"):
                    query_name = query_name.replace("pub", "").strip()
                question_path = f"{current_file_name}_{query_name}.txt"
                if llm:
                    output_path = get_signature_match_path(project_path, llm) / question_path
                else:
                    output_path = signature_output_path / question_path
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write("")
            continue
        
        # 创建 BM25 索引
        try:
            bm25 = BM25Plus(tokenized_corpus)
        except Exception as e:
            print(f"错误: 创建 BM25 索引失败: {e}")
            print(f"  跳过 BM25 匹配，将使用空匹配结果")
            # 创建空的匹配结果文件（同上）
            signature_output_path = get_signature_match_path(project_path)
            signature_output_path.mkdir(parents=True, exist_ok=True)
            if llm:
                signature_output_path_llm = get_signature_match_path(project_path, llm)
                signature_output_path_llm.mkdir(parents=True, exist_ok=True)
            
            for query in translated_function_signatures:
                query_name = query.split("(")[0].split("fn")[-1].strip()
                if query_name.startswith("pub"):
                    query_name = query_name.replace("pub", "").strip()
                question_path = f"{current_file_name}_{query_name}.txt"
                if llm:
                    output_path = get_signature_match_path(project_path, llm) / question_path
                else:
                    output_path = signature_output_path / question_path
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write("")
            continue

        # 使用新的工作空间路径
        signature_output_path = get_signature_match_path(project_path)
        signature_output_path.mkdir(parents=True, exist_ok=True)
        
        # 如果提供了LLM参数，也创建 translate_by_{llm} 目录结构
        if llm:
            signature_output_path_llm = get_signature_match_path(project_path, llm)
            signature_output_path_llm.mkdir(parents=True, exist_ok=True)
        
        # 获取请求
        for query in translated_function_signatures:
            
            tokenized_query = tokenize_code(query.split("(")[0].split("fn")[1])
            # 获取相关性评分
            scores = bm25.get_scores(tokenized_query)
            # 获取最相关的前几个函数定义
            top_n = 5
            match_results_index = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
            # 精简输出：移除详细匹配信息
            # print(scores[match_results_index[0]])
            # print(function_codes[match_results_index[0]])
            match_file = function_file2code[function_codes[match_results_index[0]]]
            if match_file in translated_function_file2code and translated_function_file2code[match_file]["score"] > scores[match_results_index[0]]:
                # 精简输出：只在出错时输出
                # print("error")
                # print(match_file + ": " + query)
                # print(tokenized_query)
                # print(translated_function_file2code[match_file])
                pass
            else:
                translated_function_file2code[match_file] = {
                    "match code":query,
                    "score":scores[match_results_index[0]]
                }
                # 保存到两个位置以兼容新旧代码
                output_file = signature_output_path / match_file
                try:
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(query)
                except Exception as write_err:
                    print(f"错误: 写入签名文件失败 {output_file}: {write_err}")
                    raise
                if llm:
                    output_file_llm = signature_output_path_llm / match_file
                    try:
                        with open(output_file_llm, 'w', encoding='utf-8') as f:
                            f.write(query)
                    except Exception as write_err:
                        print(f"错误: 写入签名文件失败 {output_file_llm}: {write_err}")
                        raise
                # 精简输出：移除 success 输出
                # print("success")

def get_dependencies_match(llm=None):
    project_path = PROJECT_NAME
    corpus_files_path = get_functions_path(project_path)

    # 获取匹配池子
    function_code2file_name = read_corpus(str(corpus_files_path))
    # print(function_code2file_name)
    dependencies_not_in_file_dir = get_dependencies_not_in_file_path(project_path)
    dependencies_not_in_file_rs_dir = get_dependencies_not_in_file_rs_path(project_path)
    
    if not dependencies_not_in_file_dir.exists():
        print(f"警告: 依赖目录不存在: {dependencies_not_in_file_dir}")
        return
    
    file_path_list = list(dependencies_not_in_file_dir.glob("*.txt"))
    # 精简输出：只在开始时输出一次
    print(f"开始处理 {len(file_path_list)} 个依赖文件...")
    
    dependencies_not_in_file_rs_dir.mkdir(parents=True, exist_ok=True)

    for file_path in file_path_list:
        if "dependency_from_other_project" in file_path.name:
            continue
        
        with open(file_path, 'r', encoding='utf-8') as f:
            functions_content = f.read()
        functions = get_function(functions_content)
        dependencies_code_signature = []
        
        signature_match_dir = get_signature_match_path(project_path)
        skeleton_dir = get_skeleton_path(project_path) / "src"
        
        for function_code in functions.values():
            # print(function_code)
            function_code_stripped = function_code.strip()
            
            # 尝试查找匹配的文件名（支持多种匹配方式）
            dependency_file_name = None
            if function_code_stripped in function_code2file_name:
                dependency_file_name = function_code2file_name[function_code_stripped]
            else:
                # 尝试不区分大小写匹配
                for key, value in function_code2file_name.items():
                    if key.strip().lower() == function_code_stripped.lower():
                        dependency_file_name = value
                        break
                
                # 如果还是找不到，尝试部分匹配（匹配函数签名）
                if dependency_file_name is None:
                    # 提取函数签名（第一行）
                    first_line = function_code_stripped.split('\n')[0].strip()
                    for key, value in function_code2file_name.items():
                        if first_line in key or key.split('\n')[0].strip() in first_line:
                            dependency_file_name = value
                            break
            
            if dependency_file_name is None:
                # 精简输出：移除详细警告信息
                # print(f"警告: 无法找到函数对应的文件: {function_code_stripped[:100]}...")
                pass
                continue
            
            # print(file_path, " ", dependency_file_name)
            
            # 检查是否有 LLM 子目录
            signature_file = None
            if llm:
                signature_file_llm = get_signature_match_path(project_path, llm) / dependency_file_name
                if signature_file_llm.exists():
                    signature_file = signature_file_llm
            
            if signature_file is None:
                signature_file = signature_match_dir / dependency_file_name
            
            if not signature_file.exists():
                # 精简输出：移除签名文件不存在的警告（这些警告太多，影响查看）
                # print(f"警告: 签名文件不存在: {signature_file}")
                pass
                continue
            
            with open(signature_file, 'r', encoding='utf-8') as f:
                dependency_code_signature = f.read().strip()
            
            # 从文件名提取基础名称（去掉序号）
            base_name = "_".join(dependency_file_name.split("_")[:-1])
            skeleton_file = skeleton_dir / f"{base_name}.rs"
            
            if not skeleton_file.exists():
                print(f"警告: 骨架文件不存在: {skeleton_file}")
                continue
            
            with open(skeleton_file, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            total_dependency_code = get_struct_code(source_code, dependency_code_signature)
            dependencies_code_signature.append(total_dependency_code)
        
        output_file = dependencies_not_in_file_rs_dir / file_path.name
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("\n\n".join(dependencies_code_signature))

if __name__ == "__main__":
    import sys
    import os
    # 支持从命令行参数或环境变量获取LLM名称（可选）
    llm = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("LLM_NAME", None)
    exit_code = 0
    try:
        get_mapping(llm=llm)
    except Exception as exc:
        print(f"错误: 函数签名匹配失败: {exc}")
        traceback.print_exc()
        exit_code = 1
    try:
        get_dependencies_match(llm=llm)
    except Exception as exc:
        print(f"错误: 依赖匹配失败: {exc}")
        traceback.print_exc()
        exit_code = 1
    sys.exit(exit_code)
