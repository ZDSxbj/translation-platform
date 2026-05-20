# 文件名: build_knowledge_base.py
# 作用: 替代 ES 索引。扫描 C/Rust 对，构建并保存一个 BM25 索引。

import os
import json
import re
import pickle
from rank_bm25 import BM25Plus
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tqdm import tqdm

# --- 配置 ---
# 1. process_judgments.py 脚本的输出目录
CODE_PAIRS_DIR = "../qwen3_coder/filtered_code_pairs" 

# 2. 使用新的工作空间路径
from workspace_config import RAG_KNOWLEDGE_BASE, RAG_BM25_INDEX, RAG_DIR

# 确保 RAG 目录存在
RAG_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_KB_JSON = str(RAG_KNOWLEDGE_BASE)
OUTPUT_BM25_INDEX = str(RAG_BM25_INDEX)

# 3. NLTK 文本规范化 (与 get_dependencies_match.py 保持一致)
def normalize_text(text):
    words = re.findall(r'[a-zA-Z0-9]+|[^\s\w]+', text)
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    pattern = r'[A-Z][a-z]+|[a-z]+'
    new_words = []
    for word in words:
        tmp = re.findall(pattern, word)
        tmp = [word.lower() for word in tmp]
        new_words.extend(tmp)
    if len(new_words) != 0:
        words = new_words
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    symbol_pattern = re.compile(r'^[^\w\s]+$')
    words = [word for word in words if not symbol_pattern.match(word)]
    return words

def build_knowledge_base():
    """
    遍历 code_pairs 目录，构建一个 JSON 知识库和 BM25 索引。
    [V10 升级] 同时读取 LLM 提取的知识文件 (__Full.json, __Partial_*.json, __API_*.json)
    """
    print(f"--- 开始构建知识库 (V10 知识增强版) ---")
    print(f"正在扫描: {CODE_PAIRS_DIR}")

    if not os.path.exists(CODE_PAIRS_DIR):
        print(f"错误: 目录 '{CODE_PAIRS_DIR}' 未找到。")
        print("请先运行 'extract_yes_pairs.py' 脚本来生成代码对。")
        return

    # 1. 查找所有 C/Rust 文件对
    # 文件名格式: Full__funcA__idx1_rank1.c / .rs / __Full.json / __API_0.json 等
    file_map = {} # { "base_filename": { "c_files": [...], "rs": "...", "knowledge_files": [...] } }
    
    for filename in os.listdir(CODE_PAIRS_DIR):
        filepath = os.path.join(CODE_PAIRS_DIR, filename)
        
        # 跳过目录
        if os.path.isdir(filepath):
            continue
            
        # 解析文件名，提取 base_name
        # 知识文件格式: base__Full.json, base__Partial_0.json, base__API_0.json
        # 代码文件格式: base.c, base.rs
        
        if filename.endswith('.json'):
            # 这是一个知识文件，找到它的 base_name
            # 例如: Full__funcA__idx1_rank1__API_0.json -> base = Full__funcA__idx1_rank1
            parts = filename.rsplit('__', 1)  # 从右边分割一次
            if len(parts) == 2 and parts[1].replace('.json', '').split('_')[0] in ['Full', 'Partial', 'API']:
                base_name = parts[0]
                if base_name not in file_map:
                    file_map[base_name] = {'c_files': [], 'knowledge_files': []}
                file_map[base_name]['knowledge_files'].append(filepath)
        else:
            # 这是一个代码文件 (.c, .h, .rs)
            base_name, ext = os.path.splitext(filename)
            ext = ext.lower()

            if base_name not in file_map:
                file_map[base_name] = {'c_files': [], 'knowledge_files': []}

            if ext in ['.c', '.h']:
                file_map[base_name]['c_files'].append(filepath)
            elif ext == '.rs':
                file_map[base_name]['rs'] = filepath
            else:
                # 其他扩展名一律忽略（目录中可能还有日志/中间文件等）
                continue

    # 2. 读取文件内容并准备 BM25
    knowledge_base = [] # 存储 {"c_code": "...", "rust_code": "...", "extracted_knowledge": [...]}
    c_code_corpus = []  # 存储用于 BM25 索引的 C 代码

    print("正在读取文件对并构建语料库...")
    knowledge_count = 0
    
    for base_name, paths in tqdm(file_map.items()):
        if paths['c_files'] and 'rs' in paths:
            try:
                # 合并所有 C/H 文件内容
                c_code_parts = []
                for c_file_path in paths['c_files']:
                    with open(c_file_path, 'r', encoding='utf-8') as f_c:
                        c_code_parts.append(f_c.read())
                
                # 用分隔符连接多个 C 文件
                c_code = "\n\n// ===== 文件分隔 =====\n\n".join(c_code_parts)
                
                with open(paths['rs'], 'r', encoding='utf-8') as f_rs:
                    rust_code = f_rs.read()
                
                if not c_code.strip() or not rust_code.strip():
                    continue
                
                # [V10 新增] 读取所有关联的知识文件
                extracted_knowledge = []
                for knowledge_file in paths.get('knowledge_files', []):
                    try:
                        with open(knowledge_file, 'r', encoding='utf-8') as f_k:
                            knowledge_data = json.load(f_k)
                            extracted_knowledge.append(knowledge_data)
                            knowledge_count += 1
                    except Exception as e:
                        print(f"警告: 读取知识文件 {knowledge_file} 失败: {e}")

                knowledge_base.append({
                    "c_code": c_code,
                    "rust_code": rust_code,
                    "original_file": base_name,
                    "c_file_count": len(paths['c_files']),
                    "extracted_knowledge": extracted_knowledge  # [V10 新增] 知识列表
                })
                c_code_corpus.append(c_code)
                
            except Exception as e:
                print(f"警告: 读取文件 {base_name} 失败: {e}")

    if not c_code_corpus:
        print("错误: 未能在知识库中找到任何有效的 C/Rust 代码对。")
        return

    print(f"已加载 {len(c_code_corpus)} 个有效的 C/Rust 翻译对。")
    print(f"已加载 {knowledge_count} 个细粒度知识条目。")

    # 3. Tokenize 语料库
    print("正在 Tokenizing 语料库 (用于 BM25)...")
    tokenized_corpus = [normalize_text(doc) for doc in tqdm(c_code_corpus)]

    # 4. 构建 BM25 索引
    print("正在构建 BM25 索引...")
    bm25 = BM25Plus(tokenized_corpus)
    print("BM25 索引构建完毕。")

    # 5. 保存知识库和索引
    print(f"正在保存知识库到: {OUTPUT_KB_JSON}")
    with open(OUTPUT_KB_JSON, 'w', encoding='utf-8') as f:
        json.dump(knowledge_base, f, indent=2, ensure_ascii=False)

    print(f"正在保存 BM25 索引到: {OUTPUT_BM25_INDEX}")
    with open(OUTPUT_BM25_INDEX, 'wb') as f_idx:
        pickle.dump(bm25, f_idx)

    print("--- 知识库构建完成 (V10 知识增强版) ---")

if __name__ == "__main__":
    # 确保 NLTK 数据已下载
    try:
        stopwords.words('english')
    except LookupError:
        import nltk
        print("正在下载 NLTK stopwords...")
        nltk.download('stopwords')
    
    build_knowledge_base()
