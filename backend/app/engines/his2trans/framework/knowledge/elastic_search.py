# 文件名: elastic_search.py (已修改)
# 作用: (RAG 步骤1) 为每个 C++ 查询函数，从知识库中检索 Top-100 的 C 代码范例。

import os
import json
import sys
import pickle
from rank_bm25 import BM25Plus
from tqdm import tqdm
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

# --- 配置 ---
from workspace_config import (
    RAG_KNOWLEDGE_BASE, RAG_BM25_INDEX, RAG_ELASTIC_SEARCH_RESULTS,
    get_functions_path, get_elastic_search_path
)
from project_config import PROJECT_NAME

# 使用新的工作空间路径
FUNCTIONS_PATH = get_functions_path(PROJECT_NAME)  # C++ 查询函数的来源
KNOWLEDGE_BASE_JSON = str(RAG_KNOWLEDGE_BASE)
BM25_INDEX = str(RAG_BM25_INDEX)
OUTPUT_RAG_DIR = str(RAG_ELASTIC_SEARCH_RESULTS)  # BM25 结果的输出目录
TOP_N = 100 # BM25 检索的数量

# 额外知识库（独立可删除）：通过环境变量指定目录（支持多个，逗号/分号/冒号分隔）
# 目录内应包含:
#   - knowledge_base.json
#   - bm25_index.pkl
EXTRA_KB_DIRS_ENV = os.environ.get("C2R_EXTRA_RAG_KB_DIR", "").strip()

# 批量处理配置
BATCH_SIZE = 64  # BM25 批量查询大小
IO_WORKERS = 8   # 并行 I/O 线程数

# --- NLTK 文本规范化 (必须与 build_knowledge_base.py 中的一致) ---
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


def _parse_extra_kb_dirs(raw: str):
    if not raw:
        return []
    parts = re.split(r"[,:;]", raw)
    out = []
    for p in parts:
        s = (p or "").strip()
        if not s:
            continue
        out.append(s)
    return out


def _load_kb_and_bm25(kb_json_path: str, bm25_path: str):
    with open(kb_json_path, 'r', encoding='utf-8') as f:
        knowledge_base = json.load(f)
    with open(bm25_path, 'rb') as f_idx:
        bm25: BM25Plus = pickle.load(f_idx)
    return knowledge_base, bm25

# --- 主逻辑 ---
def main():
    print("--- (RAG 步骤 1) BM25 检索开始 ---")
    
    # 1. 加载构建好的知识库和索引（支持额外 KB 并行检索 + 合并）
    kb_sources = []  # [{label, knowledge_base, bm25}]
    try:
        print(f"正在加载知识库: {KNOWLEDGE_BASE_JSON}")
        print(f"正在加载 BM25 索引: {BM25_INDEX}")
        knowledge_base, bm25 = _load_kb_and_bm25(KNOWLEDGE_BASE_JSON, BM25_INDEX)
        kb_sources.append({
            "label": "base",
            "knowledge_base": knowledge_base,
            "bm25": bm25,
        })
            
    except FileNotFoundError:
        print("错误: 找不到知识库文件 (rag_knowledge_base.json / rag_bm25_index.pkl)。")
        print("请先运行 'build_knowledge_base.py' 脚本。")
        sys.exit(1)
        
    print(f"基础知识库加载完毕，包含 {len(knowledge_base)} 个翻译对。")

    # 1.1 额外知识库（可选）
    extra_dirs = _parse_extra_kb_dirs(EXTRA_KB_DIRS_ENV)
    for d in extra_dirs:
        try:
            d = os.path.abspath(os.path.expanduser(d))
            kb_json = os.path.join(d, "knowledge_base.json")
            bm25_pkl = os.path.join(d, "bm25_index.pkl")
            if not (os.path.exists(kb_json) and os.path.exists(bm25_pkl)):
                print(f"警告: 跳过额外 KB（缺文件）: {d}")
                continue
            kbe, bm25e = _load_kb_and_bm25(kb_json, bm25_pkl)
            kb_sources.append({
                "label": f"extra:{os.path.basename(os.path.normpath(d))}",
                "knowledge_base": kbe,
                "bm25": bm25e,
            })
            print(f"已加载额外 KB: {d} (pairs={len(kbe)})")
        except Exception as e:
            print(f"警告: 加载额外 KB 失败，跳过: {d} ({e})")

    # 2. 遍历所有 C++ 查询函数 (来自 functions 目录)
    if not FUNCTIONS_PATH.exists():
        print(f"错误: 函数目录不存在: {FUNCTIONS_PATH}")
        sys.exit(1)
    
    project = PROJECT_NAME
    project_in_path = FUNCTIONS_PATH
    project_out_path = get_elastic_search_path(project)
    project_out_path.mkdir(parents=True, exist_ok=True)
    
    query_files = list(project_in_path.glob("*.txt"))
    print(f"\n正在处理项目: {project} ({len(query_files)} 个查询函数)")
    
    # 过滤掉已处理的文件
    pending_files = []
    for qf in query_files:
        response_path = project_out_path / qf.name
        if not response_path.exists():
            pending_files.append(qf)
    
    if not pending_files:
        print("所有查询已处理，跳过")
        print("--- (RAG 步骤 1) BM25 检索完成 ---")
        return
    
    print(f"需要处理 {len(pending_files)} 个查询（跳过 {len(query_files) - len(pending_files)} 个已完成）")
    
    # === 优化：批量处理 ===
    def read_file(file_path):
        """并行读取文件"""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return file_path, f.read()
    
    def write_result(args):
        """并行写入结果"""
        response_path, query_content, candidates = args
        with open(response_path, 'w', encoding='utf-8', errors='ignore') as f_out:
            f_out.write(f"target function is :\n{query_content}\n")
            f_out.write("-" * 50 + "\n")
            
            for cand in candidates:
                src_i = cand["src_i"]
                idx = cand["idx"]
                raw_score = cand["raw_score"]
                norm_score = cand["norm_score"]
                src_label = kb_sources[src_i]["label"]
                kb_entry = kb_sources[src_i]["knowledge_base"][idx]
                
                f_out.write(f"[KB_SOURCE]\n{src_label}\n[/KB_SOURCE]\n")
                
                f_out.write(f"[C_CODE]\n{kb_entry['c_code']}\n[/C_CODE]\n")
                f_out.write(f"[RUST_CODE]\n{kb_entry['rust_code']}\n[/RUST_CODE]\n")
                
                extracted_knowledge = kb_entry.get('extracted_knowledge', [])
                if extracted_knowledge:
                    f_out.write(f"[EXTRACTED_KNOWLEDGE]\n")
                    f_out.write(json.dumps(extracted_knowledge, ensure_ascii=False))
                    f_out.write(f"\n[/EXTRACTED_KNOWLEDGE]\n")
                
                f_out.write(f"BM25 Score: {raw_score:.4f} (norm {norm_score:.4f})\n")
                f_out.write("-" * 50 + "\n")
        return True
    
    # 批量处理
    for batch_start in tqdm(range(0, len(pending_files), BATCH_SIZE), 
                            desc=f"BM25 Search ({project})", 
                            total=(len(pending_files) + BATCH_SIZE - 1) // BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(pending_files))
        batch_files = pending_files[batch_start:batch_end]
        
        # 1. 并行读取文件
        file_contents = {}
        with ThreadPoolExecutor(max_workers=IO_WORKERS) as executor:
            futures = {executor.submit(read_file, f): f for f in batch_files}
            for future in as_completed(futures):
                try:
                    file_path, content = future.result()
                    file_contents[file_path] = content
                except Exception as e:
                    print(f"读取文件失败: {e}")
        
        # 2. 批量 Tokenize
        tokenized_queries = []
        ordered_files = []
        for qf in batch_files:
            if qf in file_contents:
                tokenized_queries.append(normalize_text(file_contents[qf]))
                ordered_files.append(qf)
        
        if not tokenized_queries:
            continue
        
        # 3. BM25 查询（逐个查询，BM25 本身很快）
        write_tasks = []
        for idx, qf in enumerate(ordered_files):
            candidates = []

            # 每个 KB 各取 TOP_N，再做归一化合并（不同 KB 的 BM25 分数不可直接比较）
            for src_i, src in enumerate(kb_sources):
                doc_scores = src["bm25"].get_scores(tokenized_queries[idx])
                if doc_scores is None:
                    continue
                # 取 Top-N
                top_idx = np.argsort(doc_scores)[::-1][:TOP_N].tolist()
                # 归一化（避免 KB 间分数尺度差异）
                max_s = float(max((doc_scores[i] for i in top_idx), default=0.0))
                denom = max_s if max_s > 0 else 1.0
                for i2 in top_idx:
                    raw = float(doc_scores[i2])
                    candidates.append({
                        "src_i": src_i,
                        "idx": int(i2),
                        "raw_score": raw,
                        "norm_score": raw / denom,
                    })

            # 合并后按 norm_score 排序，再截断到 TOP_N
            candidates.sort(key=lambda x: (x["norm_score"], x["raw_score"]), reverse=True)
            candidates = candidates[:TOP_N]
            
            response_path = project_out_path / qf.name
            write_tasks.append((
                response_path, 
                file_contents[qf], 
                candidates,
            ))
        
        # 4. 并行写入结果
        with ThreadPoolExecutor(max_workers=IO_WORKERS) as executor:
            list(executor.map(write_result, write_tasks))
                    
    print("--- (RAG 步骤 1) BM25 检索完成 ---")

if __name__ == "__main__":
    main()
