# 文件名: resort_by_unixcoder.py (已修改)
# 作用: (RAG 步骤 2) 加载 jina-reranker-v3，重排 BM25 的 Top-100 结果。
# 优化: 支持多项目并行处理
# 策略: 
#   - 每个项目使用一个GPU（通过项目名hash分配）
#   - 允许多个项目同时运行（最多4个，如果有4个GPU）
#   - 每个GPU使用独立的文件锁，避免冲突
#   - 单个项目内可以使用多进程并行处理文件（提高GPU利用率）

import os
import sys
import re
import json
import time
import fcntl
import torch
from transformers import AutoModel
from tqdm import tqdm
from multiprocessing import Process, Queue, Manager, set_start_method, get_start_method
from pathlib import Path

# --- 配置 ---
from workspace_config import (
    RAG_ELASTIC_SEARCH_RESULTS, RAG_RERANKED_RESULTS,
    get_elastic_search_path, get_reranked_path
)
from project_config import PROJECT_NAME

# 使用新的工作空间路径
RAG_PATH = get_elastic_search_path(PROJECT_NAME)  # BM25 结果的输入目录
OUTPUT_PATH = get_reranked_path(PROJECT_NAME)  # 重排结果的输出目录
def _get_env_int(name: str, default: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        v = int(raw)
        return v if v > 0 else default
    except Exception:
        return default

# 最终保留的范例数量（默认 10；可通过 C2R_RAG_TOPK 覆盖，用于 RQ3.2 top-k 敏感性实验）
TOP_K = _get_env_int("C2R_RAG_TOPK", 10)

# Jina-reranker 加载设置
DEFAULT_MODEL_ID = "jinaai/jina-reranker-v3"
SCRIPT_DIR = Path(__file__).parent.resolve()
MY_CACHE_PATH = str(SCRIPT_DIR / "data" / "my-huggingface")
os.environ["HF_HOME"] = MY_CACHE_PATH
os.environ["TRANSFORMERS_CACHE"] = MY_CACHE_PATH
os.environ["HF_HUB_CACHE"] = str(Path(MY_CACHE_PATH) / "hub")

# 由外部队列调度器分配 GPU（run_jina_reranker_queued.py）；此时本脚本不应再做“跨项目的全局 GPU 槽位/调度”
EXTERNAL_GPU_SCHEDULER = os.environ.get("C2R_JINA_EXTERNAL_SCHEDULER", "").lower() in ("1", "true", "yes")


def _detect_local_snapshot():
    """自动探测 data/my-huggingface 下的本地 reranker 模型"""
    base = Path(os.environ.get(
        "JINA_RERANKER_CACHE_DIR",
        SCRIPT_DIR / "data" / "my-huggingface" / "models--jinaai--jina-reranker-v3"
    )).expanduser()
    snapshot_root = base / "snapshots"
    if not snapshot_root.exists():
        return None
    for candidate in sorted(snapshot_root.iterdir()):
        if candidate.is_dir() and (candidate / "config.json").exists():
            return candidate
    return None


def _resolve_model_path():
    """
    获取应该加载的模型路径：
    1. 优先使用 JINA_RERANKER_LOCAL_DIR 指定的本地目录；
    2. 其次使用 JINA_RERANKER_MODEL（若其指向本地路径）；
    3. 再次尝试自动探测 data/my-huggingface/snapshots；
    4. 最后回退到 huggingface id（可通过 JINA_RERANKER_MODEL 覆盖）。
    """
    env_local = os.environ.get("JINA_RERANKER_LOCAL_DIR")
    env_model = os.environ.get("JINA_RERANKER_MODEL")

    for candidate in [env_local, env_model]:
        if candidate:
            candidate_path = Path(candidate).expanduser()
            if candidate_path.exists():
                return str(candidate_path), True

    detected = _detect_local_snapshot()
    if detected:
        return str(detected), True

    return env_model or DEFAULT_MODEL_ID, False


MODEL_NAME, MODEL_IS_LOCAL = _resolve_model_path()
# 可通过 JINA_RERANKER_LOCAL_ONLY=0 允许回退网络
LOCAL_ONLY = os.environ.get(
    "JINA_RERANKER_LOCAL_ONLY",
    "1" if MODEL_IS_LOCAL else "0"
).lower() not in ("0", "false", "no")

# 多 GPU 配置
NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 1
USE_MULTI_GPU = NUM_GPUS > 1 and torch.cuda.is_available()

# GPU 内存阈值配置（可通过环境变量调整）
# 对于 49GB 显存的 GPU（如 RTX 5880 Ada），每个 Jina Reranker 约需 6-8GB
# 因此每个 GPU 可以支持 4-6 个并行任务
MIN_FREE_MEMORY_GB = float(os.environ.get("GPU_MIN_FREE_MEMORY_GB", "6.0"))  # 降低到 6GB
BATCH_SIZE_AUTO = os.environ.get("GPU_BATCH_SIZE_AUTO", "1").lower() in ("1", "true", "yes")

# 默认每个 GPU 最大并行任务数
# 可通过 JINA_MAX_SLOTS_PER_GPU 环境变量覆盖
DEFAULT_MAX_SLOTS_PER_GPU = 3  # 每个 GPU 最多 3 个并行任务，避免显存爆炸

def get_gpu_free_memory(gpu_id: int) -> float:
    """获取 GPU 空闲内存（GB）"""
    if not torch.cuda.is_available():
        return 0.0

    # 首选：CUDA runtime 的 mem_get_info（不依赖 NVML；且在 CUDA_VISIBLE_DEVICES 被设置时语义正确）
    try:
        free_bytes, _total_bytes = torch.cuda.mem_get_info(gpu_id)
        return float(free_bytes) / (1024**3)
    except Exception:
        pass

    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits', f'--id={gpu_id}'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return float(result.stdout.strip()) / 1024  # MB -> GB
    except Exception:
        pass
    
    # 回退：使用 PyTorch API
    try:
        props = torch.cuda.get_device_properties(gpu_id)
        allocated = torch.cuda.memory_allocated(gpu_id)
        return (props.total_memory - allocated) / (1024**3)
    except Exception:
        return 0.0

def get_gpu_running_tasks(gpu_id: int, max_slots: int = 2) -> int:
    """
    获取指定 GPU 上当前运行的任务数（通过检查锁文件）
    
    Args:
        gpu_id: GPU ID
        max_slots: 每个 GPU 的最大槽位数
    
    Returns:
        当前运行的任务数
    """
    running_count = 0
    for slot_id in range(max_slots):
        lock_file_path = SCRIPT_DIR / f".jina_reranker_gpu{gpu_id}_slot{slot_id}.lock"
        if lock_file_path.exists():
            try:
                # 尝试非阻塞获取锁
                lock_file = open(lock_file_path, 'r')
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                # 成功获取锁说明槽位空闲
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                lock_file.close()
            except (IOError, OSError):
                # 无法获取锁说明槽位被占用
                running_count += 1
            except Exception:
                pass
    return running_count


# GPU 调度阈值配置
# 对于 50GB 显存的 GPU，当空闲内存低于此值时不再分配新任务
MIN_FREE_MEMORY_FOR_NEW_TASK = float(os.environ.get("JINA_MIN_FREE_MEMORY_GB", "25.0"))

def get_best_gpu_for_task() -> int:
    """
    选择最适合任务的 GPU（综合考虑运行任务数和空闲内存）
    
    调度策略：
    1. 添加随机延迟，避免所有进程同时选择同一 GPU
    2. 只选择空闲内存 >= 25GB 的 GPU（可通过 JINA_MIN_FREE_MEMORY_GB 调整）
    3. 优先选择运行任务最少的 GPU
    4. 任务数相同时，选择空闲内存最多的 GPU

    Returns:
        GPU ID，如果没有可用 GPU 则返回空闲内存最多的
    """
    import random
    import time
    
    if not torch.cuda.is_available() or NUM_GPUS == 0:
        return 0

    # 检查是否通过环境变量指定了 GPU
    # 注意：当 CUDA_VISIBLE_DEVICES 被设置后，PyTorch 只能看到有限的 GPU
    # 这些 GPU 的索引从 0 开始，不管物理 GPU ID 是多少
    # 例如：CUDA_VISIBLE_DEVICES="2" 意味着物理 GPU 2 变成了 cuda:0
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible is not None and cuda_visible != "":
        # CUDA_VISIBLE_DEVICES 被设置，返回 0（逻辑 GPU 索引）
        return 0
    
    # 添加随机延迟（0-2秒），避免多个进程同时启动时都选择同一 GPU
    random_delay = random.uniform(0, 2.0)
    time.sleep(random_delay)

    # 每个 GPU 最大槽位数（保守设置为 2）
    max_slots_per_gpu = int(os.environ.get("JINA_MAX_SLOTS_PER_GPU", "2"))
    
    # 收集每个 GPU 的状态
    gpu_status = []
    for gpu_id in range(NUM_GPUS):
        running_tasks = get_gpu_running_tasks(gpu_id, max_slots_per_gpu)
        free_memory = get_gpu_free_memory(gpu_id)
        # 同时检查槽位和内存阈值
        has_capacity = (running_tasks < max_slots_per_gpu) and (free_memory >= MIN_FREE_MEMORY_FOR_NEW_TASK)
        gpu_status.append({
            'gpu_id': gpu_id,
            'running_tasks': running_tasks,
            'free_memory': free_memory,
            'has_capacity': has_capacity
        })
    
    # 打印当前 GPU 状态（调试用）
    print(f"[GPU 调度] 当前状态 (内存阈值: {MIN_FREE_MEMORY_FOR_NEW_TASK}GB):")
    for status in gpu_status:
        slots_info = f"{status['running_tasks']}/{max_slots_per_gpu}"
        mem_info = f"{status['free_memory']:.1f}GB"
        capacity_info = "✓可用" if status['has_capacity'] else "✗满载"
        print(f"  GPU {status['gpu_id']}: 槽位 {slots_info}, 空闲 {mem_info}, {capacity_info}")
    
    # 优先选择有容量的 GPU（槽位 + 内存都满足）
    available_gpus = [s for s in gpu_status if s['has_capacity']]
    
    if available_gpus:
        # 按 (运行任务数, -空闲内存) 排序
        available_gpus.sort(key=lambda x: (x['running_tasks'], -x['free_memory']))
        
        # 如果有多个 GPU 任务数相同，随机选择一个（避免热点）
        min_tasks = available_gpus[0]['running_tasks']
        candidates = [g for g in available_gpus if g['running_tasks'] == min_tasks]
        
        if len(candidates) > 1:
            best = random.choice(candidates)
            print(f"[GPU 调度] 从 {len(candidates)} 个候选中随机选择 GPU {best['gpu_id']} (任务: {best['running_tasks']}, 空闲: {best['free_memory']:.1f}GB)")
        else:
            best = candidates[0]
            print(f"[GPU 调度] 选择 GPU {best['gpu_id']} (任务: {best['running_tasks']}, 空闲: {best['free_memory']:.1f}GB)")
        return best['gpu_id']
    else:
        # 所有 GPU 都满载或内存不足，选择空闲内存最多的（等待资源释放）
        gpu_status.sort(key=lambda x: -x['free_memory'])
        best = gpu_status[0]
        print(f"[GPU 调度] ⚠ 所有 GPU 满载，选择内存最多的 GPU {best['gpu_id']} (空闲: {best['free_memory']:.1f}GB)")
        return best['gpu_id']

def get_adaptive_batch_size(gpu_id: int) -> int:
    """
    根据 GPU 空闲内存动态计算批量大小
    
    对于 Jina Reranker v3：
    - 模型本身约需 4-5GB 显存
    - 每个 batch item 约需 0.2-0.3GB 显存（取决于文本长度）
    
    Args:
        gpu_id: GPU ID
        
    Returns:
        建议的批量大小
    """
    if not BATCH_SIZE_AUTO:
        return 16  # 默认批量大小（提高）
    
    free_memory = get_gpu_free_memory(gpu_id)
    
    # 留出 5GB 给模型本身，剩余用于批量处理
    available_for_batch = max(0, free_memory - 5.0)
    
    # 每个 batch item 约 0.25GB（更激进的估计）
    batch_size = int(available_for_batch / 0.25)
    
    # 限制范围：最小 4，最大 64
    batch_size = max(4, min(64, batch_size))
    
    return batch_size

# 多项目并行配置：每个项目使用一个GPU
# 通过项目名hash分配GPU，确保不同项目使用不同GPU
def get_project_gpu_id(project_name: str) -> int:
    """
    根据项目名分配GPU ID

    优先级：
    1. 环境变量 CUDA_VISIBLE_DEVICES（设置后只能看到 1 个 GPU，索引为 0）
    2. 选择空闲内存最多的 GPU
    3. 使用项目名 hash 分配

    Args:
        project_name: 项目名称

    Returns:
        分配的GPU ID (0 到 NUM_GPUS-1)
    """
    if not torch.cuda.is_available() or NUM_GPUS == 0:
        return 0

    # 检查是否通过环境变量指定了 GPU
    # 注意：设置 CUDA_VISIBLE_DEVICES 后，可见的 GPU 索引从 0 开始
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible is not None and cuda_visible != "":
        # CUDA_VISIBLE_DEVICES 被设置，只能看到有限的 GPU，使用索引 0
        return 0
    
    # 尝试选择空闲内存最多的 GPU
    best_gpu = get_best_gpu_for_task()
    
    # 检查这个 GPU 是否被锁定（其他项目在用）
    lock_file_path = Path(__file__).parent / f".jina_reranker_gpu{best_gpu}.lock"
    if lock_file_path.exists():
        # 尝试非阻塞获取锁
        try:
            with open(lock_file_path, 'r') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                # 成功获取锁，说明 GPU 空闲
                return best_gpu
        except (IOError, OSError):
            # GPU 被占用，使用 hash 分配
            pass
    else:
        return best_gpu
    
    # 回退：使用项目名的hash值来分配GPU
    import hashlib
    hash_value = int(hashlib.md5(project_name.encode()).hexdigest(), 16)
    gpu_id = hash_value % NUM_GPUS
    return gpu_id

# 获取当前项目分配的GPU
PROJECT_GPU_ID = get_project_gpu_id(PROJECT_NAME) if PROJECT_NAME else 0

# --- 1. 加载模型 ---
def load_model(device_id=0, local_files_only=None, max_retries=3, retry_delay=5):
    """
    加载 Jina reranker 模型到指定的 GPU
    
    Args:
        device_id: GPU 设备 ID (0, 1, 2, 3...)
        local_files_only: 是否只使用本地缓存（避免重复下载）
        max_retries: 最大重试次数（用于处理 OOM）
        retry_delay: 重试间隔（秒）
    
    Returns:
        加载的模型和设备
    """
    import time
    import gc
    
    print(f"[DEBUG] load_model 开始: device_id={device_id}, local_files_only={local_files_only}, max_retries={max_retries}")
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    print(f"[DEBUG] 使用设备: {device}")
    if local_files_only is None:
        local_files_only = LOCAL_ONLY
    
    # 设置环境变量，减少重复下载提示
    os.environ["HF_HUB_DISABLE_EXPERIMENTAL_WARNING"] = "1"
    
    last_error = None
    for attempt in range(max_retries):
        try:
            # 在尝试加载前先清理 GPU 内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                
                # 检查 GPU 内存使用情况（仅在重试时检查）
                if attempt > 0:
                    try:
                        free_memory = torch.cuda.get_device_properties(device_id).total_memory - torch.cuda.memory_allocated(device_id)
                        free_gb = free_memory / (1024**3)
                        if free_gb < 8:  # 如果可用内存少于 8GB，等待
                            print(f"[GPU {device_id}] 可用内存不足 ({free_gb:.1f}GB)，等待...")
                            time.sleep(retry_delay)
                            torch.cuda.empty_cache()
                            gc.collect()
                    except Exception:
                        pass
    
            print(f"[DEBUG] 尝试加载模型 {MODEL_NAME} (attempt={attempt+1}/{max_retries})")
            model = AutoModel.from_pretrained(
                MODEL_NAME,
                dtype="auto",
                trust_remote_code=True,
                local_files_only=local_files_only,
            )
            print(f"[DEBUG] 模型加载成功，移动到设备 {device}")
            model.to(device)
            model.eval()
            print(f"[DEBUG] 模型已就绪")
            return model, device
            
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            last_error = e
            error_str = str(e).lower()
            if "out of memory" in error_str or "cuda" in error_str:
                print(f"[GPU {device_id}] 加载模型失败 (尝试 {attempt + 1}/{max_retries}): CUDA OOM")
                # 清理内存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                
                if attempt < max_retries - 1:
                    # 指数退避
                    wait_time = retry_delay * (2 ** attempt)
                    print(f"[GPU {device_id}] 等待 {wait_time}s 后重试...")
                    time.sleep(wait_time)
            else:
                raise  # 其他错误直接抛出
    
    # 所有重试都失败，抛出最后的错误
    raise last_error

def preload_model():
    """
    主进程预加载模型（确保模型已下载到本地缓存）
    这样子进程就可以使用 local_files_only=True 避免重复下载
    
    注意：预加载时使用 CPU 而不是 GPU，以避免与后续 worker 进程争抢 GPU 内存
    """
    print(f"正在预加载 reranker 模型: {MODEL_NAME}")
    if LOCAL_ONLY and not Path(MODEL_NAME).exists():
        print(f"错误: 未找到本地模型目录 {MODEL_NAME}，请检查 JINA_RERANKER_LOCAL_DIR/JINA_RERANKER_MODEL 设置")
        return False
    
    # 检查 GPU 内存是否已被占用，如果是则跳过预加载
    if torch.cuda.is_available():
        try:
            allocated = torch.cuda.memory_allocated(0)
            if allocated > 1e9:  # 超过 1GB 已被占用
                print(f"GPU 0 已有 {allocated / 1e9:.2f}GB 内存被占用，跳过预加载以避免 OOM")
                return True  # 返回 True 表示可以继续，worker 会自己加载
        except Exception:
            pass
    
    try:
        # 使用 CPU 进行预加载，避免占用 GPU 内存
        # 这样只是确保模型文件已下载到本地缓存
        device = torch.device("cpu")
        print("使用 CPU 进行模型预加载（仅验证缓存，避免占用 GPU 内存）...")
        model = AutoModel.from_pretrained(
            MODEL_NAME,
            dtype="auto",
            trust_remote_code=True,
            local_files_only=LOCAL_ONLY,
        )
        model.to(device)
        model.eval()
        del model  # 释放内存
        
        # 如果有 CUDA，清理缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        import gc
        gc.collect()
        
        print("模型预加载完成，已缓存到本地")
        return True
    except Exception as e:
        print(f"模型预加载失败: {e}")
        # 预加载失败不应阻止程序继续，worker 可能仍能工作
        return True  # 返回 True 让程序继续

# --- 2. 重排逻辑 ---
# Jina Reranker v3 的 token 限制是 131072
# 代码约 3-4 字符/token，预留 10% 安全边际
JINA_MAX_TOKENS = 131072
JINA_SAFE_MAX_TOKENS = int(JINA_MAX_TOKENS * 0.9)  # 约 118000 tokens
CHARS_PER_TOKEN = 4  # 代码约 4 字符/token

# 预防性截断阈值
# 总字符数限制：约 118000 tokens * 4 chars/token = 472000 chars
MAX_TOTAL_CHARS = JINA_SAFE_MAX_TOKENS * CHARS_PER_TOKEN

# 策略：Query 不限制，Documents 按相关性顺序添加直到达到限制
# 这样可以保证查询完整，同时尽可能多地包含相关文档

# Token 超长时的回退截断阈值（更激进，用于出错后重试）
FALLBACK_MAX_QUERY_CHARS = 50000   # query 回退截断阈值（通常不会触发）
FALLBACK_MAX_DOC_CHARS = 20000     # 每个 document 回退截断阈值

def truncate_text(text: str, max_chars: int) -> str:
    """
    截断过长的文本
    
    Args:
        text: 输入文本
        max_chars: 最大字符数
        
    Returns:
        截断后的文本
    """
    if len(text) <= max_chars:
        return text
    # 保留开头部分（函数签名和主要逻辑通常在开头）
    return text[:max_chars] + "\n... [truncated]"

def truncate_for_rerank(query: str, documents: list, 
                        max_query_chars: int = FALLBACK_MAX_QUERY_CHARS, 
                        max_doc_chars: int = FALLBACK_MAX_DOC_CHARS) -> tuple:
    """
    截断 query 和 documents（仅在需要时调用）
    
    Args:
        query: 查询文本
        documents: 文档列表
        max_query_chars: query 最大字符数
        max_doc_chars: 每个 document 最大字符数
        
    Returns:
        (truncated_query, truncated_documents)
    """
    truncated_query = truncate_text(query, max_query_chars)
    truncated_docs = [truncate_text(doc, max_doc_chars) for doc in documents]
    return truncated_query, truncated_docs

def estimate_tokens(text: str) -> int:
    """估算文本的 token 数量"""
    return len(text) // CHARS_PER_TOKEN if text else 0

def preemptive_truncate_for_rerank(query: str, documents: list) -> tuple:
    """
    预防性截断，在调用 rerank 之前就确保不超过模型限制
    
    策略：
    1. Query 完整保留（不做任何限制，因为这是我们的目标查询内容）
    2. Documents 从 BM25 筛选的最相关的开始添加
    3. 一直添加直到：添加完成 或者 添加后会超过最大 token 限制
    
    这样可以保证：
    - 查询函数完整性
    - 尽可能多地包含相关文档（按相关性排序）
    - 不超过模型限制
    
    Jina Reranker v3 的限制是 131072 tokens
    我们预留 10% 安全边际，目标是 ~118000 tokens
    
    Args:
        query: 查询文本（完整保留）
        documents: 文档列表（按 BM25 相关性排序，索引 0 最相关）
        
    Returns:
        (query, selected_documents, original_indices, was_truncated)
        - query: 原始 query（不截断）
        - selected_documents: 选中的文档列表
        - original_indices: 选中文档在原始列表中的索引
        - was_truncated: 是否进行了截断
    """
    if not documents:
        return query, [], [], False
    
    # Query 完整保留，不做截断
    query_tokens = estimate_tokens(query)
    
    # 计算 documents 可用的 token 预算
    remaining_tokens = JINA_SAFE_MAX_TOKENS - query_tokens
    
    if remaining_tokens <= 0:
        # Query 本身就超过了限制，但我们仍然保留完整 query
        # 只添加第一个文档（最相关的）并截断
        print(f"  ⚠ Query 过长 (~{query_tokens} tokens)，只保留最相关的 1 个文档")
        first_doc = documents[0]
        # 给文档留一点空间（至少 1000 tokens）
        max_doc_chars = max(4000, remaining_tokens * CHARS_PER_TOKEN)
        truncated_doc = truncate_text(first_doc, max_doc_chars)
        return query, [truncated_doc], [0], True
    
    # 从最相关的开始添加文档，直到达到 token 限制
    selected_docs = []
    selected_indices = []
    current_tokens = query_tokens
    
    for idx, doc in enumerate(documents):
        doc_tokens = estimate_tokens(doc)
        
        # 检查添加这个文档后是否会超过限制
        if current_tokens + doc_tokens <= JINA_SAFE_MAX_TOKENS:
            # 可以完整添加
            selected_docs.append(doc)
            selected_indices.append(idx)
            current_tokens += doc_tokens
        else:
            # 添加后会超过限制
            # 检查是否还有剩余空间可以添加截断版本
            remaining_chars = (JINA_SAFE_MAX_TOKENS - current_tokens) * CHARS_PER_TOKEN
            
            if remaining_chars >= 2000:  # 至少要有 2000 字符（约 500 tokens）才值得添加
                # 截断当前文档并添加
                truncated_doc = truncate_text(doc, remaining_chars)
                selected_docs.append(truncated_doc)
                selected_indices.append(idx)
            
            # 达到限制，停止添加
            break
    
    was_truncated = len(selected_docs) < len(documents)
    
    return query, selected_docs, selected_indices, was_truncated

def is_token_length_error(error: Exception) -> bool:
    """检查是否是 token 超长错误"""
    error_msg = str(error).lower()
    return ('token' in error_msg and 'length' in error_msg) or \
           ('sequence length' in error_msg) or \
           ('longer than' in error_msg and 'maximum' in error_msg)

def is_cuda_oom_error(error: Exception) -> bool:
    """检查是否是 CUDA OOM 错误"""
    error_msg = str(error).lower()
    return 'cuda out of memory' in error_msg or 'out of memory' in error_msg

def clear_cuda_cache():
    """清理 CUDA 缓存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc
        gc.collect()

def reorder_with_jina(model, query_function: str, c_code_documents: list) -> list:
    """
    使用 Jina-reranker 对 BM25 检索到的 C 代码进行重排。
    
    策略：
    1. Query 完整保留（不截断）
    2. Documents 从最相关的开始添加，直到达到 token 限制
    3. 如果仍遇到 token 超长错误，回退到更激进的截断重试
    
    Args:
        model: 加载的 jina-reranker-v3 模型
        query_function: 原始 C++ 查询函数（完整保留）
        c_code_documents: 从 BM25 结果中提取的 C 代码范例列表（按相关性排序）
        
    Returns:
        一个重排后的列表，包含 (score, original_index)
        注意：original_index 是相对于原始 c_code_documents 的索引
    """
    if not c_code_documents:
        return []
    
    # 预防性截断（避免超过模型限制）
    # Query 完整保留，Documents 从最相关的开始添加
    query, selected_docs, selected_indices, was_truncated = preemptive_truncate_for_rerank(
        query_function, c_code_documents
    )
    
    if was_truncated:
        num_orig = len(c_code_documents)
        num_selected = len(selected_docs)
        if num_selected < num_orig:
            print(f"  📏 文档筛选: {num_orig} -> {num_selected} 个 (保留最相关的)")
    
    if not selected_docs:
        return []
    
    # 第一次尝试
    try:
        with torch.no_grad():
            results = model.rerank(
                query=query,
                documents=selected_docs,
                top_n=min(TOP_K, len(selected_docs))
            )
    except Exception as e:
        if is_token_length_error(e):
            # 仍然超长，回退到更激进的截断
            truncated_query, truncated_docs = truncate_for_rerank(query, selected_docs)
            total_chars = len(truncated_query) + sum(len(d) for d in truncated_docs)
            print(f"  ⚠ Token 仍超长，激进截断后重试 ({total_chars//1000}K chars)")
            
            with torch.no_grad():
                results = model.rerank(
                    query=truncated_query,
                    documents=truncated_docs,
                    top_n=min(TOP_K, len(truncated_docs))
                )
        else:
            raise  # 其他错误直接抛出
        
    # 'results' 是一个列表: [{'index': 5, 'relevance_score': 0.95}, ...]
    # 注意：res['index'] 是相对于 selected_docs 的索引
    # 需要映射回原始 c_code_documents 的索引
    reranked_results = []
    for res in results:
        selected_idx = res['index']  # 在 selected_docs 中的索引
        # 映射回原始索引
        original_idx = selected_indices[selected_idx] if selected_idx < len(selected_indices) else selected_idx
        reranked_results.append({
            "score": res['relevance_score'],
            "original_index": original_idx
        })
    return reranked_results


def batch_reorder_with_jina(model, batch_data: list) -> list:
    """
    批量使用 Jina-reranker 对多个 query 进行重排。
    
    策略：
    1. Query 完整保留，Documents 从最相关的开始添加直到达到 token 限制
    2. 如果仍遇到 token 超长错误，回退到更激进的截断重试
    3. 遇到 OOM 错误时，清理缓存后用截断版本重试
    
    Args:
        model: 加载的 jina-reranker-v3 模型
        batch_data: 列表，每个元素是 (query_function, c_code_documents)
        
    Returns:
        列表，每个元素是对应 query 的重排结果（索引映射回原始文档列表）
    """
    results_list = []
    
    with torch.no_grad():
        for query_function, c_code_documents in batch_data:
            if not c_code_documents:
                results_list.append([])
                continue
            
            try:
                results = None
                
                # 预防性截断（避免超过模型限制）
                # Query 完整保留，Documents 从最相关的开始添加
                query, selected_docs, selected_indices, was_truncated = preemptive_truncate_for_rerank(
                    query_function, c_code_documents
                )
                
                if not selected_docs:
                    results_list.append([])
                    continue
                
                # 第一次尝试
                try:
                    results = model.rerank(
                        query=query,
                        documents=selected_docs,
                        top_n=min(TOP_K, len(selected_docs))
                    )
                except Exception as e:
                    if is_token_length_error(e):
                        # 仍然超长，回退到更激进的截断
                        truncated_query, truncated_docs = truncate_for_rerank(query, selected_docs)
                        results = model.rerank(
                            query=truncated_query,
                            documents=truncated_docs,
                            top_n=min(TOP_K, len(truncated_docs))
                        )
                    elif is_cuda_oom_error(e):
                        # OOM 错误，清理缓存后用截断版本重试
                        clear_cuda_cache()
                        truncated_query, truncated_docs = truncate_for_rerank(
                            query, selected_docs,
                            max_query_chars=50000,  # Query 保持完整
                            max_doc_chars=15000     # 更激进的文档截断
                        )
                        try:
                            results = model.rerank(
                                query=truncated_query,
                                documents=truncated_docs,
                                top_n=min(TOP_K, len(truncated_docs))
                            )
                        except Exception:
                            # 仍然失败，跳过此 query
                            results_list.append([])
                            continue
                    else:
                        raise
                
                # 将 rerank 结果的索引映射回原始文档索引
                reranked_results = []
                for res in results:
                    selected_idx = res['index']  # 在 selected_docs 中的索引
                    # 映射回原始索引
                    original_idx = selected_indices[selected_idx] if selected_idx < len(selected_indices) else selected_idx
                    reranked_results.append({
                        "score": res['relevance_score'],
                        "original_index": original_idx
                    })
                results_list.append(reranked_results)
            except Exception as e:
                print(f"Batch rerank 单个 query 失败: {e}")
                results_list.append([])
    
    return results_list

# --- 3. 文件解析与主循环 ---
def parse_bm25_file(file_content: str):
    """
    解析 elastic_search.py 输出的文件
    [V10 升级] 支持解析 extracted_knowledge 字段
    """
    try:
        header, rest = file_content.split("-" * 50 + "\n", 1)
        query_function = header.replace("target function is :", "").strip()
        
        # 使用正则表达式提取 C 和 Rust 代码块
        c_pattern = re.compile(r"\[C_CODE\]\n(.*?)\n\[/C_CODE\]", re.DOTALL)
        r_pattern = re.compile(r"\[RUST_CODE\]\n(.*?)\n\[/RUST_CODE\]", re.DOTALL)
        k_pattern = re.compile(r"\[EXTRACTED_KNOWLEDGE\]\n(.*?)\n\[/EXTRACTED_KNOWLEDGE\]", re.DOTALL)
        
        c_docs = c_pattern.findall(rest)
        r_docs = r_pattern.findall(rest)
        k_docs = k_pattern.findall(rest)
        
        if len(c_docs) != len(r_docs):
            print(f"警告: C/Rust 代码块数量不匹配。 C: {len(c_docs)}, Rust: {len(r_docs)}")
            return query_function, [], [], []
        
        # 解析知识 JSON（如果存在）
        knowledge_list = []
        for k_str in k_docs:
            try:
                knowledge_list.append(json.loads(k_str))
            except json.JSONDecodeError:
                knowledge_list.append([])
        
        # 如果知识列表长度不匹配，用空列表补齐
        while len(knowledge_list) < len(c_docs):
            knowledge_list.append([])
            
        return query_function, c_docs, r_docs, knowledge_list
        
    except Exception as e:
        print(f"错误: 解析 BM25 文件失败: {e}")
        return None, [], [], []

def load_file_data(query_file, project_out_path):
    """
    加载单个文件的数据（用于并行 I/O）
    
    Returns:
        (query_file, query_func, c_docs_list, r_docs_list, k_docs_list) 或 None（如果文件已存在或失败）
    """
    output_file_path = project_out_path / query_file.name
    
    # 跳过已存在的文件
    if output_file_path.exists():
        return None
    
    try:
        with open(query_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        return None
    
    query_func, c_docs_list, r_docs_list, k_docs_list = parse_bm25_file(content)
    
    if not query_func or not c_docs_list:
        return None
    
    return (query_file, query_func, c_docs_list, r_docs_list, k_docs_list)


def save_rerank_result(query_file, reranked_results, c_docs_list, r_docs_list, k_docs_list, project_out_path):
    """
    保存重排结果（用于并行 I/O）
    """
    output_file_path = project_out_path / query_file.name
    
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            for res in reranked_results:
                score = res['score']
                idx = res['original_index']
                c_code = c_docs_list[idx]
                rust_code = r_docs_list[idx]
                knowledge = k_docs_list[idx] if idx < len(k_docs_list) else []
                
                f.write(f"C_Code: \n{c_code}\n")
                f.write(f"Function: \n{rust_code}\n")
                
                if knowledge:
                    f.write(f"Extracted_Knowledge: \n{json.dumps(knowledge, ensure_ascii=False)}\n")
                
                f.write(f"Unixcoder Score: {score}\n")
                f.write("-" * 50 + "\n")
        return True
    except Exception as e:
        print(f"保存结果失败 {query_file}: {e}")
        return False


def efficient_batch_process(model, query_files, project_in_path, project_out_path, batch_size=8, num_io_workers=4):
    """
    高效批量处理：单模型实例 + 多线程 I/O + 批量推理
    
    这种方式比多进程加载多个模型更高效：
    1. 只加载一个模型，节省显存
    2. 多线程并行读取文件（I/O 密集型，不需要多进程）
    3. 批量推理，提高 GPU 利用率
    4. 多线程并行写入结果
    
    Args:
        model: 已加载的模型
        query_files: 待处理的文件列表
        project_in_path: 输入路径
        project_out_path: 输出路径
        batch_size: 批量大小（同时处理的 query 数量）
        num_io_workers: I/O 线程数
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import gc
    
    total_files = len(query_files)
    processed = 0
    failed = 0
    
    # 使用线程池并行读取文件
    with ThreadPoolExecutor(max_workers=num_io_workers) as io_executor:
        # 分批处理
        for batch_start in range(0, total_files, batch_size):
            batch_end = min(batch_start + batch_size, total_files)
            batch_files = query_files[batch_start:batch_end]
            
            # 1. 并行读取这一批文件
            load_futures = {
                io_executor.submit(load_file_data, qf, project_out_path): qf 
                for qf in batch_files
            }
            
            batch_data = []
            file_metadata = []  # 保存每个文件的元数据，用于写入结果
            
            for future in as_completed(load_futures):
                result = future.result()
                if result is not None:
                    query_file, query_func, c_docs_list, r_docs_list, k_docs_list = result
                    batch_data.append((query_func, c_docs_list))
                    file_metadata.append({
                        'query_file': query_file,
                        'c_docs_list': c_docs_list,
                        'r_docs_list': r_docs_list,
                        'k_docs_list': k_docs_list
                    })
            
            if not batch_data:
                processed += len(batch_files)
                continue
            
            # 2. 批量推理
            try:
                rerank_results = batch_reorder_with_jina(model, batch_data)
            except Exception as e:
                print(f"批量推理失败: {e}")
                torch.cuda.empty_cache()
                gc.collect()
                failed += len(batch_data)
                processed += len(batch_files)
                continue
            
            # 3. 并行写入结果
            save_futures = []
            for i, (metadata, results) in enumerate(zip(file_metadata, rerank_results)):
                if results:  # 只写入有结果的文件
                    future = io_executor.submit(
                        save_rerank_result,
                        metadata['query_file'],
                        results,
                        metadata['c_docs_list'],
                        metadata['r_docs_list'],
                        metadata['k_docs_list'],
                        project_out_path
                    )
                    save_futures.append(future)
            
            # 等待写入完成
            for future in as_completed(save_futures):
                if not future.result():
                    failed += 1
            
            processed += len(batch_files)
            
            # 定期清理 GPU 内存
            if processed % (batch_size * 10) == 0:
                torch.cuda.empty_cache()
                gc.collect()
    
    return processed, failed


def process_worker(device_id, file_queue, result_queue, project_in_path, project_out_path, load_failed_event=None):
    """
    工作进程：处理分配给它的文件
    
    Args:
        device_id: GPU 设备 ID
        file_queue: 文件队列
        result_queue: 结果队列（用于进度跟踪）
        project_in_path: 输入路径
        project_out_path: 输出路径
        load_failed_event: 可选的事件，用于通知模型加载失败
    """
    import time
    import gc
    
    try:
        # 确保子进程也设置正确的缓存路径
        os.environ["HF_HOME"] = MY_CACHE_PATH
        os.environ["TRANSFORMERS_CACHE"] = MY_CACHE_PATH
        os.environ["HF_HUB_CACHE"] = str(Path(MY_CACHE_PATH) / "hub")
        
        # 错开启动时间，避免所有 GPU 同时加载模型导致 OOM
        # 只需要轻微错开（1秒），避免同时分配大块内存
        startup_delay = device_id * 1  # 每个 GPU 错开 1 秒
        if startup_delay > 0:
            time.sleep(startup_delay)
        
        # 清理 GPU 内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        # 每个进程加载自己的模型实例
        # 使用本地缓存加载模型（主进程已预加载）
        try:
            model, device = load_model(device_id, local_files_only=True, max_retries=3, retry_delay=15)
            print(f"[GPU {device_id}] 模型加载完成")
        except Exception as e:
            print(f"[GPU {device_id}] 模型加载最终失败: {e}")
            # 通知主进程加载失败
            if load_failed_event:
                load_failed_event.set()
            # 将队列中剩余的文件标记为失败
            while True:
                try:
                    item = file_queue.get_nowait()
                    if item is None:
                        break
                    result_queue.put(0)  # 标记为失败
                except:
                    break
            return
        
        processed = 0
        while True:
            # 从队列获取文件
            item = file_queue.get()
            if item is None:  # 结束信号
                break
            
            query_file = item
            output_file_path = project_out_path / query_file.name

            # 跳过已存在的文件
            if output_file_path.exists():
                result_queue.put(1)  # 已存在
                continue

            # 1. 读取 BM25 结果文件
            try:
                with open(query_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
            except Exception as e:
                print(f"[GPU {device_id}] 无法读取 {query_file}: {e}")
                result_queue.put(0)  # 失败
                continue

            # 2. 解析查询函数和 C/Rust 文档
            query_func, c_docs_list, r_docs_list, k_docs_list = parse_bm25_file(content)
            
            if not query_func or not c_docs_list:
                result_queue.put(0)  # 跳过
                continue

            # 3. 使用 Jina-Reranker 重排
            try:
                reranked_results = reorder_with_jina(model, query_func, c_docs_list)
            except Exception as e:
                print(f"[GPU {device_id}] Jina Rerank 失败 (文件: {query_file}): {e}")
                torch.cuda.empty_cache()
                result_queue.put(0)  # 失败
                continue

            # 4. 写入重排后的结果（C→C 重排，保存时补齐对应 Rust 和知识）
            try:
                with open(output_file_path, 'w', encoding='utf-8') as f:
                    for res in reranked_results:
                        score = res['score']
                        idx = res['original_index']
                        c_code = c_docs_list[idx]
                        rust_code = r_docs_list[idx]
                        knowledge = k_docs_list[idx] if idx < len(k_docs_list) else []
                        
                        # 为兼容下游，保留原有"Function"字段指向 Rust；同时额外输出 C_Code 方便排查
                        f.write(f"C_Code: \n{c_code}\n")
                        f.write(f"Function: \n{rust_code}\n")
                        
                        # [V10 新增] 写入提取的知识
                        if knowledge:
                            f.write(f"Extracted_Knowledge: \n{json.dumps(knowledge, ensure_ascii=False)}\n")
                        
                        f.write(f"Unixcoder Score: {score}\n")
                        f.write("-" * 50 + "\n")
                
                processed += 1
                result_queue.put(1)  # 成功
            except Exception as e:
                print(f"[GPU {device_id}] 写入失败 (文件: {output_file_path}): {e}")
                result_queue.put(0)  # 失败
        
        print(f"[GPU {device_id}] 处理完成，共处理 {processed} 个文件")
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"[GPU {device_id}] 工作进程错误: {e}")
        import traceback
        traceback.print_exc()

def wait_for_gpu_memory(gpu_id: int, min_free_gb: float = 12.0, max_wait_seconds: int = 600) -> bool:
    """
    等待 GPU 有足够的空闲内存
    
    Args:
        gpu_id: GPU ID
        min_free_gb: 最小空闲内存（GB）
        max_wait_seconds: 最大等待时间（秒）
        
    Returns:
        是否成功获取到足够内存
    """
    import gc
    
    waited = 0
    check_interval = 10
    
    while waited < max_wait_seconds:
        # 清理缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        free_memory = get_gpu_free_memory(gpu_id)
        
        if free_memory >= min_free_gb:
            print(f"[GPU {gpu_id}] 空闲内存: {free_memory:.1f}GB >= {min_free_gb}GB，可以开始")
            return True
        
        if waited == 0:
            print(f"[GPU {gpu_id}] 空闲内存不足: {free_memory:.1f}GB < {min_free_gb}GB")
            print(f"[GPU {gpu_id}] 等待其他进程释放内存（最多等待 {max_wait_seconds}s）...")
        
        time.sleep(check_interval)
        waited += check_interval
        
        if waited % 60 == 0:
            print(f"[GPU {gpu_id}] 已等待 {waited}s，当前空闲: {free_memory:.1f}GB")
    
    print(f"[GPU {gpu_id}] 等待超时，当前空闲内存: {get_gpu_free_memory(gpu_id):.1f}GB")
    return False


def acquire_gpu_slot(gpu_id: int, max_slots: int = 2, max_wait_seconds: int = 1800) -> tuple:
    """
    获取 GPU 槽位（每个 GPU 最多允许 max_slots 个实例同时运行）
    
    Args:
        gpu_id: GPU ID
        max_slots: 每个 GPU 最大并行数
        max_wait_seconds: 最大等待时间
        
    Returns:
        (slot_id, lock_file) 或 (None, None) 如果获取失败
    """
    waited = 0
    wait_interval = 3
    
    while waited < max_wait_seconds:
        # 尝试获取任意一个空闲槽位
        for slot_id in range(max_slots):
            lock_file_path = SCRIPT_DIR / f".jina_reranker_gpu{gpu_id}_slot{slot_id}.lock"
            try:
                lock_file = open(lock_file_path, 'w')
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                # 成功获取锁
                return slot_id, lock_file
            except (IOError, OSError):
                # 这个槽位被占用，尝试下一个
                try:
                    lock_file.close()
                except:
                    pass
                continue
        
        # 所有槽位都被占用，等待
        if waited == 0:
            print(f"[GPU {gpu_id}] 所有 {max_slots} 个槽位都被占用，排队等待...")
        
        time.sleep(wait_interval)
        waited += wait_interval
        
        if waited % 30 == 0:
            print(f"[GPU {gpu_id}] 已等待 {waited}s...")
    
    print(f"[GPU {gpu_id}] 等待槽位超时（{max_wait_seconds}s）")
    return None, None


def release_gpu_slot(lock_file):
    """释放 GPU 槽位"""
    if lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            lock_file.close()
        except:
            pass


def main():
    """
    主函数：支持多项目并行运行
    每个 GPU 最多允许 DEFAULT_MAX_SLOTS_PER_GPU 个 Jina Reranker 实例同时运行
    对于 49GB 显存的 GPU（如 RTX 5880 Ada），默认为 4 个
    """
    # 每个 GPU 最大并行数（可通过环境变量调整）
    max_slots_per_gpu = int(os.environ.get("JINA_MAX_SLOTS_PER_GPU", str(DEFAULT_MAX_SLOTS_PER_GPU)))
    
    slot_id = None
    lock_file = None
    
    try:
        if torch.cuda.is_available() and NUM_GPUS > 0:
            global PROJECT_GPU_ID
            if EXTERNAL_GPU_SCHEDULER:
                # 外部调度器已通过 CUDA_VISIBLE_DEVICES 将本进程绑定到某个“物理 GPU”，此时本进程只应使用逻辑 cuda:0。
                # 不能再用本脚本的“全局锁文件”做跨项目排队，否则不同物理 GPU 会互相误伤（都写同一个 gpu0_slot*.lock）。
                PROJECT_GPU_ID = 0
                cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
                if cvd:
                    print(f"[项目 {PROJECT_NAME}] 外部调度模式: CUDA_VISIBLE_DEVICES={cvd} (逻辑使用 cuda:0)")
            else:
                # 选择最佳 GPU（脚本独立运行时才启用内部调度/锁槽）
                best_gpu = get_best_gpu_for_task()
                PROJECT_GPU_ID = best_gpu

                print(f"[项目 {PROJECT_NAME}] 分配到 GPU {PROJECT_GPU_ID}/{NUM_GPUS}")
                print(f"[项目 {PROJECT_NAME}] 每个 GPU 最大并行数: {max_slots_per_gpu}")

                # 获取 GPU 槽位
                slot_id, lock_file = acquire_gpu_slot(PROJECT_GPU_ID, max_slots=max_slots_per_gpu)

                if slot_id is not None:
                    print(f"[项目 {PROJECT_NAME}] 已获取 GPU {PROJECT_GPU_ID} 槽位 {slot_id}")
                else:
                    print(f"[项目 {PROJECT_NAME}] 无法获取 GPU 槽位，强制继续（可能导致 OOM）...")

                # 等待 GPU 内存释放
                # 每个 Jina Reranker 进程约需 6GB 内存
                min_required_memory = float(os.environ.get("JINA_MIN_GPU_MEMORY_GB", "6.0"))
                if not wait_for_gpu_memory(PROJECT_GPU_ID, min_required_memory, max_wait_seconds=300):
                    print(f"[警告] GPU {PROJECT_GPU_ID} 内存不足（需要 {min_required_memory}GB），尝试继续运行...")

            # 确保使用 spawn 方法（CUDA 要求）
            current_method = get_start_method(allow_none=True)
            if current_method != 'spawn':
                try:
                    set_start_method('spawn', force=True)
                    print(f"设置 multiprocessing 启动方法为 'spawn'（CUDA 要求）")
                except RuntimeError as e:
                    print(f"警告: 无法设置启动方法: {e}")
        
        # 运行 Jina Reranker（GPU 或 CPU 模式）
        _run_jina_rerank()
        
    finally:
        # 释放 GPU 槽位
        if lock_file:
            release_gpu_slot(lock_file)
            print(f"[项目 {PROJECT_NAME}] 已释放 GPU {PROJECT_GPU_ID} 槽位 {slot_id}")

def _run_jina_rerank():
    """实际的 Jina Reranker 处理逻辑"""
    print(f"--- (RAG 步骤 2) Jina 重排开始 ---")
    print(f"项目: {PROJECT_NAME}")
    print(f"BM25 结果输入: {RAG_PATH}")
    print(f"重排结果输出: {OUTPUT_PATH}")
    if torch.cuda.is_available() and NUM_GPUS > 0:
        print(f"检测到 {NUM_GPUS} 个 GPU，项目 '{PROJECT_NAME}' 使用 GPU {PROJECT_GPU_ID}")
        print(f"提示: 不同项目会自动分配到不同GPU，最多可并行运行 {NUM_GPUS} 个项目")
    else:
        print(f"使用 CPU 模式")

    if not RAG_PATH.exists():
        print(f"错误: BM25 结果目录不存在: {RAG_PATH}")
        return
    
    project = PROJECT_NAME
    project_in_path = RAG_PATH
    project_out_path = OUTPUT_PATH
    project_out_path.mkdir(parents=True, exist_ok=True)
    
    query_files = list(project_in_path.glob("*.txt"))
    # 过滤掉已处理的文件
    query_files = [f for f in query_files if not (project_out_path / f.name).exists()]
    
    print(f"\n正在处理项目: {project} ({len(query_files)} 个查询函数)")

    if not query_files:
        print("所有文件已处理完成，跳过。")
        return

    # 使用高效批量处理模式：单模型实例 + 多线程 I/O + 批量推理
    # 这种方式比多进程加载多个模型更高效，因为：
    # 1. 只加载一个模型，节省显存（约 2-8GB）
    # 2. 多线程并行读取/写入文件（I/O 密集型）
    # 3. 批量推理，充分利用 GPU 并行能力
    
    USE_EFFICIENT_BATCH = True  # 开启高效批量模式
    
    if USE_EFFICIENT_BATCH and len(query_files) > 5:
        # 高效批量处理模式
        device_id = PROJECT_GPU_ID if torch.cuda.is_available() else -1
        
        print(f"使用高效批量处理模式 (GPU {device_id})...")
        print(f"  - 单模型实例（节省显存）")
        print(f"  - 批量推理（自适应 batch_size）")
        print(f"  - 多线程 I/O")
        
        # 检查 GPU 内存
        if device_id >= 0:
            free_memory = get_gpu_free_memory(device_id)
            print(f"  - GPU {device_id} 空闲内存: {free_memory:.1f} GB")
            
            if free_memory < 10.0:
                print(f"[警告] GPU 空闲内存不足 ({free_memory:.1f}GB < 10GB)，等待释放...")
                wait_for_gpu_memory(device_id, min_free_gb=10.0, max_wait_seconds=120)
        
        # 加载模型
        model, device = load_model(device_id)
        print(f"模型已加载到: {device}")
        
        # 计算最佳批量大小（根据 GPU 内存动态调整）
        if device_id >= 0 and BATCH_SIZE_AUTO:
            batch_size = get_adaptive_batch_size(device_id)
            print(f"  - 自适应 batch_size: {batch_size} (基于 GPU 空闲内存)")
        else:
            batch_size = min(32, max(8, len(query_files) // 10))
        
        # 限制 batch_size（根据显存动态调整，不再硬编码为 8）
        # 对于 49GB 显存的 GPU，可以使用更大的批量
        max_batch_size = int(os.environ.get("JINA_MAX_BATCH_SIZE", "32"))
        batch_size = min(batch_size, max_batch_size)
        num_io_workers = min(8, len(query_files) // 5 + 1)  # 增加 I/O workers
        
        print(f"批量大小: {batch_size}, I/O 线程数: {num_io_workers}")
        
        # 使用 tqdm 显示进度
        with tqdm(total=len(query_files), desc=f"Jina Rerank ({project}, GPU {device_id}, batch={batch_size})") as pbar:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            import gc
            
            processed = 0
            
            # 分批处理
            for batch_start in range(0, len(query_files), batch_size):
                batch_end = min(batch_start + batch_size, len(query_files))
                batch_files = query_files[batch_start:batch_end]
                
                # 1. 并行读取这一批文件
                with ThreadPoolExecutor(max_workers=num_io_workers) as io_executor:
                    load_futures = {
                        io_executor.submit(load_file_data, qf, project_out_path): qf 
                        for qf in batch_files
                    }
                    
                    batch_data = []
                    file_metadata = []
                    skipped = 0
                    
                    for future in as_completed(load_futures):
                        result = future.result()
                        if result is not None:
                            query_file, query_func, c_docs_list, r_docs_list, k_docs_list = result
                            batch_data.append((query_func, c_docs_list))
                            file_metadata.append({
                                'query_file': query_file,
                                'c_docs_list': c_docs_list,
                                'r_docs_list': r_docs_list,
                                'k_docs_list': k_docs_list
                            })
                        else:
                            skipped += 1
                
                if not batch_data:
                    pbar.update(len(batch_files))
                    continue
                
                # 2. 批量推理（带 OOM 恢复）
                try:
                    rerank_results = batch_reorder_with_jina(model, batch_data)
                except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                    error_str = str(e).lower()
                    if "out of memory" in error_str or "cuda" in error_str:
                        print(f"\n[OOM] 批量推理失败，清理内存并重试单条处理...")
                        torch.cuda.empty_cache()
                        gc.collect()
                        time.sleep(5)  # 等待内存释放
                        
                        # 逐条处理（降级模式）
                        rerank_results = []
                        for query_func, c_docs_list in batch_data:
                            try:
                                result = reorder_with_jina(model, query_func, c_docs_list)
                                rerank_results.append(result)
                                torch.cuda.empty_cache()
                            except Exception as inner_e:
                                print(f"  单条处理也失败: {inner_e}")
                                rerank_results.append([])
                    else:
                        print(f"\n批量推理失败: {e}")
                        torch.cuda.empty_cache()
                        gc.collect()
                        pbar.update(len(batch_files))
                        continue
                except Exception as e:
                    print(f"\n批量推理失败: {e}")
                    torch.cuda.empty_cache()
                    gc.collect()
                    pbar.update(len(batch_files))
                    continue
                
                # 3. 并行写入结果
                with ThreadPoolExecutor(max_workers=num_io_workers) as io_executor:
                    save_futures = []
                    for metadata, results in zip(file_metadata, rerank_results):
                        if results:
                            future = io_executor.submit(
                                save_rerank_result,
                                metadata['query_file'],
                                results,
                                metadata['c_docs_list'],
                                metadata['r_docs_list'],
                                metadata['k_docs_list'],
                                project_out_path
                            )
                            save_futures.append(future)
                    
                    for future in as_completed(save_futures):
                        future.result()
                
                pbar.update(len(batch_files))
                
                # 定期清理 GPU 内存
                if (batch_start + batch_size) % (batch_size * 5) == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
        
        print(f"高效批量处理完成")
    
    elif USE_MULTI_GPU and len(query_files) > 10:
        # 备用：多进程模式（当高效模式不适用时）
        num_workers_per_gpu = min(2, len(query_files) // 20 + 1)
        
        print(f"使用多进程模式 (GPU {PROJECT_GPU_ID}, {num_workers_per_gpu} workers)...")
        
        preload_model()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            import gc
            gc.collect()
        
        manager = Manager()
        file_queue = manager.Queue()
        result_queue = manager.Queue()
        load_failed_event = manager.Event()
        
        for query_file in query_files:
            file_queue.put(query_file)
        
        for _ in range(num_workers_per_gpu):
            file_queue.put(None)
        
        processes = []
        for worker_id in range(num_workers_per_gpu):
            p = Process(
                target=process_worker,
                args=(PROJECT_GPU_ID, file_queue, result_queue, project_in_path, project_out_path, load_failed_event)
            )
            p.start()
            processes.append(p)
        
        with tqdm(total=len(query_files), desc=f"Jina Rerank ({project}, GPU {PROJECT_GPU_ID}, {num_workers_per_gpu} workers)") as pbar:
            completed = 0
            while completed < len(query_files):
                result_queue.get()
                completed += 1
                pbar.update(1)
        
        for p in processes:
            p.join()
        
        print(f"GPU {PROJECT_GPU_ID} 处理完成")
    else:
        # 单进程处理（文件数量较少时）
        device_id = PROJECT_GPU_ID if torch.cuda.is_available() else -1
        model, device = load_model(device_id)
        print(f"使用设备: {device} (项目 {project})")

        for query_file in tqdm(query_files, desc=f"Jina Rerank ({project})"):
            output_file_path = project_out_path / query_file.name

            if output_file_path.exists():
                continue

            # 1. 读取 BM25 结果文件
            try:
                with open(query_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
            except Exception as e:
                print(f"无法读取 {query_file}: {e}")
                continue

            # 2. 解析查询函数和 C/Rust 文档
            query_func, c_docs_list, r_docs_list, k_docs_list = parse_bm25_file(content)
            
            if not query_func or not c_docs_list:
                continue

            # 3. 使用 Jina-Reranker 重排
            try:
                reranked_results = reorder_with_jina(model, query_func, c_docs_list)
            except Exception as e:
                print(f"Jina Rerank 失败 (文件: {query_file}): {e}")
                torch.cuda.empty_cache()
                continue

            # 4. 写入重排后的结果（C→C 重排，保存时补齐对应 Rust 和知识）
            with open(output_file_path, 'w', encoding='utf-8') as f:
                for res in reranked_results:
                    score = res['score']
                    idx = res['original_index']
                    c_code = c_docs_list[idx]
                    rust_code = r_docs_list[idx]
                    knowledge = k_docs_list[idx] if idx < len(k_docs_list) else []
                    
                    # 为兼容下游，保留原有"Function"字段指向 Rust；同时额外输出 C_Code 方便排查
                    f.write(f"C_Code: \n{c_code}\n")
                    f.write(f"Function: \n{rust_code}\n")
                    
                    # [V10 新增] 写入提取的知识
                    if knowledge:
                        f.write(f"Extracted_Knowledge: \n{json.dumps(knowledge, ensure_ascii=False)}\n")
                    
                    f.write(f"Unixcoder Score: {score}\n")
                    f.write("-" * 50 + "\n")
                    
    print("--- (RAG 步骤 2) Jina 重排完成 ---")

if __name__ == "__main__":
    # 必须在主模块中设置启动方法为 'spawn'
    # 这是 PyTorch CUDA 与 multiprocessing 的要求
    # fork 方法会导致 CUDA 无法在子进程中重新初始化
    import multiprocessing
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        # 如果已经设置过，忽略错误
        pass
    main()
