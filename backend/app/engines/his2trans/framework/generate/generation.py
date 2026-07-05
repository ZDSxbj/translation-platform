from openai import OpenAI
import concurrent.futures
import sys
import json
import re
import os
import multiprocessing
import time
from threading import Semaphore
# from tenacity import retry, stop_after_attempt, wait_fixed

# --- vLLM 并发控制 ---
# 限制同时发送到 vLLM 的请求数，避免多项目并行时请求过多
VLLM_CONCURRENT_LIMIT = int(os.environ.get("VLLM_CONCURRENT_LIMIT", "120"))
_vllm_semaphore = Semaphore(VLLM_CONCURRENT_LIMIT)

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

# def extract_code(content):
    

#     pattern = r'\[translated result\](.*?)\[#END\]'
#     translated_result = re.findall(pattern, content, re.DOTALL)[0].strip()

#     translated_code = None

#     patterns = [r'```rust(.*?)```',r'```Rust(.*?)```']

#     for pattern in patterns:
#         if not translated_code:
#             try:
#                 translated_code = re.findall(pattern, translated_result, re.DOTALL)[0].strip()
                
#             except:
#                 translated_code = None
#         else:
#             break
    
#     if not translated_code:
#         translated_code = translated_result
    

#     return translated_code

# --- vLLM 配置 ---
# 支持环境变量配置，默认使用 vLLM。
#
# ⚠️ 重要：vLLM 的 OpenAI 兼容接口里 `model=` 必须使用 vLLM 暴露的 "served model name"
# 而不是 HuggingFace 的仓库名/本地路径。
# 例如 vLLM 可能以 served_model_name=["qwen3_coder"] 启动，则这里必须传 "qwen3_coder"。
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "EMPTY")
VLLM_MODEL_NAME = (
    os.environ.get("VLLM_MODEL_NAME")
    or os.environ.get("LLM_NAME")
    or "qwen3_coder"
)
VLLM_NUM_WORKERS = int(os.environ.get("VLLM_NUM_WORKERS", "48"))
VLLM_REQUEST_TIMEOUT = float(os.environ.get("VLLM_REQUEST_TIMEOUT", os.environ.get("VLLM_TIMEOUT", "600.0")))
VLLM_MAX_RETRIES = int(os.environ.get("VLLM_MAX_RETRIES", "3"))
VLLM_RETRY_BACKOFF_SEC = float(os.environ.get("VLLM_RETRY_BACKOFF_SEC", "2.0"))
VLLM_RETRY_BACKOFF_MAX_SEC = float(os.environ.get("VLLM_RETRY_BACKOFF_MAX_SEC", "30.0"))

# --- 外部 API 配置 (当 USE_VLLM=false 时使用) ---
# EXTERNAL_API_KEY = os.environ.get("EXTERNAL_API_KEY", "<YOUR_API_KEY>")
# EXTERNAL_API_BASE_URL = os.environ.get("EXTERNAL_API_BASE_URL", "https://api.agicto.cn/v1")
# EXTERNAL_API_MODEL = os.environ.get("EXTERNAL_API_MODEL", "deepseek-v3.2")
# EXTERNAL_API_MAX_TOKENS = int(os.environ.get("EXTERNAL_API_MAX_TOKENS", "16384"))
# EXTERNAL_API_TEMPERATURE = float(os.environ.get("EXTERNAL_API_TEMPERATURE", "0.0"))
# EXTERNAL_API_TOP_P = float(os.environ.get("EXTERNAL_API_TOP_P", "1.0"))


# External API mode (USE_VLLM=false):
# - For open-source release, do NOT hard-code any API keys.
# - Users must provide EXTERNAL_API_KEY via environment variable.
EXTERNAL_API_KEY = os.environ.get("EXTERNAL_API_KEY", "")
EXTERNAL_API_BASE_URL = os.environ.get("EXTERNAL_API_BASE_URL", "https://api.deepseek.com/beta")
EXTERNAL_API_MODEL = os.environ.get("EXTERNAL_API_MODEL", "deepseek-coder")
EXTERNAL_API_MAX_TOKENS = int(os.environ.get("EXTERNAL_API_MAX_TOKENS", "8192"))
EXTERNAL_API_TEMPERATURE = float(os.environ.get("EXTERNAL_API_TEMPERATURE", "0.0"))
EXTERNAL_API_TOP_P = float(os.environ.get("EXTERNAL_API_TOP_P", "1.0"))

# EXTERNAL_API_KEY = os.environ.get("EXTERNAL_API_KEY", "<YOUR_API_KEY>")
# EXTERNAL_API_BASE_URL = os.environ.get("EXTERNAL_API_BASE_URL", "https://api.kourichat.com/v1")
# EXTERNAL_API_MODEL = os.environ.get("EXTERNAL_API_MODEL", "claude-opus-4-5-20251101")
# EXTERNAL_API_MAX_TOKENS = int(os.environ.get("EXTERNAL_API_MAX_TOKENS", "16384"))
# EXTERNAL_API_TEMPERATURE = float(os.environ.get("EXTERNAL_API_TEMPERATURE", "0.0"))


# NOTE: DeepSeek API only supports max_tokens in [1, 8192], agicto may support higher.
# Use 8192 as default for broad compatibility. Override with EXTERNAL_API_MAX_TOKENS if needed.
# EXTERNAL_API_MAX_TOKENS = int(os.environ.get("EXTERNAL_API_MAX_TOKENS", "16384"))
EXTERNAL_API_MAX_RETRIES = int(os.environ.get("EXTERNAL_API_MAX_RETRIES", "3"))
EXTERNAL_API_RETRY_DELAY = float(os.environ.get("EXTERNAL_API_RETRY_DELAY", "2.0"))
EXTERNAL_API_RETRY_MAX_DELAY = float(os.environ.get("EXTERNAL_API_RETRY_MAX_DELAY", "30.0"))
EXTERNAL_API_TIMEOUT = float(os.environ.get("EXTERNAL_API_TIMEOUT", "600.0"))


def _is_timeout_error(e: Exception) -> bool:
    msg = str(e).lower()
    return ("timed out" in msg) or ("timeout" in msg)


def _coerce_message_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return str(value)


def _extract_response_content(response):
    if not response or not getattr(response, "choices", None):
        return "", False

    message = getattr(response.choices[0], "message", None)
    if not message:
        return "", False

    content = _coerce_message_text(getattr(message, "content", None)).strip()
    if content:
        return content, False

    reasoning_content = _coerce_message_text(getattr(message, "reasoning_content", None)).strip()
    if reasoning_content:
        print("外部 API 返回了 reasoning_content，使用该字段作为响应内容", file=sys.stderr)
        return reasoning_content, True

    return "", False

# 是否使用 vLLM（默认启用）
USE_VLLM = os.environ.get("USE_VLLM", "true").lower() in ("true", "1", "yes")

# 如果使用 vLLM，设置 multiprocessing start method
if USE_VLLM:
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # 如果已经设置过，忽略错误

# vLLM 采样参数
# 代码翻译任务推荐使用低温度以获得更确定性的输出
VLLM_SAMPLING_PARAMS = {
    "temperature": float(os.environ.get("VLLM_TEMPERATURE", "0.0")),  # 0 = 最确定性输出，适合代码翻译
    "top_p": float(os.environ.get("VLLM_TOP_P", "1.0")),              # 1.0 = 不使用 nucleus sampling（配合 temperature=0）
    "top_k": int(os.environ.get("VLLM_TOP_K", "-1")),                 # -1 = 禁用 top_k（配合 temperature=0）
    "repetition_penalty": float(os.environ.get("VLLM_REPETITION_PENALTY", "1.0")),  # 1.0 = 不惩罚重复（代码中重复是正常的）
    # NOTE: keep this reasonably high to avoid truncation in long function bodies.
    "max_tokens": int(os.environ.get("VLLM_MAX_TOKENS", "16384"))
}

# Worker 全局变量（用于多进程）
WORKER_CLIENT = None

def init_vllm_worker():
    """初始化 vLLM worker（用于多进程）"""
    global WORKER_CLIENT
    worker_id = multiprocessing.current_process()._identity[0] - 1 if hasattr(multiprocessing.current_process(), '_identity') else 0
    try:
        WORKER_CLIENT = OpenAI(
            base_url=VLLM_BASE_URL,
            api_key=VLLM_API_KEY,
            timeout=VLLM_REQUEST_TIMEOUT
        )
        # 测试连接
        WORKER_CLIENT.models.list()
        print(f"[Worker {worker_id}]: vLLM 客户端初始化成功，已连接到 {VLLM_BASE_URL}")
    except Exception as e:
        print(f"[Worker {worker_id}]: vLLM 客户端初始化失败: {e}", file=sys.stderr)
        print(f"请确保 vLLM 服务器正在 {VLLM_BASE_URL} 运行", file=sys.stderr)
        sys.exit(1)

class Claude():
    """
    兼容旧版本的 Claude 类
    现在支持 vLLM 和外部 API 两种模式
    """
    def __init__(
        self,
        api_keys : list = None,
    ):
        self.use_vllm = USE_VLLM

        if self.use_vllm:
            # 使用 vLLM
            self.model_name = VLLM_MODEL_NAME
            self.base_url = VLLM_BASE_URL
            self.api_key = VLLM_API_KEY
            # 创建单个客户端用于单线程调用
            try:
                self.client = OpenAI(
                    base_url=self.base_url,
                    api_key=self.api_key,
                    timeout=VLLM_REQUEST_TIMEOUT
                )
            except Exception as e:
                print(f"警告: vLLM 客户端初始化失败: {e}", file=sys.stderr)
                print(f"回退到外部 API 模式", file=sys.stderr)
                self.use_vllm = False

        if not self.use_vllm:
            # 外部 API 模式
            self.model_name = EXTERNAL_API_MODEL
            self.base_url = EXTERNAL_API_BASE_URL
            self.api_key = EXTERNAL_API_KEY
            self.temperature = EXTERNAL_API_TEMPERATURE
            self.top_p = globals().get('EXTERNAL_API_TOP_P', None)
            self.max_tokens = EXTERNAL_API_MAX_TOKENS
            self.max_retries = EXTERNAL_API_MAX_RETRIES
            self.retry_delay = EXTERNAL_API_RETRY_DELAY
            self.retry_max_delay = EXTERNAL_API_RETRY_MAX_DELAY
            self.timeout = EXTERNAL_API_TIMEOUT
            try:
                self.client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                    timeout=self.timeout
                )
                print(f"外部 API 客户端初始化成功: {self.base_url}, 模型: {self.model_name}")
            except Exception as e:
                print(f"外部 API 客户端初始化失败: {e}", file=sys.stderr)
                raise

    
    def generation_slice_in_parallel(self, generator):
        """遗留方法：并行生成（已更新为使用单客户端）"""
        max_workers = 8  # 外部 API 限制并发数
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {
                executor.submit(self.generation, [{"role": "user", "content": message}]): (path, message, query_func, cnt)
                for i, (path, message, query_func, cnt) in enumerate(generator)
            }

            for future in concurrent.futures.as_completed(future_to_path):
                path, message, query_func, cnt = future_to_path[future]
                response = future.result()
                translated_code = extract_code(response)
                with open(path, 'r', encoding='utf-8', errors='ignore') as file:
                    content = json.load(file)
                content.append(translated_code)
                with open(path, 'w', encoding='utf-8', errors='ignore') as file:
                    json.dump(content, file, indent=2)

                response_path = path.replace(".json", f"_{cnt}.txt")
                try:
                    with open(response_path, 'w', encoding='utf-8', errors='ignore') as output_file:
                        output_file.write(f"<message>\n{message}\n</message>\n")
                        output_file.write(f"<function>\n{query_func}\n</function>\n<response>\n{response}</response>")
                    print(f"Successfully processed: {response_path}")
                except Exception as e:
                    print(f"Error processing {response_path}: {e}")

    def generation_in_parallel(self, generator):
        """遗留方法：并行生成（已更新为使用单客户端）"""
        max_workers = 8  # 外部 API 限制并发数
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {
                executor.submit(self.generation, [{"role": "user", "content": message}]): (path, message, query_func, original_function_name)
                for i, (path, message, query_func, original_function_name) in enumerate(generator)
            }

            for future in concurrent.futures.as_completed(future_to_path):
                path, message, query_func, original_function_name = future_to_path[future]
                response = future.result()

                translated_code = extract_code(response)
                new_function_name = translated_code.split("fn ")[1].split("{")[0].strip()
                translated_code = translated_code.replace(new_function_name, original_function_name, 1)

                with open(path, 'w', encoding='utf-8', errors='ignore') as output_file:
                    output_file.write(f"<message>\n{message}\n</message>\n")
                    output_file.write(f"<response>\n{response}\n</response>\n")
                    output_file.write(f"<function>\n{query_func}\n</function>\n<translated function>\n{translated_code}</translated function>")
                print(f"Successfully processed: {path}")

    # def generation(self, content, client, temperature=0):
    #     # print(content)
    #     response = client.chat.completions.create(
    #         model=self.model_name, 
    #         messages=[
    #             {
    #                 "role": "user", 
    #                 "content": content
    #             }
    #         ],
    #         temperature=temperature
    #     )
    #     # print(response)
    #     if response.choices[0].message.content:
    #         # print(response.choices[0].message.content)
    #         return response.choices[0].message.content 
    #     else:
    #         raise ValueError("Empty response from API")
    
    # @retry(stop=stop_after_attempt(5), wait=wait_fixed(10))
    def generation(self, messages, temperature=0, return_usage=False):
        """
        生成翻译结果

        Args:
            messages: 消息列表（OpenAI 格式）
            temperature: 温度参数（vLLM 模式下使用配置中的值）
            return_usage: 是否返回 token 使用信息

        Returns:
            如果 return_usage=True，返回 {"content": ..., "usage": {...}}
            否则返回字符串内容
        """
        if self.use_vllm:
            # 使用 vLLM - 通过 Semaphore 控制并发
            _vllm_semaphore.acquire()
            try:
                # 准备 API 参数
                api_params = {
                    "temperature": VLLM_SAMPLING_PARAMS.get("temperature"),
                    "top_p": VLLM_SAMPLING_PARAMS.get("top_p"),
                    "max_tokens": VLLM_SAMPLING_PARAMS.get("max_tokens")
                }
                extra_body_params = {
                    "top_k": VLLM_SAMPLING_PARAMS.get("top_k"),
                    "repetition_penalty": VLLM_SAMPLING_PARAMS.get("repetition_penalty")
                }
                # Disable reasoning/thinking mode for DeepSeek v4 models to avoid
                # slow responses and truncated output.  Other models ignore this.
                if "deepseek-v4" in str(self.model_name).lower():
                    extra_body_params["thinking"] = {"type": "disabled"}

                last_err = None
                for attempt in range(1, max(1, VLLM_MAX_RETRIES) + 1):
                    try:
                        response = self.client.chat.completions.create(
                            model=self.model_name,
                            messages=messages,
                            stop=["<|im_end|>"],
                            **api_params,
                            extra_body=extra_body_params
                        )

                        content, _used_reasoning = _extract_response_content(response)
                        if response.choices and content:
                            if return_usage:
                                usage = response.usage
                                return {
                                    "content": content,
                                    "usage": {
                                        "prompt_tokens": usage.prompt_tokens if usage else 0,
                                        "completion_tokens": usage.completion_tokens if usage else 0,
                                        "total_tokens": usage.total_tokens if usage else 0
                                    }
                                }
                            return content

                        raise ValueError("Empty response from vLLM")
                    except Exception as e:
                        last_err = e
                        if attempt < max(1, VLLM_MAX_RETRIES) and _is_timeout_error(e):
                            backoff = min(VLLM_RETRY_BACKOFF_SEC * (2 ** (attempt - 1)), VLLM_RETRY_BACKOFF_MAX_SEC)
                            print(
                                f"vLLM 调用超时，{backoff:.1f}s 后重试 (attempt {attempt}/{VLLM_MAX_RETRIES}): {e}",
                                file=sys.stderr,
                            )
                            time.sleep(backoff)
                            continue
                        print(f"vLLM 调用失败: {e}", file=sys.stderr)
                        raise

                # 理论上不会到这里
                raise last_err
            finally:
                _vllm_semaphore.release()
        else:
            # 外部 API 模式（带重试机制）
            last_err = None
            for attempt in range(1, self.max_retries + 1):
                try:
                    kwargs = {
                        "model": self.model_name,
                        "messages": messages,
                        "max_tokens": self.max_tokens,
                        "stream": False
                    }
                    # 只有当 temperature 不为 None 时才加入参数
                    # (注意：如果 EXTERNAL_API_TEMPERATURE 定义为 0.0，if 0.0 可能会被判为 False，
                    # 建议显式判断 is not None)
                    if getattr(self, 'temperature', None) is not None:
                        kwargs["temperature"] = self.temperature
                        
                    # 只有当 top_p 有值时才加入参数
                    if self.top_p is not None:
                        kwargs["top_p"] = self.top_p

                    # Disable reasoning/thinking for DeepSeek v4 models
                    if "deepseek-v4" in str(self.model_name).lower():
                        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
                    # 使用 **kwargs 解包参数发送请求
                    _vllm_semaphore.acquire()
                    try:
                        response = self.client.chat.completions.create(**kwargs)
                    finally:
                        _vllm_semaphore.release()
                    
                    response_error = getattr(response, "error", None) if response else None
                    if response_error:
                        error_code = getattr(response_error, "code", None) or "unknown_error"
                        error_message = getattr(response_error, "message", None) or str(response_error)
                        raise RuntimeError(f"External API error {error_code}: {error_message}")

                    if response and response.choices and len(response.choices) > 0:
                        content, _used_reasoning = _extract_response_content(response)
                        if content:
                            if return_usage:
                                usage = response.usage
                                return {
                                    "content": content,
                                    "usage": {
                                        "prompt_tokens": getattr(usage, 'prompt_tokens', 0) if usage else 0,
                                        "completion_tokens": getattr(usage, 'completion_tokens', 0) if usage else 0,
                                        "total_tokens": getattr(usage, 'total_tokens', 0) if usage else 0
                                    }
                                }
                            return content

                    # 空响应，重试
                    if attempt < self.max_retries:
                        backoff = min(self.retry_delay * (2 ** (attempt - 1)), self.retry_max_delay)
                        print(f"外部 API 返回空响应，{backoff:.1f}s 后重试 (attempt {attempt}/{self.max_retries})", file=sys.stderr)
                        time.sleep(backoff)
                        continue
                    raise ValueError("Empty response from external API")

                except Exception as e:
                    last_err = e
                    error_str = str(e).lower()
                    # 限流错误或超时：重试
                    if any(keyword in error_str for keyword in ['rate', 'limit', '429', 'too many', 'timeout', 'timed out']):
                        if attempt < self.max_retries:
                            backoff = min(self.retry_delay * (2 ** (attempt - 1)), self.retry_max_delay)
                            print(f"外部 API 调用失败 ({e})，{backoff:.1f}s 后重试 (attempt {attempt}/{self.max_retries})", file=sys.stderr)
                            time.sleep(backoff)
                            continue
                    print(f"外部 API 调用失败: {e}", file=sys.stderr)
                    raise

            # 所有重试都失败
            if last_err:
                raise last_err
            raise ValueError("External API call failed after all retries")


# 创建全局实例
claude = Claude()

def generation(message, return_usage=False):
    """
    生成翻译结果（单线程版本）
    
    Args:
        message: 消息列表（OpenAI 格式）或单个消息字符串
        return_usage: 是否返回 token 使用信息
    
    Returns:
        如果 return_usage=True，返回 {"content": ..., "usage": {...}}
        否则返回字符串内容
    """
    # 如果 message 是字符串，转换为消息格式（向后兼容）
    if isinstance(message, str):
        messages = [{"role": "user", "content": message}]
    elif isinstance(message, list):
        messages = message
    else:
        raise ValueError("message 必须是字符串或消息列表")
    
    return claude.generation(messages, return_usage=return_usage)

def generation_in_parallel_vllm(tasks, max_workers=None):
    """
    使用 vLLM 多进程并行生成（新函数）
    
    Args:
        tasks: 任务列表，每个任务是一个字典，包含 "messages" 键
        max_workers: 最大工作进程数，默认使用 VLLM_NUM_WORKERS
    
    Returns:
        结果列表，与 tasks 顺序对应
    """
    if not USE_VLLM:
        raise ValueError("generation_in_parallel_vllm 需要启用 vLLM (USE_VLLM=true)")
    
    if max_workers is None:
        max_workers = VLLM_NUM_WORKERS
    
    def process_single_task(task_data):
        """处理单个任务（用于多进程）"""
        global WORKER_CLIENT
        if WORKER_CLIENT is None:
            init_vllm_worker()
        
        try:
            messages = task_data.get("messages", [])
            if not messages:
                return {"error": "No messages in task", "task": task_data}
            
            # 准备 API 参数
            api_params = {
                "temperature": VLLM_SAMPLING_PARAMS.get("temperature"),
                "top_p": VLLM_SAMPLING_PARAMS.get("top_p"),
                "max_tokens": VLLM_SAMPLING_PARAMS.get("max_tokens")
            }
            extra_body_params = {
                "top_k": VLLM_SAMPLING_PARAMS.get("top_k"),
                "repetition_penalty": VLLM_SAMPLING_PARAMS.get("repetition_penalty")
            }
            
            response = WORKER_CLIENT.chat.completions.create(
                model=VLLM_MODEL_NAME,
                messages=messages,
                stop=["<|im_end|>"],
                **api_params,
                extra_body=extra_body_params
            )
            
            content, _used_reasoning = _extract_response_content(response)
            if content:
                result = {
                    "content": content,
                    "task": task_data
                }
                if response.usage:
                    result["usage"] = {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    }
                return result
            else:
                return {"error": "Empty response", "task": task_data}
        except Exception as e:
            return {"error": str(e), "task": task_data}
    
    # 使用多进程池
    with multiprocessing.Pool(processes=max_workers, initializer=init_vllm_worker) as pool:
        results = pool.map(process_single_task, tasks)
    
    return results

def generation_in_parallel(message):
    """
    并行生成（向后兼容，保留旧接口）
    如果启用 vLLM，使用多进程；否则使用传统方式
    """
    if USE_VLLM:
        # 注意：这个函数原本接受 generator，但为了简化，暂时不支持
        # 如果需要，可以后续扩展
        raise NotImplementedError("generation_in_parallel 在 vLLM 模式下需要使用 generation_in_parallel_vllm")
    else:
        return claude.generation_in_parallel(message)
