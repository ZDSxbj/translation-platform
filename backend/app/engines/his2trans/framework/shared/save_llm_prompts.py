"""
LLM 提示词保存工具

用于保存每次调用 LLM 时的真实提示词，方便后续分析和优化。
"""
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any
import os


def get_llm_prompts_dir(project_name: str, llm_name: str = None) -> Path:
    """
    获取 LLM 提示词保存目录
    
    Args:
        project_name: 项目名称
        llm_name: LLM 名称（可选）
        
    Returns:
        提示词保存目录路径
    """
    from workspace_config import get_llm_prompts_path
    path = get_llm_prompts_path(project_name, llm_name)
    # 确保目录存在
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_llm_prompt(
    messages: List[Dict[str, str]],
    project_name: str,
    llm_name: str,
    task_type: str,
    function_name: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    output_dir: Optional[Path] = None
) -> Path:
    """
    保存 LLM 提示词到文件
    
    Args:
        messages: LLM 消息列表（OpenAI 格式），包含 system 和 user 消息
        project_name: 项目名称
        llm_name: LLM 名称
        task_type: 任务类型，如 "translate", "repair", "skeleton_translate" 等
        function_name: 函数名称（可选）
        metadata: 额外的元数据（可选），如上下文信息、错误信息等
        output_dir: 自定义输出目录（可选）
        
    Returns:
        保存的文件路径
    """
    # 确定输出目录
    if output_dir is None:
        output_dir = get_llm_prompts_dir(project_name, llm_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # 毫秒级时间戳
    if function_name:
        # 清理函数名，移除特殊字符
        safe_func_name = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in function_name)
        filename = f"{task_type}_{safe_func_name}_{timestamp}.json"
    else:
        filename = f"{task_type}_{timestamp}.json"
    
    file_path = output_dir / filename
    
    # 提取 system 和 user prompt
    system_prompt = ""
    user_prompt = ""
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system":
            system_prompt = content
        elif role == "user":
            user_prompt = content
    
    # 构建保存的数据
    prompt_data = {
        "timestamp": datetime.now().isoformat(),
        "project_name": project_name,
        "llm_name": llm_name,
        "task_type": task_type,
        "function_name": function_name,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "messages": messages,  # 保存完整的消息列表
        "metadata": metadata or {}
    }
    
    # 保存为 JSON 文件
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(prompt_data, f, ensure_ascii=False, indent=2)
        return file_path
    except Exception as e:
        # 如果保存失败，尝试使用简单的文件名
        simple_filename = f"{task_type}_{timestamp}.json"
        simple_file_path = output_dir / simple_filename
        try:
            with open(simple_file_path, 'w', encoding='utf-8') as f:
                json.dump(prompt_data, f, ensure_ascii=False, indent=2)
            return simple_file_path
        except Exception as e2:
            # 如果还是失败，打印错误但不中断程序
            import logging
            logging.warning(f"保存提示词失败: {e2}")
            return None


def save_llm_prompt_text(
    messages: List[Dict[str, str]],
    project_name: str,
    llm_name: str,
    task_type: str,
    function_name: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    output_dir: Optional[Path] = None
) -> Path:
    """
    保存 LLM 提示词为可读的文本格式（除了 JSON，也保存为文本）
    
    Args:
        同 save_llm_prompt
        
    Returns:
        保存的文件路径
    """
    # 确定输出目录
    if output_dir is None:
        output_dir = get_llm_prompts_dir(project_name, llm_name)
    else:
        # 如果提供了自定义目录，也确保它存在
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    if function_name:
        safe_func_name = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in function_name)
        filename = f"{task_type}_{safe_func_name}_{timestamp}.txt"
    else:
        filename = f"{task_type}_{timestamp}.txt"
    
    file_path = output_dir / filename
    
    # 构建文本内容
    lines = [
        f"# LLM Prompt - {task_type}",
        f"Timestamp: {datetime.now().isoformat()}",
        f"Project: {project_name}",
        f"LLM: {llm_name}",
        f"Function: {function_name or 'N/A'}",
        "",
        "=" * 80,
        "SYSTEM PROMPT",
        "=" * 80,
        ""
    ]
    
    # 提取 system 和 user prompt
    system_prompt = ""
    user_prompt = ""
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system":
            system_prompt = content
        elif role == "user":
            user_prompt = content
    
    lines.append(system_prompt)
    lines.append("")
    lines.append("=" * 80)
    lines.append("USER PROMPT")
    lines.append("=" * 80)
    lines.append("")
    lines.append(user_prompt)
    
    # 添加元数据
    if metadata:
        lines.append("")
        lines.append("=" * 80)
        lines.append("METADATA")
        lines.append("=" * 80)
        lines.append("")
        lines.append(json.dumps(metadata, ensure_ascii=False, indent=2))
    
    # 保存为文本文件
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))
        return file_path
    except Exception as e:
        import logging
        logging.warning(f"保存提示词文本失败: {e}")
        return None

