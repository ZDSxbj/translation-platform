"""
项目配置模块
用于统一管理项目名称和路径配置，避免在多个文件中硬编码
"""
import os
import sys
import glob
from pathlib import Path

# 默认项目名称
DEFAULT_PROJECT_NAME = "dlp_fuse"

# C/C++ 文件扩展名
C_EXTENSIONS = ['.c', '.cpp', '.cc', '.cxx', '.C']
H_EXTENSIONS = ['.h', '.hpp', '.hh', '.hxx', '.H']

def _env_flag(name: str, default: str = "0") -> bool:
    v = os.environ.get(name, default)
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")

def _parse_csv_env(name: str, default_csv: str) -> set:
    raw = os.environ.get(name, default_csv)
    parts = []
    for p in str(raw).split(","):
        p = p.strip()
        if p:
            parts.append(p.lower())
    return set(parts)

def _is_in_excluded_dir(path: Path, project_root: Path) -> bool:
    """
    Exclude sources/headers under test directories by default.

    This is a *file selection* policy for translation/truth extraction, not an include-path rule.
    """
    if not _env_flag("C2R_EXCLUDE_TEST_DIRS", "1"):
        return False
    excluded = _parse_csv_env("C2R_EXCLUDED_DIR_NAMES", "test,tests,unittest,unittests")
    try:
        rel = path.resolve().relative_to(project_root.resolve())
    except Exception:
        rel = path
    return any(str(part).lower() in excluded for part in rel.parts)

def get_project_name():
    """
    获取项目名称，优先级：
    1. 命令行参数 --project/-p
    2. 环境变量 PROJECT_NAME
    3. 默认值 "dlp_fuse"
    """
    # 检查命令行参数
    for i, arg in enumerate(sys.argv):
        if arg in ("--project", "-p") and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if arg.startswith("--project="):
            return arg.split("=", 1)[1]
    
    # 检查环境变量
    env_project = os.environ.get("PROJECT_NAME", "").strip()
    if env_project:
        return env_project
    
    # 返回默认值
    return DEFAULT_PROJECT_NAME

def get_project_root():
    """
    获取项目根目录路径
    优先级：
    1. 环境变量 PROJECT_ROOT
    2. 项目名称（相对路径）
    """
    env_root = os.environ.get("PROJECT_ROOT", "").strip()
    if env_root and os.path.exists(env_root):
        return env_root
    
    project_name = get_project_name()
    if os.path.exists(project_name):
        return project_name
    
    return project_name

def find_cpp_files(project_root, extensions=None):
    """
    递归查找项目中的所有C/C++源文件
    
    Args:
        project_root: 项目根目录
        extensions: 文件扩展名列表，默认为 C_EXTENSIONS
    
    Returns:
        list: 所有找到的源文件路径列表
    """
    if extensions is None:
        extensions = C_EXTENSIONS
    
    cpp_files = []
    project_path = Path(project_root)
    
    if not project_path.exists():
        print(f"警告: 项目目录不存在: {project_root}")
        return cpp_files
    
    for ext in extensions:
        pattern = f"**/*{ext}"
        cpp_files.extend(project_path.glob(pattern))
    
    # 转换为字符串并排序
    cpp_files = [f for f in cpp_files if not _is_in_excluded_dir(f, project_path)]
    return sorted([str(f) for f in cpp_files])

def find_header_files(project_root, extensions=None):
    """
    递归查找项目中的所有头文件
    
    Args:
        project_root: 项目根目录
        extensions: 文件扩展名列表，默认为 H_EXTENSIONS
    
    Returns:
        list: 所有找到的头文件路径列表
    """
    if extensions is None:
        extensions = H_EXTENSIONS
    
    header_files = []
    project_path = Path(project_root)
    
    if not project_path.exists():
        return header_files
    
    for ext in extensions:
        pattern = f"**/*{ext}"
        header_files.extend(project_path.glob(pattern))
    
    # 转换为字符串并排序
    header_files = [f for f in header_files if not _is_in_excluded_dir(f, project_path)]
    return sorted([str(f) for f in header_files])

def find_matching_header(source_file, header_files):
    """
    为源文件查找匹配的头文件
    
    策略（按优先级）：
    1. 首先尝试同名文件（在 include 目录或同目录）
    2. 然后尝试通过 #include 语句查找同名文件（优先）
    3. 最后尝试通过 #include 语句查找其他文件
    4. 返回 None（表示没有找到）
    
    Args:
        source_file: 源文件路径
        header_files: 所有头文件路径列表
    
    Returns:
        str or None: 匹配的头文件路径，如果没找到返回 None
    """
    source_path = Path(source_file)
    source_dir = source_path.parent
    source_stem = source_path.stem  # 不含扩展名的文件名
    
    # 策略1: 查找同名头文件（优先检查 include 目录，然后同目录）
    # 1.1: 检查 include 目录（src -> include）
    if 'src' in str(source_dir):
        include_dir = source_dir.parent / 'include'
        same_name_header = include_dir / f'{source_stem}.h'
        if same_name_header.exists():
            return str(same_name_header)
    
    # 1.2: 检查同目录下同名文件
    same_name_header = source_dir / f'{source_stem}.h'
    if same_name_header.exists():
        return str(same_name_header)
    
    # 1.3: 在所有头文件中查找同名文件
    for header_file in header_files:
        header_path = Path(header_file)
        if header_path.stem == source_stem:
            return header_file
    
    # 策略2: 读取源文件，查找 #include 语句
    try:
        with open(source_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # 查找 #include "xxx.h" 或 #include <xxx.h>
        import re
        include_pattern = r'#include\s+["<]([^">]+)[">]'
        includes = re.findall(include_pattern, content)
        
        # 优先查找同名文件的 include
        for include_name in includes:
            include_basename = Path(include_name).name
            include_stem = Path(include_basename).stem
            
            # 如果 include 的文件名与源文件名相同，优先返回
            if include_stem == source_stem:
                for header_file in header_files:
                    header_path = Path(header_file)
                    if header_path.name == include_basename or header_path.stem == include_stem:
                        return header_file
        
        # 然后查找其他 include 文件
        for include_name in includes:
            include_basename = Path(include_name).name
            include_stem = Path(include_basename).stem
            
            # 跳过同名文件（已经在上面处理了）
            if include_stem == source_stem:
                continue
            
            # 在头文件列表中查找
            for header_file in header_files:
                header_path = Path(header_file)
                if header_path.name == include_basename or header_path.stem == include_stem:
                    return header_file
    except Exception as e:
        print(f"警告: 读取源文件 {source_file} 时出错: {e}")
    
    # 策略3: 没找到匹配的头文件
    return None

def get_all_source_files_with_headers(project_root=None):
    """
    获取所有源文件及其对应的头文件
    
    Returns:
        list: [(source_file, header_file or None), ...]
    """
    if project_root is None:
        project_root = get_project_root()
    
    source_files = find_cpp_files(project_root)
    header_files = find_header_files(project_root)
    
    result = []
    for source_file in source_files:
        header_file = find_matching_header(source_file, header_files)
        result.append((source_file, header_file))
    
    return result

# 导出项目名称和根目录
PROJECT_NAME = get_project_name()
PROJECT_ROOT = get_project_root()
