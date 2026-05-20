"""
工作空间路径配置模块
统一管理所有中间过程的输入输出路径
支持通过环境变量 C2R_WORKSPACE_ROOT 自定义工作空间根目录
"""
import os
import re
from pathlib import Path
from typing import Optional

# 工作空间根目录（所有中间过程的输入输出）
# 支持通过环境变量覆盖，方便在不同目录下运行翻译任务
_workspace_root_env = os.environ.get("C2R_WORKSPACE_ROOT")
if _workspace_root_env:
    WORKSPACE_ROOT = Path(_workspace_root_env).expanduser().resolve()
else:
    WORKSPACE_ROOT = Path("workspace").resolve()

# 项目源代码目录（待翻译的项目）
PROJECTS_DIR = WORKSPACE_ROOT / "projects"

# 提取的代码目录
EXTRACTED_DIR = WORKSPACE_ROOT / "extracted"

# 翻译骨架目录
SKELETONS_DIR = WORKSPACE_ROOT / "skeletons"

# 翻译结果目录
TRANSLATED_DIR = WORKSPACE_ROOT / "translated"

# 测试结果目录
TEST_RESULTS_DIR = WORKSPACE_ROOT / "test_results"

# 修复结果目录
REPAIR_RESULTS_DIR = WORKSPACE_ROOT / "repair_results"

# 修复历史目录（保存每次修复尝试的错误信息和翻译代码）
REPAIR_HISTORY_DIR = WORKSPACE_ROOT / "repair_history"

# RAG 相关目录
RAG_DIR = WORKSPACE_ROOT / "rag"
RAG_KNOWLEDGE_BASE = RAG_DIR / "knowledge_base.json"
RAG_BM25_INDEX = RAG_DIR / "bm25_index.pkl"
RAG_ELASTIC_SEARCH_RESULTS = RAG_DIR / "elastic_search_results"
RAG_RERANKED_RESULTS = RAG_DIR / "reranked_results"

# 函数签名匹配目录
SIGNATURE_MATCH_DIR = WORKSPACE_ROOT / "signature_matches"

# 源代码骨架目录（用于修复）
SOURCE_SKELETON_DIR = WORKSPACE_ROOT / "source_skeletons"

# 最终合并项目目录（包含所有编译通过函数的完整项目）
FINAL_PROJECTS_DIR = WORKSPACE_ROOT / "final_projects"

# LLM 提示词保存目录
LLM_PROMPTS_DIR = WORKSPACE_ROOT / "llm_prompts"


def get_workspace_root() -> Path:
    """Return the workspace root directory."""
    return WORKSPACE_ROOT


def get_project_path(project_name: str) -> Path:
    """获取项目源代码路径"""
    return PROJECTS_DIR / project_name


def get_extracted_path(project_name: str) -> Path:
    """获取提取的代码路径"""
    return EXTRACTED_DIR / project_name


def get_functions_path(project_name: str) -> Path:
    """获取提取的函数路径"""
    return get_extracted_path(project_name) / "functions"


def get_dependencies_path(project_name: str) -> Path:
    """获取依赖关系路径"""
    return get_extracted_path(project_name) / "dependencies"


def get_dependencies_not_in_file_path(project_name: str) -> Path:
    """获取不在当前文件的依赖路径"""
    return get_extracted_path(project_name) / "dependencies_not_in_current_file"


def get_dependencies_not_in_file_rs_path(project_name: str) -> Path:
    """获取 Rust 格式的依赖路径"""
    return get_extracted_path(project_name) / "dependencies_not_in_current_file_rs"


def get_skeleton_path(project_name: str) -> Path:
    """获取翻译骨架路径"""
    return SKELETONS_DIR / project_name


def get_source_skeleton_path(project_name: str) -> Path:
    """获取源代码骨架路径（用于修复）"""
    return SOURCE_SKELETON_DIR / project_name


def get_signature_match_path(project_name: str, llm_name: str = None) -> Path:
    """获取签名匹配路径"""
    if llm_name:
        return SIGNATURE_MATCH_DIR / project_name / f"translate_by_{llm_name}"
    return SIGNATURE_MATCH_DIR / project_name


def get_translated_path(project_name: str, llm_name: str) -> Path:
    """获取翻译结果路径"""
    return TRANSLATED_DIR / project_name / f"translate_by_{llm_name}"


def get_test_results_path(project_name: str, llm_name: str) -> Path:
    """获取测试结果路径"""
    return TEST_RESULTS_DIR / project_name / f"translate_by_{llm_name}"


def get_repair_results_path(project_name: str, llm_name: str, round_num: int = None) -> Path:
    """获取修复结果路径"""
    if round_num:
        return REPAIR_RESULTS_DIR / project_name / f"translate_by_{llm_name}" / f"round_{round_num}"
    return REPAIR_RESULTS_DIR / project_name / f"translate_by_{llm_name}"


def get_elastic_search_path(project_name: str) -> Path:
    """获取 BM25 检索结果路径"""
    return RAG_ELASTIC_SEARCH_RESULTS / project_name


def get_reranked_path(project_name: str) -> Path:
    """获取重排结果路径"""
    return RAG_RERANKED_RESULTS / project_name


def get_final_project_path(project_name: str, llm_name: str) -> Path:
    """获取最终合并项目路径（包含所有编译通过函数的完整项目）"""
    return FINAL_PROJECTS_DIR / project_name / f"translate_by_{llm_name}"


def get_llm_prompts_path(project_name: str, llm_name: str = None) -> Path:
    """获取 LLM 提示词保存路径"""
    if llm_name:
        return LLM_PROMPTS_DIR / project_name / f"translate_by_{llm_name}"
    return LLM_PROMPTS_DIR / project_name


def get_repair_history_path(project_name: str, llm_name: str) -> Path:
    """获取修复历史保存路径"""
    return REPAIR_HISTORY_DIR / project_name / f"translate_by_{llm_name}"


def ensure_dirs(project_name: str, llm_name: str = None):
    """确保所有必要的目录存在"""
    dirs = [
        WORKSPACE_ROOT,
        PROJECTS_DIR,
        EXTRACTED_DIR,
        SKELETONS_DIR,
        TRANSLATED_DIR,
        TEST_RESULTS_DIR,
        REPAIR_RESULTS_DIR,
        FINAL_PROJECTS_DIR,
        RAG_DIR,
        RAG_ELASTIC_SEARCH_RESULTS,
        RAG_RERANKED_RESULTS,
        SIGNATURE_MATCH_DIR,
        SOURCE_SKELETON_DIR,
        LLM_PROMPTS_DIR,
        REPAIR_HISTORY_DIR,
        get_extracted_path(project_name),
        get_functions_path(project_name),
        get_dependencies_path(project_name),
        get_dependencies_not_in_file_path(project_name),
        get_dependencies_not_in_file_rs_path(project_name),
        get_skeleton_path(project_name),
        get_source_skeleton_path(project_name),
        get_signature_match_path(project_name),
    ]
    
    if llm_name:
        dirs.extend([
            get_translated_path(project_name, llm_name),
            get_signature_match_path(project_name, llm_name),
            get_test_results_path(project_name, llm_name),
            get_repair_results_path(project_name, llm_name),
            get_final_project_path(project_name, llm_name),
            get_repair_history_path(project_name, llm_name),
        ])
    
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)


def get_compile_commands_path(ohos_root: Path = None) -> Optional[Path]:
    """
    查找 compile_commands.json 文件
    
    优先级：
    1. 环境变量 OHOS_COMPILE_COMMANDS（最高优先级；支持 out_dir profile 自动选择的解压产物）
    2. 硬编码的默认路径（仅作为最后兜底，避免固定到 rk3568）
    3. 如果提供了 ohos_root，检查常见默认位置
    4. 递归搜索
    
    Args:
        ohos_root: OpenHarmony 源码根目录
    
    Returns:
        compile_commands.json 的路径，如果不存在返回 None
    """
    # 默认路径（仅兜底；开源版优先使用 his2trans/data/ohos/ohos_root_min）
    DEFAULT_OHOS_ROOT = (Path(__file__).resolve().parent.parent.parent / "data" / "ohos" / "ohos_root_min")
    DEFAULT_COMPILE_COMMANDS = DEFAULT_OHOS_ROOT / "out" / "rk3568" / "compile_commands.json"
    
    # 1. 检查环境变量（允许 pipeline/profile 选择器显式指定）
    env_path = os.environ.get("OHOS_COMPILE_COMMANDS")
    if env_path:
        path = Path(env_path)
        if path.exists():
            return path.resolve()

    # 2. 检查硬编码默认路径（兜底）
    if DEFAULT_COMPILE_COMMANDS.exists():
        return DEFAULT_COMPILE_COMMANDS.resolve()
    
    # 3. 如果提供了 ohos_root，检查默认位置
    if ohos_root:
        ohos_root = Path(ohos_root)
        default_paths = [
            ohos_root / "out" / "rk3568" / "compile_commands.json",
            ohos_root / "out" / "sdk" / "compile_commands.json",
            ohos_root / "compile_commands.json",
        ]
        
        for path in default_paths:
            if path.exists():
                return path.resolve()
    
    # 4. 如果 ohos_root 未提供，尝试使用硬编码的默认根目录
    if DEFAULT_OHOS_ROOT.exists():
        default_paths = [
            DEFAULT_OHOS_ROOT / "out" / "rk3568" / "compile_commands.json",
            DEFAULT_OHOS_ROOT / "out" / "sdk" / "compile_commands.json",
            DEFAULT_OHOS_ROOT / "compile_commands.json",
        ]
        
        for path in default_paths:
            if path.exists():
                return path.resolve()
    
    # 5. 递归搜索（从当前目录开始）
    # 注意：递归搜索可能很慢，只在其他方法都失败时使用
    current_dir = Path.cwd()
    search_count = 0
    for path in current_dir.rglob("compile_commands.json"):
        search_count += 1
        if search_count == 1:  # 只返回第一个找到的
            return path.resolve()
        # 如果找到多个，可以选择更合适的

    return None


def safe_module_name(project_root: Path, src_file: Path) -> str:
    """
    根据文件相对路径生成安全的模块名/文件前缀

    这是唯一的命名实现，所有需要生成安全文件名/模块名的地方都应该调用此函数，
    确保 extracted 函数文件前缀、source_skeletons、skeletons 的模块名完全一致。

    命名规则：
    1. 计算相对于项目根目录的路径
    2. 去掉扩展名 (.c, .cpp, .h 等)
    3. 将路径分隔符转换为下划线
    4. 清理非法字符（只保留 a-zA-Z0-9_）
    5. 处理以数字开头的情况（加前缀 _）

    示例：
    - /abs/path/src/linux/ipc.c -> src_linux_ipc
    - /abs/path/src/liteos_m/ipc.c -> src_liteos_m_ipc
    - /abs/path/3rdparty/utils.c -> _3rdparty_utils
    - /abs/path/src/my-lib.c -> src_my_lib

    Args:
        project_root: 项目根目录
        src_file: 源文件路径

    Returns:
        安全的模块名/文件前缀（符合 Rust 标识符规则）
    """
    # 重要：统一路径语义（resolve），避免 project_root 是软链接而 src_file 是真实路径时
    # relative_to() 失败，导致模块名退化为仅文件名（进而造成 extracted/skeletons/source_skeletons 命名不一致）。
    project_root = Path(project_root).expanduser().resolve()
    src_file = Path(src_file).expanduser().resolve()

    try:
        # 计算相对于项目根目录的路径
        rel_path = src_file.relative_to(project_root)
    except ValueError:
        # 如果文件不在项目根目录下（罕见），回退到文件名
        module_name = src_file.stem.replace('.', '_').replace('-', '_')
        # 清理非法字符
        module_name = re.sub(r'[^a-zA-Z0-9_]', '_', module_name)
        # 处理以数字开头的情况
        if module_name and module_name[0].isdigit():
            module_name = f"_{module_name}"
        return module_name

    # 去掉扩展名 (.c, .cpp, .h 等)
    rel_path_no_ext = rel_path.with_suffix('')

    # 将路径分隔符转换为下划线
    module_name = str(rel_path_no_ext).replace('/', '_').replace('\\', '_')

    # 清理非法字符，确保是合法的 Rust 标识符
    # 只保留 a-zA-Z0-9_
    module_name = re.sub(r'[^a-zA-Z0-9_]', '_', module_name)

    # 处理以数字开头的情况
    if module_name and module_name[0].isdigit():
        module_name = f"_{module_name}"

    return module_name


def safe_module_name_legacy(project_root: Path, src_file: Path) -> str:
    """
    旧版命名规则（仅用于向后兼容查找旧输出文件）

    旧规则只做了路径分隔符替换，没有：
    - 清理非法字符
    - 处理数字开头

    Args:
        project_root: 项目根目录
        src_file: 源文件路径

    Returns:
        旧版命名规则生成的文件前缀
    """
    project_root = Path(project_root)
    src_file = Path(src_file)

    try:
        rel_path = src_file.relative_to(project_root)
        # 旧规则：只做路径分隔符替换，去掉扩展名
        return str(rel_path.with_suffix('')).replace('/', '_').replace('\\', '_')
    except ValueError:
        return src_file.stem
