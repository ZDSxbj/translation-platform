#!/usr/bin/env python3
"""
增量式翻译与验证脚本

核心流程：
1. 拓扑排序：基于依赖关系决定翻译顺序（自底向上）
2. 构建上下文：提取骨架中的定义作为 LLM 上下文
3. 受限翻译：要求 LLM 仅生成函数体，且必须符合骨架签名
4. 物理注入：将生成的代码替换掉占位符
5. 即时编译验证：填一个，测一个
6. 错误驱动修复：利用编译器报错进行局部微调
"""
import os
import sys
import re
import json
import shlex
import hashlib
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional, Any, Iterable
from collections import defaultdict
from dataclasses import dataclass
import logging
from threading import Semaphore

# 导入日志配置
from log_config import ensure_logging_setup, LogPrinter
ensure_logging_setup()
logger = logging.getLogger(__name__)
log = LogPrinter(__name__)

# vLLM 并发控制信号量 - 限制同时最多 N 个请求（全局共享，跨进程）
# 从环境变量获取配置，默认 4（降低并发避免超时）
# 原来默认 120 导致大量请求超时
VLLM_CONCURRENT_LIMIT = int(os.environ.get("VLLM_CONCURRENT_LIMIT", "120"))
_vllm_semaphore = Semaphore(VLLM_CONCURRENT_LIMIT)


def _atomic_write_text(path: Path, text: str) -> None:
    """
    Write text via temp file + atomic rename to avoid 0-byte/partial artifacts
    when the process is interrupted (OOM kill, crash, SIGKILL, etc.).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.tmp.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp_name, path)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass

from tree_sitter import Language, Parser, Query, QueryCursor
import tree_sitter_cpp as tscpp
try:
    import tree_sitter_rust as tsrust  # type: ignore
    _HAS_TREE_SITTER_RUST = True
except Exception:
    tsrust = None  # type: ignore
    _HAS_TREE_SITTER_RUST = False
from workspace_config import get_reranked_path, get_project_path
from scripts.semantic_slice import extract_semantic_slice
from scripts.dependency_prober import collect_global_declarations
from scripts.usage_index import ensure_usage_examples

# types.rs context slicing (reduce prompt size deterministically)
try:
    from context_slicer import (
        TypesRsRegistry,
        extract_identifiers as _extract_identifiers_for_slice,
        extract_missing_symbols_from_rustc_output as _extract_missing_symbols_for_slice,
        extract_types_from_rust_signature as _extract_types_from_signature_for_slice,
    )

    TYPES_SLICE_AVAILABLE = True
except Exception:
    TypesRsRegistry = None  # type: ignore
    _extract_identifiers_for_slice = None  # type: ignore
    _extract_missing_symbols_for_slice = None  # type: ignore
    _extract_types_from_signature_for_slice = None  # type: ignore
    TYPES_SLICE_AVAILABLE = False

# C accessor shims (Scheme-B): deterministic field-access fallback
try:
    from c_accessor_shims import CAccessorShimManager
    C_ACCESSOR_SHIMS_AVAILABLE = True
except Exception:
    CAccessorShimManager = None
    C_ACCESSOR_SHIMS_AVAILABLE = False

# 导入规则修复模块
try:
    from rule_fix import RuleFixManager, try_rule_fix
    RULE_FIX_AVAILABLE = True
except ImportError:
    RULE_FIX_AVAILABLE = False
    logging.getLogger(__name__).warning("规则修复模块不可用，将跳过规则修复阶段")

# 允许通过环境变量显式关闭规则修复（直接走增量修复/LLM 修复）
# - C2R_ENABLE_RULE_FIX=0/false/off/no 可关闭
_enable_rule_fix_env = os.environ.get("C2R_ENABLE_RULE_FIX", "1").strip().lower()
if _enable_rule_fix_env in ("0", "false", "off", "no"):
    RULE_FIX_AVAILABLE = False
    logging.getLogger(__name__).info("已禁用规则修复（C2R_ENABLE_RULE_FIX=0），将直接进入增量修复/LLM 修复路径")

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
# 屏蔽 openai 和 httpx 的 INFO 日志
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


CONTEXT_PREFIX_CHAR_LIMIT = 8000

# Token 限制常量（防止超过模型最大上下文窗口）
# Qwen3-Coder-30B 的最大上下文是 262144 tokens (256K)
# 预留空间给系统提示词(~3000)、C代码(~10000)、输出(~16000)等
# 实际可用于骨架上下文：约 230000 tokens (~920000 chars)
# 保留完整上下文以获得更好的翻译效果
# 骨架上下文长度上限（字符数）
#
# 说明：
# - skeleton_context 通常是 prompt 的主体；过大（例如 30k~80k tokens）会显著增加 vLLM 的
#   KV cache 压力与排队时间，导致 Request timed out + 重试风暴。
# - 允许通过环境变量覆盖；默认不主动截断（保持历史行为），由“选择性上下文”策略来降 token。
MAX_SKELETON_CONTEXT_CHARS = int(os.environ.get("MAX_SKELETON_CONTEXT_CHARS", "2000000"))
MAX_TOTAL_PROMPT_CHARS = int(os.environ.get("MAX_TOTAL_PROMPT_CHARS", "2000000"))  # ~500K tokens

def estimate_tokens(text: str) -> int:
    """粗略估计文本的 token 数量（英文约 4 字符/token，中文约 2 字符/token）"""
    if not text:
        return 0
    # 简单估算：代码主要是英文，按 4 字符/token 估算
    return len(text) // 4

def smart_truncate_context(context: str, max_chars: int, priority_patterns: List[str] = None) -> str:
    """
    智能截断上下文，保留重要部分

    优先级：
    1. 类型定义 (struct, enum, type)
    2. 全局变量声明 (static)
    3. 函数签名
    """
    if len(context) <= max_chars:
        return context

    # C2R: 添加详细的截断日志
    logger.warning(f"[截断警告] 骨架上下文过长: {len(context)} 字符 (~{len(context)//4} tokens)")
    logger.warning(f"[截断警告] 将截断到: {max_chars} 字符 (~{max_chars//4} tokens)")

    # 按段落分割
    sections = context.split("\n\n")

    # 分类段落
    type_sections = []      # 类型定义
    static_sections = []    # 全局变量
    func_sections = []      # 函数签名
    other_sections = []     # 其他

    for section in sections:
        section_stripped = section.strip()
        if not section_stripped:
            continue
        # 检查是否是注释行开头（段落标记）
        if section_stripped.startswith("// From types.rs"):
            type_sections.append(section)
        elif section_stripped.startswith("// From globals.rs"):
            static_sections.append(section)
        elif "pub struct " in section or "pub enum " in section or "pub type " in section:
            type_sections.append(section)
        elif "static " in section or "static mut " in section:
            static_sections.append(section)
        elif "fn " in section and "extern" not in section:
            func_sections.append(section)
        else:
            other_sections.append(section)

    # C2R: 记录各类别的统计信息
    logger.info(f"[截断统计] 类型定义: {len(type_sections)} 段, 全局变量: {len(static_sections)} 段, "
                f"函数签名: {len(func_sections)} 段, 其他: {len(other_sections)} 段")

    # 按优先级重新组装，直到达到限制
    result_parts = []
    current_length = 0
    dropped_counts = {"types": 0, "statics": 0, "funcs": 0, "others": 0}

    # 1. 优先添加类型定义
    for section in type_sections:
        if current_length + len(section) + 2 <= max_chars:
            result_parts.append(section)
            current_length += len(section) + 2
        else:
            dropped_counts["types"] += 1

    # 2. 添加全局变量
    for section in static_sections:
        if current_length + len(section) + 2 <= max_chars:
            result_parts.append(section)
            current_length += len(section) + 2
        else:
            dropped_counts["statics"] += 1

    # 3. 添加函数签名
    for section in func_sections:
        if current_length + len(section) + 2 <= max_chars:
            result_parts.append(section)
            current_length += len(section) + 2
        else:
            dropped_counts["funcs"] += 1

    # 4. 添加其他（如果还有空间）
    for section in other_sections:
        if current_length + len(section) + 2 <= max_chars:
            result_parts.append(section)
            current_length += len(section) + 2
        else:
            dropped_counts["others"] += 1

    truncated_context = "\n\n".join(result_parts)

    # C2R: 详细的截断统计日志
    logger.info(f"[截断结果] 最终长度: {len(truncated_context)} 字符 (~{len(truncated_context)//4} tokens)")
    logger.info(f"[截断结果] 保留段落: {len(result_parts)} 个")
    if any(dropped_counts.values()):
        logger.warning(f"[截断丢弃] 类型定义: {dropped_counts['types']}, 全局变量: {dropped_counts['statics']}, "
                      f"函数签名: {dropped_counts['funcs']}, 其他: {dropped_counts['others']}")

    return truncated_context


# 初始化 tree-sitter（兼容不同版本的 tree_sitter Python 绑定）
try:
    CPP_LANGUAGE = Language(tscpp.language(), "cpp")
except TypeError:
    CPP_LANGUAGE = Language(tscpp.language())
cpp_parser = Parser()
try:
    cpp_parser.set_language(CPP_LANGUAGE)  # older API
except AttributeError:
    cpp_parser.language = CPP_LANGUAGE  # newer API

if _HAS_TREE_SITTER_RUST:
    try:
        RUST_LANGUAGE = Language(tsrust.language(), "rust")
    except TypeError:
        RUST_LANGUAGE = Language(tsrust.language())
    rust_parser = Parser()
    try:
        rust_parser.set_language(RUST_LANGUAGE)  # older API
    except AttributeError:
        rust_parser.language = RUST_LANGUAGE  # newer API
else:
    RUST_LANGUAGE = None
    rust_parser = None


def _query_captures_compat(query, root_node):
    """Compat wrapper: tree_sitter >=0.25 removed Query.captures().

    Returns a normalized list of (node, capture_name_str) tuples.
    """
    cursor = QueryCursor(query)
    captures = cursor.captures(root_node)
    if isinstance(captures, dict):
        out = []
        for cap_name, nodes in captures.items():
            for node in nodes:
                out.append((node, cap_name))
        return out
    return captures or []


@dataclass
class FunctionInfo:
    """函数信息"""
    name: str                    # 函数名
    file_name: str               # 所属文件名（不含扩展名）
    index: int                   # 函数索引
    c_code: str                  # C 源代码
    rust_signature: str          # Rust 签名
    dependencies: Set[str]       # 依赖的其他函数名
    translated: bool = False     # 是否已翻译
    semantic_data: Optional[Dict[str, Any]] = None
    semantic_cache_path: Optional[Path] = None
    # 新增：从 extracted/<proj>/dependencies/<func_key>.txt 提取的被调函数签名提示（用于减少 *mut/*const 调用错误）
    callee_signatures_hint: Optional[str] = None
    # 新增：从 signature_matches 提取的 Rust 级被调函数签名提示（用于减少 E0308 / unsafe / Option<fn> 调用错误）
    callee_rust_signatures_hint: Optional[str] = None
    compiled: bool = False       # 是否编译通过
    # 新增：精确位置信息（用于确定性注入）
    location: Optional[Dict[str, Any]] = None  # {file, start_byte, end_byte, line_start, line_end}
    injection_failed: bool = False  # 是否注入失败


@dataclass
class FunctionLocation:
    """函数在 Rust 文件中的精确位置"""
    file_path: Path              # .rs 文件路径
    func_name: str               # 函数名
    start_byte: int              # 起始字节偏移
    end_byte: int                # 结束字节偏移
    line_start: int              # 起始行号（1-based）
    line_end: int                # 结束行号
    signature: str               # 完整签名（用于校验）


class CallGraphAnalyzer:
    """调用图分析器，用于拓扑排序"""
    
    def __init__(
        self,
        functions_dir: Path,
        dependencies_dir: Path,
        target_func_files: Optional[Set[str]] = None,
    ):
        self.functions_dir = functions_dir
        self.dependencies_dir = dependencies_dir
        self.target_func_files = set(target_func_files) if target_func_files else None
        self.functions: Dict[str, FunctionInfo] = {}
        self.call_graph: Dict[str, Set[str]] = defaultdict(set)  # caller -> callees
        self.reverse_graph: Dict[str, Set[str]] = defaultdict(set)  # callee -> callers
    
    def analyze(self) -> List[str]:
        """
        分析调用图并返回拓扑排序后的函数列表（自底向上）
        
        返回: 按依赖顺序排列的函数文件名列表（叶子函数在前）
        """
        # 1. 加载所有函数
        self._load_functions()
        
        # 2. 构建调用图
        self._build_call_graph()
        
        # 3. 拓扑排序
        sorted_functions = self._topological_sort()
        
        return sorted_functions
    
    def _load_functions(self):
        """加载所有函数信息"""
        if self.target_func_files:
            func_files: List[Path] = []
            for stem in sorted(self.target_func_files):
                p = self.functions_dir / f"{stem}.txt"
                if p.exists():
                    func_files.append(p)
                else:
                    logger.debug(f"函数文件不存在，跳过: {p}")
        else:
            func_files = list(self.functions_dir.glob("*.txt"))

        for func_file in func_files:
            func_name = func_file.stem  # e.g., "cJsonMock_1"
            
            with open(func_file, 'r', encoding='utf-8', errors='ignore') as f:
                c_code = f.read()
            
            # 提取实际函数名
            actual_name = self._extract_function_name(c_code)
            
            # 解析文件名获取基础信息
            parts = func_name.rsplit('_', 1)
            if len(parts) == 2:
                file_name, idx_str = parts
                try:
                    index = int(idx_str)
                except ValueError:
                    index = 0
            else:
                file_name = func_name
                index = 0
            
            self.functions[func_name] = FunctionInfo(
                name=actual_name or func_name,
                file_name=file_name,
                index=index,
                c_code=c_code,
                rust_signature="",
                dependencies=set()
            )
    
    def _extract_function_name(self, c_code: str) -> Optional[str]:
        """从 C 代码中提取函数名"""
        def _unwrap_to_identifier(decl_node):
            """
            tree-sitter 的 declarator 可能是多层嵌套（pointer_declarator / parenthesized_declarator 等）。
            这里沿着 declarator 字段向下剥离，直到拿到 identifier。
            """
            cur = decl_node
            visited = set()
            while cur is not None and id(cur) not in visited:
                visited.add(id(cur))
                if cur.type in ("identifier", "field_identifier"):
                    return cur

                nxt = cur.child_by_field_name("declarator")
                if nxt is not None:
                    cur = nxt
                    continue

                # 兜底：在当前节点子树里找第一个 identifier（只在 declarator 子树内搜索，避免误匹配参数名）
                queue = list(getattr(cur, "children", []) or [])
                while queue:
                    n = queue.pop(0)
                    if n.type in ("identifier", "field_identifier"):
                        return n
                    queue.extend(getattr(n, "children", []) or [])
                break
            return None

        try:
            tree = cpp_parser.parse(bytes(c_code, "utf-8"))

            # 查找函数定义：捕获 function_declarator 的 declarator 根节点，然后剥离到 identifier。
            query = Query(CPP_LANGUAGE, 
                """
                (function_definition
                    declarator: (function_declarator
                        declarator: (_) @decl))
                """
            )
            for node, capture_name in _query_captures_compat(query, tree.root_node):
                if capture_name != "decl":
                    continue
                ident = _unwrap_to_identifier(node)
                if ident is not None:
                    return c_code[ident.start_byte : ident.end_byte]
        except Exception as e:
            logger.debug(f"提取函数名失败: {e}")

        # 兜底（更弱但更稳）：只在第一行做轻量正则，避免误匹配函数体里的调用
        try:
            first_line = (c_code or "").strip().splitlines()[0]
            m = re.search(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", first_line)
            if m:
                return m.group(1)
        except Exception:
            pass

        return None
    
    def _build_call_graph(self):
        """构建调用图（优先从步骤1生成的调用图文件读取）"""
        import json
        from collections import defaultdict

        # 优先尝试读取步骤1生成的调用图 JSON 文件
        call_graph_file = self.dependencies_dir.parent / "call_graph.json"
        # 兼容：当使用 --use-libclang 且未运行 get_dependencies.py 时，调用图可能由 translate.py 写在 source_skeletons 中
        if not call_graph_file.exists():
            try:
                project_name = self.dependencies_dir.parent.name
                workspace_root = self.dependencies_dir.parent.parent.parent
                fallback = workspace_root / "source_skeletons" / project_name / "call_graph.json"
                if fallback.exists():
                    call_graph_file = fallback
            except Exception:
                pass

        if call_graph_file.exists():
            try:
                with open(call_graph_file, 'r', encoding='utf-8') as f:
                    graph_data = json.load(f)

                schema_version = graph_data.get("schema_version", "1.0")
                call_graph_data = graph_data.get("call_graph", {})
                function_names_data = graph_data.get("function_names", {})
                name_index_data = graph_data.get("name_index", {})
                functions_data = graph_data.get("functions", {})

                # ============ 新格式 (schema v2.0): uid 作为主键 ============
                if schema_version == "2.0":
                    logger.info("检测到调用图 schema v2.0 (uid 格式)")

                    # 构建 uid -> func_file_name 的映射
                    # uid 格式: file:line:name
                    # func_file_name 格式: filename_idx
                    uid_to_func_file = {}

                    # 方法1: 通过函数名匹配
                    func_name_to_files = defaultdict(list)
                    for func_file_name, func_info in self.functions.items():
                        func_name_to_files[func_info.name].append(func_file_name)

                    # 遍历 name_index，建立映射
                    for func_name, uids in name_index_data.items():
                        matching_files = func_name_to_files.get(func_name, [])
                        if matching_files:
                            # 尝试更精确匹配：通过文件名和行号
                            for uid in uids:
                                uid_info = functions_data.get(uid, {})
                                uid_file = uid_info.get("file", "")
                                uid_line = uid_info.get("start_line", 0)

                                # 在 matching_files 中找最佳匹配
                                best_match = None
                                for func_file in matching_files:
                                    func_file_stem = func_file.rsplit('_', 1)[0] if '_' in func_file else func_file
                                    if uid_file and uid_file in func_file_stem:
                                        best_match = func_file
                                        break
                                    if not best_match:
                                        best_match = func_file

                                if best_match:
                                    uid_to_func_file[uid] = best_match

                    # 加载调用图边
                    matched_nodes = 0
                    loaded_edges = 0
                    for caller_uid, callee_uids in call_graph_data.items():
                        caller_file = uid_to_func_file.get(caller_uid)
                        if not caller_file or caller_file not in self.functions:
                            continue
                        matched_nodes += 1
                        for callee_uid in (callee_uids or []):
                            callee_file = uid_to_func_file.get(callee_uid)
                            if callee_file and callee_file in self.functions:
                                self.call_graph[caller_file].add(callee_file)
                                self.reverse_graph[callee_file].add(caller_file)
                                self.functions[caller_file].dependencies.add(callee_file)
                                loaded_edges += 1

                    if matched_nodes > 0:
                        # 检查并报告同名函数冲突
                        duplicate_names = {name: uids for name, uids in name_index_data.items() if len(uids) > 1}
                        if duplicate_names:
                            logger.info(f"调用图中有 {len(duplicate_names)} 个同名函数（已通过 uid 区分）")
                        logger.info(
                            f"从调用图文件加载了 {matched_nodes} 个函数（{loaded_edges} 条边，uid 格式 v2.0）"
                        )
                        return

                    logger.warning("schema v2.0 格式但无法映射，尝试降级处理")

                # ============ 旧格式 (schema v1.0): 文件名或函数名作为键 ============
                # 兼容两种格式：
                # 1) 旧格式：call_graph 的 key 是函数文件名（例如 src_xxx_1）
                # 2) 新格式：call_graph 的 key 是函数名（例如 CreateFrameVec），需要用 function_names 做映射
                #
                # 优先尝试旧格式（文件名键）
                matched_nodes = 0
                loaded_edges = 0
                if isinstance(call_graph_data, dict):
                    for caller_key, callee_keys in call_graph_data.items():
                        if caller_key not in self.functions:
                            continue
                        matched_nodes += 1
                        for callee_key in (callee_keys or []):
                            if callee_key in self.functions:
                                self.call_graph[caller_key].add(callee_key)
                                self.reverse_graph[callee_key].add(caller_key)
                                self.functions[caller_key].dependencies.add(callee_key)
                                loaded_edges += 1
                if matched_nodes > 0:
                    if schema_version != "2.0":
                        logger.warning(
                            f"使用旧版调用图格式 (schema v1.0)，存在函数名冲突风险，建议重新生成调用图"
                        )
                    logger.info(
                        f"从调用图文件加载了 {matched_nodes} 个函数（{loaded_edges} 条边，文件名键）"
                    )
                    return

                # 再尝试新格式（函数名键）
                name_to_files: dict[str, list[str]] = defaultdict(list)
                if isinstance(function_names_data, dict) and function_names_data:
                    # function_names: {func_file_name -> actual_function_name}
                    for func_file_name, actual_name in function_names_data.items():
                        if not isinstance(func_file_name, str) or not isinstance(actual_name, str):
                            continue
                        name_to_files[actual_name].append(func_file_name)
                else:
                    # 回退：用已加载的 functions_dir 信息构建映射
                    for func_file_name, func_info in self.functions.items():
                        name_to_files[str(func_info.name)].append(func_file_name)

                matched_nodes = 0
                loaded_edges = 0
                if isinstance(call_graph_data, dict):
                    for caller_name, callee_names in call_graph_data.items():
                        caller_files = [f for f in name_to_files.get(str(caller_name), []) if f in self.functions]
                        if not caller_files:
                            continue
                        matched_nodes += len(caller_files)
                        for caller_file in caller_files:
                            for callee_name in (callee_names or []):
                                for callee_file in name_to_files.get(str(callee_name), []):
                                    if callee_file not in self.functions:
                                        continue
                                    self.call_graph[caller_file].add(callee_file)
                                    self.reverse_graph[callee_file].add(caller_file)
                                    self.functions[caller_file].dependencies.add(callee_file)
                                    loaded_edges += 1
                if matched_nodes > 0:
                    logger.info(
                        f"从调用图文件加载了 {matched_nodes} 个函数（{loaded_edges} 条边，函数名键）"
                    )
                    return

                logger.warning("调用图文件存在但格式不匹配或无法映射，回退到字符串匹配方式")
            except Exception as e:
                logger.warning(f"读取调用图文件失败: {e}，回退到原有字符串匹配方式")
        
        # 回退到原有逻辑（向后兼容）
        logger.info("使用字符串匹配方式构建调用图（向后兼容）")
        for func_name, func_info in self.functions.items():
            # 读取依赖文件
            dep_file = self.dependencies_dir / f"{func_name}.txt"
            if dep_file.exists():
                with open(dep_file, 'r', encoding='utf-8', errors='ignore') as f:
                    dep_content = f.read()
                
                # 解析依赖的函数
                for other_name, other_info in self.functions.items():
                    if other_name == func_name:
                        continue
                    # 检查是否调用了其他函数
                    if other_info.name in dep_content or other_info.name in func_info.c_code:
                        self.call_graph[func_name].add(other_name)
                        self.reverse_graph[other_name].add(func_name)
                        func_info.dependencies.add(other_name)
    
    def _topological_sort(self) -> List[str]:
        """
        拓扑排序，返回自底向上的顺序（叶子函数在前）
        """
        # 计算入度（被多少函数调用）
        in_degree = {name: 0 for name in self.functions}
        for caller, callees in self.call_graph.items():
            for callee in callees:
                if callee in in_degree:
                    in_degree[callee] += 1
        
        # 从叶子函数开始（入度为 0，即不被任何函数调用）
        # 注意：这里我们反向思考，先翻译"不调用其他函数的函数"
        # 所以我们计算的是"出度"
        out_degree = {name: len(self.call_graph.get(name, set())) for name in self.functions}
        
        result = []
        queue = [name for name, degree in out_degree.items() if degree == 0]
        visited = set()
        
        while queue:
            # 按文件名和索引排序，保持稳定性
            queue.sort(key=lambda x: (self.functions[x].file_name, self.functions[x].index))
            current = queue.pop(0)
            
            if current in visited:
                continue
            
            visited.add(current)
            result.append(current)
            
            # 处理依赖于当前函数的函数
            for caller in self.reverse_graph.get(current, set()):
                if caller not in visited:
                    # 检查该函数的所有依赖是否都已处理
                    deps = self.call_graph.get(caller, set())
                    if all(d in visited for d in deps):
                        queue.append(caller)
        
        # 处理剩余的函数（可能存在循环依赖）
        for name in self.functions:
            if name not in visited:
                result.append(name)
        
        return result
    
    def get_parallel_layers(self) -> List[List[str]]:
        """
        返回分层的函数列表，每层内的函数可以并行处理
        
        原理：
        - 第1层：出度为0的函数（叶子函数，不调用其他函数）
        - 第2层：依赖的函数都在第1层的函数
        - 第N层：依赖的函数都在前N-1层的函数
        
        Returns:
            List[List[str]]: 每层是一个可并行处理的函数列表
        """
        out_degree = {name: len(self.call_graph.get(name, set())) for name in self.functions}
        visited = set()
        layers = []
        
        while len(visited) < len(self.functions):
            # 找出当前可以处理的函数（所有依赖都已在之前的层处理完）
            current_layer = []
            for name in self.functions:
                if name in visited:
                    continue
                deps = self.call_graph.get(name, set())
                if all(d in visited for d in deps):
                    current_layer.append(name)
            
            if not current_layer:
                # 有循环依赖，将剩余函数放入最后一层
                remaining = [n for n in self.functions if n not in visited]
                if remaining:
                    layers.append(remaining)
                break
            
            # 按文件名和索引排序
            current_layer.sort(key=lambda x: (self.functions[x].file_name, self.functions[x].index))
            layers.append(current_layer)
            visited.update(current_layer)
        
        return layers


class IncrementalTranslator:
    """增量式翻译器"""
    
    def __init__(
        self,
        project_name: str,
        llm_name: str,
        max_repair_attempts: int = 5
    ):
        from workspace_config import (
            get_functions_path, get_dependencies_path, get_skeleton_path,
            get_translated_path, get_signature_match_path, get_final_project_path,
            get_repair_history_path, get_source_skeleton_path, WORKSPACE_ROOT
        )

        self.project_name = project_name
        self.llm_name = llm_name
        self.max_repair_attempts = max_repair_attempts
        self.truth_mode = os.environ.get("C2R_TRUTH_MODE", "0").strip().lower() in ("1", "true", "yes", "y", "on")
        # func_file -> FunctionInfo (filled in run(); used for deterministic callee module-path hints)
        self._functions_by_func_file: Dict[str, FunctionInfo] = {}
        # Deterministic internal symbol resolver (E0425): symbol -> {module_stem}
        self._internal_symbol_to_modules: Dict[str, Set[str]] = {}
        self._internal_symbol_index_built = False
        
        # 路径配置
        self.functions_dir = get_functions_path(project_name)
        self.dependencies_dir = get_dependencies_path(project_name)
        self.skeleton_dir = get_skeleton_path(project_name)
        self.translated_dir = get_translated_path(project_name, llm_name)
        # 签名目录：优先使用不带 llm_name 的目录（包含独立函数签名）
        self.signature_dir = get_signature_match_path(project_name)  # 不带 llm_name
        self.final_project_dir = get_final_project_path(project_name, llm_name)
        self.workspace_root = WORKSPACE_ROOT
        # 预处理后的 C 代码（.i 文件）目录：用于补足“外部函数声明/宏相关”的原型信息
        # 位置通常为 <workspace>/.preprocessed/*.i
        self.preprocessed_dir = self.workspace_root / ".preprocessed"
        # TU 上下文映射（由 get_dependencies.py 生成），用于在各阶段严格复用同一套 flags/宏/include 顺序
        self._tu_context_map_path = self.preprocessed_dir / "tu_context_map.json"
        self._tu_context_map: Dict[str, Any] = {}
        self._tu_context_files: Dict[str, Dict[str, Any]] = {}
        # 记录无法闭包（缺 compile_commands / 预处理失败 / .i 缺失等）的 file group，供最终汇总打印
        self._tu_closure_issues: Dict[str, str] = {}
        # 项目级 TU/输入闭包摘要（由 _load_tu_context_map 计算；用于 run() 末尾输出）
        self._tu_context_gap_summary: Dict[str, Any] = {}
        # RAG 结果目录
        self.rag_reranked_dir = get_reranked_path(project_name)
        # Whether to inject RAG knowledge into LLM prompts (independent of whether reranking was re-run)
        self._enable_rag_context = os.environ.get("C2R_USE_RAG_CONTEXT", "true").strip().lower() in (
            "1",
            "true",
            "yes",
            "y",
            "on",
        )

        # ===================== RQ3.3: Knowledge channel ablation (C0-C6) =====================
        # Default behavior MUST remain unchanged unless the env vars are explicitly set:
        # - C2R_RQ3_API_MAPPING_MODE: none|predicted|oracle (default: predicted)
        # - C2R_RQ3_PARTIAL_IDIOM_MODE: none|predicted|oracle (default: predicted)
        # - C2R_ORACLE_KNOWLEDGE_ROOT: oracle root dir (default: <repo>/oracle_rust)
        # - C2R_ORACLE_AUTO_EXTRACT: 1/0 (default: 0)
        def _norm_mode(raw: str) -> str:
            v = (raw or "").strip().lower()
            if v in ("none", "n", "no", "off", "false", "0"):
                return "none"
            if v in ("oracle", "o"):
                return "oracle"
            # default fallback: predicted
            return "predicted"

        self._rq3_api_mapping_mode = _norm_mode(os.environ.get("C2R_RQ3_API_MAPPING_MODE", ""))
        self._rq3_partial_idiom_mode = _norm_mode(os.environ.get("C2R_RQ3_PARTIAL_IDIOM_MODE", ""))

        _oracle_root_env = (os.environ.get("C2R_ORACLE_KNOWLEDGE_ROOT") or "").strip()
        if _oracle_root_env:
            self._oracle_knowledge_root = Path(_oracle_root_env).expanduser()
        else:
            # incremental_translate.py lives at repo root; keep oracle_rust beside it by default.
            self._oracle_knowledge_root = Path(__file__).resolve().parent.parent / "oracle_rust"

        self._oracle_auto_extract = (os.environ.get("C2R_ORACLE_AUTO_EXTRACT", "0").strip().lower() in ("1", "true", "yes", "y", "on"))
        self._oracle_index_loaded = False
        self._oracle_funcname_to_path: Dict[str, Path] = {}
        self._oracle_funckey_to_path: Dict[str, Path] = {}
        self.project_src_dir = get_project_path(project_name)
        self.source_skeleton_dir = get_source_skeleton_path(project_name)
        
        # 工作目录（用于增量编译测试）
        self.work_dir = self.workspace_root / "incremental_work" / project_name / f"translate_by_{llm_name}"
        
        # 修复历史目录（保存每次修复尝试的错误信息和翻译代码）
        self.repair_history_dir = get_repair_history_path(project_name, llm_name)

        # 人工修复工件（避免“为编译通过而删内容”；始终保留 LLM 的翻译尝试供人工复检）
        self.manual_fix_root = self.repair_history_dir / "_manual_fix"
        self.manual_fix_root.mkdir(parents=True, exist_ok=True)
        self.manual_fix_manifest_path = self.manual_fix_root / "manifest.jsonl"
        # 失败注释模式：off|simple|llm（默认 simple，避免额外 LLM 调用拖慢整体吞吐）
        self.failure_comment_mode = os.environ.get("C2R_FAILURE_COMMENT_MODE", "simple").strip().lower()
        
        # LLM 提示词保存目录（保存到 output 目录）
        # 优先使用 workspace_root/llm_prompts，如果不存在则使用默认位置
        self.llm_prompts_dir = self.workspace_root / "llm_prompts" / project_name / f"translate_by_{llm_name}"
        self.llm_prompts_dir.mkdir(parents=True, exist_ok=True)
        self.context_cache_dir = self.workspace_root / "context_cache" / project_name
        self.context_cache_dir.mkdir(parents=True, exist_ok=True)
        self.usage_examples_file = self.context_cache_dir / "_usage_examples.json"
        self._source_file_cache: Dict[str, Optional[Path]] = {}
        # cache: preprocessed .i path -> {name: [prototype,...]}
        self._preprocessed_decl_index: Dict[Path, Dict[str, List[str]]] = {}
        self._load_tu_context_map()
        self._apply_selected_profile_env()

        # ========== Preprocessed function slices (macro-expanded C as primary source) ==========
        # When available, replace extracted `functions/*.txt` with the corresponding function slice
        # from the exact preprocessed `.i` TU (stage1-selected). This avoids "macro-as-function"
        # symbol issues and stabilizes pointer/const/mut analysis.
        self._enable_preprocessed_function_slices = os.environ.get(
            "C2R_USE_PREPROCESSED_FUNCTION_SLICES", "true"
        ).strip().lower() in ("1", "true", "yes")
        self._functions_manifest_by_file: Dict[str, Dict[str, Any]] = {}
        self._preprocessed_text_cache: Dict[Path, str] = {}
        # pre_path -> {(name, basename(source_file), start_line): [(start_byte,end_byte), ...]}
        self._preprocessed_fn_index_cache: Dict[Path, Dict[Tuple[str, str, int], List[Tuple[int, int]]]] = {}
        # pre_path -> {static_inline_fn_name,...} (best-effort, used to avoid generating extern decls for inline helpers)
        self._preprocessed_static_inline_names_cache: Dict[Path, Set[str]] = {}
        if self._enable_preprocessed_function_slices:
            self._load_functions_manifest()

        # ========== types.rs slicing (reduce prompt length deterministically) ==========
        self._enable_types_slice = os.environ.get("C2R_ENABLE_TYPES_SLICE", "true").lower() in ("1", "true", "yes")
        self._types_registry = None
        self._skeleton_non_types_context: str = ""
        # Extracted helper maps from types.rs (best-effort, used for prompt hints)
        self._types_alias_map: Dict[str, str] = {}
        self._types_const_names: Set[str] = set()
        self._types_const_type_map: Dict[str, str] = {}
        try:
            self._types_slice_max_other_seeds = int(os.environ.get("C2R_TYPES_SLICE_MAX_OTHER_SEEDS", "120"))
        except Exception:
            self._types_slice_max_other_seeds = 120

        # ========== 调用签名提示（从 extracted/<proj>/dependencies/<func_key>.txt） ==========
        # 目的：减少 LLM 在“调用点”把 *mut/*const 搞反导致的 E0308（mutability mismatch）
        self._enable_dep_signature_hints = os.environ.get("C2R_ENABLE_DEP_SIGNATURE_HINTS", "true").lower() in (
            "1",
            "true",
            "yes",
        )
        try:
            self._dep_signature_hints_max = int(os.environ.get("C2R_DEP_SIGNATURE_HINTS_MAX", "32"))
        except Exception:
            self._dep_signature_hints_max = 32
        # 额外：从预处理 .i 文件中提取“外部函数声明”原型（解决 E0425: missing function/macro 声明）
        self._enable_preprocessed_decl_hints = os.environ.get(
            "C2R_ENABLE_PREPROCESSED_DECL_HINTS",
            "true",
        ).lower() in ("1", "true", "yes")
        try:
            self._preprocessed_decl_hints_max_per_name = int(
                os.environ.get("C2R_PREPROCESSED_DECL_HINTS_MAX_PER_NAME", "3")
            )
        except Exception:
            self._preprocessed_decl_hints_max_per_name = 3
        # 额外：从预处理 .i 文件中提取 `static inline` 函数定义（避免把宏/inline 当 extern）
        self._enable_preprocessed_inline_helper_hints = os.environ.get(
            "C2R_ENABLE_PREPROCESSED_INLINE_HELPER_HINTS",
            "true",
        ).lower() in ("1", "true", "yes")
        try:
            self._preprocessed_inline_helper_hints_max = int(
                os.environ.get("C2R_PREPROCESSED_INLINE_HELPER_HINTS_MAX", "8")
            )
        except Exception:
            self._preprocessed_inline_helper_hints_max = 8
        try:
            self._preprocessed_inline_helper_hints_max_chars = int(
                os.environ.get("C2R_PREPROCESSED_INLINE_HELPER_HINTS_MAX_CHARS", "12000")
            )
        except Exception:
            self._preprocessed_inline_helper_hints_max_chars = 12000

        # ========== Step 2.55: extern decl prelude (bindgen allowlist; compile-safe by construction) ==========
        # Prefer bindgen to generate *exact* Rust signatures for external callees, and backfill missing typedefs
        # into `types.rs` before baseline compile. This avoids global blockers like:
        # - compat.rs references missing typedefs (e.g., StorageHookFunction)
        # - wrong mut/const in extern signatures from string-based mapping
        self._enable_bindgen_extern_prelude = os.environ.get(
            "C2R_ENABLE_BINDGEN_EXTERN_DECLS",
            "true",
        ).lower() in ("1", "true", "yes")
        # Legacy fallback: string-mapping from preprocessed `.i` prototypes.
        # Default OFF: this path is inherently brittle (parameter names can be missing/ambiguous) and can generate
        # invalid Rust like `crate::types::T: *mut crate::types::T` (E0130).
        self._enable_extern_decls_string_fallback = os.environ.get(
            "C2R_ENABLE_EXTERN_DECLS_STRING_FALLBACK",
            "false",
        ).lower() in ("1", "true", "yes")
        try:
            self._bindgen_extern_timeout_sec = int(os.environ.get("C2R_BINDGEN_EXTERN_TIMEOUT_SEC", "90"))
        except Exception:
            self._bindgen_extern_timeout_sec = 90
        try:
            self._bindgen_extern_max_funcs_per_tu = int(os.environ.get("C2R_BINDGEN_EXTERN_MAX_FUNCS_PER_TU", "200"))
        except Exception:
            self._bindgen_extern_max_funcs_per_tu = 200
        # cache: (preprocessed .i path, hash(callee set)) -> {"raw_rs": ..., "types": {...}, "fns": {...}, ...}
        # NOTE: the callee set must be part of the key; otherwise a "small allowlist run" can poison cache and
        # make later runs miss extern decls (then we accidentally fall back to brittle string-mapping).
        self._bindgen_extern_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}

        # ========== Step 2.56: extern vars/consts prelude (bindgen allowlist; compile-safe by construction) ==========
        # When Rust code refers to a global C variable/const that wasn't generated into `globals.rs`/`types.rs`,
        # we can deterministically backfill its declaration from the pinned TU `.i` via bindgen.
        # This is for TRUE external symbols / constified globals (not C macros / header-only inlines).
        self._enable_bindgen_extern_vars_prelude = os.environ.get(
            "C2R_ENABLE_BINDGEN_EXTERN_VARS",
            "true",
        ).lower() in ("1", "true", "yes")
        try:
            self._bindgen_extern_vars_timeout_sec = int(os.environ.get("C2R_BINDGEN_EXTERN_VARS_TIMEOUT_SEC", "90"))
        except Exception:
            self._bindgen_extern_vars_timeout_sec = 90
        try:
            self._bindgen_extern_max_vars_per_pass = int(os.environ.get("C2R_BINDGEN_EXTERN_MAX_VARS_PER_PASS", "200"))
        except Exception:
            self._bindgen_extern_max_vars_per_pass = 200
        # cache: (preprocessed .i path, hash(var/const allowlist set)) -> {"raw_rs": ..., "types": {...}, "consts": {...}, "vars": {...}}
        self._bindgen_extern_vars_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._extern_vars_report_path: Optional[Path] = None
        self._extern_vars_missing: List[Dict[str, Any]] = []

        # ========== C2Rust fallback for failed functions ==========
        # If a function fails after all LLM repair attempts and would be rolled back to `unimplemented!()`,
        # we can optionally fill it with a deterministic C2Rust transpile fallback (mechanical, unsafe OK).
        # To avoid breaking the project, we only keep the fallback if `cargo check` succeeds.
        self._enable_c2rust_fallback = os.environ.get("C2R_ENABLE_C2RUST_FALLBACK", "true").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        try:
            self._c2rust_fallback_timeout_sec = int(os.environ.get("C2R_C2RUST_FALLBACK_TIMEOUT_SEC", "180"))
        except Exception:
            self._c2rust_fallback_timeout_sec = 180
        # cache: safe_file_group -> generated fallback module path (under work_dir/src/__c2r_generated)
        self._c2rust_fallback_modules: Dict[str, Path] = {}
        # cache: safe_file_group -> set(function names available in c2rust output) (best-effort)
        self._c2rust_fallback_function_index: Dict[str, Set[str]] = {}

        # ========== Rust callee signature hints (signature_matches + typedef expansion) ==========
        self._enable_rust_callee_signature_hints = os.environ.get(
            "C2R_ENABLE_RUST_CALLEE_SIGNATURE_HINTS",
            "true",
        ).lower() in ("1", "true", "yes")
        try:
            self._rust_callee_signature_hints_max = int(os.environ.get("C2R_RUST_CALLEE_SIGNATURE_HINTS_MAX", "16"))
        except Exception:
            self._rust_callee_signature_hints_max = 16
        self._rust_signature_index: Optional[Dict[str, List[str]]] = None

        # ========== Typed constants/macros + field access hints (for E0308 / struct-field guidance) ==========
        self._enable_typed_const_hints = os.environ.get("C2R_ENABLE_TYPED_CONST_HINTS", "true").lower() in (
            "1",
            "true",
            "yes",
        )
        try:
            self._typed_const_hints_max = int(os.environ.get("C2R_TYPED_CONST_HINTS_MAX", "32"))
        except Exception:
            self._typed_const_hints_max = 32
        self._enable_c_field_access_hints = os.environ.get("C2R_ENABLE_C_FIELD_ACCESS_HINTS", "true").lower() in (
            "1",
            "true",
            "yes",
        )
        try:
            self._c_field_access_max_vars = int(os.environ.get("C2R_C_FIELD_ACCESS_MAX_VARS", "24"))
        except Exception:
            self._c_field_access_max_vars = 24
        try:
            self._c_field_access_max_fields = int(os.environ.get("C2R_C_FIELD_ACCESS_MAX_FIELDS", "12"))
        except Exception:
            self._c_field_access_max_fields = 12

        # ========== Pointer mutability/nullability contracts (inferred from C) ==========
        self._enable_c_pointer_contract_hints = os.environ.get("C2R_ENABLE_C_POINTER_CONTRACT_HINTS", "true").lower() in (
            "1",
            "true",
            "yes",
        )
        try:
            self._c_pointer_contract_max_lines = int(os.environ.get("C2R_C_POINTER_CONTRACT_MAX_LINES", "32"))
        except Exception:
            self._c_pointer_contract_max_lines = 32

        # ========== 新增：加载确定性签名映射（阶段1生成） ==========
        self.func_file_to_rust_sig: Dict[str, str] = {}
        self._load_deterministic_signature_mapping()

        # Step 2.55 extern decl report path (populated by `_ensure_extern_decls_from_preprocessed`).
        self._extern_decls_report_path: Optional[Path] = None
        # Bindgen allowlist missing report (best-effort; populated by `_ensure_extern_decls_from_bindgen_allowlist`).
        self._extern_decls_missing: List[Dict[str, Any]] = []

        # ========== Injection protocol hardening (structure validation + auto retry) ==========
        # Goal: avoid `injection_failed` due to truncated/unclosed LLM output.
        self._enable_injection_retry = os.environ.get("C2R_ENABLE_INJECTION_RETRY", "true").lower() in (
            "1",
            "true",
            "yes",
        )
        try:
            self._injection_retry_max = int(os.environ.get("C2R_INJECTION_RETRY_MAX", "2"))
        except Exception:
            self._injection_retry_max = 2

        # 统计信息
        self.stats = {
            "total": 0,
            "translated": 0,
            "compiled": 0,
            "repaired": 0,
            "c2rust_fallback": 0,  # 新增：LLM 失败后使用 C2Rust 兜底并通过编译的函数数
            "failed": 0,
            "skipped": 0,
            "injection_failed": 0,
            "still_unimplemented": 0,  # 新增：编译通过但函数体仍是 unimplemented!()
            "timeout_failed": 0  # 新增：LLM 超时失败的函数数
        }

        # 超时失败的函数列表（用于重试）
        self.timeout_failed_funcs: List[str] = []
        
        # 不透明类型集合（动态检测）
        self.opaque_types: Set[str] = set()

        # RustMap 风格 safe globals（来自 skeleton 的 globals_accessors.json）
        # 映射: 原始变量名 -> {get/set/with/cell/rust_type/...}
        self.safe_globals: Dict[str, Dict] = {}
        
        # 调用图上下文提供器（可选）
        self.call_graph_context_provider = None
        self._init_call_graph_context()

        # Scheme-B: field accessor shims (lazy init after work_dir is created)
        self._field_shims = None

        # ========== 新增：按需类型/结构恢复（编译错误触发） ==========
        self.type_recovery = None
        try:
            enable_type_recovery = os.environ.get("C2R_ENABLE_TYPE_RECOVERY", "true").lower() in ("true", "1", "yes")
            enable_llm_struct_infer = os.environ.get("C2R_ENABLE_LLM_STRUCT_INFER", "true").lower() in ("true", "1", "yes")
            if enable_type_recovery:
                from type_recovery import TypeRecoveryManager

                self.type_recovery = TypeRecoveryManager(
                    project_name=self.project_name,
                    project_root=self.project_src_dir,
                    work_dir=self.work_dir,
                    context_cache_dir=self.context_cache_dir,
                    llm_name=self.llm_name,
                    enable_llm=enable_llm_struct_infer,
                )
                logger.info("已启用按需类型恢复: C2R_ENABLE_TYPE_RECOVERY=true")
        except Exception as e:
            logger.debug(f"初始化 type_recovery 失败，跳过: {e}")

    def _apply_selected_profile_env(self) -> None:
        """
        Ensure stage3 (incremental translation / type recovery) uses the same OpenHarmony build profile
        (out_dir keyed compile_commands.json) selected in stage1.

        Motivation:
        - `get_dependencies.py` selects a profile from compile_commands_all and records it in `tu_context_map.json`.
        - Later stages run in separate processes, so we must re-apply the selection to the environment,
          otherwise helpers like `TypeRecoveryManager` may fall back to the hard-coded rk3568 compile DB.
        """
        # If the caller already pinned compile_commands explicitly, keep it.
        try:
            env_cc = (os.environ.get("OHOS_COMPILE_COMMANDS") or "").strip() or (os.environ.get("COMPILE_COMMANDS_PATH") or "").strip()
            if env_cc:
                p = Path(env_cc).expanduser()
                if p.exists():
                    return
        except Exception:
            pass

        sel: Dict[str, Any] = {}
        try:
            if isinstance(getattr(self, "_tu_context_map", None), dict):
                sel = self._tu_context_map.get("selected_profile") or {}
        except Exception:
            sel = {}

        def _set_env(cc_path: Optional[Path], ohos_root: Optional[Path], *, out_dir: str = "", label: str = "") -> None:
            if not cc_path:
                return
            try:
                cc_path = Path(cc_path).expanduser().resolve()
            except Exception:
                return
            if not cc_path.exists():
                return
            os.environ.setdefault("OHOS_COMPILE_COMMANDS", str(cc_path))
            os.environ.setdefault("COMPILE_COMMANDS_PATH", str(cc_path))  # legacy compatibility
            if ohos_root:
                try:
                    oh = Path(ohos_root).expanduser().resolve()
                    if oh.exists():
                        os.environ.setdefault("OHOS_ROOT", str(oh))
                except Exception:
                    pass
            if out_dir:
                os.environ.setdefault("TARGET_OUT_DIR", str(out_dir))
            if label:
                os.environ.setdefault("C2R_SELECTED_PROFILE_LABEL", str(label))

        # 1) Prefer stage1-selected profile (tu_context_map.json)
        try:
            cc = (sel.get("compile_commands_json") or "").strip()
            oh = (sel.get("ohos_root") or "").strip()
            out_dir = (sel.get("out_dir") or "").strip()
            label = (sel.get("label") or "").strip()
            if cc:
                cc_path = Path(cc).expanduser()
                oh_path = Path(oh).expanduser() if oh else None
                _set_env(cc_path, oh_path, out_dir=out_dir, label=label)
                if os.environ.get("OHOS_COMPILE_COMMANDS"):
                    return
        except Exception:
            pass

        # 2) If stage1 mapping is missing (e.g., libclang mode), re-run the same deterministic profile selector.
        try:
            from ohos_build_profile import select_profile_for_project

            selected = select_profile_for_project(
                project_name=str(self.project_name),
                project_root=Path(self.project_src_dir),
                require_registry=False,
            )
            if selected:
                cc_path = Path(selected.compile_commands_json)
                oh_path = Path(selected.ohos_root) if selected.ohos_root else None
                _set_env(cc_path, oh_path, out_dir=str(selected.profile.out_dir or ""), label=str(selected.profile.label or ""))
        except Exception:
            return

    def _load_tu_context_map(self) -> None:
        """
        Load TU context mapping produced by stage1 (get_dependencies.py).

        This mapping is the glue that ensures stage3 reuses the exact same TU flags/macros/include order
        (via the exact preprocessed `.i` path), rather than heuristically picking the "first glob match".
        """
        self._tu_context_map = {}
        self._tu_context_files = {}
        self._tu_context_gap_summary = {}
        p = self._tu_context_map_path
        if not p or not p.exists():
            self._tu_context_gap_summary = {"enabled": False, "reason": "tu_context_map_missing", "path": str(p) if p else None}
            return
        try:
            data = json.loads(p.read_text(encoding="utf-8", errors="ignore") or "{}")
            if isinstance(data, dict):
                self._tu_context_map = data
                files = data.get("files")
                if isinstance(files, dict):
                    # Keys are safe module names (file groups).
                    self._tu_context_files = {str(k): (v if isinstance(v, dict) else {}) for k, v in files.items()}
                # Build a small project-level summary for end-of-run reporting.
                total = 0
                included = 0
                excluded = 0
                entry_present = 0
                entry_missing = 0
                missing_samples: List[str] = []
                pre_ok = 0
                pre_missing = 0
                pre_failed = 0
                fail_samples: List[Dict[str, Any]] = []
                if isinstance(files, dict):
                    for safe_name, rec in files.items():
                        if not isinstance(rec, dict):
                            continue
                        total += 1
                        if rec.get("excluded_by_compile_commands"):
                            excluded += 1
                            continue
                        included += 1
                        entry = rec.get("compile_commands_entry")
                        has_entry = isinstance(entry, dict) and bool(entry.get("command") or entry.get("arguments"))
                        if has_entry:
                            entry_present += 1
                        else:
                            entry_missing += 1
                            if len(missing_samples) < 10:
                                missing_samples.append(str(safe_name))

                        pre = rec.get("preprocessed_file")
                        err = (rec.get("error") or "").strip()
                        if pre:
                            try:
                                pp = Path(str(pre))
                                if pp.exists():
                                    pre_ok += 1
                                else:
                                    pre_missing += 1
                            except Exception:
                                pre_missing += 1
                        else:
                            if err:
                                pre_failed += 1
                                if len(fail_samples) < 10:
                                    fail_samples.append(
                                        {
                                            "file_group": str(safe_name),
                                            "source_file_rel": rec.get("source_file_rel"),
                                            "error": err[:240],
                                        }
                                    )
                self._tu_context_gap_summary = {
                    "enabled": True,
                    "path": str(p),
                    "use_preprocessing": bool(data.get("use_preprocessing")),
                    "preprocessing_enabled": bool(data.get("preprocessing_enabled")),
                    "counts": {
                        "file_groups_total": total,
                        "file_groups_included": included,
                        "excluded_by_compile_commands": excluded,
                        "compile_commands_entry_present": entry_present,
                        "compile_commands_entry_missing": entry_missing,
                        "preprocessed_ok": pre_ok,
                        "preprocessed_failed": pre_failed,
                        "preprocessed_file_missing": pre_missing,
                    },
                    "missing_entry_samples": missing_samples,
                    "failure_samples": fail_samples,
                }
        except Exception as e:
            logger.debug(f"读取 TU 上下文映射失败: {p}: {e}")
            self._tu_context_map = {}
            self._tu_context_files = {}
            self._tu_context_gap_summary = {"enabled": False, "reason": f"tu_context_map_parse_failed: {e}", "path": str(p)}

    def _print_tu_context_gap_summary(self) -> None:
        """
        Print project-level TU/input-closure gaps.

        NOTE: These are not translation bugs; they indicate build context is incomplete
        (e.g., compile_commands missing/partial, preprocessing failed, or recorded .i missing).
        """
        tc = getattr(self, "_tu_context_gap_summary", None)
        if not isinstance(tc, dict) or not tc:
            return
        if not tc.get("enabled"):
            # Only surface missing map when the user expected preprocessing (use_preprocessing=true).
            reason = (tc.get("reason") or "").strip()
            if reason in ("tu_context_map_missing", "tu_context_map_no_files"):
                print(f"[TU闭包] ⚠ 未发现 TU 上下文映射: {tc.get('path')}")
            return
        counts = tc.get("counts") or {}
        miss = int(counts.get("compile_commands_entry_missing", 0) or 0)
        pre_failed = int(counts.get("preprocessed_failed", 0) or 0)
        pre_missing = int(counts.get("preprocessed_file_missing", 0) or 0)
        if miss <= 0 and pre_failed <= 0 and pre_missing <= 0:
            return
        total = int(counts.get("file_groups_included", counts.get("file_groups_total", 0) or 0) or 0)
        print("[TU闭包] ⚠ 输入闭包不完整（不是翻译 bug）:")
        print(
            f"  - compile_commands 缺失条目: {miss} / {total}；"
            f"预处理失败: {pre_failed}；预处理产物缺失: {pre_missing}"
        )
        print(f"  - 详情: {tc.get('path')}")
        samples = tc.get("missing_entry_samples") or []
        if isinstance(samples, list) and samples:
            print(f"  - 缺失条目样例(file_group): {', '.join(str(x) for x in samples[:8])}")
        fail_samples = tc.get("failure_samples") or []
        if isinstance(fail_samples, list) and fail_samples:
            s0 = fail_samples[0] if isinstance(fail_samples[0], dict) else {}
            err0 = (s0.get('error') or '').strip()
            src0 = s0.get('source_file_rel') or s0.get('file_group') or '-'
            if err0:
                print(f"  - 预处理失败样例: {src0}: {err0[:160]}")

    def _load_functions_manifest(self) -> None:
        """
        Load stage1 `functions_manifest.json` (best-effort).

        We use it to map `func_file -> (source_file, start_line, end_line, name)` so we can
        locate the exact function definition inside the preprocessed `.i` TU.
        """
        self._functions_manifest_by_file = {}
        candidates: List[Path] = []
        try:
            candidates.append(self.functions_dir.parent / "functions_manifest.json")
        except Exception:
            pass
        try:
            candidates.append(self.dependencies_dir.parent / "functions_manifest.json")
        except Exception:
            pass
        # Legacy nested path from older stage1 layout.
        try:
            candidates.append(self.dependencies_dir.parent / "extracted" / self.project_name / "functions_manifest.json")
        except Exception:
            pass

        for p in candidates:
            try:
                if not p or not p.exists():
                    continue
                data = json.loads(p.read_text(encoding="utf-8", errors="ignore") or "{}")
                funcs = data.get("functions") if isinstance(data, dict) else None
                if not isinstance(funcs, list):
                    continue
                for meta in funcs:
                    if not isinstance(meta, dict):
                        continue
                    func_file = meta.get("func_file")
                    if isinstance(func_file, str) and func_file:
                        self._functions_manifest_by_file[func_file] = meta
                if self._functions_manifest_by_file:
                    logger.info(f"[PreprocessedSlices] 已加载 functions_manifest: {p} (n={len(self._functions_manifest_by_file)})")
                return
            except Exception:
                continue

    @staticmethod
    def _map_line_directives_for_target_lines(i_text: str, target_lines: Set[int]) -> Dict[int, Tuple[str, int]]:
        """
        Best-effort mapping: `.i` line -> (original_file, original_line), but only for `target_lines`.

        This avoids storing a full per-line mapping for large `.i` files.
        """
        if not i_text or not target_lines:
            return {}
        mapping: Dict[int, Tuple[str, int]] = {}
        current_file: Optional[str] = None
        current_line: int = 0
        for i, line in enumerate(i_text.splitlines(), start=1):
            m = re.match(r'^#\s+(\d+)\s+"([^"]+)"', line)
            if m:
                try:
                    current_line = int(m.group(1))
                    current_file = m.group(2)
                except Exception:
                    current_file = None
                    current_line = 0
                continue
            if current_file:
                if i in target_lines:
                    mapping[i] = (current_file, current_line)
                current_line += 1
        return mapping

    @staticmethod
    def _unwrap_declarator_to_identifier(decl_node):
        """
        tree-sitter declarator nodes can be nested (pointer_declarator / parenthesized_declarator / etc).
        Peel `declarator:` fields until reaching an identifier.
        """
        def _node_key(n) -> Tuple[str, int, int]:
            # Avoid `id(node)` due to potential wrapper reuse across bindings/GC.
            try:
                return (str(getattr(n, "type", "")), int(getattr(n, "start_byte", -1)), int(getattr(n, "end_byte", -1)))
            except Exception:
                return (str(getattr(n, "type", "")), -1, -1)

        def _iter_children(n) -> List[Any]:
            # Compat across tree_sitter python bindings: some expose `.children`, some only `child_count/child(i)`.
            try:
                kids = getattr(n, "children", None)
                if kids is not None:
                    return list(kids)
            except Exception:
                pass
            out: List[Any] = []
            try:
                cnt = int(getattr(n, "child_count", 0) or 0)
            except Exception:
                cnt = 0
            for i in range(cnt):
                try:
                    out.append(n.child(i))
                except Exception:
                    continue
            return out

        cur = decl_node
        visited: Set[Tuple[str, int, int]] = set()
        while cur is not None:
            key = _node_key(cur)
            if key in visited:
                break
            visited.add(key)

            if getattr(cur, "type", None) in ("identifier", "field_identifier"):
                return cur

            try:
                nxt = cur.child_by_field_name("declarator")
            except Exception:
                nxt = None
            if nxt is not None:
                cur = nxt
                continue

            # Fallback: BFS within subtree (best-effort).
            queue = _iter_children(cur)
            q_i = 0
            seen: Set[Tuple[str, int, int]] = set()
            while q_i < len(queue):
                n = queue[q_i]
                q_i += 1
                nk = _node_key(n)
                if nk in seen:
                    continue
                seen.add(nk)
                if getattr(n, "type", None) in ("identifier", "field_identifier"):
                    return n
                queue.extend(_iter_children(n))
            break

        return None

    def _build_preprocessed_fn_index(
        self, pre_path: Path, i_text: str
    ) -> Dict[Tuple[str, str, int], List[Tuple[int, int]]]:
        """
        Build an index for a preprocessed TU:
          (fn_name, basename(orig_file), orig_start_line) -> [(start_byte, end_byte), ...]
        """
        if not i_text:
            return {}
        try:
            tree = cpp_parser.parse(bytes(i_text, "utf-8"))
        except Exception as e:
            logger.debug(f"[PreprocessedSlices] 解析 .i 失败 {pre_path}: {e}")
            return {}

        # Collect function_definition nodes.
        try:
            q = Query(CPP_LANGUAGE, "(function_definition) @fn")
            captured = _query_captures_compat(q, tree.root_node)
            fn_nodes = [n for (n, cap) in captured if cap == "fn"]
        except Exception:
            fn_nodes = []

        if not fn_nodes:
            return {}

        # Need original mapping for the start line of each function node.
        start_lines_i: Set[int] = set()
        for n in fn_nodes:
            try:
                start_lines_i.add(int(n.start_point[0]) + 1)
            except Exception:
                continue
        line_map = self._map_line_directives_for_target_lines(i_text, start_lines_i)

        index: Dict[Tuple[str, str, int], List[Tuple[int, int]]] = defaultdict(list)
        for n in fn_nodes:
            try:
                decl = n.child_by_field_name("declarator")
                if decl is None:
                    continue
                ident = self._unwrap_declarator_to_identifier(decl)
                if ident is None:
                    continue
                fn_name = i_text[ident.start_byte : ident.end_byte]
                if not fn_name:
                    continue
                start_line_i = int(n.start_point[0]) + 1
                mapped = line_map.get(start_line_i)
                if not mapped:
                    continue
                orig_file, orig_line = mapped
                base = Path(str(orig_file)).name
                key = (fn_name, base, int(orig_line))
                index[key].append((int(n.start_byte), int(n.end_byte)))
            except Exception:
                continue

        return index

    def _get_preprocessed_static_inline_names(self, pre_path: Path) -> Set[str]:
        """
        Best-effort: return the set of `static inline` function names defined in the given preprocessed TU.

        We use this to avoid generating `extern "C"` declarations for header inlines, which are not stable link
        symbols and should be translated/inlined from their bodies instead.
        """
        if not pre_path:
            return set()
        try:
            p = Path(pre_path)
        except Exception:
            return set()
        if not p.exists():
            return set()

        cached = self._preprocessed_static_inline_names_cache.get(p)
        if cached is not None:
            return set(cached)

        names: Set[str] = set()

        i_text = self._preprocessed_text_cache.get(p)
        if i_text is None:
            try:
                i_text = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                i_text = ""
            self._preprocessed_text_cache[p] = i_text
        if not i_text:
            self._preprocessed_static_inline_names_cache[p] = set()
            return set()

        fn_index = self._preprocessed_fn_index_cache.get(p)
        if fn_index is None:
            fn_index = self._build_preprocessed_fn_index(p, i_text)
            self._preprocessed_fn_index_cache[p] = fn_index
        if not fn_index:
            self._preprocessed_static_inline_names_cache[p] = set()
            return set()

        # Scan indexed function_definition slices and pick those that look like `static inline`.
        for k, spans in fn_index.items():
            try:
                name = k[0] if isinstance(k, tuple) and len(k) >= 1 else ""
            except Exception:
                name = ""
            if not name or name in names:
                continue
            for start_b, end_b in (spans or []):
                try:
                    start_b = int(start_b)
                    end_b = int(end_b)
                except Exception:
                    continue
                if start_b < 0 or end_b <= start_b or end_b > len(i_text):
                    continue
                snippet = i_text[start_b:end_b].strip()
                if not snippet:
                    continue
                # Avoid huge blobs; inline helpers are usually small.
                if len(snippet) > 8000:
                    continue
                if self._looks_like_static_inline_c_function(snippet):
                    names.add(str(name))
                    break

        self._preprocessed_static_inline_names_cache[p] = set(names)
        return set(names)

    def _maybe_replace_c_code_with_preprocessed_slices(self, functions: Dict[str, FunctionInfo]) -> None:
        """
        Replace `FunctionInfo.c_code` with the corresponding slice from the preprocessed `.i` TU (best-effort).
        """
        if not self._enable_preprocessed_function_slices:
            return
        if not functions or not self._functions_manifest_by_file:
            return

        # Group by TU (safe module name)
        by_safe: Dict[str, List[Tuple[str, FunctionInfo]]] = defaultdict(list)
        for func_file, fi in functions.items():
            by_safe[str(getattr(fi, "file_name", "") or "")].append((func_file, fi))

        replaced = 0
        total = 0

        try:
            require_tu = (os.environ.get("C2R_REQUIRE_TU_CLOSURE", "1") or "1").strip().lower() not in (
                "0",
                "false",
                "no",
            )
        except Exception:
            require_tu = True

        for safe_name, items in by_safe.items():
            if not safe_name:
                continue
            rec = self._tu_context_files.get(safe_name) if getattr(self, "_tu_context_files", None) else None
            pre = rec.get("preprocessed_file") if isinstance(rec, dict) else None
            if not pre:
                if require_tu and safe_name:
                    # Missing pinned `.i` means we cannot use macro-expanded truth input for this TU.
                    self._tu_closure_issues.setdefault(safe_name, "preprocessed_file_not_recorded")
                continue
            try:
                pre_path = Path(str(pre)).expanduser()
            except Exception:
                if require_tu and safe_name:
                    self._tu_closure_issues.setdefault(safe_name, "preprocessed_file_invalid_path")
                continue
            if not pre_path.exists():
                if require_tu and safe_name:
                    self._tu_closure_issues.setdefault(safe_name, f"preprocessed_file_missing: {pre_path}")
                continue

            # Load TU text + index (cached).
            i_text = self._preprocessed_text_cache.get(pre_path)
            if i_text is None:
                try:
                    i_text = pre_path.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    i_text = ""
                self._preprocessed_text_cache[pre_path] = i_text
            if not i_text:
                if require_tu and safe_name:
                    self._tu_closure_issues.setdefault(safe_name, "preprocessed_file_empty")
                continue

            fn_index = self._preprocessed_fn_index_cache.get(pre_path)
            if fn_index is None:
                fn_index = self._build_preprocessed_fn_index(pre_path, i_text)
                self._preprocessed_fn_index_cache[pre_path] = fn_index
            if not fn_index:
                if require_tu and safe_name:
                    self._tu_closure_issues.setdefault(safe_name, "preprocessed_fn_index_empty")
                continue

            local_total = len(items)
            local_replaced = 0

            for func_file, fi in items:
                total += 1
                meta = self._functions_manifest_by_file.get(func_file) or {}
                want_name = str(meta.get("name") or getattr(fi, "name", "") or "").strip()
                # NOTE: stage1 `functions_manifest.json` may store `source_file` as the *safe_file_name*
                # (e.g. "src_disk") instead of a real path like "disk.c". In that case, prefer TU context map.
                want_src = str(meta.get("source_file") or "").strip()
                want_base = ""
                try:
                    # Heuristic: treat as a file path only if it looks like it has an extension.
                    if want_src and Path(want_src).suffix:
                        want_base = Path(want_src).name
                except Exception:
                    want_base = ""
                if not want_base and isinstance(rec, dict):
                    mapped = rec.get("source_for_cc_abs") or rec.get("source_file_abs") or rec.get("source_file_rel")
                    if mapped:
                        try:
                            want_base = Path(str(mapped)).name
                        except Exception:
                            want_base = ""
                want_line = int(meta.get("start_line") or 0) if str(meta.get("start_line") or "").isdigit() else 0
                if not want_name or not want_base or not want_line:
                    continue

                key = (want_name, want_base, want_line)
                spans = fn_index.get(key) or []
                if not spans:
                    # Best-effort nearest-line fallback within the same file basename.
                    candidates = [
                        (k, v)
                        for k, v in fn_index.items()
                        if k and k[0] == want_name and (not want_base or k[1] == want_base)
                    ]
                    best_span = None
                    best_dist = None
                    for k, v in candidates:
                        try:
                            dist = abs(int(k[2]) - int(want_line))
                        except Exception:
                            continue
                        if best_dist is None or dist < best_dist:
                            best_dist = dist
                            best_span = v[0] if v else None
                    if best_span and best_dist is not None and best_dist <= 5:
                        spans = [best_span]

                # Fallback 2: 仅按函数名查找（忽略文件名）
                # 用于处理 manifest 记录的 source_file 与 .i 行号指令不一致的情况
                # 例如：manifest 记录 url.c 但实际代码在 url.h 中
                if not spans:
                    candidates = [(k, v) for k, v in fn_index.items() if k and k[0] == want_name]
                    if len(candidates) == 1:
                        # 唯一匹配，使用它
                        spans = [candidates[0][1][0]] if candidates[0][1] else []
                    elif len(candidates) > 1:
                        # 多个匹配，尝试找最接近的行号
                        best_span = None
                        best_dist = None
                        for k, v in candidates:
                            try:
                                dist = abs(int(k[2]) - int(want_line))
                            except Exception:
                                continue
                            if best_dist is None or dist < best_dist:
                                best_dist = dist
                                best_span = v[0] if v else None
                        if best_span:
                            spans = [best_span]

                if not spans:
                    continue
                start_b, end_b = spans[0]
                if start_b < 0 or end_b <= start_b or end_b > len(i_text):
                    continue
                sliced = i_text[start_b:end_b].strip()
                if not sliced:
                    continue
                fi.c_code = sliced
                replaced += 1
                local_replaced += 1

            if require_tu and safe_name and local_total and local_replaced < local_total:
                # Strict truth mode: every translated function must come from the pinned `.i` TU slice.
                # Otherwise we'd silently fall back to non-expanded C (macro/inline gaps).
                self._tu_closure_issues.setdefault(
                    safe_name,
                    f"preprocessed_function_slices_incomplete: {local_replaced}/{local_total}",
                )

        if total and replaced:
            logger.info(f"[PreprocessedSlices] 使用 .i 函数切片替换 C 代码: {replaced}/{total}")

    def _get_field_shim_manager(self) -> Optional["CAccessorShimManager"]:
        """
        Lazy init per-workdir shim manager.
        - Uses `project_src_dir` as the primary C header search root.
        """
        if not C_ACCESSOR_SHIMS_AVAILABLE:
            return None
        if self._field_shims is not None:
            return self._field_shims
        try:
            search_dirs = [self.project_src_dir]
            extra = os.environ.get("C2R_SHIM_C_SEARCH_DIRS", "").strip()
            if extra:
                for part in re.split(r"[,:]", extra):
                    p = part.strip()
                    if not p:
                        continue
                    try:
                        search_dirs.append(Path(p).expanduser().resolve())
                    except Exception:
                        continue

            self._field_shims = CAccessorShimManager(
                rust_project_root=self.work_dir,
                c_search_dirs=search_dirs,
            )
            return self._field_shims
        except Exception as e:
            logger.debug(f"初始化 C accessor shims 失败: {e}")
            self._field_shims = None
            return None

    @staticmethod
    def _strip_rust_type_name(type_expr: str) -> str:
        """
        Best-effort: reduce a Rust type expr to a likely C type name.
        Examples:
          `crate::types::Foo` -> Foo
          `*mut crate::types::Foo` -> Foo
          `&mut Foo` -> Foo
        """
        s = (type_expr or "").strip()
        if not s:
            return ""
        s = s.strip("`")
        # Drop refs/pointers
        s = re.sub(r"^\*const\s+", "", s)
        s = re.sub(r"^\*mut\s+", "", s)
        s = re.sub(r"^&(?:mut\s+)?", "", s)
        s = s.strip()
        # Drop common module prefixes
        for pref in ("crate::types::", "types::"):
            if s.startswith(pref):
                s = s[len(pref) :]
        # Last path segment
        if "::" in s:
            s = s.split("::")[-1]
        # Remove generic wrappers (best-effort)
        s = re.sub(r"<.*>$", "", s).strip()
        # Keep identifier-ish
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)$", s)
        return m.group(1) if m else s

    @staticmethod
    def _find_base_start_for_field(line: str, dot_index: int) -> int:
        """
        Heuristic backward scan to find the start of the base expression for `base.field`.
        Works well for common patterns:
          - x.field
          - (*p).field
          - a.b[c].field
          - foo().field
        """
        i = dot_index - 1
        # skip trailing whitespace before '.'
        while i >= 0 and line[i].isspace():
            i -= 1

        stack: List[str] = []
        pairs = {")": "(", "]": "[", "}": "{"}
        stops = set(";=,{")

        while i >= 0:
            ch = line[i]
            if stack:
                if ch in ")]}":
                    stack.append(ch)
                elif ch in "([{":
                    if stack and pairs.get(stack[-1]) == ch:
                        stack.pop()
                i -= 1
                continue

            if ch in ")]}":
                stack.append(ch)
                i -= 1
                continue

            # Stop at obvious expression separators (only at nesting level 0)
            if ch in stops:
                break
            i -= 1

        return max(0, i + 1)

    def _try_fix_missing_fields_with_c_shims(self, error_msg: str) -> bool:
        """
        Deterministic fix for E0609 (no field on type):
        - Generate a C shim returning the field address as `void*`.
        - Rewrite the failing Rust line to dereference that pointer (casts to `*mut _`).

        Returns True if any source file was modified.
        """
        if not error_msg:
            return False

        shim_mgr = self._get_field_shim_manager()
        if shim_mgr is None:
            return False

        # Capture blocks like:
        # error[E0609]: no field `foo` on type `Bar`
        #   --> src/file.rs:10:20
        pat = re.compile(
            r"error\\[E0609\\]:\\s+no field `(?P<field>[A-Za-z_][A-Za-z0-9_]*)` on type `(?P<ty>[^`]+)`.*?\\n\\s*-->\\s+(?P<file>[^:\\s]+):(?P<line>\\d+):(?P<col>\\d+)",
            re.DOTALL,
        )

        changed_any = False
        for m in pat.finditer(error_msg):
            field = m.group("field")
            ty_expr = m.group("ty")
            rel_file = m.group("file")
            line_no = int(m.group("line"))
            col_no = int(m.group("col"))

            c_type = self._strip_rust_type_name(ty_expr)
            if not c_type or not field:
                continue

            # Ensure shim exists (may fail if header cannot be found).
            shim = shim_mgr.ensure_field_ptr(c_type=c_type, field=field)
            if shim is None:
                continue

            rust_path = Path(rel_file)
            if not rust_path.is_absolute():
                rust_path = self.work_dir / rel_file
            if not rust_path.exists():
                continue

            try:
                lines = rust_path.read_text(encoding="utf-8", errors="ignore").splitlines(True)
            except Exception:
                continue
            if line_no < 1 or line_no > len(lines):
                continue

            original_line = lines[line_no - 1].rstrip("\n")
            if f".{field}" not in original_line:
                continue

            # Locate the target occurrence using column if possible.
            idx = max(0, col_no - 1)  # 0-based column
            # rustc points at field ident; find '.' before it
            dot_index = original_line.rfind(".", 0, min(len(original_line), idx + 1))
            if dot_index < 0:
                dot_index = original_line.find(f".{field}")
            if dot_index < 0:
                continue

            # Base expr extraction
            base_start = self._find_base_start_for_field(original_line, dot_index)
            base_expr = original_line[base_start:dot_index].strip()
            if not base_expr:
                continue

            field_end = dot_index + 1 + len(field)
            prefix = original_line[:base_start]
            suffix = original_line[field_end:]

            base_ptr = f"std::ptr::addr_of!({base_expr}) as *mut ::core::ffi::c_void"
            call = f"crate::compat::{shim.function}({base_ptr})"

            # If this is a simple assignment statement with only indentation before base,
            # rewrite the whole line so the LHS remains a place expression.
            assign_m = re.match(r"\s*(<<=|>>=|\+=|-=|\*=|/=|%=|&=|\|=|\^=|=(?![=>]))", suffix)
            indent_only = prefix.strip() == ""
            if assign_m and indent_only:
                op = assign_m.group(1)
                rhs = suffix[assign_m.end():].lstrip()
                new_line = f"{prefix}unsafe {{ *({call} as *mut _) {op} {rhs} }}"
            else:
                replacement = f"unsafe {{ *({call} as *mut _) }}"
                new_line = prefix + replacement + suffix

            if new_line.rstrip("\n") != original_line:
                lines[line_no - 1] = new_line + "\n"
                try:
                    rust_path.write_text("".join(lines), encoding="utf-8")
                    changed_any = True
                except Exception:
                    continue

        return changed_any

    def _extract_opaque_field_accesses(self, c_code: str, opaque_types: Set[str]) -> List[Tuple[str, str]]:
        """
        分析 C 代码，提取对 opaque 类型的字段访问

        返回: [(type_name, field_name), ...]
        """
        if not c_code or not opaque_types:
            return []

        field_accesses = []

        # 常见的 C 变量声明模式: TypeName* varname; 或 struct TypeName* varname;
        # 然后查找 varname->field 或 varname.field
        var_type_map = {}  # var_name -> type_name

        # 模式1: 直接类型声明 (TypeName* var, TypeName var, const TypeName* var)
        decl_pat = re.compile(
            r'(?:const\s+)?(?:struct\s+)?(' + '|'.join(re.escape(t) for t in opaque_types) + r')\s*\*?\s+(\w+)',
            re.IGNORECASE
        )
        for m in decl_pat.finditer(c_code):
            type_name = m.group(1)
            var_name = m.group(2)
            # 查找与 opaque_types 匹配的类型（忽略大小写）
            for ot in opaque_types:
                if ot.lower() == type_name.lower():
                    var_type_map[var_name] = ot
                    break

        # 模式2: 函数参数 (const TypeName* param)
        param_pat = re.compile(
            r'(?:const\s+)?(?:struct\s+)?(' + '|'.join(re.escape(t) for t in opaque_types) + r')\s*\*\s*(\w+)\s*[,)]',
            re.IGNORECASE
        )
        for m in param_pat.finditer(c_code):
            type_name = m.group(1)
            var_name = m.group(2)
            for ot in opaque_types:
                if ot.lower() == type_name.lower():
                    var_type_map[var_name] = ot
                    break

        # 模式3: 查找 var->field 或 (*var).field 访问
        for var_name, type_name in var_type_map.items():
            # var->field
            arrow_pat = re.compile(rf'\b{re.escape(var_name)}\s*->\s*(\w+)')
            for m in arrow_pat.finditer(c_code):
                field_accesses.append((type_name, m.group(1)))

            # (*var).field
            deref_pat = re.compile(rf'\(\s*\*\s*{re.escape(var_name)}\s*\)\s*\.\s*(\w+)')
            for m in deref_pat.finditer(c_code):
                field_accesses.append((type_name, m.group(1)))

        # 模式4: 直接类型转换后访问 ((TypeName*)ptr)->field
        for ot in opaque_types:
            cast_pat = re.compile(
                rf'\(\s*(?:struct\s+)?{re.escape(ot)}\s*\*\s*\)\s*\w+\s*->\s*(\w+)',
                re.IGNORECASE
            )
            for m in cast_pat.finditer(c_code):
                field_accesses.append((ot, m.group(1)))

        # 去重并返回
        return list(set(field_accesses))

    def _generate_accessor_shims_proactively(
        self,
        func_info: "FunctionInfo",
        opaque_types: Set[str]
    ) -> List[Tuple[str, str, str]]:
        """
        在翻译前主动生成 accessor shims

        返回: [(type_name, field_name, shim_function_name), ...]
        """
        if not C_ACCESSOR_SHIMS_AVAILABLE:
            return []

        shim_mgr = self._get_field_shim_manager()
        if shim_mgr is None:
            return []

        # 从 C 代码提取字段访问
        c_code = func_info.c_code or ""
        field_accesses = self._extract_opaque_field_accesses(c_code, opaque_types)

        if not field_accesses:
            return []

        generated_shims = []
        for type_name, field_name in field_accesses:
            try:
                shim = shim_mgr.ensure_field_ptr(c_type=type_name, field=field_name)
                if shim:
                    generated_shims.append((type_name, field_name, shim.function))
                    logger.debug(f"生成 accessor shim: {shim.function} for {type_name}.{field_name}")
            except Exception as e:
                logger.debug(f"生成 accessor shim 失败 ({type_name}.{field_name}): {e}")

        return generated_shims

    def _build_accessor_shim_hints(self, shims: List[Tuple[str, str, str]]) -> str:
        """
        构建 accessor shim 使用说明，用于 LLM 提示
        """
        if not shims:
            return ""

        hints = ["""
## ACCESSOR SHIMS AVAILABLE (for opaque type field access)

The following C accessor shim functions have been generated to help you access fields of opaque types.
These are declared in `crate::compat` and can be called via FFI.

**USAGE PATTERN:**
Instead of directly accessing `obj->field` (which will fail for opaque types), use:
```rust
// Get field address via C shim, then dereference
let field_ptr = unsafe {
    crate::compat::c2r_field_ptr_TypeName__fieldName(
        obj as *mut ::core::ffi::c_void
    )
};
let field_value = unsafe { *(field_ptr as *const FieldType) };

// Or for assignment:
unsafe {
    *(crate::compat::c2r_field_ptr_TypeName__fieldName(obj as *mut ::core::ffi::c_void) as *mut FieldType) = new_value;
}
```

**AVAILABLE SHIMS:**
"""]

        for type_name, field_name, func_name in shims:
            hints.append(f"- `crate::compat::{func_name}(base: *mut c_void) -> *mut c_void` - Access `{type_name}.{field_name}`")

        hints.append("""
**IMPORTANT:**
1. Cast the struct pointer to `*mut ::core::ffi::c_void` before passing to shim
2. Cast the returned `*mut c_void` to the appropriate field type pointer
3. Wrap all shim calls in `unsafe {{ }}`
""")

        return "\n".join(hints)

    def _init_call_graph_context(self):
        """初始化调用图上下文提供器"""
        try:
            from call_graph import CallGraphBuilder, LLMContextProvider
            
            # 尝试加载预构建的调用图
            # 优先使用 source_skeletons 中的调用图（包含 signature/body），其次使用 extracted 中的调用图
            call_graph_path = self.source_skeleton_dir / "call_graph.json"
            fallback_path = self.dependencies_dir.parent / "call_graph.json"
            if not call_graph_path.exists() and fallback_path.exists():
                call_graph_path = fallback_path
            
            if call_graph_path.exists():
                call_graph = CallGraphBuilder.load(call_graph_path)
                self.call_graph_context_provider = LLMContextProvider(call_graph)
                logger.info(f"已加载调用图上下文: {call_graph.stats.total_functions} 个函数 ({call_graph_path})")
                try:
                    sig_count = sum(1 for info in call_graph.functions.values() if getattr(info, "signature", ""))
                    if call_graph.stats.total_functions > 0 and sig_count == 0:
                        logger.info("调用图未包含函数签名（上下文增强将较弱）：建议使用 source_skeletons/call_graph.json")
                except Exception:
                    pass
            else:
                logger.debug(f"调用图文件不存在: {call_graph_path} / {fallback_path}")
        except ImportError:
            logger.debug("call_graph 模块不可用，跳过调用图上下文")
        except Exception as e:
            logger.warning(f"初始化调用图上下文失败: {e}")

    def _load_deterministic_signature_mapping(self):
        """
        加载确定性的函数签名映射（阶段1生成）

        从 source_skeletons/<project>/func_file_to_rust_sig.json 加载映射
        这避免了使用BM25猜测C函数到Rust签名的对应关系
        """
        mapping_path = self.source_skeleton_dir / "func_file_to_rust_sig.json"

        if not mapping_path.exists():
            logger.debug(f"确定性签名映射文件不存在: {mapping_path}")
            logger.debug("提示：运行 'python generate_signature_mappings.py' 生成映射文件")
            return

        try:
            with open(mapping_path, 'r', encoding='utf-8') as f:
                self.func_file_to_rust_sig = json.load(f)

            # 统计有效映射数量
            valid_count = sum(1 for sig in self.func_file_to_rust_sig.values() if sig is not None)
            logger.info(f"加载确定性签名映射: {valid_count}/{len(self.func_file_to_rust_sig)} 个有效映射")
            print(f"  ✓ 加载确定性签名映射: {valid_count} 个函数")
        except Exception as e:
            logger.warning(f"加载确定性签名映射失败: {e}")
            self.func_file_to_rust_sig = {}

    def _load_signature_from_mapping(self, func_name: str) -> Optional[str]:
        """
        从确定性映射文件中加载函数签名

        Args:
            func_name: 函数文件名（例如 "src_disk_12"）

        Returns:
            Rust 签名，如果未找到则返回 None
        """
        if not self.func_file_to_rust_sig:
            return None

        # func_name 就是 func_file（例如 "src_disk_12"）
        return self.func_file_to_rust_sig.get(func_name)

    def _select_target_func_files(self) -> Optional[Set[str]]:
        """
        选择本次增量翻译应处理的函数集合（func_file 级别）。

        目标：
        - 避免对大量“无 Rust 签名/不可注入”的函数做无意义的 LLM 调用与编译尝试
        - 让增量翻译的函数集合与阶段2(签名匹配)/分层骨架保持一致
        """
        # 1) 优先使用确定性映射（如果存在）
        if self.func_file_to_rust_sig:
            targets = {k for k, v in self.func_file_to_rust_sig.items() if v}
            if targets:
                return targets

        # 2) 其次使用签名匹配阶段产物（signature_matches/<project>/*.txt）
        try:
            sig_files = list(self.signature_dir.glob("*.txt"))
            if sig_files:
                return {p.stem for p in sig_files}
        except Exception as e:
            logger.debug(f"读取签名匹配目录失败，跳过过滤: {e}")

        # 3) 回退：从 functions_manifest.json 获取项目自有函数
        #    （排除头文件内联的辅助函数如 DListHeadInit 等）
        try:
            import json as _json
            manifest_candidates = [
                self.functions_dir.parent / "functions_manifest.json",
                self.dependencies_dir.parent / "functions_manifest.json",
            ]
            for mp in manifest_candidates:
                if mp.is_file():
                    mf = _json.loads(mp.read_text(encoding="utf-8"))
                    func_files = {f["func_file"].replace(".txt", "")
                                  for f in mf.get("functions", [])
                                  if f.get("func_file")}
                    if func_files:
                        logger.info(
                            "[target_select] 从 manifest 获取目标函数: "
                            f"{len(func_files)} 个")
                        return func_files
                    break
        except Exception as e:
            logger.debug(f"从 manifest 读取目标函数失败: {e}")

        return None
    
    def _get_call_graph_context(self, func_info: FunctionInfo) -> str:
        """
        获取函数的调用图上下文（被调用函数的签名或已翻译代码）
        
        Args:
            func_info: 函数信息
            
        Returns:
            调用上下文字符串
        """
        if not self.call_graph_context_provider:
            return ""
        
        try:
            context = self.call_graph_context_provider.get_context_for_translation(
                func_info.name,
                file_hint=func_info.file_name
            )
            return context
        except Exception as e:
            logger.debug(f"获取调用上下文失败 {func_info.name}: {e}")
            return ""

    def _build_internal_callee_module_path_hints(self, func_info: FunctionInfo) -> str:
        """
        Deterministic symbol resolution hint for internal (in-crate) callees.

        We use the call-graph edges (func_file dependencies) + per-function metadata to provide
        `callee_name -> crate::<module>::<callee_name>` mappings to the LLM.
        This is NOT heuristic “rule fixing”; it is a stable mapping from extracted truth.
        """
        try:
            callee_func_files = set(func_info.dependencies or set())
        except Exception:
            callee_func_files = set()
        if not callee_func_files:
            return ""

        if not self._functions_by_func_file:
            return ""

        lines: List[str] = []
        for callee_func_file in sorted(callee_func_files):
            callee = self._functions_by_func_file.get(callee_func_file)
            if not callee:
                continue
            callee_name = (callee.name or "").strip()
            callee_mod = (callee.file_name or "").strip()
            if not callee_name or not callee_mod:
                continue
            lines.append(f"{callee_func_file} ({callee_name}) -> crate::{callee_mod}::{callee_name}")

        if not lines:
            return ""

        # Keep it compact: direct callees only, limited lines.
        # 强化提示：明确告诉 LLM 必须使用完整路径
        return (
            "\n## Internal Callees: Deterministic Rust Module Paths (CRITICAL - E0425 FIX)\n"
            "**⚠️ YOU MUST USE THE FULL PATH when calling these functions!**\n"
            "- WRONG: `ZopfliCalculateEntropy(...)` → ERROR: E0425 cannot find function\n"
            "- CORRECT: `crate::src_tree::ZopfliCalculateEntropy(...)`\n\n"
            "Mapping (use the RIGHT side path):\n"
            "```text\n"
            + "\n".join(lines[:80])
            + "\n```\n"
        )

    def _ensure_internal_symbol_module_index(self) -> None:
        """
        Build a best-effort index of in-crate *public* functions: `symbol -> {module_stem}`.
        Used for deterministic repair of E0425 "not found in this scope" by qualifying call sites.
        """
        if getattr(self, "_internal_symbol_index_built", False):
            return
        src_dir = self.work_dir / "src"
        if not src_dir.exists():
            self._internal_symbol_to_modules = {}
            self._internal_symbol_index_built = True
            return

        special_files = {"types.rs", "globals.rs", "compat.rs", "main.rs", "lib.rs", "compatibility.rs"}
        idx: Dict[str, Set[str]] = defaultdict(set)
        for rs_file in src_dir.glob("*.rs"):
            if rs_file.name in special_files:
                continue
            try:
                mod = rs_file.stem
                text = rs_file.read_text(encoding="utf-8", errors="ignore")
                # Keep only callable public functions (avoid private items; those would be E0603, not E0425).
                for m in re.finditer(
                    r'(?m)^\s*pub(?:\s*\(crate\))?\s+(?:unsafe\s+)?(?:extern\s+"C"\s+)?fn\s+([A-Za-z_]\w*)\b',
                    text,
                ):
                    name = (m.group(1) or "").strip()
                    if name:
                        idx[name].add(mod)
            except Exception:
                continue
        self._internal_symbol_to_modules = dict(idx)
        self._internal_symbol_index_built = True

    def _try_fix_missing_internal_symbols(self, cargo_err: str) -> bool:
        """
        Deterministic fix for E0425 (missing function in scope):
        - If the missing symbol is defined in another in-crate module (pub fn),
          rewrite call sites to `crate::<module>::<symbol>(...)`.

        This avoids spending LLM attempts on simple missing-import issues.
        """
        if not cargo_err:
            return False

        lines = (cargo_err or "").splitlines()
        missing: List[Tuple[str, str]] = []  # (symbol, file_rel)
        for i, ln in enumerate(lines):
            m = re.search(
                r"error\[E0425\]:\s+cannot find function, tuple struct or tuple variant `([^`]+)` in this scope",
                ln,
            )
            if not m:
                continue
            name = (m.group(1) or "").strip()
            if not name:
                continue
            file_rel = ""
            for j in range(i + 1, min(i + 14, len(lines))):
                mm = re.search(r"-->\s+([^\s:]+):\d+:\d+", lines[j])
                if mm:
                    file_rel = (mm.group(1) or "").strip()
                    break
            if not file_rel:
                continue
            # Don't patch generated fallback modules.
            if "src/__c2r_generated/" in file_rel or "src\\__c2r_generated\\" in file_rel:
                continue
            if not (file_rel.startswith("src/") or Path(file_rel).is_absolute()):
                continue
            missing.append((name, file_rel))

        if not missing:
            return False

        self._ensure_internal_symbol_module_index()
        if not self._internal_symbol_to_modules:
            return False

        by_file: Dict[str, Set[str]] = defaultdict(set)
        for name, file_rel in missing:
            by_file[file_rel].add(name)

        changed_any = False
        for file_rel, names in by_file.items():
            file_path = Path(file_rel)
            if not file_path.is_absolute():
                file_path = self.work_dir / file_rel
            if not file_path.exists() or not file_path.is_file():
                continue
            try:
                text = file_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            new_text = text

            for name in sorted(names):
                mods = self._internal_symbol_to_modules.get(name) or set()
                # Ambiguous (same symbol defined in multiple modules): skip to avoid incorrect rewrites.
                if len(mods) != 1:
                    continue
                mod = next(iter(mods))
                qual = f"crate::{mod}::{name}"
                # Replace only unqualified call sites (`Name(...)`), not definitions (`fn Name(...)`) nor paths (`foo::Name(...)`).
                pat = re.compile(rf"(?<!::)(?<!fn )\b{re.escape(name)}\b(\s*)\(")
                new_text = pat.sub(lambda mm: f"{qual}{mm.group(1)}(", new_text)

            if new_text != text:
                try:
                    file_path.write_text(new_text, encoding="utf-8", errors="ignore")
                    changed_any = True
                except Exception:
                    continue

        return changed_any
    
    def _register_translated_function(self, func_info: FunctionInfo, rust_code: str):
        """
        注册已翻译的函数（用于为后续翻译提供上下文）
        
        Args:
            func_info: 函数信息
            rust_code: 翻译后的 Rust 代码
        """
        if self.call_graph_context_provider:
            try:
                self.call_graph_context_provider.register_translated(
                    func_info.name,
                    rust_code,
                    file_hint=func_info.file_name
                )
            except Exception as e:
                logger.debug(f"注册已翻译函数失败 {func_info.name}: {e}")
    
    def run(self) -> bool:
        """
        运行增量式翻译流程
        
        返回: 是否全部成功
        """
        print(f"\n{'='*60}")
        print(f"增量式翻译开始")
        print(f"{'='*60}")
        print(f"项目: {self.project_name}")
        print(f"LLM: {self.llm_name}")
        print(f"{'='*60}\n")
        
        # 1. 分析调用图并获取拓扑排序
        print("步骤 1: 分析调用图...")
        target_funcs = self._select_target_func_files()
        analyzer = CallGraphAnalyzer(self.functions_dir, self.dependencies_dir, target_func_files=target_funcs)
        sorted_functions = analyzer.analyze()
        if target_funcs:
            print(f"  (已启用目标函数过滤: {len(target_funcs)} 个 func_file)")
            if len(sorted_functions) == 0:
                logger.warning("目标函数过滤后为空，回退到全量函数列表（可能缺少签名映射）")
                analyzer = CallGraphAnalyzer(self.functions_dir, self.dependencies_dir)
                sorted_functions = analyzer.analyze()

        # Save the full func_file -> FunctionInfo mapping for deterministic prompt hints.
        self._functions_by_func_file = dict(analyzer.functions or {})

        # Best-effort: use macro-expanded `.i` function slices as the primary C source for translation/repair.
        try:
            self._maybe_replace_c_code_with_preprocessed_slices(analyzer.functions)
        except Exception as e:
            logger.debug(f"[PreprocessedSlices] 替换 .i 函数切片失败（已忽略）: {e}")

        # TU-closure gate (strict): if preprocessing is incomplete or the TU map can't be applied,
        # do NOT proceed to translation (otherwise we'd silently translate from non-expanded C).
        try:
            require_tu = (os.environ.get("C2R_REQUIRE_TU_CLOSURE", "1") or "1").strip().lower() not in (
                "0",
                "false",
                "no",
            )
        except Exception:
            require_tu = True
        if require_tu:
            # Strict mode requires stage1 TU mapping; otherwise we'd have to re-pick a TU/profile (non-truth).
            if not getattr(self, "_tu_context_files", None):
                print("\n✗ 未发现/未能加载 tu_context_map.json（C2R_REQUIRE_TU_CLOSURE=1），跳过本项目以保持 TU 真值一致性")
                self._print_tu_context_gap_summary()
                return False
            tc = getattr(self, "_tu_context_gap_summary", None)
            counts = (tc or {}).get("counts") if isinstance(tc, dict) else None
            if isinstance(counts, dict):
                miss = int(counts.get("compile_commands_entry_missing", 0) or 0)
                pre_failed = int(counts.get("preprocessed_failed", 0) or 0)
                pre_missing = int(counts.get("preprocessed_file_missing", 0) or 0)
                if miss > 0 or pre_failed > 0 or pre_missing > 0:
                    print("\n✗ TU 闭包不完整（缺 TU 条目/预处理失败/.i 缺失），跳过本项目以保持 TU 真值一致性")
                    self._print_tu_context_gap_summary()
                    return False
            if getattr(self, "_tu_closure_issues", None):
                print("\n✗ TU 闭包不完整（.i 缺失/映射不一致），跳过本项目以保持 TU 真值一致性")
                self._print_tu_context_gap_summary()
                return False
        self.stats["total"] = len(sorted_functions)
        print(f"  找到 {len(sorted_functions)} 个函数")
        print(f"  翻译顺序: 自底向上（叶子函数优先）")
        
        # 2. 准备工作目录（复制骨架）
        print("\n步骤 2: 准备工作目录...")
        self._prepare_work_dir()
        
        # 2.5 验证骨架文件内容
        print("\n  验证骨架文件...")
        self._validate_skeleton_files()

        # 2.55 预生成外部 C API 声明（FFI prelude）
        # 目的：避免在函数体注入后出现大量 E0425（缺失外部符号声明），把“声明存在”前置为确定性步骤。
        try:
            predecl_count = self._ensure_extern_decls_from_bindgen_allowlist(analyzer.functions)
            used_string_fallback = False
            if (not predecl_count) and self._enable_extern_decls_string_fallback:
                # Optional fallback: legacy string-mapping from `.i` prototypes (less accurate, but may unblock).
                used_string_fallback = True
                predecl_count = self._ensure_extern_decls_from_preprocessed(analyzer.functions)
            if predecl_count:
                extra = f"，详情: {self._extern_decls_report_path}" if self._extern_decls_report_path else ""
                src = "string fallback" if used_string_fallback else "bindgen allowlist"
                warn = ""
                try:
                    if self._extern_decls_missing:
                        warn = f"（bindgen 缺失 {len(self._extern_decls_missing)} 个 TU 记录）"
                except Exception:
                    warn = ""
                print(f"\n步骤 2.55: 预生成外部 C API 声明... ✓ {predecl_count} 个（{src}；已写入 compat.rs{extra}）{warn}")
            else:
                # Even if we didn't inject any decls, bindgen may have recorded "missing closure" diagnostics.
                if self._extern_decls_report_path and self._extern_decls_missing:
                    print(
                        f"\n步骤 2.55: 预生成外部 C API 声明... ⚠ bindgen 未能生成所需 extern 声明（见 {self._extern_decls_report_path}）"
                    )
        except Exception as e:
            logger.warning(f"预生成 extern 声明失败（已忽略，回退到按错误修复/LLM 处理）: {e}")

        # 2.6 基线编译检查：如果骨架工程本身无法通过 cargo check，
        # 后续对每个函数的“修复”都会被同一个全局错误阻塞，导致修复次数爆炸并产生大量无效 LLM 请求。
        print("\n步骤 2.6: 基线编译检查...")
        baseline_ok = self._compile_project()
        if not baseline_ok:
            baseline_error = self._get_compile_error()
            print("  ✗ 基线编译失败：骨架工程本身无法编译，跳过函数翻译以避免无效修复循环")
            if baseline_error:
                print(baseline_error.rstrip("\n"))
            try:
                err_path = self.work_dir / "baseline_compile_error.log"
                _atomic_write_text(err_path, baseline_error or "")
                print(f"  完整错误已保存: {err_path}")
            except Exception:
                pass

            allow = os.environ.get("C2R_ALLOW_BASELINE_COMPILE_FAILURE", "0").strip().lower() in ("1", "true", "yes")
            if not allow:
                # 输入闭包诊断：即使基线编译失败，也尽量输出 closure 缺口摘要帮助定位
                self._print_tu_context_gap_summary()
                return False
            print("  ⚠️ 已设置 C2R_ALLOW_BASELINE_COMPILE_FAILURE=1，继续执行（可能产生大量失败/修复尝试）")
        
        # 3. 提取骨架上下文
        print("\n步骤 3: 提取骨架上下文...")
        # 默认启用 types.rs slice：不把整个 types.rs 塞进每次 prompt（过大且导致 vLLM 超时/排队）
        # - non-types context: 共享（globals.rs + 其他 .rs 的签名/小型定义）
        # - types.rs: 每个函数按需切片（TypesRsRegistry）
        skeleton_context = self._extract_skeleton_context(include_types_rs_defs=not self._enable_types_slice)
        self._skeleton_non_types_context = skeleton_context
        if self._enable_types_slice:
            self._reload_types_registry()
            if self._types_registry:
                logger.info(
                    f"[TypesSlice] non-types context: {len(self._skeleton_non_types_context)} chars, "
                    f"types symbols: {len(self._types_registry.names)}"
                )
        
        # 4. 增量翻译与验证
        print("\n步骤 4: 增量翻译与验证...")
        
        # 检查是否启用并行翻译（从环境变量读取）
        enable_parallel = os.environ.get("PARALLEL_TRANSLATE", "1").lower() in ("1", "true", "yes")
        max_parallel_workers = int(os.environ.get("MAX_PARALLEL_WORKERS", "4"))
        
        if enable_parallel and len(sorted_functions) > 3:
            # 分层并行翻译模式
            layers = analyzer.get_parallel_layers()
            print(f"  启用分层并行翻译: {len(layers)} 层, 最大 {max_parallel_workers} 并行")
            
            # 统计每层函数数量
            layer_sizes = [len(layer) for layer in layers]
            print(f"  各层函数数量: {layer_sizes}")
            
            processed = 0
            for layer_idx, layer in enumerate(layers, 1):
                print(f"\n--- 第 {layer_idx}/{len(layers)} 层 ({len(layer)} 个函数) ---")
                
                if len(layer) == 1:
                    # 单个函数，串行处理
                    func_name = layer[0]
                    processed += 1
                    print(f"\n[{processed}/{len(sorted_functions)}] 处理: {func_name}")
                    
                    success = self._process_function(
                        func_name,
                        analyzer.functions[func_name],
                        skeleton_context
                    )
                    if not getattr(analyzer.functions[func_name], "skipped", False):
                        if success:
                            self.stats["compiled"] += 1
                            print(f"  ✓ 编译通过")
                        else:
                            self.stats["failed"] += 1
                            print(f"  ✗ 编译失败（已回退）")
                else:
                    # 多个函数，并行翻译（但串行注入和编译以避免文件冲突）
                    # 第一步：并行调用 LLM 获取翻译结果
                    from concurrent.futures import ThreadPoolExecutor, as_completed
                    
                    print(f"  并行翻译 {len(layer)} 个函数（LLM 调用）...")
                    
                    # 并行翻译函数（只调用 LLM，不注入）
                    translation_results = {}
                    with ThreadPoolExecutor(max_workers=min(max_parallel_workers, len(layer))) as executor:
                        future_to_func = {}
                        for func_name in layer:
                            func_info = analyzer.functions[func_name]
                            future = executor.submit(
                                self._translate_function_only,
                                func_name,
                                func_info,
                                skeleton_context
                            )
                            future_to_func[future] = func_name
                        
                        for future in as_completed(future_to_func):
                            func_name = future_to_func[future]
                            try:
                                translated_code = future.result()
                                translation_results[func_name] = translated_code
                            except Exception as e:
                                logger.error(f"并行翻译 {func_name} 失败: {e}")
                                translation_results[func_name] = None
                    
                    # 第二步：串行注入和编译（避免文件冲突）
                    print(f"  串行注入和编译 {len(layer)} 个函数...")
                    for func_name in layer:
                        processed += 1
                        print(f"\n[{processed}/{len(sorted_functions)}] 注入: {func_name}")
                        
                        translated_code = translation_results.get(func_name)
                        func_info = analyzer.functions[func_name]
                        if translated_code is None:
                            # When translation-only stage decides to skip (e.g., no Rust signature),
                            # it sets `func_info.skipped = True` but returns None. Treat it as skipped
                            # (NOT failed), otherwise the summary will be misleading.
                            if getattr(func_info, "skipped", False):
                                try:
                                    self.stats["skipped"] += 1
                                except Exception:
                                    pass
                                reason = getattr(func_info, "skip_reason", "") or ""
                                if reason:
                                    print(f"  [跳过] {reason}")
                                else:
                                    print("  [跳过]")
                            else:
                                self.stats["failed"] += 1
                                print(f"  ✗ 翻译失败")
                            continue
                        
                        # 注入并编译
                        success = self._inject_and_compile_with_repair(
                            func_name,
                            func_info,
                            translated_code,
                            skeleton_context
                        )
                        
                        if getattr(func_info, "skipped", False):
                            continue
                        
                        if success:
                            self.stats["compiled"] += 1
                            print(f"  ✓ 编译通过")
                        else:
                            self.stats["failed"] += 1
                            print(f"  ✗ 编译失败（已回退）")
        else:
            # 串行翻译模式（原有逻辑）
            for i, func_name in enumerate(sorted_functions, 1):
                print(f"\n[{i}/{len(sorted_functions)}] 处理: {func_name}")
                
                success = self._process_function(
                    func_name,
                    analyzer.functions[func_name],
                    skeleton_context
                )
                if getattr(analyzer.functions[func_name], "skipped", False):
                    continue
                
                if success:
                    self.stats["compiled"] += 1
                    print(f"  ✓ 编译通过")
                else:
                    self.stats["failed"] += 1
                    print(f"  ✗ 编译失败（已回退）")

        # 4.5 重试超时失败的函数（最多重试2轮）
        max_retry_rounds = int(os.environ.get("MAX_TIMEOUT_RETRY_ROUNDS", "2"))
        for retry_round in range(max_retry_rounds):
            if not self.timeout_failed_funcs:
                break

            retry_funcs = list(set(self.timeout_failed_funcs))  # 去重
            self.timeout_failed_funcs.clear()

            print(f"\n--- 超时重试第 {retry_round + 1}/{max_retry_rounds} 轮 ({len(retry_funcs)} 个函数) ---")
            logger.info(f"超时重试第 {retry_round + 1} 轮: {len(retry_funcs)} 个函数")

            for func_file_name in retry_funcs:
                # 在 analyzer.functions 中查找对应函数
                func_info = None
                for fn, fi in analyzer.functions.items():
                    if fi.file_name == func_file_name:
                        func_info = fi
                        func_name = fn
                        break

                if not func_info:
                    logger.warning(f"重试时找不到函数: {func_file_name}")
                    continue

                print(f"\n[重试] 处理: {func_name}")
                success = self._process_function(func_name, func_info, skeleton_context)

                if not getattr(func_info, "skipped", False):
                    if success:
                        self.stats["compiled"] += 1
                        self.stats["failed"] -= 1  # 从失败计数中减去
                        print(f"  ✓ 重试成功")
                    else:
                        print(f"  ✗ 重试失败")

            if self.timeout_failed_funcs:
                print(f"  本轮仍有 {len(self.timeout_failed_funcs)} 个函数超时")

        # 5. 自动填充 trait impl 方法（如果对应的独立函数已翻译）
        print("\n步骤 5: 自动填充 trait impl 方法...")
        self._fill_trait_impl_methods()
        
        # 6. 统计功能完整性（检查 unimplemented!() 占位符）
        print("\n步骤 6: 保存统计信息...")
        self._count_unimplemented_functions()
        self._save_translation_stats()
        
        # 7. 保存最终结果
        print("\n步骤 7: 保存最终结果...")
        save_success = self._save_final_project()
        if not save_success:
            print(f"  ⚠ 警告: 保存最终项目失败，但继续输出统计信息")
            logger.warning("保存最终项目失败")
        
        # 8. 输出统计
        print(f"\n{'='*60}")
        print(f"增量式翻译完成")
        print(f"{'='*60}")
        print(f"总函数数: {self.stats['total']}")
        print(f"翻译成功: {self.stats['translated']}")
        print(f"编译通过: {self.stats['compiled']}")
        print(f"修复成功: {self.stats['repaired']}")
        print(f"最终失败: {self.stats['failed']}")
        print(f"跳过: {self.stats['skipped']}")
        print(f"注入失败: {self.stats['injection_failed']}")
        if self.stats.get('timeout_failed', 0) > 0:
            print(f"⚠️ LLM超时: {self.stats['timeout_failed']} (已尝试重试)")

        # 功能完整性检查
        still_unimplemented = self.stats.get('still_unimplemented', 0)
        if still_unimplemented > 0:
            print(f"⚠️ 仍未实现: {still_unimplemented} (编译通过但函数体是占位符)")
        
        if self.stats['total'] > 0:
            success_rate = (self.stats['compiled'] / self.stats['total']) * 100
            # 计算真正的功能完整率（排除仍是占位符的函数）
            functional_count = self.stats['compiled'] - still_unimplemented
            if functional_count < 0:
                functional_count = 0
            functional_rate = (functional_count / self.stats['total']) * 100
            print(f"编译成功率: {success_rate:.1f}%")
            if still_unimplemented > 0:
                print(f"功能完整率: {functional_rate:.1f}% (实际实现了功能的函数)")
        else:
            success_rate = 0
            print("成功率: N/A (总函数数为 0)")
        print(f"{'='*60}\n")

        # 输入闭包诊断：compile_commands/预处理产物缺失等（不是翻译 bug）
        self._print_tu_context_gap_summary()
        
        # 成功判断：
        # - 完全成功 (返回 2): 最终项目能编译 + 所有函数都成功 + 无占位符函数体（unimplemented!）
        # - 部分成功 (返回 1): 最终项目能编译 + 但有函数失败/跳过/注入失败/仍为占位符
        # - 失败 (返回 0): 最终项目无法编译
        final_compile_success = self._compile_project()

        still_unimplemented = int(self.stats.get("still_unimplemented", 0) or 0)
        skipped = int(self.stats.get("skipped", 0) or 0)
        injection_failed = int(self.stats.get("injection_failed", 0) or 0)

        if final_compile_success and self.stats["failed"] == 0 and still_unimplemented == 0 and skipped == 0 and injection_failed == 0:
            print("✓ 完全成功：所有函数已实现且项目编译通过")
            return 2  # 完全成功
        elif final_compile_success:
            print(
                f"⚠ 部分成功：failed={self.stats['failed']}, still_unimplemented={still_unimplemented}, "
                f"skipped={skipped}, injection_failed={injection_failed}（项目仍可编译）"
            )
            return 1  # 部分成功
        else:
            print("✗ 失败：项目无法编译")
            return 0  # 失败
    
    def _prepare_work_dir(self):
        """准备工作目录"""
        if self.work_dir.exists():
            shutil.rmtree(self.work_dir)
        shutil.copytree(self.skeleton_dir, self.work_dir)
        print(f"  工作目录: {self.work_dir}")
        # Load RustMap-style globals accessors map (if any) for deterministic rewrite.
        self._load_globals_accessors()
        # 构建函数位置索引（用于确定性注入）
        self._build_function_location_index()

    def _load_globals_accessors(self) -> None:
        """
        加载骨架阶段生成的 globals_accessors.json（RustMap safe globals 映射）。

        该文件由 skeleton_builder.py 在生成 globals.rs 时写入：src/globals_accessors.json
        """
        self.safe_globals = {}
        try:
            p = self.work_dir / "src" / "globals_accessors.json"
            if not p.exists():
                return
            data = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
            safe_list = data.get("safe_globals") or []
            if isinstance(safe_list, list):
                for item in safe_list:
                    name = (item or {}).get("name")
                    if isinstance(name, str) and name:
                        self.safe_globals[name] = dict(item)
        except Exception as e:
            logger.debug(f"读取 globals_accessors.json 失败: {e}")

    def _build_function_location_index(self) -> Dict[str, FunctionLocation]:
        """
        使用 tree-sitter 扫描骨架 .rs 文件，建立函数位置索引。

        这是确保注入 100% 成功的关键：通过精确的字节偏移定位，
        而不是脆弱的正则匹配。

        Returns:
            Dict[func_name -> FunctionLocation]
        """
        self.function_locations: Dict[str, FunctionLocation] = {}

        # 初始化 tree-sitter Rust 解析器
        try:
            if RUST_LANGUAGE is None or rust_parser is None:
                logger.warning("tree-sitter Rust 不可用，位置索引将为空")
                return self.function_locations
        except (NameError, AttributeError):
            return self.function_locations

        src_dir = self.work_dir / "src"
        if not src_dir.exists():
            return self.function_locations

        # 扫描所有 .rs 文件（排除特殊文件）
        special_files = {"types.rs", "globals.rs", "main.rs", "compat.rs", "compatibility.rs"}

        for rs_file in src_dir.glob("*.rs"):
            if rs_file.name in special_files:
                continue

            try:
                content = rs_file.read_text(encoding="utf-8", errors="ignore")
                content_bytes = content.encode("utf-8")
                tree = rust_parser.parse(content_bytes)

                # 遍历所有 function_item 节点
                self._extract_functions_from_tree(
                    tree.root_node, content_bytes, rs_file, content
                )
            except Exception as e:
                logger.debug(f"解析 {rs_file} 失败: {e}")

        logger.info(f"函数位置索引已建立: {len(self.function_locations)} 个函数")
        return self.function_locations

    def _extract_functions_from_tree(
        self,
        node,
        content_bytes: bytes,
        rs_file: Path,
        content: str,
        inside_extern: bool = False
    ):
        """递归提取函数位置信息"""
        # 跳过 extern "C" 块内的函数声明（这些是 FFI 声明）
        if node.type == "extern_block":
            inside_extern = True

        if node.type == "function_item" and not inside_extern:
            name_node = node.child_by_field_name("name")
            if name_node:
                func_name = content_bytes[name_node.start_byte:name_node.end_byte].decode("utf-8", errors="ignore")

                # 提取完整签名（从函数开始到函数体开始）
                body_node = node.child_by_field_name("body")
                if body_node:
                    sig_end = body_node.start_byte
                else:
                    sig_end = node.end_byte

                signature = content_bytes[node.start_byte:sig_end].decode("utf-8", errors="ignore").strip()

                # 计算行号
                line_start = content[:node.start_byte].count('\n') + 1
                line_end = content[:node.end_byte].count('\n') + 1

                location = FunctionLocation(
                    file_path=rs_file,
                    func_name=func_name,
                    start_byte=node.start_byte,
                    end_byte=node.end_byte,
                    line_start=line_start,
                    line_end=line_end,
                    signature=signature
                )

                # 使用 "文件名:函数名" 作为唯一键（处理同名函数）
                file_stem = rs_file.stem  # 如 "src_dbinder_invoker"
                key = f"{file_stem}:{func_name}"
                self.function_locations[key] = location

                # 也保存纯函数名映射（用于后备查找）
                if func_name not in self.function_locations:
                    self.function_locations[func_name] = location

        # 递归处理子节点
        for child in node.children:
            self._extract_functions_from_tree(
                child, content_bytes, rs_file, content, inside_extern
            )

    def _find_function_in_tree(
        self,
        node,
        content_bytes: bytes,
        target_func_name: str,
        inside_extern: bool = False
    ) -> Optional[Dict[str, int]]:
        """
        在 tree-sitter 语法树中查找目标函数的精确位置。

        Returns:
            Dict with keys: start, end, body_start (byte offsets)
            或 None 如果未找到
        """
        if node.type == "extern_block":
            inside_extern = True

        if node.type == "function_item" and not inside_extern:
            name_node = node.child_by_field_name("name")
            if name_node:
                func_name = content_bytes[name_node.start_byte:name_node.end_byte].decode("utf-8", errors="ignore")
                if func_name == target_func_name:
                    body_node = node.child_by_field_name("body")
                    body_start = body_node.start_byte if body_node else node.end_byte
                    return {
                        'start': node.start_byte,
                        'end': node.end_byte,
                        'body_start': body_start
                    }

        # 递归搜索子节点
        for child in node.children:
            result = self._find_function_in_tree(child, content_bytes, target_func_name, inside_extern)
            if result:
                return result

        return None
    
    def _validate_skeleton_files(self):
        """验证骨架文件是否有有效内容"""
        src_dir = self.work_dir / "src"
        if not src_dir.exists():
            print(f"  ⚠ 警告: 骨架目录不存在: {src_dir}")
            return
        
        rs_files = list(src_dir.glob("*.rs"))
        empty_files = []
        valid_files = []
        special_files = []  # types.rs, globals.rs 等特殊文件
        total_functions = 0
        total_types = 0
        total_statics = 0
        
        for rs_file in rs_files:
            with open(rs_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # 检查是否只有注释或空白
            code_lines = [line for line in content.splitlines() 
                         if line.strip() and not line.strip().startswith("//")]
            
            # 特殊文件处理（分层骨架）
            if rs_file.name == 'types.rs':
                # 统计类型定义数量
                type_count = len(re.findall(r'\b(struct|enum|union|type)\s+\w+', content))
                total_types += type_count
                if type_count > 0 or len(code_lines) > 3:
                    special_files.append((rs_file.name, f"{type_count} types"))
                continue
            
            if rs_file.name == 'globals.rs':
                # 统计静态变量数量
                static_count = len(re.findall(r'\bstatic\s+(?:mut\s+)?\w+', content))
                total_statics += static_count
                if static_count > 0 or len(code_lines) > 3:
                    special_files.append((rs_file.name, f"{static_count} statics"))
                continue
            
            if rs_file.name in ['main.rs', 'lib.rs']:
                # 跳过入口文件
                continue
            
            # 统计函数定义数量
            fn_count = len(re.findall(r'\bfn\s+\w+\s*\(', content))
            total_functions += fn_count
            
            if len(code_lines) < 3 or fn_count == 0:
                empty_files.append(rs_file.name)
            else:
                valid_files.append((rs_file.name, fn_count))
        
        # 输出验证结果
        print(f"  ✓ 骨架文件: {len(rs_files)} 个")
        print(f"    - 模块文件: {len(valid_files)} 个有效, {len(empty_files)} 个空/无函数")
        print(f"    - 函数签名总数: {total_functions}")
        
        if special_files:
            print(f"    - 特殊文件 (分层骨架):")
            for name, info in special_files:
                print(f"      ✓ {name}: {info}")
            if total_types > 0:
                print(f"    - 类型定义总数: {total_types}")
            if total_statics > 0:
                print(f"    - 静态变量总数: {total_statics}")
        
        if empty_files:
            print(f"  ⚠ 警告: 以下骨架文件为空或无函数定义（可能导致签名匹配失败）:")
            for name in empty_files[:5]:
                print(f"      - {name}")
            if len(empty_files) > 5:
                print(f"      ... 还有 {len(empty_files) - 5} 个文件")
    
    def _extract_skeleton_context(self, *, include_types_rs_defs: bool = True) -> str:
        """
        提取骨架中的类型定义和函数签名作为上下文
        同时检测不透明类型
        
        支持分层骨架结构:
        - types.rs: bindgen 生成的类型定义（优先处理）
        - globals.rs: tree-sitter 提取的全局变量
        - 其他 .rs 文件: 函数签名和模块级定义
        """
        context_parts = []
        
        src_dir = self.work_dir / "src"
        if not src_dir.exists():
            return ""
        
        # 清空并重新检测不透明类型
        self.opaque_types.clear()
        
        # 优先处理 types.rs（分层骨架的类型定义）
        types_file = src_dir / "types.rs"
        if types_file.exists():
            with open(types_file, 'r', encoding='utf-8', errors='ignore') as f:
                types_content = f.read()
            
            # 检测不透明类型
            file_opaque_types = self._detect_opaque_types(types_content)
            self.opaque_types.update(file_opaque_types)

            if include_types_rs_defs:
                # 提取类型定义（⚠️ types.rs 可能非常大；增量翻译默认会用 types slice 代替全量注入）
                type_defs = self._extract_type_definitions(types_content)
                if type_defs:
                    context_parts.append("// From types.rs (bindgen-generated types)")
                    context_parts.extend(type_defs)
        
        # 处理 globals.rs（分层骨架的全局变量）
        globals_file = src_dir / "globals.rs"
        if globals_file.exists():
            with open(globals_file, 'r', encoding='utf-8', errors='ignore') as f:
                globals_content = f.read()
            
            # 提取 static 变量声明
            static_defs = self._extract_static_declarations(globals_content)
            if static_defs:
                context_parts.append("// From globals.rs (global/static variables)")
                context_parts.extend(static_defs)

            # RustMap safe globals 会额外生成 get_/set_/with_ API；把这些签名也提供给 LLM。
            accessor_sigs = self._extract_function_signatures(globals_content)
            if accessor_sigs:
                context_parts.append("// From globals.rs (accessor functions)")
                context_parts.extend(accessor_sigs)
        
        # 处理其他模块文件
        for rs_file in src_dir.glob("*.rs"):
            # 跳过已处理的特殊文件和 main.rs
            if rs_file.name in ['types.rs', 'globals.rs', 'main.rs', 'lib.rs']:
                continue
            
            with open(rs_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # 检测不透明类型
            file_opaque_types = self._detect_opaque_types(content)
            self.opaque_types.update(file_opaque_types)
            
            # 提取 struct、enum、type 定义
            type_defs = self._extract_type_definitions(content)
            if type_defs:
                context_parts.append(f"// From {rs_file.name}")
                context_parts.extend(type_defs)
            
            # 提取函数签名
            func_sigs = self._extract_function_signatures(content)
            if func_sigs:
                context_parts.extend(func_sigs)
        
        return "\n\n".join(context_parts)

    def _reload_types_registry(self) -> None:
        """
        Build (or rebuild) the types.rs symbol registry for prompt slicing.

        This does NOT change any files; it only prepares a fast index for context slicing.
        """
        self._types_registry = None
        self._types_alias_map = {}
        self._types_const_names = set()
        self._types_const_type_map = {}
        if not self._enable_types_slice:
            return
        if not TYPES_SLICE_AVAILABLE or TypesRsRegistry is None:
            logger.debug("[TypesSlice] context_slicer not available; fallback to full context")
            return
        try:
            types_rs = self.work_dir / "src" / "types.rs"
            if not types_rs.exists():
                return
            self._types_registry = TypesRsRegistry.from_types_rs(types_rs)
            logger.info(f"[TypesSlice] registry ready: {len(self._types_registry.names)} symbols")
            # Build lightweight alias/const maps for prompt hints (best-effort).
            try:
                text = self._types_registry.full_text if self._types_registry else ""
                if text:
                    for m in re.finditer(
                        r"(?m)^\\s*pub\\s+type\\s+([A-Za-z_][A-Za-z0-9_]*)\\s*=\\s*(.+?)\\s*;\\s*$",
                        text,
                    ):
                        name = m.group(1)
                        rhs = (m.group(2) or "").strip()
                        if name and rhs:
                            self._types_alias_map[name] = rhs
                    for m in re.finditer(
                        r"(?m)^\\s*pub\\s+const\\s+([A-Za-z_][A-Za-z0-9_]*)\\s*:\\s*([^=;]+?)\\s*=.*;\\s*$",
                        text,
                    ):
                        name = m.group(1)
                        ty = (m.group(2) or "").strip()
                        if name and ty:
                            self._types_const_names.add(name)
                            self._types_const_type_map[name] = ty
            except Exception:
                self._types_alias_map = {}
                self._types_const_names = set()
                self._types_const_type_map = {}
        except Exception as e:
            logger.warning(f"[TypesSlice] failed to build registry: {e}")
            self._types_registry = None

    def _seed_symbols_for_types_slice(self, func_info: FunctionInfo) -> Set[str]:
        """
        Collect seed symbols for a function-specific types.rs slice.

        Sources (deterministic):
        - Rust signature explicit types::X references (highest precision)
        - Identifier intersection between C snippet and known types.rs symbols
        - semantic_slice unresolved symbols (when available)
        """
        if not self._types_registry or not TYPES_SLICE_AVAILABLE:
            return set()

        signature_seeds: Set[str] = set()
        try:
            if _extract_types_from_signature_for_slice:
                signature_seeds = self._types_registry.filter_existing(
                    _extract_types_from_signature_for_slice(func_info.rust_signature)
                )
        except Exception:
            signature_seeds = set()

        other_seeds: Set[str] = set()

        # C snippet identifiers: cheap, works even without compile_commands/libclang.
        try:
            if _extract_identifiers_for_slice:
                other_seeds.update(self._types_registry.filter_existing(_extract_identifiers_for_slice(func_info.c_code)))
        except Exception:
            pass

        # semantic_slice unresolved symbols (cached per function)
        try:
            data = func_info.semantic_data or self._load_semantic_data(func_info)
            if data:
                unresolved = [sym for sym in (data.get("unresolved_symbols") or []) if sym]
                other_seeds.update(self._types_registry.filter_existing(unresolved))
        except Exception:
            pass

        # Keep the slice bounded; correctness is recovered by compile-error-driven expansion.
        other_seeds.difference_update(signature_seeds)
        max_other = max(0, int(self._types_slice_max_other_seeds or 0))
        if max_other and len(other_seeds) > max_other:
            other_seeds = set(sorted(other_seeds)[:max_other])

        return set(signature_seeds) | set(other_seeds)

    def _extract_missing_symbols_for_types_slice(self, error_msg: str) -> Set[str]:
        """
        Expand slice seeds based on rustc errors (E0412/E0425-like).
        """
        if not self._types_registry or not TYPES_SLICE_AVAILABLE:
            return set()
        try:
            if not _extract_missing_symbols_for_slice:
                return set()
            missing = _extract_missing_symbols_for_slice(error_msg or "")
            return self._types_registry.filter_existing(missing)
        except Exception:
            return set()

    def _build_skeleton_context_for_function(
        self,
        func_info: FunctionInfo,
        non_types_context: str,
        *,
        extra_types_symbols: Iterable[str] = (),
    ) -> str:
        """
        Build per-function skeleton context:
        - types.rs: sliced (small, per-function)
        - other files: shared non-types context (globals, helper fns, etc.)
        """
        parts: List[str] = []

        if self._types_registry and TYPES_SLICE_AVAILABLE:
            seeds = self._seed_symbols_for_types_slice(func_info)
            try:
                types_slice = self._types_registry.render_slice(seeds, extra=extra_types_symbols, header=True).strip()
            except Exception:
                types_slice = ""
            if types_slice:
                parts.append(types_slice)

        if non_types_context:
            parts.append(non_types_context.strip())

        return "\n\n".join([p for p in parts if p]).strip()

    # ---------------------------------------------------------------------
    # Dependency signature hints (C-level): extracted/<proj>/dependencies/<func_key>.txt
    # ---------------------------------------------------------------------
    def _build_dependency_signature_hints(self, func_info: FunctionInfo) -> str:
        """
        Build a compact "called function signature" section for LLM prompts.

        Source of truth: extracted/<project>/dependencies/<func_key>.txt (already generated by stage1).

        Motivation:
        - LLM frequently makes call-site mistakes like passing `&var` where callee expects `*mut T`,
          causing `E0308: types differ in mutability`.
        - Showing callee prototypes (C constness) + a short mapping rule reduces these failures.
        """
        if not self._enable_dep_signature_hints:
            return ""

        cached = getattr(func_info, "callee_signatures_hint", None)
        if cached is not None:
            return cached or ""

        func_key = f"{func_info.file_name}_{func_info.index}"
        dep_file = self.dependencies_dir / f"{func_key}.txt"
        dep_text = ""
        if dep_file.exists():
            try:
                dep_text = dep_file.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                dep_text = ""

        called_names = self._extract_called_identifiers_from_c(func_info.c_code or "")
        called_names.discard(func_info.name)
        if not called_names:
            func_info.callee_signatures_hint = ""
            return ""

        # prototypes from in-file dependency blocks (usually static/local helper defs)
        dep_prototypes_by_name = self._extract_c_function_prototypes(dep_text) if dep_text else {}
        prototypes_by_name: Dict[str, List[str]] = defaultdict(list)
        for name, protos in (dep_prototypes_by_name or {}).items():
            if not protos:
                continue
            prototypes_by_name[name].extend(protos)

        # prototypes from preprocessed .i headers (usually extern declarations for cross-file calls)
        pre_decl_index = self._get_preprocessed_decl_prototypes(func_info)
        if pre_decl_index:
            for name in called_names:
                decls = pre_decl_index.get(name) or []
                if not decls:
                    continue
                prototypes_by_name[name].extend(decls)

        if not prototypes_by_name:
            func_info.callee_signatures_hint = ""
            return ""

        local_types = self._extract_local_type_names_from_c(func_info.c_code or "")

        picked: List[str] = []
        seen: Set[str] = set()
        for name in sorted(called_names):
            protos = prototypes_by_name.get(name) or []
            if not protos:
                continue
            best = self._choose_best_c_prototype(protos, local_types)
            if not best:
                continue
            # Keep this section focused, but also include preprocessed-only extern decls
            # (they are the key to fixing E0425 missing symbol issues).
            is_preprocessed_only = name not in (dep_prototypes_by_name or {})
            if not is_preprocessed_only and ("*" not in best and "const" not in best):
                continue
            if best in seen:
                continue
            seen.add(best)
            picked.append(best)
            if self._dep_signature_hints_max and len(picked) >= self._dep_signature_hints_max:
                break

        if not picked:
            func_info.callee_signatures_hint = ""
            return ""

        hint = (
            "\n## Called C APIs (dependency blocks + preprocessed headers; pay attention to const/mut)\n"
            "```c\n"
            + "\n".join(picked)
            + "\n```\n"
            "Rule: C `const T*` => Rust `*const T` (pass `&T`), C `T*` => Rust `*mut T` (pass `&mut T`).\n"
            "If a called C symbol is missing in Rust (`E0425`), do NOT guess an `extern` decl in the function; ensure it is generated in `compat.rs` via bindgen allowlist (Step 2.55).\n"
        )

        func_info.callee_signatures_hint = hint
        return hint

    @staticmethod
    def _looks_like_static_inline_c_function(defn: str) -> bool:
        """
        Best-effort check for `static inline` C function definitions.
        We use this to avoid treating header inlines as external link symbols.
        """
        if not defn:
            return False
        head = defn.lstrip()
        if not head:
            return False
        prefix = head.split("{", 1)[0]
        return bool(
            re.search(r"\bstatic\b", prefix)
            and (re.search(r"\binline\b", prefix) or ("__inline" in prefix))
        )

    def _build_preprocessed_inline_helper_hints(self, func_info: FunctionInfo) -> str:
        """
        Provide `static inline` helper definitions from the preprocessed TU when referenced by this function.

        Motivation:
        - `static inline` helpers are not stable link symbols and should NOT be declared as `extern "C"` in Rust.
        - When we translate from `.i` (macro-expanded) code, these helpers often remain as calls (unlike macros),
          so we surface their bodies to the LLM to inline/translate them correctly.
        """
        if not self._enable_preprocessed_inline_helper_hints:
            return ""

        try:
            max_defs = max(0, int(self._preprocessed_inline_helper_hints_max or 0))
        except Exception:
            max_defs = 0
        try:
            max_chars = max(0, int(self._preprocessed_inline_helper_hints_max_chars or 0))
        except Exception:
            max_chars = 0
        if max_defs <= 0 or max_chars <= 0:
            return ""

        pre_path = self._locate_preprocessed_file(func_info)
        if not pre_path or not Path(pre_path).exists():
            return ""

        p = Path(pre_path)
        i_text = self._preprocessed_text_cache.get(p)
        if i_text is None:
            try:
                i_text = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                i_text = ""
            self._preprocessed_text_cache[p] = i_text
        if not i_text:
            return ""

        fn_index = self._preprocessed_fn_index_cache.get(p)
        if fn_index is None:
            fn_index = self._build_preprocessed_fn_index(p, i_text)
            self._preprocessed_fn_index_cache[p] = fn_index
        if not fn_index:
            return ""

        called = self._extract_called_identifiers_from_c(func_info.c_code or "")
        called.discard(func_info.name)
        if not called:
            return ""

        picked: List[str] = []
        seen: Set[str] = set()
        total_chars = 0

        # Prefer deterministic order.
        for name in sorted(called):
            if len(picked) >= max_defs:
                break

            # Find any function_definition for this name inside the `.i` TU.
            spans: List[Tuple[int, int]] = []
            for k, v in fn_index.items():
                if not k or k[0] != name:
                    continue
                for span in (v or []):
                    if isinstance(span, tuple) and len(span) == 2:
                        spans.append((int(span[0]), int(span[1])))
            if not spans:
                continue

            # Pick the smallest `static inline` candidate (avoid dragging huge headers into prompt).
            best = ""
            best_len = None
            for start_b, end_b in spans:
                if start_b < 0 or end_b <= start_b or end_b > len(i_text):
                    continue
                snippet = i_text[start_b:end_b].strip()
                if not snippet:
                    continue
                if not self._looks_like_static_inline_c_function(snippet):
                    continue
                # Hard cap per item to prevent explosion.
                if len(snippet) > 4000:
                    continue
                if best_len is None or len(snippet) < best_len:
                    best = snippet
                    best_len = len(snippet)

            if not best or best in seen:
                continue
            if total_chars + len(best) > max_chars:
                break
            picked.append(best)
            seen.add(best)
            total_chars += len(best) + 2

        if not picked:
            return ""

        return (
            "\n## Inline C Helpers (from preprocessed TU; NOT extern symbols)\n"
            "```c\n"
            + "\n\n".join(picked)
            + "\n```\n"
            "Note: these are `static inline` helper definitions from headers. In Rust, inline/translate their logic; do NOT declare them as `extern \"C\"`.\n"
        )

    # ---------------------------------------------------------------------
    # Rust callee signature hints (Rust-level): signature_matches/<proj>/*.txt
    # ---------------------------------------------------------------------
    def _ensure_rust_signature_index(self) -> None:
        """
        Build a map: callee_name -> [rust_signature_str, ...] from signature_matches.

        This is project-local and used to provide precise Rust-level callee signatures
        (mutability, return type, extern/unsafe) to reduce call-site E0308 mistakes.
        """
        if self._rust_signature_index is not None:
            return
        index: Dict[str, List[str]] = defaultdict(list)
        try:
            if self.signature_dir.exists():
                for sig_file in sorted(self.signature_dir.glob("*.txt")):
                    try:
                        sig = sig_file.read_text(encoding="utf-8", errors="ignore").strip()
                    except Exception:
                        continue
                    if not sig:
                        continue
                    m = re.search(r"\bfn\s+([A-Za-z_][A-Za-z0-9_]*)\b", sig)
                    if not m:
                        continue
                    name = m.group(1)
                    index[name].append(sig)
        except Exception:
            index = {}
        self._rust_signature_index = dict(index)

    @staticmethod
    def _choose_best_rust_signature(candidates: List[str]) -> str:
        if not candidates:
            return ""

        def score(sig: str) -> Tuple[int, int]:
            s = 0
            if sig.startswith("pub "):
                s += 20
            if 'extern "C"' in sig:
                s += 80
            if sig.startswith("pub unsafe") or " unsafe " in sig:
                s += 30
            if "->" in sig:
                s += 5
            return (s, len(sig))

        return max(candidates, key=score)

    def _collect_type_alias_expansions(self, rust_sigs: Iterable[str]) -> List[str]:
        """
        Collect small `types.rs` alias expansions relevant to given Rust signatures.
        Example: `c_int = i32`, `size_t = usize`.
        """
        if not rust_sigs or not self._types_alias_map:
            return []
        used: Set[str] = set()
        for sig in rust_sigs:
            for m in re.finditer(r"(?:crate::)?types::([A-Za-z_][A-Za-z0-9_]*)", sig or ""):
                used.add(m.group(1))

        lines: List[str] = []
        for name in sorted(used):
            rhs = self._types_alias_map.get(name)
            if not rhs:
                continue
            # Keep it compact; skip very long aliases.
            if len(rhs) > 140:
                continue
            lines.append(f"type {name} = {rhs};")
        return lines

    def _build_called_rust_signature_hints(self, func_info: FunctionInfo) -> str:
        """
        Build Rust-level callee signature hints from signature_matches.

        This complements C-level prototypes by giving the exact Rust signature the call must satisfy.
        """
        if not self._enable_rust_callee_signature_hints:
            return ""

        cached = getattr(func_info, "callee_rust_signatures_hint", None)
        if cached is not None:
            return cached or ""

        called_names = self._extract_called_identifiers_from_c(func_info.c_code or "")
        called_names.discard(func_info.name)
        if not called_names:
            func_info.callee_rust_signatures_hint = ""
            return ""

        self._ensure_rust_signature_index()
        if not self._rust_signature_index:
            func_info.callee_rust_signatures_hint = ""
            return ""

        picked: List[str] = []
        for name in sorted(called_names):
            sigs = self._rust_signature_index.get(name) or []
            if not sigs:
                continue
            best = self._choose_best_rust_signature(sigs)
            if not best:
                continue
            picked.append(best)
            if self._rust_callee_signature_hints_max and len(picked) >= self._rust_callee_signature_hints_max:
                break

        if not picked:
            func_info.callee_rust_signatures_hint = ""
            return ""

        alias_expansions = self._collect_type_alias_expansions(picked)

        parts: List[str] = []
        parts.append("\n## Called Rust APIs (signatures; match mutability/unsafe EXACTLY)\n```rust")
        parts.append("\n".join(picked))
        parts.append("```\n")
        if alias_expansions:
            parts.append("## Type Alias Expansions (from types.rs)\n```rust")
            parts.append("\n".join(alias_expansions[:40]))
            parts.append("```\n")

        hint = "\n".join(parts)
        func_info.callee_rust_signatures_hint = hint
        return hint

    def _build_typed_constants_hints(self, func_info: FunctionInfo) -> str:
        """
        Build a compact "constants/macros with Rust types" section.

        Motivation:
        - A common E0308 source is integer signedness mismatch, especially when comparing against bindgen
          constants (e.g., `NSTACKX_EOK: u32` vs `errorCode: i32`).
        """
        if not self._enable_typed_const_hints:
            return ""
        if not self._types_const_names or not self._types_const_type_map:
            return ""
        if not func_info.c_code:
            return ""

        try:
            if _extract_identifiers_for_slice:
                idents = set(_extract_identifiers_for_slice(func_info.c_code))
            else:
                idents = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", func_info.c_code))
        except Exception:
            idents = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", func_info.c_code))

        used = sorted(idents & self._types_const_names)
        if not used:
            return ""
        if self._typed_const_hints_max and len(used) > self._typed_const_hints_max:
            used = used[: self._typed_const_hints_max]

        lines: List[str] = []
        for name in used:
            ty = self._types_const_type_map.get(name)
            if not ty:
                continue
            lines.append(f"{name}: {ty}")

        if not lines:
            return ""

        return (
            "\n## Constants/Macros with Rust Types (from bindgen types.rs)\n"
            "```text\n"
            + "\n".join(lines)
            + "\n```\n"
            "Tip: make both sides the same type explicitly (e.g., cast) before comparing.\n"
        )

    def _build_c_field_access_hints(self, func_info: FunctionInfo) -> str:
        """
        Extract observed C/C++ field accesses like `x->field` / `x.field` / `(*x).field`.

        This helps the LLM map which fields are actually used, and avoid inventing fields on opaque types.
        """
        if not self._enable_c_field_access_hints:
            return ""
        c = func_info.c_code or ""
        if not c:
            return ""

        stop_vars = {
            "if",
            "for",
            "while",
            "switch",
            "return",
            "sizeof",
            "alignof",
            "struct",
            "enum",
            "union",
        }
        stop_fields = {
            "size",
            "len",
        }

        access: Dict[str, Set[str]] = defaultdict(set)

        for m in re.finditer(r"\(\s*\*\s*([A-Za-z_]\w*)\s*\)\s*\.\s*([A-Za-z_]\w*)", c):
            base, field = m.group(1), m.group(2)
            if base in stop_vars or field in stop_fields:
                continue
            access[base].add(field)
        for m in re.finditer(r"\b([A-Za-z_]\w*)\s*->\s*([A-Za-z_]\w*)", c):
            base, field = m.group(1), m.group(2)
            if base in stop_vars or field in stop_fields:
                continue
            access[base].add(field)
        for m in re.finditer(r"\b([A-Za-z_]\w*)\s*\.\s*([A-Za-z_]\w*)", c):
            base, field = m.group(1), m.group(2)
            if base in stop_vars or field in stop_fields:
                continue
            access[base].add(field)

        if not access:
            return ""

        vars_sorted = sorted(access.keys())
        if self._c_field_access_max_vars and len(vars_sorted) > self._c_field_access_max_vars:
            vars_sorted = vars_sorted[: self._c_field_access_max_vars]

        lines: List[str] = []
        for v in vars_sorted:
            fields = sorted(access[v])
            if self._c_field_access_max_fields and len(fields) > self._c_field_access_max_fields:
                fields = fields[: self._c_field_access_max_fields]
            if not fields:
                continue
            lines.append(f"- {v}: {', '.join(fields)}")

        if not lines:
            return ""

        return (
            "\n## Observed C Field Accesses (map these to Rust types/fields)\n"
            + "\n".join(lines)
            + "\nNote: If the base type is opaque (`type X = c_void`), DO NOT access fields; use accessor shims or return defaults.\n"
        )

    @staticmethod
    def _split_c_comma_list(text: str) -> List[str]:
        """Split a C comma-separated list, respecting (), {}, [] nesting (best-effort)."""
        if text is None:
            return []
        s = text.strip()
        if not s:
            return []
        out: List[str] = []
        buf: List[str] = []
        paren = brace = bracket = 0
        in_str: Optional[str] = None
        escape = False
        for ch in s:
            if in_str:
                buf.append(ch)
                if escape:
                    escape = False
                    continue
                if ch == "\\":
                    escape = True
                    continue
                if ch == in_str:
                    in_str = None
                continue
            if ch in ("'", '"'):
                in_str = ch
                buf.append(ch)
                continue
            if ch == "(":
                paren += 1
            elif ch == ")":
                paren = max(0, paren - 1)
            elif ch == "{":
                brace += 1
            elif ch == "}":
                brace = max(0, brace - 1)
            elif ch == "[":
                bracket += 1
            elif ch == "]":
                bracket = max(0, bracket - 1)
            if ch == "," and paren == 0 and brace == 0 and bracket == 0:
                item = "".join(buf).strip()
                if item:
                    out.append(item)
                buf = []
                continue
            buf.append(ch)
        tail = "".join(buf).strip()
        if tail:
            out.append(tail)
        return out

    @staticmethod
    def _extract_c_params_from_code(c_code: str, func_name: str) -> List[str]:
        """
        Extract raw C parameter declaration strings for the target function.
        Best-effort, but works well for typical extracted single-function snippets.
        """
        if not c_code or not func_name:
            return []
        code = c_code

        # Find the function name followed by '(' and a matching ')' before '{'.
        pat = re.compile(rf"\b{re.escape(func_name)}\s*\(", re.MULTILINE)
        m = pat.search(code)
        if not m:
            return []
        start = m.end()  # position after '('
        i = start
        depth = 1
        in_str: Optional[str] = None
        escape = False
        while i < len(code) and depth > 0:
            ch = code[i]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == in_str:
                    in_str = None
                i += 1
                continue
            if ch in ("'", '"'):
                in_str = ch
                i += 1
                continue
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            i += 1

        if depth != 0:
            return []
        params_str = code[start : i - 1].strip()
        if not params_str or params_str == "void":
            return []
        return IncrementalTranslator._split_c_comma_list(params_str)

    @staticmethod
    def _extract_c_param_name(decl: str) -> str:
        if not decl:
            return ""
        # function pointer: void (*cb)(int)
        m = re.search(r"\(\s*\*\s*([A-Za-z_]\w*)\s*\)", decl)
        if m:
            return m.group(1)
        # array: T name[...]
        m = re.search(r"\b([A-Za-z_]\w*)\s*(?:\[[^\]]*\])\s*$", decl.strip())
        if m:
            return m.group(1)
        # generic: take last identifier
        tokens = re.findall(r"[A-Za-z_]\w*", decl)
        if not tokens:
            return ""
        return tokens[-1]

    @staticmethod
    def _strip_c_expr_to_ident(expr: str) -> str:
        """Best-effort: normalize an argument expression to a bare identifier name."""
        if not expr:
            return ""
        s = expr.strip()
        # drop trailing comments
        s = re.sub(r"//.*$", "", s).strip()
        s = re.sub(r"/\*.*?\*/", "", s, flags=re.DOTALL).strip()
        # drop casts like (Type*) or (void*)
        s = re.sub(r"^\(\s*[^)]+\s*\)\s*", "", s).strip()
        # remove unary &/* and parens
        s = s.lstrip("&*").strip()
        s = s.strip("()")
        # take first identifier token
        m = re.match(r"^([A-Za-z_]\w*)$", s)
        return m.group(1) if m else ""

    def _build_c_pointer_contract_hints(self, func_info: FunctionInfo) -> str:
        """
        Infer pointer mutability / NULL contracts from C code.

        - Detect pointer params
        - Detect NULL checks
        - Detect direct writes (e.g., *p=, p->field=, memset(p,...))
        - Detect passing a param to a callee parameter that is a non-const pointer (from dependency prototypes)
        """
        if not self._enable_c_pointer_contract_hints:
            return ""
        c_code = func_info.c_code or ""
        if not c_code or not func_info.name:
            return ""

        params = self._extract_c_params_from_code(c_code, func_info.name)
        if not params:
            return ""

        param_meta: Dict[str, Dict[str, Any]] = {}
        for decl in params:
            d = (decl or "").strip()
            if not d or d == "void" or d == "...":
                continue
            name = self._extract_c_param_name(d)
            if not name:
                continue
            is_ptr = ("*" in d) or ("[" in d) or bool(re.search(r"\(\s*\*\s*\w+\s*\)", d))
            if not is_ptr:
                continue
            # Best-effort: treat any 'const' as const-pointee (good enough for mutability guidance).
            is_const = "const" in d
            param_meta[name] = {
                "decl": d,
                "is_const": is_const,
                "null_checked": False,
                "written": False,
                "passed_as_mut": set(),  # callee names
            }

        if not param_meta:
            return ""

        # NULL checks + direct writes
        for name, info in param_meta.items():
            if re.search(rf"\bif\s*\(\s*!\s*{re.escape(name)}\s*\)", c_code):
                info["null_checked"] = True
            if re.search(rf"\bif\s*\(\s*{re.escape(name)}\s*==\s*NULL\s*\)", c_code):
                info["null_checked"] = True
            if re.search(rf"\bif\s*\(\s*NULL\s*==\s*{re.escape(name)}\s*\)", c_code):
                info["null_checked"] = True

            # direct writes / out-params
            write_patterns = [
                rf"\*\s*\(?\s*{re.escape(name)}\s*\)?\s*=",
                rf"{re.escape(name)}\s*\[[^\]]*\]\s*=",
                rf"{re.escape(name)}\s*->\s*[A-Za-z_]\w*\s*=",
                rf"\(\s*\*\s*{re.escape(name)}\s*\)\s*\.\s*[A-Za-z_]\w*\s*=",
                rf"\bmemcpy\s*\(\s*{re.escape(name)}\b",
                rf"\bmemset\s*\(\s*{re.escape(name)}\b",
                rf"\bstrcpy\s*\(\s*{re.escape(name)}\b",
                rf"\bstrncpy\s*\(\s*{re.escape(name)}\b",
            ]
            if any(re.search(p, c_code) for p in write_patterns):
                info["written"] = True

        # Passing pointer params to callees expecting non-const pointers (from dependency prototypes)
        func_key = f"{func_info.file_name}_{func_info.index}"
        dep_file = self.dependencies_dir / f"{func_key}.txt"
        dep_text = ""
        try:
            if dep_file.exists():
                dep_text = dep_file.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            dep_text = ""
        prototypes_by_name = self._extract_c_function_prototypes(dep_text) if dep_text else {}
        local_types = self._extract_local_type_names_from_c(c_code)

        def proto_param_requires_mut(p: str) -> bool:
            if not p:
                return False
            # non-const pointer param => requires mutable pointer in Rust call-site
            return ("*" in p) and ("const" not in p)

        called_names = self._extract_called_identifiers_from_c(c_code)
        for callee in sorted(called_names):
            protos = prototypes_by_name.get(callee) or []
            if not protos:
                continue
            best = self._choose_best_c_prototype(protos, local_types)
            m = re.search(r"\((.*)\)", best)
            if not m:
                continue
            proto_params = self._split_c_comma_list(m.group(1))
            req_mut = [proto_param_requires_mut(p) for p in proto_params]
            if not any(req_mut):
                continue

            # parse call sites: callee(...)
            for call_m in re.finditer(rf"\b{re.escape(callee)}\s*\(", c_code):
                start = call_m.end()
                i = start
                depth = 1
                while i < len(c_code) and depth > 0:
                    ch = c_code[i]
                    if ch == "(":
                        depth += 1
                    elif ch == ")":
                        depth -= 1
                    i += 1
                if depth != 0:
                    continue
                args_str = c_code[start : i - 1]
                args = self._split_c_comma_list(args_str)
                for idx, arg in enumerate(args):
                    if idx >= len(req_mut) or not req_mut[idx]:
                        continue
                    ident = self._strip_c_expr_to_ident(arg)
                    if ident and ident in param_meta:
                        param_meta[ident]["passed_as_mut"].add(callee)

        # Build report
        lines: List[str] = []
        for name in sorted(param_meta.keys()):
            info = param_meta[name]
            kind = "const ptr" if info.get("is_const") else "ptr"
            flags: List[str] = [kind]
            if info.get("null_checked"):
                flags.append("NULL-checked")
            if info.get("written"):
                flags.append("written")
            callees = sorted(info.get("passed_as_mut") or [])
            if callees:
                flags.append("passed-as-mutable-to: " + ", ".join(callees[:6]))
            lines.append(f"- {name}: " + "; ".join(flags))
            if self._c_pointer_contract_max_lines and len(lines) >= self._c_pointer_contract_max_lines:
                break

        if not lines:
            return ""

        return (
            "\n## Pointer Mutability / NULL Contracts (inferred from C)\n"
            + "\n".join(lines)
            + "\nRule: if callee expects `*mut T`, pass a mutable pointer (`&mut` / `as *mut T`). If it expects `*const T`, pass immutable (`&` / `as *const T`).\n"
        )

    @staticmethod
    def _extract_called_identifiers_from_c(c_code: str) -> Set[str]:
        if not c_code:
            return set()
        stop = {
            "if",
            "for",
            "while",
            "switch",
            "return",
            "sizeof",
            "alignof",
            "typedef",
            "struct",
            "enum",
            "union",
            "case",
            "break",
            "continue",
        }
        called: Set[str] = set()
        for m in re.finditer(r"\b([A-Za-z_]\w*)\s*\(", c_code):
            name = m.group(1)
            if name in stop:
                continue
            called.add(name)
        return called

    @staticmethod
    def _extract_local_type_names_from_c(c_code: str) -> Set[str]:
        if not c_code:
            return set()
        # Heuristic: collect base identifiers that appear in local declarations / params.
        # This is only used to disambiguate same-name prototypes (e.g., TransGetFileListener variants).
        types: Set[str] = set()
        stop = {
            "const",
            "volatile",
            "static",
            "inline",
            "unsigned",
            "signed",
            "struct",
            "enum",
            "union",
            "return",
        }
        for m in re.finditer(r"\b(?:const\s+)?([A-Za-z_]\w*)\s*(?:\*+\s*)*([A-Za-z_]\w*)\s*(?:[;=,\)])", c_code):
            t = (m.group(1) or "").strip()
            if not t or t in stop:
                continue
            types.add(t)
        return types

    @staticmethod
    def _extract_c_function_prototypes(dep_text: str) -> Dict[str, List[str]]:
        """
        Extract C function prototypes from concatenated dependency code blocks.
        Returns: {func_name: [prototype_str, ...]}
        """
        if not dep_text:
            return {}
        protos: Dict[str, List[str]] = defaultdict(list)
        # Match function definitions like:
        #   static void Foo(int a, const Bar *b)
        #   {
        #       ...
        #   }
        pattern = re.compile(
            r"(?ms)^\s*(?:static\s+)?(?:inline\s+)?(?:extern\s+)?[A-Za-z_]\w*(?:[\w\s\*]*?)\b([A-Za-z_]\w*)\s*\((.*?)\)\s*\{"
        )
        for m in pattern.finditer(dep_text):
            name = m.group(1)
            whole = m.group(0)
            if not name or not whole:
                continue
            try:
                sig = whole[: whole.rfind("{")].strip()
            except Exception:
                sig = whole.strip()
            sig = re.sub(r"\s+", " ", sig).strip()
            if not sig:
                continue
            protos[name].append(sig)
        return dict(protos)

    @staticmethod
    def _choose_best_c_prototype(prototypes: List[str], local_types: Set[str]) -> str:
        if not prototypes:
            return ""
        if not local_types:
            return max(prototypes, key=len)

        def score(proto: str) -> Tuple[int, int]:
            s = 0
            for t in local_types:
                if re.search(rf"\b{re.escape(t)}\b", proto):
                    s += 1
            return (s, len(proto))

        return max(prototypes, key=score)
    
    def _extract_static_declarations(self, content: str) -> List[str]:
        """
        从代码中提取 static 变量声明
        """
        static_defs = []
        
        # 匹配 static mut 和 static 声明
        static_pattern = r'^(\s*(?:pub\s+)?static\s+(?:mut\s+)?\w+\s*:\s*[^=;]+(?:\s*=\s*[^;]+)?;)'
        
        for match in re.finditer(static_pattern, content, re.MULTILINE):
            decl = match.group(1).strip()
            if decl and not decl.startswith('//'):
                static_defs.append(decl)
        
        return static_defs

    def _locate_source_file(self, func_info: FunctionInfo) -> Optional[Path]:
        """在项目源代码中定位对应的 C/C++ 文件。"""
        cache_key = func_info.file_name
        if cache_key in self._source_file_cache:
            return self._source_file_cache[cache_key]
        # Prefer TU context map (exact path) if available.
        rec = self._tu_context_files.get(cache_key) if getattr(self, "_tu_context_files", None) else None
        if isinstance(rec, dict):
            for k in ("source_file_abs", "source_for_cc_abs"):
                v = rec.get(k)
                if not v:
                    continue
                try:
                    p = Path(str(v)).expanduser()
                    if p.exists():
                        self._source_file_cache[cache_key] = p
                        return p
                except Exception:
                    continue
        if not self.project_src_dir.exists():
            self._source_file_cache[cache_key] = None
            return None
        search_roots = [
            self.project_src_dir / "src",
            self.project_src_dir
        ]
        extensions = [".cpp", ".cc", ".cxx", ".c", ".hpp", ".hh", ".h"]
        found: Optional[Path] = None
        for root in search_roots:
            if not root.exists():
                continue
            for ext in extensions:
                pattern = f"{cache_key}{ext}"
                try:
                    for candidate in root.rglob(pattern):
                        found = candidate
                        break
                except FileNotFoundError:
                    continue
                if found:
                    break
            if found:
                break
        self._source_file_cache[cache_key] = found
        return found

    def _locate_preprocessed_file(self, func_info: FunctionInfo) -> Optional[Path]:
        """
        Locate the preprocessed `.i` file for the C translation unit that contains this function.

        Expected layout: `<workspace>/.preprocessed/<stem>_<hash>.i`
        where `<stem>` usually matches the original C file stem (often `file_name` without leading `src_`).
        """
        if not self._enable_preprocessed_decl_hints:
            return None
        base_dir = self.preprocessed_dir
        if not base_dir or not base_dir.is_dir():
            return None

        # Strict TU-closure mode: ONLY use the stage1-pinned `.i` path from `tu_context_map.json`.
        # Do NOT fall back to glob matching (it can silently pick a different TU/profile and break truth-mode).
        try:
            require_tu = (os.environ.get("C2R_REQUIRE_TU_CLOSURE", "1") or "1").strip().lower() not in (
                "0",
                "false",
                "no",
            )
        except Exception:
            require_tu = True

        # Prefer TU context map (exact preprocessed path) to avoid multi-profile ambiguity.
        safe_name = (func_info.file_name or "").strip()
        if safe_name and getattr(self, "_tu_context_files", None):
            rec = self._tu_context_files.get(safe_name)
            if rec is None:
                # Mapping exists but this file group is missing: likely naming drift between stages.
                self._tu_closure_issues.setdefault(safe_name, "tu_context_map_missing_record")
            elif isinstance(rec, dict):
                pre = rec.get("preprocessed_file")
                if pre:
                    try:
                        p = Path(str(pre)).expanduser()
                        if p.exists():
                            return p
                        self._tu_closure_issues.setdefault(safe_name, f"preprocessed_file_missing: {p}")
                    except Exception:
                        self._tu_closure_issues.setdefault(safe_name, "preprocessed_file_invalid_path")
                else:
                    err = (rec.get("error") or "").strip()
                    if err:
                        self._tu_closure_issues.setdefault(safe_name, f"preprocess_failed: {err[:160]}")
                    else:
                        self._tu_closure_issues.setdefault(safe_name, "preprocessed_file_not_recorded")

        if require_tu:
            # In strict mode, never attempt heuristic matches.
            return None

        stem = safe_name
        if stem.startswith("src_"):
            stem = stem[len("src_") :]
        if not stem:
            return None

        # Prefer exact prefix match: <stem>_*.i
        try:
            candidates = sorted(base_dir.glob(f"{stem}_*.i"))
        except Exception:
            candidates = []
        if candidates:
            return candidates[0]

        # Fallback: any file containing stem (best-effort)
        try:
            candidates = sorted(base_dir.glob(f"*{stem}*.i"))
        except Exception:
            candidates = []
        return candidates[0] if candidates else None

    @staticmethod
    def _extract_c_function_decl_prototypes(text: str, *, max_per_name: int = 0) -> Dict[str, List[str]]:
        """
        Extract C function *declarations* (ending with `;`) from a preprocessed `.i` file.

        Returns: {name: [prototype_str, ...]}
        """
        if not text:
            return {}
        protos: Dict[str, List[str]] = defaultdict(list)
        # Require whitespace before function name to avoid matching call statements like `Foo(x);`
        pattern = re.compile(
            r"(?m)^(?!\s*typedef\b)\s*(?:extern\s+)?[A-Za-z_][\w\s\*]*?\s+([A-Za-z_]\w*)\s*\([^;\n]*\)\s*;"
        )
        for m in pattern.finditer(text):
            name = (m.group(1) or "").strip()
            whole = m.group(0) or ""
            if not name or not whole:
                continue
            if max_per_name and len(protos.get(name, [])) >= max_per_name:
                continue
            sig = re.sub(r"\s+", " ", whole).strip()
            if not sig:
                continue
            protos[name].append(sig)
        return dict(protos)

    def _get_preprocessed_decl_prototypes(self, func_info: FunctionInfo) -> Dict[str, List[str]]:
        """
        Load and cache extern-like C declaration prototypes from the preprocessed `.i` file.
        """
        if not self._enable_preprocessed_decl_hints:
            return {}
        p = self._locate_preprocessed_file(func_info)
        if not p:
            return {}
        cached = self._preprocessed_decl_index.get(p)
        if cached is not None:
            return cached

        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            self._preprocessed_decl_index[p] = {}
            return {}

        idx = self._extract_c_function_decl_prototypes(text, max_per_name=int(self._preprocessed_decl_hints_max_per_name or 0))
        self._preprocessed_decl_index[p] = idx
        return idx

    @staticmethod
    def _parse_c_decl_prototype(proto: str) -> Optional[Tuple[str, str, List[str], bool]]:
        """
        Parse a single-line C function declaration prototype like:
          `int32_t Foo(Bar *x, const Baz *y);`

        Returns: (name, ret_type, param_decls, is_variadic)
        """
        if not proto:
            return None
        s = proto.strip()
        if not s:
            return None
        s = s.rstrip(";").strip()
        # Remove common qualifiers that may appear in declarations.
        s = re.sub(r"\b(?:extern|static|inline|__inline__)\b\s*", "", s).strip()

        m = re.search(r"\b([A-Za-z_]\w*)\s*\(", s)
        if not m:
            return None
        name = m.group(1)
        ret_type = s[: m.start(1)].strip()

        # Extract params between matching parentheses.
        start = m.end()  # after '('
        i = start
        depth = 1
        in_str: Optional[str] = None
        escape = False
        while i < len(s) and depth > 0:
            ch = s[i]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == in_str:
                    in_str = None
                i += 1
                continue
            if ch in ("'", '"'):
                in_str = ch
                i += 1
                continue
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            i += 1
        if depth != 0:
            return None
        params_str = s[start : i - 1].strip()
        params = IncrementalTranslator._split_c_comma_list(params_str) if params_str else []
        is_variadic = any(p.strip() == "..." for p in params)
        params = [p for p in params if p.strip() and p.strip() != "..."]
        if len(params) == 1 and params[0].strip() == "void":
            params = []
        return name, ret_type, params, is_variadic

    @staticmethod
    def _split_c_param_decl(decl: str, *, default_name: str) -> Tuple[str, str]:
        """
        Split a C parameter declaration into (name, type_str).
        Best-effort; falls back to `default_name` if the name can't be extracted.
        """
        d = (decl or "").strip()
        if not d:
            return default_name, "void*"

        # Function pointer param: `ret (*cb)(...)`
        m = re.search(r"\(\s*\*\s*([A-Za-z_]\w*)\s*\)", d)
        if m:
            # Mapping full function-pointer types is complex; use void* as a conservative placeholder.
            return default_name, "void*"

        # Array param: `T name[...]`
        m = re.search(r"\b([A-Za-z_]\w*)\s*(?:\[[^\]]*\])\s*$", d)
        if m:
            # Drop trailing array suffix for type derivation (ignore the actual C param name in Rust to avoid collisions).
            name_in_c = m.group(1)
            c_type = re.sub(rf"\b{re.escape(name_in_c)}\b\s*(?:\[[^\]]*\])?\s*$", "", d).strip()
            return default_name, c_type if c_type else d

        # Generic params: prefer stable Rust arg names (arg0/arg1/...) to avoid E0130 patterns in extern decls.
        #
        # NOTE: C prototypes may omit parameter names entirely, making "last identifier" ambiguous.
        # We only strip a trailing identifier if it is very likely a real param name (not a type keyword/specifier).
        tokens = re.findall(r"[A-Za-z_]\w*", d)
        if not tokens:
            return default_name, d

        c_keywords = {
            "void",
            "char",
            "short",
            "int",
            "long",
            "float",
            "double",
            "signed",
            "unsigned",
            "const",
            "volatile",
            "restrict",
            "struct",
            "union",
            "enum",
            "inline",
            "__inline__",
            "__attribute__",
            "__declspec",
        }

        candidate = tokens[-1]
        can_strip_name = False
        if len(tokens) >= 2 and candidate not in c_keywords:
            # `struct Foo` (no name) is common in prototypes; don't treat Foo as a name here.
            if not (tokens[0] in {"struct", "union", "enum"} and len(tokens) == 2):
                # Only if the decl text actually ends with this identifier.
                if re.search(rf"\b{re.escape(candidate)}\b\s*$", d):
                    can_strip_name = True

        if can_strip_name:
            c_type = re.sub(rf"\b{re.escape(candidate)}\b\s*$", "", d).strip()
            return default_name, c_type if c_type else d
        return default_name, d

    @staticmethod
    def _map_c_type_to_rust_best_effort(c_type: str) -> str:
        """
        Map a C type string to a Rust type string using TypeMapper, but avoid risky fallbacks.
        """
        t = (c_type or "").strip()
        if not t or t == "void":
            return "()"
        # Heuristic: complex types (function pointers, attributes) are mapped to void*.
        if any(ch in t for ch in ("(", ")", "{", "}", "__attribute__", "__declspec")):
            return "*mut std::ffi::c_void"
        try:
            from type_mapper import TypeMapper
            return TypeMapper.map_c_type(t)
        except Exception:
            return "*mut std::ffi::c_void"

    def _c_decl_to_rust_extern_decl(self, proto: str) -> Optional[Tuple[str, str]]:
        """
        Convert a C declaration prototype into a Rust extern declaration line:
          `pub fn Foo(x: *mut T) -> i32;`

        Returns: (name, decl_line_without_indent)
        """
        parsed = self._parse_c_decl_prototype(proto)
        if not parsed:
            return None
        name, ret_type, params, is_variadic = parsed
        if not name:
            return None

        # Return type
        ret_t = (ret_type or "").strip()
        if ret_t in ("", "void"):
            rust_ret = ""
        else:
            rust_ret_ty = self._map_c_type_to_rust_best_effort(ret_t)
            rust_ret = f" -> {rust_ret_ty}"

        # Params
        rust_params: List[str] = []
        for idx, p in enumerate(params):
            pname, ptype = self._split_c_param_decl(p, default_name=f"arg{idx}")
            rust_name = pname
            try:
                from type_mapper import TypeMapper

                rust_name = TypeMapper.sanitize_identifier(pname)
            except Exception:
                pass
            rust_ty = self._map_c_type_to_rust_best_effort(ptype)
            # Rust extern doesn't accept `()` as a param type; fall back to void*
            if rust_ty == "()":
                rust_ty = "*mut std::ffi::c_void"
            rust_params.append(f"{rust_name}: {rust_ty}")

        if is_variadic:
            rust_params.append("...")
        params_str = ", ".join(rust_params)
        decl = f"pub fn {name}({params_str}){rust_ret};"
        # Preserve common aliasing attributes when available, so Rust-side calls can still link
        # to the real symbol name (e.g., weakref/alias wrappers).
        try:
            m = re.search(r'weakref\s*\(\s*"([^"]+)"\s*\)', proto or "")
            if not m:
                m = re.search(r'alias\s*\(\s*"([^"]+)"\s*\)', proto or "")
            if m:
                target = (m.group(1) or "").strip()
                if target and target != name:
                    decl = f'#[link_name = "{target}"]\n{decl}'
        except Exception:
            pass
        return name, decl

    @staticmethod
    def _find_bindgen_binary() -> Optional[str]:
        """
        Locate `bindgen` CLI.

        Note: We intentionally use the CLI to match the existing skeleton_builder pipeline.
        """
        p = shutil.which("bindgen")
        if p:
            return p
        cargo_bin = Path.home() / ".cargo" / "bin" / "bindgen"
        if cargo_bin.exists():
            return str(cargo_bin)
        return None

    @staticmethod
    def _parse_defined_type_names_from_types_rs(types_rs: str) -> Set[str]:
        """
        Extract type names that are available to callers via `crate::types::*` (best-effort).

        We count both:
        - Direct definitions: `pub struct/enum/union/type Name`
        - Re-exports: `pub use path::{T1, T2}` / `pub use path::T`

        This is intentionally conservative to avoid generating duplicate type defs in Step 2.55.
        """
        if not types_rs:
            return set()
        out: Set[str] = set(re.findall(r"\bpub\s+(?:struct|enum|union|type)\s+([A-Za-z_]\w*)\b", types_rs))

        # `pub use foo::{A, B as C};`
        for m in re.findall(r"\bpub\s+use\s+[^;]*::\s*\{([^}]+)\}\s*;", types_rs):
            inner = (m or "").strip()
            if not inner:
                continue
            for item in inner.split(","):
                s = (item or "").strip()
                if not s or s == "*" or s.endswith("::*"):
                    continue
                if s in {"self", "super"}:
                    continue
                # Handle `X as Y`
                if " as " in s:
                    try:
                        _lhs, rhs = s.split(" as ", 1)
                        s = (rhs or "").strip()
                    except Exception:
                        s = s.split()[-1].strip()
                # Drop path prefixes if present.
                s = s.split("::")[-1].strip()
                if not re.match(r"^(?:r#)?[A-Za-z_]\w*$", s):
                    continue
                out.add(s)
                if s.startswith("r#"):
                    out.add(s[2:])

        # `pub use foo::Bar;` (ignore glob re-exports)
        for m in re.findall(r"\bpub\s+use\s+[^;]*::\s*((?:r#)?[A-Za-z_]\w*)\s*;", types_rs):
            s = (m or "").strip()
            if not s or s == "*" or s.endswith("::*"):
                continue
            out.add(s)
            if s.startswith("r#"):
                out.add(s[2:])

        return out

    @staticmethod
    def _parse_bindgen_types_and_fns(bindgen_rs: str) -> Tuple[Dict[str, str], Dict[str, str]]:
        """
        Parse bindgen output and return:
          - type_blocks: {TypeName: "...\n"} for `pub type` / `pub struct|union|enum`
          - fn_blocks: {FnRustName: "...\n"} for `extern \"C\" { pub fn ...; }` items (including `#[link_name]`)
        """
        if not bindgen_rs:
            return {}, {}

        type_blocks: Dict[str, str] = {}
        fn_blocks: Dict[str, str] = {}

        lines = bindgen_rs.splitlines()
        i = 0
        pending_attrs: List[str] = []

        def _flush_attrs_on_blank(line: str) -> None:
            nonlocal pending_attrs
            if not line.strip():
                pending_attrs = []

        def _normalize_sig(sig_lines: List[str]) -> str:
            sig = " ".join((ln or "").strip() for ln in sig_lines if (ln or "").strip())
            return re.sub(r"\s+", " ", sig).strip()

        # Pass 1: collect type blocks (top-level)
        i = 0
        pending_attrs = []
        while i < len(lines):
            line = lines[i]
            _flush_attrs_on_blank(line)
            s = line.strip()
            if s.startswith("#["):
                pending_attrs.append(line)
                i += 1
                continue
            m_type = re.match(r"^pub\s+type\s+([A-Za-z_]\w*)\s*=", s)
            if m_type:
                name = m_type.group(1)
                buf: List[str] = []
                if pending_attrs:
                    buf.extend(pending_attrs)
                pending_attrs = []
                buf.append(line)
                i += 1
                while i < len(lines) and not (buf[-1] or "").strip().endswith(";"):
                    buf.append(lines[i])
                    i += 1
                type_blocks[name] = "\n".join(buf).rstrip() + "\n"
                continue
            m_struct = re.match(r"^pub\s+(struct|union|enum)\s+([A-Za-z_]\w*)\b", s)
            if m_struct:
                name = m_struct.group(2)
                buf = []
                if pending_attrs:
                    buf.extend(pending_attrs)
                pending_attrs = []
                buf.append(line)
                depth = line.count("{") - line.count("}")
                i += 1
                while i < len(lines) and depth > 0:
                    buf.append(lines[i])
                    depth += lines[i].count("{") - lines[i].count("}")
                    i += 1
                type_blocks[name] = "\n".join(buf).rstrip() + "\n"
                continue
            pending_attrs = []
            i += 1

        # Pass 2: collect extern fn blocks
        i = 0
        pending_attrs = []
        in_extern = False
        while i < len(lines):
            line = lines[i]
            s = line.strip()
            if re.match(r'^(?:unsafe\s+)?extern\s+"C"\s*\{$', s):
                in_extern = True
                pending_attrs = []
                i += 1
                continue
            if in_extern and s == "}":
                in_extern = False
                pending_attrs = []
                i += 1
                continue
            if not in_extern:
                i += 1
                continue
            if not s:
                pending_attrs = []
                i += 1
                continue
            if s.startswith("#["):
                pending_attrs.append(line)
                i += 1
                continue
            if s.startswith("pub fn "):
                m = re.match(r"^pub\s+fn\s+(r#[A-Za-z_]\w*|[A-Za-z_]\w*)\b", s)
                if not m:
                    pending_attrs = []
                    i += 1
                    continue
                fn_name = m.group(1)
                sig_lines = [line]
                i += 1
                while i < len(lines) and not (sig_lines[-1] or "").strip().endswith(";"):
                    sig_lines.append(lines[i])
                    i += 1
                sig = _normalize_sig(sig_lines)
                block_lines: List[str] = []
                if pending_attrs:
                    block_lines.extend(pending_attrs)
                block_lines.append(sig)
                blk = "\n".join(block_lines).rstrip() + "\n"
                fn_blocks[fn_name] = blk
                # bindgen may emit `r#keyword` for Rust-reserved identifiers; map both forms for lookup convenience.
                if fn_name.startswith("r#"):
                    fn_blocks.setdefault(fn_name[2:], blk)
                pending_attrs = []
                continue
            pending_attrs = []
            i += 1

        return type_blocks, fn_blocks

    @staticmethod
    def _parse_bindgen_consts_and_vars(bindgen_rs: str) -> Tuple[Dict[str, str], Dict[str, str]]:
        """
        Parse bindgen output and extract:
          - const blocks: `pub const NAME: Ty = ...;` (may be multi-line)
          - extern var blocks: `pub static (mut)? NAME: Ty;` inside `extern "C" { ... }`

        Returns: (const_blocks, var_blocks) where each value is a normalized block string.

        Notes:
        - Keep attributes directly above each item (best-effort).
        - bindgen may emit `r#keyword` identifiers; map both `r#x` and `x` for lookup convenience.
        """
        if not bindgen_rs:
            return {}, {}

        const_blocks: Dict[str, str] = {}
        var_blocks: Dict[str, str] = {}

        lines = bindgen_rs.splitlines()

        def _normalize_sig(sig_lines: List[str]) -> str:
            sig = " ".join((ln or "").strip() for ln in sig_lines if (ln or "").strip())
            return re.sub(r"\s+", " ", sig).strip()

        i = 0
        pending_attrs: List[str] = []
        in_extern = False
        while i < len(lines):
            line = lines[i]
            s = (line or "").strip()

            if not s:
                pending_attrs = []
                i += 1
                continue

            if s.startswith("#["):
                pending_attrs.append(line)
                i += 1
                continue

            if s in {'extern "C" {', 'unsafe extern "C" {'}:
                in_extern = True
                pending_attrs = []
                i += 1
                continue

            if in_extern and s == "}":
                in_extern = False
                pending_attrs = []
                i += 1
                continue

            if (not in_extern) and s.startswith("pub const "):
                m = re.match(r"^pub\s+const\s+([A-Za-z_]\w*|r#[A-Za-z_]\w*)\b", s)
                if not m:
                    pending_attrs = []
                    i += 1
                    continue
                name = m.group(1)
                sig_lines = [line]
                i += 1
                while i < len(lines) and not (sig_lines[-1] or "").strip().endswith(";"):
                    sig_lines.append(lines[i])
                    i += 1
                sig = _normalize_sig(sig_lines)
                block_lines: List[str] = []
                if pending_attrs:
                    block_lines.extend(pending_attrs)
                block_lines.append(sig)
                blk = "\n".join(block_lines).rstrip() + "\n"
                const_blocks[name] = blk
                if name.startswith("r#"):
                    const_blocks.setdefault(name[2:], blk)
                pending_attrs = []
                continue

            if in_extern and s.startswith("pub static "):
                m = re.match(r"^pub\s+static(?:\s+mut)?\s+([A-Za-z_]\w*|r#[A-Za-z_]\w*)\b", s)
                if not m:
                    pending_attrs = []
                    i += 1
                    continue
                name = m.group(1)
                sig_lines = [line]
                i += 1
                while i < len(lines) and not (sig_lines[-1] or "").strip().endswith(";"):
                    sig_lines.append(lines[i])
                    i += 1
                sig = _normalize_sig(sig_lines)
                block_lines = []
                if pending_attrs:
                    block_lines.extend(pending_attrs)
                block_lines.append(sig)
                blk = "\n".join(block_lines).rstrip() + "\n"
                var_blocks[name] = blk
                if name.startswith("r#"):
                    var_blocks.setdefault(name[2:], blk)
                pending_attrs = []
                continue

            pending_attrs = []
            i += 1

        return const_blocks, var_blocks

    @staticmethod
    def _prefix_types_in_rust_fn_decl(fn_block: str, type_names: Set[str]) -> str:
        """
        Rewrite a bindgen `pub fn ...` decl block to reference `crate::types::X` for known type names.

        This keeps call sites using `crate::types::*` compatible (no shadow "duplicate" type defs in compat.rs).
        """
        if not fn_block or not type_names:
            return fn_block or ""

        def _split_top_level_commas(s: str) -> List[str]:
            parts: List[str] = []
            buf: List[str] = []
            depth_paren = 0
            depth_angle = 0
            depth_brack = 0
            depth_brace = 0
            for ch in s:
                if ch == "(":
                    depth_paren += 1
                elif ch == ")":
                    depth_paren = max(0, depth_paren - 1)
                elif ch == "<":
                    depth_angle += 1
                elif ch == ">":
                    depth_angle = max(0, depth_angle - 1)
                elif ch == "[":
                    depth_brack += 1
                elif ch == "]":
                    depth_brack = max(0, depth_brack - 1)
                elif ch == "{":
                    depth_brace += 1
                elif ch == "}":
                    depth_brace = max(0, depth_brace - 1)
                if ch == "," and depth_paren == 0 and depth_angle == 0 and depth_brack == 0 and depth_brace == 0:
                    parts.append("".join(buf))
                    buf = []
                    continue
                buf.append(ch)
            parts.append("".join(buf))
            return parts

        def _prefix_type_expr(expr: str) -> str:
            """
            Prefix standalone type identifiers in a Rust type expression.

            IMPORTANT: only used on the *type* side; never on parameter names (avoids E0130 in extern decls).
            """
            if not expr:
                return ""
            s = expr
            res: List[str] = []
            i = 0
            last_token = ""
            while i < len(s):
                if s.startswith("::", i):
                    res.append("::")
                    i += 2
                    last_token = "::"
                    continue
                m = re.match(r"[A-Za-z_][A-Za-z0-9_]*", s[i:])
                if m:
                    ident = m.group(0)
                    if ident in type_names and last_token != "::":
                        res.append(f"crate::types::{ident}")
                    else:
                        res.append(ident)
                    i += len(ident)
                    last_token = ident
                    continue
                res.append(s[i])
                i += 1
            return "".join(res)

        out_lines: List[str] = []
        for raw in (fn_block or "").splitlines():
            line = raw
            if not line.strip().startswith("pub fn "):
                out_lines.append(line)
                continue

            # Parse and rewrite: ONLY touch the type positions.
            indent = line[: len(line) - len(line.lstrip())]
            s = line.strip()
            # Find the parameter list bounds (handle nested function pointer types).
            try:
                fn_pos = s.find("fn ")
                start = s.find("(", fn_pos if fn_pos != -1 else 0)
            except Exception:
                start = -1
            if start < 0:
                out_lines.append(line)
                continue
            depth = 0
            end = None
            for j in range(start, len(s)):
                if s[j] == "(":
                    depth += 1
                elif s[j] == ")":
                    depth -= 1
                    if depth == 0:
                        end = j
                        break
            if end is None:
                out_lines.append(line)
                continue
            head = s[:start].rstrip()
            params_raw = s[start + 1 : end].strip()
            tail = s[end + 1 :].strip()
            semicolon = ";" if tail.endswith(";") else ""
            if semicolon:
                tail = tail[:-1].rstrip()

            ret_raw = ""
            if "->" in tail:
                _, ret_raw = tail.split("->", 1)
                ret_raw = ret_raw.strip()

            params_out: List[str] = []
            if params_raw:
                for item in _split_top_level_commas(params_raw):
                    item = (item or "").strip()
                    if not item:
                        continue
                    if item == "...":
                        params_out.append("...")
                        continue
                    if ":" not in item:
                        params_out.append(item)
                        continue
                    name_part, ty_part = item.split(":", 1)
                    name_part = (name_part or "").strip()
                    ty_part = _prefix_type_expr((ty_part or "").strip())
                    params_out.append(f"{name_part}: {ty_part}".strip())
            params_joined = ", ".join(params_out)
            ret_clause = f" -> {_prefix_type_expr(ret_raw)}" if ret_raw else ""
            out_lines.append(f"{indent}{head}({params_joined}){ret_clause}{semicolon}".rstrip())
        return "\n".join(out_lines).rstrip() + "\n"

    @staticmethod
    def _compute_needed_crate_types_imports(*, type_blocks: Dict[str, str], defined_types: Set[str]) -> List[str]:
        """
        Compute a minimal `use crate::types::{...};` import list needed for bindgen-emitted Rust type blocks.

        Background:
        - Step2.55 emits missing type declarations into `compat.rs` (to avoid polluting the truth-layer `types.rs`).
        - Those emitted types may still reference other types that already exist in `crate::types`.
        - If we keep those references as bare identifiers, baseline compile fails with E0412.
          Example (host): `HdfDriverEntry` references `HdfDeviceObject` which is re-exported in `types.rs`,
          but not in scope inside `compat.rs`.
        """
        if not type_blocks or not defined_types:
            return []
        locally_defined = {k for k in (type_blocks or {}).keys() if isinstance(k, str) and k.strip()}
        needed: Set[str] = set()
        for blk in (type_blocks or {}).values():
            for ident in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", blk or ""):
                if ident in defined_types and ident not in locally_defined:
                    needed.add(ident)
        return sorted(needed)

    @staticmethod
    def _qualify_types_in_rust_code(code: str, defined_types: Set[str]) -> str:
        """
        Replace bare type names with fully qualified `crate::types::TypeName` in Rust code.

        This is used for extern vars/statics where `use` imports may not work inside `extern "C"` blocks.

        Args:
            code: Rust code snippet (e.g., `pub static mut named_colors: [named_color; 149];`)
            defined_types: Set of type names defined in `crate::types`

        Returns:
            Code with type references qualified with `crate::types::`
        """
        if not code or not defined_types:
            return code

        # Types that should NOT be qualified (core/primitive types)
        core_types = {
            # Rust primitives
            'i8', 'i16', 'i32', 'i64', 'i128', 'isize',
            'u8', 'u16', 'u32', 'u64', 'u128', 'usize',
            'f32', 'f64', 'bool', 'char', 'str',
            # core::ffi types
            'c_void', 'c_char', 'c_schar', 'c_uchar', 'c_short', 'c_ushort',
            'c_int', 'c_uint', 'c_long', 'c_ulong', 'c_longlong', 'c_ulonglong',
            'c_float', 'c_double',
            # Keywords and common identifiers
            'mut', 'pub', 'static', 'const', 'extern', 'unsafe', 'fn',
            'Option', 'Some', 'None', 'Result', 'Ok', 'Err',
        }

        # Parse the code to find type positions (after `:`, inside `[]`, after `*mut`/`*const`)
        # We look for identifiers that are in defined_types and in type context
        result = code

        for type_name in defined_types:
            if type_name in core_types:
                continue
            # Pattern: word boundary + type_name + word boundary
            # But NOT preceded by :: (already qualified) or followed by :: (path prefix)
            # Type context patterns:
            # 1. After `:` - e.g., `: TypeName` or `: [TypeName; N]`
            # 2. Inside `[]` - e.g., `[TypeName; N]`
            # 3. After `*mut` or `*const` - e.g., `*mut TypeName`

            # Simple approach: replace `TypeName` with `crate::types::TypeName` if not already qualified
            # Use negative lookbehind to avoid replacing already qualified names
            pattern = rf'(?<!::)\b{re.escape(type_name)}\b(?!::)'
            replacement = f'crate::types::{type_name}'

            # Only replace in type contexts (after : or inside [ ] or after *mut/*const)
            # For safety, we do a more targeted replacement
            # IMPORTANT: Use negative lookbehind to avoid re-qualifying already qualified types

            # Pattern 1: After `:` (type annotation context) - but not after `types::`
            result = re.sub(rf'(?<!types:)(:\s*(?:\[)?){re.escape(type_name)}\b(?!::)', rf'\1crate::types::{type_name}', result)
            # Pattern 2: After `*mut ` or `*const ` - but not after `types::`
            result = re.sub(rf'(?<!types::)(\*(?:mut|const)\s+){re.escape(type_name)}\b(?!::)', rf'\1crate::types::{type_name}', result)
            # Pattern 3: Inside `[` (array type) - but not after `types::`
            result = re.sub(rf'(?<!types::)(\[\s*){re.escape(type_name)}\b(?!::)', rf'\1crate::types::{type_name}', result)

        return result

    @staticmethod
    def _upsert_marked_block(content: str, *, begin: str, end: str, block: str, anchor: Optional[str] = None) -> str:
        """
        Replace or insert a `begin..end` marked block into `content`.
        """
        if content is None:
            content = ""
        start = content.find(begin)
        stop = content.find(end)
        if start != -1 and stop != -1 and stop > start:
            stop = stop + len(end)
            return content[:start] + block.rstrip("\n") + content[stop:]
        if anchor:
            pos = content.find(anchor)
            if pos != -1:
                return content[:pos] + block + content[pos:]
        if content and not content.endswith("\n"):
            content += "\n"
        return content + block

    def _ensure_extern_decls_from_bindgen_allowlist(self, functions: Dict[str, FunctionInfo]) -> int:
        """
        Step 2.55 (preferred): generate extern decls via bindgen allowlist on preprocessed `.i`.

        - Extract external callees (called but not translated in this project).
        - Run `bindgen --allowlist-function` on the TU's `.i` file to get correct Rust signatures.
        - Backfill missing typedefs/structs into `src/types.rs` (marker block) so compat.rs can reference them.
        - Inject the extern decls into `src/compat.rs` (marker block).
        """
        if not self._enable_bindgen_extern_prelude:
            return 0
        if not self._enable_preprocessed_decl_hints:
            # `.i` files are still the most stable input for TU-level bindgen.
            return 0
        compat_path = self.work_dir / "src" / "compat.rs"
        types_path = self.work_dir / "src" / "types.rs"
        if not compat_path.exists() or not types_path.exists():
            return 0
        if not functions:
            return 0

        bindgen_bin = self._find_bindgen_binary()
        if not bindgen_bin:
            logger.debug("bindgen 未找到，跳过 bindgen extern prelude")
            return 0

        try:
            types_rs_text = types_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            types_rs_text = ""
        defined_types: Set[str] = self._parse_defined_type_names_from_types_rs(types_rs_text)

        internal_names: Set[str] = {fi.name for fi in functions.values() if getattr(fi, "name", None)}

        by_file: Dict[str, List[FunctionInfo]] = defaultdict(list)
        for fi in functions.values():
            by_file[fi.file_name].append(fi)

        extra_types_by_name: Dict[str, str] = {}
        decls_by_name: Dict[str, str] = {}
        decl_reports: List[Dict[str, Any]] = []
        missing_reports: List[Dict[str, Any]] = []
        requested_total = 0

        tmp_dir = self.work_dir / ".c2r_bindgen_extern"
        try:
            tmp_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        for file_name, fis in by_file.items():
            if not fis:
                continue
            rep = fis[0]
            pre_path = self._locate_preprocessed_file(rep)
            if not pre_path or not pre_path.exists():
                continue

            called: Set[str] = set()
            for fi in fis:
                called.update(self._extract_called_identifiers_from_c(fi.c_code or ""))
            called.difference_update(internal_names)
            # keep valid identifiers only
            called = {c for c in called if c and re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", c)}
            # `static inline` helpers are not stable link symbols; do NOT generate `extern "C"` for them.
            try:
                inline_names = self._get_preprocessed_static_inline_names(pre_path)
                if inline_names:
                    removed_inline = set(called) & set(inline_names)
                    if removed_inline:
                        called.difference_update(removed_inline)
                        logger.info(
                            f"[Step2.55] {file_name}: 跳过 static inline helper extern decls: {sorted(removed_inline)[:8]}"
                        )
            except Exception:
                pass
            if not called:
                continue
            requested_total += len(called)
            if self._bindgen_extern_max_funcs_per_tu and len(called) > self._bindgen_extern_max_funcs_per_tu:
                called = set(sorted(called)[: self._bindgen_extern_max_funcs_per_tu])

            # Cache by `.i` path + callee set (best-effort: if cache exists, reuse).
            # The allowlist set affects bindgen output; without it, a "small run" can poison later results.
            callee_key_hasher = hashlib.sha1()
            for name in sorted(called):
                callee_key_hasher.update(name.encode("utf-8", errors="ignore"))
                callee_key_hasher.update(b"\0")
            callee_key = callee_key_hasher.hexdigest()
            cache_key = (str(pre_path), callee_key)
            cached = self._bindgen_extern_cache.get(cache_key)
            if cached is None or not isinstance(cached, dict) or not cached.get("raw_rs"):
                # Run bindgen
                out_rs = tmp_dir / f"{pre_path.stem}.externs.rs"

                lang = "c"
                src = self._locate_source_file(rep)
                if src and src.suffix.lower() in {".cc", ".cpp", ".cxx", ".c++"}:
                    lang = "c++"

                cmd: List[str] = [
                    bindgen_bin,
                    str(pre_path),
                    "-o",
                    str(out_rs),
                    "--no-layout-tests",
                    "--no-doc-comments",
                    "--use-core",
                    "--default-enum-style=consts",
                    "--no-prepend-enum-name",
                    "--no-size_t-is-usize",
                ]
                for name in sorted(called):
                    cmd.extend(["--allowlist-function", rf"^{re.escape(name)}$"])
                cmd.append("--")
                if lang == "c++":
                    cmd.extend(["-x", "c++", "-std=c++17"])
                else:
                    cmd.extend(["-x", "c"])
                cmd.extend(
                    [
                        "-Wno-error",
                        "-Wno-macro-redefined",
                        "-Wno-builtin-macro-redefined",
                        "-Wno-ignored-attributes",
                    ]
                )
                # Allow env-var injection of extra clang flags (e.g. for
                # -include stubs to resolve OHOS macros in minimal ohos_root_min).
                _extra_flags = os.environ.get(
                    "C2R_BINDGEN_EXTERN_CLANG_FLAGS", ""
                ).strip()
                if _extra_flags:
                    cmd.extend(shlex.split(_extra_flags))

                try:
                    proc = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=max(10, int(self._bindgen_extern_timeout_sec or 90)),
                    )
                except Exception as e:
                    logger.debug(f"bindgen extern prelude 失败 [{file_name}]: {e}")
                    missing_reports.append(
                        {
                            "source_file_group": file_name,
                            "source_i": str(pre_path),
                            "reason": "bindgen_exception",
                            "error": str(e),
                            "requested_callees": sorted(called),
                        }
                    )
                    continue
                if proc.returncode != 0 or not out_rs.exists():
                    logger.debug(
                        f"bindgen extern prelude 失败 [{file_name}] rc={proc.returncode}: {(proc.stderr or '')[:200]}"
                    )
                    missing_reports.append(
                        {
                            "source_file_group": file_name,
                            "source_i": str(pre_path),
                            "reason": "bindgen_failed",
                            "returncode": proc.returncode,
                            "stderr_head": (proc.stderr or "")[:500],
                            "requested_callees": sorted(called),
                        }
                    )
                    continue

                try:
                    bindgen_text = out_rs.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    bindgen_text = ""

                type_blocks, fn_blocks = self._parse_bindgen_types_and_fns(bindgen_text)
                cached = {
                    "raw_rs": bindgen_text,
                    "types": type_blocks,
                    "fns": fn_blocks,
                    "source_i": str(pre_path),
                    "file_group": file_name,
                }
                self._bindgen_extern_cache[cache_key] = cached

            type_blocks = cached.get("types") or {}
            fn_blocks = cached.get("fns") or {}
            missing_callees: List[str] = []

            for callee in sorted(called):
                if callee in decls_by_name:
                    continue
                fn_block = fn_blocks.get(callee)
                if not fn_block:
                    # bindgen may rename keywords; try link_name scan as a fallback
                    # (keep simple: best-effort)
                    for rust_name, blk in fn_blocks.items():
                        if re.search(rf"\bpub\s+fn\s+{re.escape(rust_name)}\b", blk) and re.search(
                            rf'link_name\s*=\s*"{re.escape(callee)}"', blk
                        ):
                            fn_block = blk
                            break
                if not fn_block:
                    missing_callees.append(callee)
                    continue

                # Backfill missing types referenced by this fn from bindgen output.
                needed: Set[str] = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", fn_block))
                queue: List[str] = [t for t in sorted(needed) if t in type_blocks]
                seen: Set[str] = set()
                added_count = 0
                while queue:
                    tname = queue.pop(0)
                    if tname in seen:
                        continue
                    seen.add(tname)
                    if tname in defined_types or tname in extra_types_by_name:
                        continue
                    block = type_blocks.get(tname)
                    if not block:
                        continue
                    extra_types_by_name[tname] = block
                    added_count += 1
                    if added_count > 200:
                        break
                    # recursive closure (best-effort)
                    for ident in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", block):
                        if ident in type_blocks and ident not in seen:
                            queue.append(ident)

                # IMPORTANT: only prefix types that already exist in `crate::types` (truth layer).
                # Missing types are emitted into `compat.rs` (Step 2.55 extern_types) and must stay unqualified here.
                available_types = set(defined_types)
                rewritten = self._prefix_types_in_rust_fn_decl(fn_block, available_types)

                decls_by_name[callee] = rewritten.rstrip()  # keep as a block (may include attrs)
                decl_reports.append(
                    {
                        "name": callee,
                        "rust_decl": rewritten.rstrip(),
                        "source_i": cached.get("source_i"),
                        "source_file_group": cached.get("file_group"),
                    }
                )

            if missing_callees:
                missing_reports.append(
                    {
                        "source_file_group": file_name,
                        "source_i": str(pre_path),
                        "reason": "missing_in_bindgen_output",
                        "missing_callees": sorted(missing_callees),
                        "requested_callees": sorted(called),
                    }
                )

        # Always write a report for audit (even if no decls were generated), so users can diagnose closure issues.
        report_path = self.manual_fix_root / "extern_decls_from_bindgen_allowlist.json"
        try:
            payload = {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "project": getattr(self, "project_name", None),
                "llm_name": getattr(self, "llm_name", None),
                "workspace_root": str(self.workspace_root),
                "preprocessed_dir": str(self.preprocessed_dir),
                "types_rs": str(types_path),
                "compat_rs": str(compat_path),
                "requested_callees_total": int(requested_total),
                "extern_count": len(decls_by_name),
                "extra_types_count": len(extra_types_by_name),
                "decls": sorted(decl_reports, key=lambda x: (x.get("name") or "")),
                "extra_types": sorted(extra_types_by_name.keys()),
                "missing": missing_reports,
            }
            _atomic_write_text(report_path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
            self._extern_decls_report_path = report_path
            self._extern_decls_missing = missing_reports
        except Exception as e:
            logger.warning(f"写入 bindgen extern 声明报告失败（已忽略）: {e}")

        if not decls_by_name:
            return 0

        # 1) Write extra bindgen types into compat.rs (marker block)
        # Rationale: avoid polluting `types.rs` (truth layer). Only emit truly-missing type items.
        try:
            compat_text = compat_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            compat_text = ""
        if extra_types_by_name:
            types_begin = "// === C2R_EXTERN_TYPES_BEGIN ==="
            types_end = "// === C2R_EXTERN_TYPES_END ==="
            type_lines: List[str] = []
            type_lines.append(types_begin)
            type_lines.append("// Auto-generated by C2R step 2.55 (bindgen allowlist extern prelude).")
            type_lines.append("// Only includes missing type items required by extern decls below.")
            imports = self._compute_needed_crate_types_imports(type_blocks=extra_types_by_name, defined_types=defined_types)
            if imports:
                type_lines.append(f"use crate::types::{{{', '.join(imports)}}};")
                type_lines.append("")
            for name in sorted(extra_types_by_name.keys()):
                type_lines.append(extra_types_by_name[name].rstrip())
                type_lines.append("")
            type_lines.append(types_end)
            type_block = "\n".join(type_lines).rstrip() + "\n\n"
            compat_text = self._upsert_marked_block(
                compat_text,
                begin=types_begin,
                end=types_end,
                block=type_block,
                anchor="// === C2R_ACCESSOR_SHIMS_BEGIN ===",
            )

        # 2) Inject extern decls into compat.rs (marker block)
        section_lines: List[str] = []
        section_lines.append("// === C2R_EXTERN_DECLS_BEGIN ===")
        section_lines.append("// Auto-generated extern decls (C2R step 2.55; bindgen allowlist).")
        section_lines.append(f"// Source: {self.preprocessed_dir}/*.i (preprocessed translation units)")
        section_lines.append(f"// Details: {report_path}")
        section_lines.append("#[allow(improper_ctypes)]")
        section_lines.append("#[allow(non_snake_case)]")
        section_lines.append('extern "C" {')
        for name in sorted(decls_by_name.keys()):
            for ln in (decls_by_name[name] or "").splitlines():
                if not ln.strip():
                    continue
                # keep attributes unindented; indent decl itself.
                if ln.lstrip().startswith("#["):
                    section_lines.append(f"    {ln.strip()}")
                else:
                    section_lines.append(f"    {ln.strip()}")
        section_lines.append("}")
        section_lines.append("// === C2R_EXTERN_DECLS_END ===")
        section = "\n".join(section_lines).rstrip() + "\n\n"

        compat_new = self._upsert_marked_block(
            compat_text,
            begin="// === C2R_EXTERN_DECLS_BEGIN ===",
            end="// === C2R_EXTERN_DECLS_END ===",
            block=section,
            anchor="// === C2R_ACCESSOR_SHIMS_BEGIN ===",
        )
        _atomic_write_text(compat_path, compat_new)

        return len(decls_by_name)

    def _ensure_extern_vars_from_bindgen_allowlist(self, *, missing_names: Set[str]) -> int:
        """
        Step 2.56: generate extern var/const items via bindgen allowlist on preprocessed `.i`.

        Motivation:
        - Rust modules import `use crate::compat::*;` so putting `pub const` / `extern "C" { pub static ... }`
          here makes missing values resolvable without per-function guessing.
        - This should ONLY be used for true extern globals / constified globals that exist in the pinned TU `.i`.
          If bindgen cannot find the symbol, we do NOT fabricate a placeholder.
        """
        if not self._enable_bindgen_extern_vars_prelude:
            return 0
        if not missing_names:
            return 0

        compat_path = self.work_dir / "src" / "compat.rs"
        types_path = self.work_dir / "src" / "types.rs"
        if not compat_path.exists() or not types_path.exists():
            return 0

        bindgen_bin = self._find_bindgen_binary()
        if not bindgen_bin:
            return 0

        # Valid identifier candidates only.
        candidates: Set[str] = {
            n for n in (missing_names or set()) if isinstance(n, str) and re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", n)
        }
        if not candidates:
            return 0

        try:
            types_rs_text = types_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            types_rs_text = ""
        defined_types: Set[str] = self._parse_defined_type_names_from_types_rs(types_rs_text)

        try:
            compat_text = compat_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            compat_text = ""

        # Avoid duplicating items already present anywhere in compat.rs (including placeholder blocks).
        already_in_compat: Set[str] = set()
        try:
            for m in re.findall(r"\bpub\s+const\s+((?:r#)?[A-Za-z_]\w*)\b", compat_text):
                already_in_compat.add(m)
                if m.startswith("r#"):
                    already_in_compat.add(m[2:])
            for m in re.findall(r"\bpub\s+static(?:\s+mut)?\s+((?:r#)?[A-Za-z_]\w*)\b", compat_text):
                already_in_compat.add(m)
                if m.startswith("r#"):
                    already_in_compat.add(m[2:])
        except Exception:
            already_in_compat = set()

        candidates.difference_update(already_in_compat)
        if not candidates:
            return 0

        # Only use TU-pinned `.i` files that are included in the selected build (Route A).
        tu_items: List[Tuple[str, Path, str]] = []
        try:
            for safe_name, rec in sorted((self._tu_context_files or {}).items(), key=lambda kv: kv[0]):
                if not isinstance(rec, dict):
                    continue
                if rec.get("excluded_by_compile_commands"):
                    continue
                pre = rec.get("preprocessed_file")
                if not pre:
                    continue
                try:
                    pre_path = Path(str(pre)).expanduser()
                except Exception:
                    continue
                if not pre_path.exists():
                    continue
                lang = "c"
                try:
                    src = rec.get("source_for_cc_abs") or rec.get("source_file_abs") or ""
                    suf = Path(str(src)).suffix.lower()
                    if suf in {".cc", ".cpp", ".cxx", ".c++"}:
                        lang = "c++"
                except Exception:
                    lang = "c"
                tu_items.append((str(safe_name), pre_path, lang))
        except Exception:
            tu_items = []

        if not tu_items:
            return 0

        tmp_dir = self.work_dir / ".c2r_bindgen_extern_vars"
        try:
            tmp_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        report_path = self.manual_fix_root / "extern_vars_from_bindgen_allowlist.json"
        requested_total = int(len(candidates))
        remaining: Set[str] = set(candidates)
        decl_reports: List[Dict[str, Any]] = []
        missing_reports: List[Dict[str, Any]] = []

        # Merge with existing compat extern-types/vars blocks (so later passes don't lose earlier items).
        existing_extra_types_by_name: Dict[str, str] = {}
        existing_const_blocks: Dict[str, str] = {}
        existing_var_blocks: Dict[str, str] = {}
        try:
            types_begin = "// === C2R_EXTERN_TYPES_BEGIN ==="
            types_end = "// === C2R_EXTERN_TYPES_END ==="
            ts = compat_text.find(types_begin)
            te = compat_text.find(types_end)
            if ts != -1 and te != -1 and te > ts:
                block = compat_text[ts : te + len(types_end)]
                existing_extra_types_by_name, _ = self._parse_bindgen_types_and_fns(block)
        except Exception:
            existing_extra_types_by_name = {}
        try:
            vars_begin = "// === C2R_EXTERN_VARS_BEGIN ==="
            vars_end = "// === C2R_EXTERN_VARS_END ==="
            vs = compat_text.find(vars_begin)
            ve = compat_text.find(vars_end)
            if vs != -1 and ve != -1 and ve > vs:
                block = compat_text[vs : ve + len(vars_end)]
                existing_const_blocks, existing_var_blocks = self._parse_bindgen_consts_and_vars(block)
        except Exception:
            existing_const_blocks, existing_var_blocks = {}, {}

        new_extra_types_by_name: Dict[str, str] = {}
        new_const_blocks: Dict[str, str] = {}
        new_var_blocks: Dict[str, str] = {}

        # Helper to collect a minimal type-closure from bindgen output.
        def _collect_type_closure(type_blocks: Dict[str, str], seed_blocks: List[str]) -> None:
            if not type_blocks or not seed_blocks:
                return
            needed: Set[str] = set()
            for blk in seed_blocks:
                for ident in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", blk or ""):
                    if ident in type_blocks:
                        needed.add(ident)
            queue: List[str] = sorted(needed)
            seen: Set[str] = set()
            added = 0
            while queue:
                tname = queue.pop(0)
                if tname in seen:
                    continue
                seen.add(tname)
                if tname in defined_types or tname in existing_extra_types_by_name or tname in new_extra_types_by_name:
                    continue
                block = type_blocks.get(tname)
                if not block:
                    continue
                new_extra_types_by_name[tname] = block
                added += 1
                if added > 200:
                    break
                for ident in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", block):
                    if ident in type_blocks and ident not in seen:
                        queue.append(ident)

        for file_group, pre_path, lang in tu_items:
            if not remaining:
                break

            # Respect a max per pass to keep command-line length sane.
            allowlist_names = sorted(remaining)
            if self._bindgen_extern_max_vars_per_pass and len(allowlist_names) > self._bindgen_extern_max_vars_per_pass:
                allowlist_names = allowlist_names[: self._bindgen_extern_max_vars_per_pass]

            # Cache key: `.i` path + allowlist set.
            h = hashlib.sha1()
            for name in allowlist_names:
                h.update(name.encode("utf-8", errors="ignore"))
                h.update(b"\0")
            cache_key = (str(pre_path), h.hexdigest())
            cached = self._bindgen_extern_vars_cache.get(cache_key)

            if cached is None or not isinstance(cached, dict) or not cached.get("raw_rs"):
                out_rs = tmp_dir / f"{pre_path.stem}.extern_vars.rs"
                cmd: List[str] = [
                    bindgen_bin,
                    str(pre_path),
                    "-o",
                    str(out_rs),
                    "--generate",
                    "vars",
                    "--no-layout-tests",
                    "--no-doc-comments",
                    "--use-core",
                    "--default-enum-style=consts",
                    "--no-prepend-enum-name",
                    "--no-size_t-is-usize",
                    "--ignore-functions",
                ]
                for name in allowlist_names:
                    cmd.extend(["--allowlist-item", rf"^{re.escape(name)}$"])
                cmd.append("--")
                if (lang or "c").lower().startswith("c++"):
                    cmd.extend(["-x", "c++", "-std=c++17"])
                else:
                    cmd.extend(["-x", "c"])
                cmd.extend(
                    [
                        "-Wno-error",
                        "-Wno-macro-redefined",
                        "-Wno-builtin-macro-redefined",
                        "-Wno-ignored-attributes",
                    ]
                )
                # Allow env-var injection of extra clang flags (e.g. for
                # -include stubs to resolve OHOS macros in minimal ohos_root_min).
                _extra_flags = os.environ.get(
                    "C2R_BINDGEN_EXTERN_CLANG_FLAGS", ""
                ).strip()
                if _extra_flags:
                    cmd.extend(shlex.split(_extra_flags))

                try:
                    proc = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=max(10, int(self._bindgen_extern_vars_timeout_sec or 90)),
                    )
                except Exception as e:
                    missing_reports.append(
                        {
                            "source_file_group": file_group,
                            "source_i": str(pre_path),
                            "reason": "bindgen_failed",
                            "exception": str(e),
                            "requested_names": allowlist_names,
                        }
                    )
                    continue
                if proc.returncode != 0 or not out_rs.exists():
                    missing_reports.append(
                        {
                            "source_file_group": file_group,
                            "source_i": str(pre_path),
                            "reason": "bindgen_nonzero_or_no_output",
                            "rc": int(proc.returncode),
                            "stderr_head": (proc.stderr or "")[:1200],
                            "requested_names": allowlist_names,
                        }
                    )
                    continue

                try:
                    bindgen_text = out_rs.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    bindgen_text = ""

                type_blocks, _fn_blocks = self._parse_bindgen_types_and_fns(bindgen_text)
                const_blocks, var_blocks = self._parse_bindgen_consts_and_vars(bindgen_text)

                cached = {
                    "raw_rs": bindgen_text,
                    "types": type_blocks,
                    "consts": const_blocks,
                    "vars": var_blocks,
                    "source_i": str(pre_path),
                    "file_group": file_group,
                }
                self._bindgen_extern_vars_cache[cache_key] = cached

            type_blocks = cached.get("types") or {}
            const_blocks = cached.get("consts") or {}
            var_blocks = cached.get("vars") or {}

            found_blocks: List[str] = []
            for name in allowlist_names:
                if name in const_blocks and name not in new_const_blocks and name not in existing_const_blocks:
                    blk = const_blocks.get(name) or ""
                    if blk.strip():
                        new_const_blocks[name] = blk
                        found_blocks.append(blk)
                        decl_reports.append(
                            {
                                "name": name,
                                "kind": "const",
                                "rust_decl": blk.rstrip(),
                                "source_i": cached.get("source_i"),
                                "source_file_group": cached.get("file_group"),
                            }
                        )
                        remaining.discard(name)
                if name in var_blocks and name not in new_var_blocks and name not in existing_var_blocks:
                    blk = var_blocks.get(name) or ""
                    if blk.strip():
                        # Safety: only keep vars whose dependent type names are resolvable.
                        # Otherwise we'd emit an extern `static` with an unknown type and break the whole crate.
                        missing_deps: Set[str] = set()
                        try:
                            for ident in re.findall(r"\b[A-Z][A-Za-z0-9_]*\b", blk):
                                if (
                                    ident
                                    and ident not in defined_types
                                    and ident not in type_blocks
                                    and ident not in existing_extra_types_by_name
                                    and ident not in new_extra_types_by_name
                                ):
                                    missing_deps.add(ident)
                        except Exception:
                            missing_deps = set()
                        if missing_deps:
                            missing_reports.append(
                                {
                                    "source_file_group": file_group,
                                    "source_i": str(pre_path),
                                    "reason": "skip_var_unresolved_type_deps",
                                    "name": name,
                                    "kind": "static",
                                    "rust_decl": blk.rstrip(),
                                    "missing_type_deps": sorted(missing_deps),
                                }
                            )
                        else:
                            new_var_blocks[name] = blk
                            found_blocks.append(blk)
                            decl_reports.append(
                                {
                                    "name": name,
                                    "kind": "static",
                                    "rust_decl": blk.rstrip(),
                                    "source_i": cached.get("source_i"),
                                    "source_file_group": cached.get("file_group"),
                                }
                            )
                            remaining.discard(name)

            # Backfill missing types referenced by found const/vars.
            if found_blocks:
                _collect_type_closure(type_blocks, found_blocks)

        # Always write a report for audit (even if nothing was generated).
        try:
            payload = {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "project": getattr(self, "project_name", None),
                "llm_name": getattr(self, "llm_name", None),
                "workspace_root": str(self.workspace_root),
                "preprocessed_dir": str(self.preprocessed_dir),
                "types_rs": str(types_path),
                "compat_rs": str(compat_path),
                "requested_names_total": requested_total,
                "found_consts": len(new_const_blocks),
                "found_statics": len(new_var_blocks),
                "extra_types_count": len(new_extra_types_by_name),
                "decls": sorted(decl_reports, key=lambda x: (x.get("kind") or "", x.get("name") or "")),
                "extra_types": sorted(new_extra_types_by_name.keys()),
                "remaining_unresolved": sorted(remaining),
                "missing": missing_reports,
            }
            _atomic_write_text(report_path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
            self._extern_vars_report_path = report_path
            self._extern_vars_missing = missing_reports
        except Exception as e:
            logger.warning(f"写入 bindgen extern vars/const 报告失败（已忽略）: {e}")

        if not new_const_blocks and not new_var_blocks and not new_extra_types_by_name:
            return 0

        # Build updated compat.rs content: merge types + vars blocks, preserving already-injected items.
        updated_text = compat_text

        # 1) Merge extra types into C2R_EXTERN_TYPES (only if new types were discovered).
        merged_types_by_name = dict(existing_extra_types_by_name)
        for n, blk in new_extra_types_by_name.items():
            if n and blk and n not in merged_types_by_name and n not in defined_types:
                merged_types_by_name[n] = blk
        if new_extra_types_by_name:
            types_begin = "// === C2R_EXTERN_TYPES_BEGIN ==="
            types_end = "// === C2R_EXTERN_TYPES_END ==="
            type_lines: List[str] = []
            type_lines.append(types_begin)
            type_lines.append("// Auto-generated by C2R step 2.55/2.56 (bindgen allowlist extern prelude).")
            type_lines.append("// Only includes missing type items required by extern decls / extern vars below.")
            imports = self._compute_needed_crate_types_imports(type_blocks=merged_types_by_name, defined_types=defined_types)
            if imports:
                type_lines.append(f"use crate::types::{{{', '.join(imports)}}};")
                type_lines.append("")
            for name in sorted(merged_types_by_name.keys()):
                type_lines.append(merged_types_by_name[name].rstrip())
                type_lines.append("")
            type_lines.append(types_end)
            type_block = "\n".join(type_lines).rstrip() + "\n\n"
            updated_text = self._upsert_marked_block(
                updated_text,
                begin=types_begin,
                end=types_end,
                block=type_block,
                anchor="// === C2R_ACCESSOR_SHIMS_BEGIN ===",
            )

        # 2) Merge vars/consts into C2R_EXTERN_VARS marker block.
        merged_const_blocks = dict(existing_const_blocks)
        merged_const_blocks.update(new_const_blocks)
        merged_var_blocks = dict(existing_var_blocks)
        merged_var_blocks.update(new_var_blocks)

        vars_begin = "// === C2R_EXTERN_VARS_BEGIN ==="
        vars_end = "// === C2R_EXTERN_VARS_END ==="
        var_lines: List[str] = []
        var_lines.append(vars_begin)
        var_lines.append("// Auto-generated by C2R step 2.56 (bindgen allowlist vars/consts).")
        var_lines.append(f"// Source: {self.preprocessed_dir}/*.i (preprocessed translation units)")
        var_lines.append(f"// Details: {report_path}")
        imports = self._compute_needed_crate_types_imports(
            type_blocks={**merged_const_blocks, **merged_var_blocks}, defined_types=defined_types
        )
        if imports:
            var_lines.append(f"use crate::types::{{{', '.join(imports)}}};")
            var_lines.append("")

        for name in sorted(merged_const_blocks.keys()):
            blk = (merged_const_blocks.get(name) or "").rstrip()
            if blk:
                var_lines.append(blk)
                var_lines.append("")

        if merged_var_blocks:
            var_lines.append("#[allow(improper_ctypes)]")
            var_lines.append("#[allow(non_snake_case)]")
            var_lines.append('extern "C" {')
            for name in sorted(merged_var_blocks.keys()):
                for ln in (merged_var_blocks.get(name) or "").splitlines():
                    if not (ln or "").strip():
                        continue
                    # Qualify type references with crate::types:: prefix
                    qualified_ln = self._qualify_types_in_rust_code(ln.strip(), defined_types)
                    var_lines.append(f"    {qualified_ln}")
            var_lines.append("}")
            var_lines.append("")

        var_lines.append(vars_end)
        var_block = "\n".join(var_lines).rstrip() + "\n\n"

        # Prefer placing vars before extern fn decls, if that block already exists.
        anchor = "// === C2R_EXTERN_DECLS_BEGIN ===" if "// === C2R_EXTERN_DECLS_BEGIN ===" in updated_text else None
        if not anchor:
            anchor = "// === C2R_ACCESSOR_SHIMS_BEGIN ==="
        updated_text = self._upsert_marked_block(updated_text, begin=vars_begin, end=vars_end, block=var_block, anchor=anchor)

        _atomic_write_text(compat_path, updated_text)

        return int(len(new_const_blocks) + len(new_var_blocks))

    @staticmethod
    def _extract_missing_value_symbols_from_rustc_error(error_msg: str) -> Set[str]:
        """
        Extract missing VALUE symbols from rustc output (E0425-like), excluding functions.

        Examples:
        - cannot find value `FOO` in this scope
        - cannot find constant `BAR` in this scope
        - cannot find value `BAZ` in module `crate::compat`
        """
        if not error_msg:
            return set()
        out: Set[str] = set()
        try:
            for raw in re.findall(r"cannot find (?:value|constant|static) `([^`]+)`", error_msg):
                name = (raw or "").strip()
                if not name:
                    continue
                if "::" in name:
                    name = name.split("::")[-1].strip()
                if not name:
                    continue
                if name.startswith("r#"):
                    out.add(name)
                    out.add(name[2:])
                    continue
                if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
                    out.add(name)
        except Exception:
            return set()
        return out

    @staticmethod
    def _postprocess_c2rust_output_for_stable(rs_text: str) -> str:
        """
        Make c2rust output compile on stable toolchains.

        c2rust commonly emits:
        - `#![feature(extern_types)]` + `extern "C" { pub type Foo; }` (unstable on stable)
        - `#[no_mangle]` on functions (can collide with linked C objects)

        We:
        - remove all `#![feature(...)]` lines
        - remove `#[no_mangle]` attributes
        - rewrite `extern type` declarations into opaque zero-sized structs:
            `extern "C" { pub type Foo; }` -> `#[repr(C)] pub struct Foo { _unused: [u8; 0] }`
        """
        if not rs_text:
            return ""

        lines_in = (rs_text or "").splitlines()
        stripped: List[str] = []
        saw_allow_deref_nullptr = False
        for ln in lines_in:
            s = (ln or "").strip()
            if re.match(r"^#!\s*\[\s*feature\s*\(.*\)\s*\]\s*$", s):
                continue
            if re.match(r"^#\s*\[\s*no_mangle\s*\]\s*$", s):
                continue
            if re.match(r"^#!\s*\[\s*allow\s*\(\s*deref_nullptr\s*\)\s*\]\s*$", s):
                saw_allow_deref_nullptr = True
            stripped.append(ln)
        # c2rust may emit container_of/offset computations that trigger Rust's `#[deny(deref_nullptr)]`
        # and fail compilation. Since this is deterministic fallback code (NOT truth-layer), allow it.
        if not saw_allow_deref_nullptr:
            stripped.insert(0, "#![allow(deref_nullptr)]")
        text = "\n".join(stripped) + ("\n" if rs_text.endswith("\n") else "")

        # Collect already defined types to avoid duplicates.
        # NOTE: c2rust emits unstable `extern type` as `extern "C" { pub type Foo; }`.
        # We remove those `pub type Foo;` lines and re-introduce them as opaque structs.
        # Therefore, treat only real item definitions as "defined", and include `pub type Foo = ...;`
        # aliases but NOT `pub type Foo;` extern types.
        defined: Set[str] = set()
        try:
            defined.update(re.findall(r"\bpub\s+(?:struct|enum|union)\s+([A-Za-z_]\w*)\b", text))
            defined.update(re.findall(r"\bpub\s+type\s+([A-Za-z_]\w*)\s*=", text))
        except Exception:
            defined = set()

        extern_type_names: List[str] = []
        out_lines: List[str] = []
        i = 0
        in_extern = False
        extern_open_line = ""
        extern_buf: List[str] = []

        def _flush_extern_block(buf: List[str]) -> None:
            nonlocal out_lines
            if not buf:
                return
            # If the block has only the opening and closing brace, skip it.
            body_lines = [b for b in buf if (b or "").strip() not in ('extern "C" {', 'unsafe extern "C" {', "}")]
            has_real = False
            for b in body_lines:
                if (b or "").strip() and not (b or "").strip().startswith("//"):
                    has_real = True
                    break
            if has_real:
                out_lines.extend(buf)

        while i < len(stripped):
            line = stripped[i]
            s = (line or "").strip()
            if not in_extern and s in {'extern "C" {', 'unsafe extern "C" {'}:
                in_extern = True
                extern_open_line = line
                extern_buf = [line]
                i += 1
                continue
            if in_extern:
                if s == "}":
                    extern_buf.append(line)
                    in_extern = False
                    _flush_extern_block(extern_buf)
                    extern_buf = []
                    extern_open_line = ""
                    i += 1
                    continue
                # Drop `pub type Foo;` extern types and record them.
                m = re.match(r"^pub\s+type\s+([A-Za-z_]\w*)\s*;\s*$", s)
                if m:
                    extern_type_names.append(m.group(1))
                    i += 1
                    continue
                extern_buf.append(line)
                i += 1
                continue
            out_lines.append(line)
            i += 1

        # Insert opaque structs for extern types (dedup + avoid redefining).
        uniq: List[str] = []
        seen: Set[str] = set()
        for n in extern_type_names:
            if not n or n in seen or n in defined:
                continue
            seen.add(n)
            uniq.append(n)

        if not uniq:
            out_text = "\n".join(out_lines) + ("\n" if rs_text.endswith("\n") else "")
            # 2026-01-07 修复: 无 extern types 时也要添加 pub，否则早期返回会跳过 pub 修复
            try:
                out_text = re.sub(r'(?m)^(\s*)unsafe\s+extern\s+"C"\s+fn\s+', r'\1pub unsafe extern "C" fn ', out_text)
                out_text = re.sub(r'(?m)^(\s*)extern\s+"C"\s+fn\s+', r'\1pub extern "C" fn ', out_text)
                out_text = re.sub(r'(?m)^(\s*)unsafe\s+fn\s+([A-Za-z_])', r'\1pub unsafe fn \2', out_text)
                out_text = re.sub(r'(?m)^(\s*)fn\s+([A-Za-z_])', r'\1pub fn \2', out_text)
                out_text = re.sub(r'pub\s+pub\s+', 'pub ', out_text)
            except Exception:
                pass
            # c2rust may emit `compile_error!(...)` for volatile reads; turn it into a runtime failure
            # so the module can compile and be used as a mechanical fallback.
            out_text = out_text.replace("compile_error!(", "panic!(")
            return out_text

        opaque_defs: List[str] = []
        opaque_defs.append("// === C2R_C2RUST_EXTERN_TYPES_BEGIN ===")
        opaque_defs.append("// Auto-generated: downgraded c2rust `extern type` to stable-safe opaque structs.")
        for n in uniq:
            opaque_defs.append("#[repr(C)]")
            opaque_defs.append("#[derive(Copy, Clone)]")
            opaque_defs.append(f"pub struct {n} {{")
            opaque_defs.append("    _unused: [u8; 0],")
            opaque_defs.append("}")
            opaque_defs.append("")
        opaque_defs.append("// === C2R_C2RUST_EXTERN_TYPES_END ===")
        opaque_lines = opaque_defs + [""]

        # Insert after any leading inner attributes (`#![...]`) to keep them at the top.
        # c2rust often formats `#![allow(...)]` across multiple lines, so we must skip until the
        # attribute's closing `]`.
        insert_at = 0
        i = 0
        while i < len(out_lines) and (out_lines[i] or "").lstrip().startswith("#!["):
            bracket_balance = 0
            j = i
            while j < len(out_lines):
                ln = out_lines[j] or ""
                bracket_balance += ln.count("[")
                bracket_balance -= ln.count("]")
                j += 1
                if bracket_balance <= 0:
                    break
            i = j
        insert_at = i
        out_lines = out_lines[:insert_at] + opaque_lines + out_lines[insert_at:]
        out_text = "\n".join(out_lines) + ("\n" if rs_text.endswith("\n") else "")
        # c2rust emits `static` C functions as non-`pub` Rust functions. Our fallback wrapper lives
        # outside the generated module and must be able to call into it, so we promote top-level
        # function items to `pub` (safe: this is fallback-only code, not truth-layer).
        #
        # 2026-01-07 修复: 原正则 `^` 要求严格行首匹配，但 c2rust 生成的函数可能有缩进，
        # 导致无法添加 pub (如 zopfli 的 BoundaryPM)。改用 `^(\s*)` 捕获任意前导空白。
        try:
            out_text = re.sub(r'(?m)^(\s*)unsafe\s+extern\s+"C"\s+fn\s+', r'\1pub unsafe extern "C" fn ', out_text)
            out_text = re.sub(r'(?m)^(\s*)extern\s+"C"\s+fn\s+', r'\1pub extern "C" fn ', out_text)
            out_text = re.sub(r'(?m)^(\s*)unsafe\s+fn\s+([A-Za-z_])', r'\1pub unsafe fn \2', out_text)
            out_text = re.sub(r'(?m)^(\s*)fn\s+([A-Za-z_])', r'\1pub fn \2', out_text)
            # 避免重复 pub pub
            out_text = re.sub(r'pub\s+pub\s+', 'pub ', out_text)
        except Exception:
            pass
        out_text = out_text.replace("compile_error!(", "panic!(")
        return out_text

    def _ensure_compat_has_c2rust_fallback_mod(self, *, file_group: str, rel_include_path: str) -> None:
        """
        Ensure `src/compat.rs` defines `pub mod __c2rust_fallback::{file_group}` that references the
        generated c2rust fallback Rust file.

        We use a generated `src/__c2r_generated/c2rust_fallback/mod.rs` + `#[path = ...] pub mod __c2rust_fallback;`
        rather than `include!()`, because the c2rust output contains inner attributes like `#![allow(...)]`,
        which are not permitted inside `include!()`.
        """
        compat_path = self.work_dir / "src" / "compat.rs"
        if not compat_path.exists():
            return
        try:
            compat_text = compat_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            compat_text = ""

        begin = "// === C2R_C2RUST_FALLBACK_MODS_BEGIN ==="
        end = "// === C2R_C2RUST_FALLBACK_MODS_END ==="

        # Extract existing module names from the current block (if any).
        existing: Set[str] = set()
        try:
            bs = compat_text.find(begin)
            be = compat_text.find(end)
            if bs != -1 and be != -1 and be > bs:
                blk = compat_text[bs: be + len(end)]
                # Support both historical formats:
                # - `pub mod __c2rust_fallback { pub mod name { include!(...); } }`
                # - `pub mod __c2rust_fallback { #[path = ...] pub mod name; }`
                existing = set(re.findall(r"(?m)^\s{4}pub\s+mod\s+([A-Za-z_]\w*)\s*(?:\{|;)", blk))
        except Exception:
            existing = set()

        # Keep a deterministic module index next to the generated files.
        # This avoids tricky relative-path resolution rules for nested `#[path] mod ...;`.
        mod_rs_dir = self.work_dir / "src" / "__c2r_generated" / "c2rust_fallback"
        try:
            mod_rs_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        # Determine the module set from the actual generated files, not from `compat.rs`,
        # so incremental additions don't accidentally drop older modules.
        # Also skip known-bad modules that were disabled after failing compilation.
        def _is_disabled(name: str) -> bool:
            try:
                return (mod_rs_dir / f"{name}.disabled").exists()
            except Exception:
                return False

        names_set: Set[str] = {n for n in existing if n and not _is_disabled(n)}
        if file_group and not _is_disabled(file_group):
            names_set.add(file_group)
        try:
            for p in mod_rs_dir.glob("*.rs"):
                if p.name == "mod.rs":
                    continue
                if p.stem and not _is_disabled(p.stem):
                    names_set.add(p.stem)
        except Exception:
            pass
        names = sorted(names_set)

        mod_rs_path = mod_rs_dir / "mod.rs"
        try:
            mod_rs_lines: List[str] = []
            mod_rs_lines.append("// Auto-generated by C2R (C2Rust fallback module index).")
            mod_rs_lines.append("// This is NOT truth-layer; it is a deterministic mechanical fallback.")
            mod_rs_lines.append("")
            for n in names:
                mod_rs_lines.append(f"pub mod {n};")
            mod_rs_lines.append("")
            _atomic_write_text(mod_rs_path, "\n".join(mod_rs_lines))
        except Exception:
            pass

        lines: List[str] = []
        lines.append(begin)
        lines.append("// Auto-generated: C2Rust transpile fallback modules (used when LLM translation fails).")
        lines.append("// NOTE: This is NOT truth-layer; it is a deterministic mechanical fallback.")
        # Use a single `#[path] mod` for stable inner-attribute handling.
        # The content lives in `src/__c2r_generated/c2rust_fallback/mod.rs`.
        lines.append("#[path = \"__c2r_generated/c2rust_fallback/mod.rs\"]")
        lines.append("pub mod __c2rust_fallback;")
        lines.append(end)
        block = "\n".join(lines).rstrip() + "\n\n"

        updated = self._upsert_marked_block(
            compat_text,
            begin=begin,
            end=end,
            block=block,
            anchor="// === C2R_COMPAT_PLACEHOLDERS_BEGIN ===",
        )
        if updated != compat_text:
            _atomic_write_text(compat_path, updated)

    def _c2rust_fallback_disabled_marker_path(self, *, file_group: str) -> Path:
        return self.work_dir / "src" / "__c2r_generated" / "c2rust_fallback" / f"{file_group}.disabled"

    def _is_c2rust_fallback_module_disabled(self, *, file_group: str) -> bool:
        if not file_group:
            return False
        try:
            return self._c2rust_fallback_disabled_marker_path(file_group=file_group).exists()
        except Exception:
            return False

    def _disable_c2rust_fallback_module(self, *, file_group: str, reason: str, error_msg: str) -> None:
        """
        Disable a c2rust fallback module so it won't be included into the crate again.

        This is used as a safety valve: c2rust output may contain patterns that fail `cargo check`
        under project lint settings. In that case we prefer keeping `unimplemented!()` placeholders.
        """
        if not file_group:
            return
        marker = self._c2rust_fallback_disabled_marker_path(file_group=file_group)
        try:
            marker.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        try:
            payload = {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "project": getattr(self, "project_name", None),
                "file_group": file_group,
                "reason": reason,
                "error_head": (error_msg or "")[:2000],
            }
            _atomic_write_text(marker, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        except Exception:
            try:
                _atomic_write_text(marker, f"reason: {reason}\n\n{(error_msg or '')[:2000]}\n")
            except Exception:
                pass

    def _get_compile_commands_includes(self, file_group: str) -> List[str]:
        """Extract -I include paths from the TU context map for *file_group*."""
        incs: List[str] = []
        try:
            tc_files = getattr(self, "_tu_context_files", None) or {}
            entry = tc_files.get(file_group)
            if not entry:
                return incs
            ce = entry.get("compile_commands_entry") if isinstance(entry, dict) else None
            if not isinstance(ce, dict):
                return incs
            cmd = (ce.get("command") or ce.get("arguments") or "")
            if isinstance(cmd, list):
                cmd = " ".join(str(p) for p in cmd)
        except Exception:
            return incs
        if not cmd:
            return incs

        for part in str(cmd).split():
            if part.startswith("-I"):
                inc_path = part[2:]
                if inc_path and os.path.isdir(inc_path):
                    incs.append(os.path.abspath(inc_path))
        return incs

    def _run_c2rust_transpile(self, *, preprocessed_i: Path, file_group: str) -> Optional[str]:
        """
        Run `c2rust transpile` on a pinned TU preprocessed `.i` (copied to `.c` to avoid `-x cpp-output` issues).
        Return the raw Rust output, or None on failure.
        """
        c2rust_bin = shutil.which("c2rust")
        if not c2rust_bin:
            return None
        if not preprocessed_i or not preprocessed_i.exists():
            return None

        tmp_dir = self.work_dir / ".c2r_c2rust_fallback" / file_group
        try:
            tmp_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            return None

        tmp_c = tmp_dir / f"{preprocessed_i.stem}.c"
        tmp_rs = tmp_dir / f"{preprocessed_i.stem}.rs"

        # Copy the preprocessed file, stripping known-problematic variadic
        # calls (HiLogPrint) that trigger an assertion bug in c2rust v0.22.
        try:
            raw = preprocessed_i.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            raw = ""
        if raw:
            # Replace ((void)HiLogPrint(...)); → (void)0; (multiline-safe).
            import re as _re
            raw = _re.sub(
                r'\(\(void\)HiLogPrint\([^)]*?(?:\([^)]*\)[^)]*?)*\)\)\s*;',
                '(void)0;', raw, flags=_re.DOTALL
            )
            tmp_c.write_text(raw, encoding="utf-8")
        else:
            try:
                shutil.copy(preprocessed_i, tmp_c)
            except Exception as e:
                logger.debug(f"c2rust fallback: 复制 .i 失败: {e}")
                return None

        cmd = [
            c2rust_bin,
            "transpile",
            str(tmp_c),
            "--",
            "-Wno-error",
            "-Wno-macro-redefined",
            "-Wno-builtin-macro-redefined",
            "-Wno-ignored-attributes",
        ]

        # Pass include paths from the TU context so c2rust can resolve
        # external declarations (e.g. HiLogPrint, HdfSbufWriteString).
        _tu_incs = self._get_compile_commands_includes(file_group)
        if _tu_incs:
            for inc in _tu_incs:
                cmd.extend(["-I", inc])
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=max(10, int(self._c2rust_fallback_timeout_sec or 180)),
                cwd=str(tmp_dir),
            )
        except Exception as e:
            logger.debug(f"c2rust fallback: transpile 失败: {e}")
            return None

        # Save logs for audit.
        try:
            log_path = self.manual_fix_root / f"c2rust_fallback_{file_group}_{preprocessed_i.stem}.log"
            _atomic_write_text(
                log_path,
                "\n".join(
                    [
                        f"cmd: {' '.join(cmd)}",
                        f"returncode: {proc.returncode}",
                        "---- stdout ----",
                        proc.stdout or "",
                        "---- stderr ----",
                        proc.stderr or "",
                    ]
                )
                + "\n",
            )
        except Exception:
            pass

        if proc.returncode != 0 or not tmp_rs.exists():
            return None
        try:
            return tmp_rs.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return None

    @staticmethod
    def _extract_missing_names_from_cargo_error_for_file(cargo_err: str, *, file_path_substr: str) -> Set[str]:
        """
        Best-effort: extract missing symbol names for common errors (E0425/E0433) that point to a given file.

        We only use this to bridge *c2rust fallback modules* (non-truth). Any bridges are still gated by:
        - "symbol must be present in pinned `.i`" (checked separately)
        - compile-gate `cargo check` must succeed
        """
        if not cargo_err or not file_path_substr:
            return set()
        lines = (cargo_err or "").splitlines()
        missing: Set[str] = set()
        for i, ln in enumerate(lines):
            name: Optional[str] = None
            m = re.search(
                r"error\[E0425\]:\s+cannot find function, tuple struct or tuple variant `([^`]+)` in this scope",
                ln,
            )
            if m:
                name = (m.group(1) or "").strip()
            if not name:
                m = re.search(r"error\[E0425\]:\s+cannot find value `([^`]+)` in this scope", ln)
                if m:
                    name = (m.group(1) or "").strip()
            if not name:
                # E0433 is much broader; we still collect the name, but later we will require it to exist in `.i`.
                m = re.search(
                    r"error\[E0433\]:\s+failed to resolve:\s+use of undeclared crate or module `([^`]+)`",
                    ln,
                )
                if m:
                    name = (m.group(1) or "").strip()
            if not name:
                continue
            # Look ahead for the location line.
            hit = False
            for j in range(i + 1, min(i + 14, len(lines))):
                if "-->" in lines[j] and file_path_substr in lines[j]:
                    hit = True
                    break
            if hit:
                missing.add(name)
        return missing

    @staticmethod
    def _file_contains_token(path: Path, token: str) -> bool:
        """
        Best-effort streaming substring search.
        Used to ensure we only bridge symbols that are present in the pinned TU `.i`.
        """
        if not path or not path.exists() or not token:
            return False
        try:
            needle = token.encode("utf-8", errors="ignore")
            if not needle:
                return False
            with open(path, "rb") as f:
                while True:
                    chunk = f.read(1024 * 1024)
                    if not chunk:
                        break
                    if needle in chunk:
                        return True
        except Exception:
            return False
        return False

    @staticmethod
    def _extract_bindgen_bridge_items(bindgen_rs: str, *, names: Set[str]) -> Tuple[List[str], List[str]]:
        """
        Extract `extern "C"` decls + `pub const` definitions for a set of names from bindgen output.

        Returns: (extern_items, const_items), each a list of multi-line snippets (deduped, best-effort).
        """
        if not bindgen_rs or not names:
            return [], []
        lines = (bindgen_rs or "").splitlines()
        want = set(names or [])

        const_items: List[str] = []
        i = 0
        while i < len(lines):
            ln = lines[i]
            m = re.match(r"^\s*pub\s+const\s+([A-Za-z_]\w*)\b", ln)
            if m and m.group(1) in want:
                start = i
                j = i
                while j < len(lines) and not (lines[j] or "").strip().endswith(";"):
                    j += 1
                if j < len(lines):
                    j += 1
                const_items.append("\n".join(lines[start:j]).rstrip())
                i = j
                continue
            i += 1

        extern_items: List[str] = []
        in_extern = False
        attrs: List[str] = []
        i = 0
        while i < len(lines):
            s = (lines[i] or "").strip()
            if not in_extern and s in {'extern "C" {', 'unsafe extern "C" {'}:
                in_extern = True
                attrs = []
                i += 1
                continue
            if in_extern:
                if s == "}":
                    in_extern = False
                    attrs = []
                    i += 1
                    continue
                if s.startswith("#["):
                    attrs.append(lines[i])
                    i += 1
                    continue
                m_fn = re.match(r"^\s*(?:pub\s+)?fn\s+([A-Za-z_]\w*)\s*\(", lines[i] or "")
                m_static = re.match(
                    r"^\s*(?:pub\s+)?static(?:\s+mut)?\s+([A-Za-z_]\w*)\b",
                    lines[i] or "",
                )
                item_name = ""
                if m_fn:
                    item_name = m_fn.group(1) or ""
                elif m_static:
                    item_name = m_static.group(1) or ""
                if item_name:
                    item_lines: List[str] = []
                    item_lines.extend(attrs)
                    attrs = []
                    item_lines.append(lines[i])
                    j = i + 1
                    while j < len(lines):
                        item_lines.append(lines[j])
                        if (lines[j] or "").strip().endswith(";"):
                            j += 1
                            break
                        j += 1
                    if item_name in want:
                        extern_items.append("\n".join(item_lines).rstrip())
                    i = j
                    continue
                # Avoid leaking attrs to later items.
                if s:
                    attrs = []
                i += 1
                continue
            i += 1

        def _dedup(snips: List[str]) -> List[str]:
            out: List[str] = []
            seen: Set[str] = set()
            for s in snips:
                k = (s or "").strip()
                if not k or k in seen:
                    continue
                seen.add(k)
                out.append(s)
            return out

        return _dedup(extern_items), _dedup(const_items)

    def _upsert_c2rust_fallback_bridge_block(
        self,
        *,
        module_path: Path,
        extern_items: List[str],
        const_items: List[str],
    ) -> bool:
        """
        Insert/update a marked bridge block inside a generated c2rust fallback module.
        Returns True if the file content changed.
        """
        if not module_path or not module_path.exists():
            return False
        try:
            before = module_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return False

        begin = "// === C2R_C2RUST_FALLBACK_BRIDGES_BEGIN ==="
        end = "// === C2R_C2RUST_FALLBACK_BRIDGES_END ==="

        def _reindent_snippet(snip: str, *, base: str = "    ") -> List[str]:
            ls = (snip or "").splitlines()
            min_indent: Optional[int] = None
            for l in ls:
                if not (l or "").strip():
                    continue
                n = len(l) - len(l.lstrip(" "))
                if min_indent is None or n < min_indent:
                    min_indent = n
            if min_indent is None:
                min_indent = 0
            out: List[str] = []
            for l in ls:
                if not (l or "").strip():
                    out.append("")
                else:
                    out.append(base + l[min_indent:])
            return out

        lines: List[str] = []
        lines.append(begin)
        lines.append("// Auto-generated: bindgen bridge decls for missing symbols in c2rust fallback module.")
        lines.append("// NOTE: This is NOT truth-layer; it is only used to make mechanical fallback compile.")
        lines.append("")
        if const_items:
            lines.append("// ---- consts (from bindgen on pinned `.i`) ----")
            for snip in const_items:
                lines.append((snip or "").rstrip())
                lines.append("")
        if extern_items:
            lines.append('extern "C" {')
            for snip in extern_items:
                lines.extend(_reindent_snippet(snip))
                lines.append("")
            # trim last blank line inside extern
            if lines and lines[-1] == "":
                lines.pop()
            lines.append("}")
            lines.append("")
        lines.append(end)
        block = "\n".join(lines).rstrip() + "\n\n"

        after = self._upsert_marked_block(before, begin=begin, end=end, block=block)
        if after == before:
            return False
        try:
            _atomic_write_text(module_path, after)
            return True
        except Exception:
            return False

    def _try_patch_c2rust_fallback_module_with_bindgen_bridge(
        self,
        *,
        file_group: str,
        module_path: Path,
        preprocessed_i: Path,
        cargo_error: str,
    ) -> bool:
        """
        Best-effort: when a c2rust fallback module fails `cargo check` due to missing symbols (E0425/E0433),
        try to generate bridge decls via bindgen allowlist on the *same pinned `.i`* and inject them into
        the fallback module (NOT truth-layer), then let the caller retry `cargo check`.

        Returns True if the fallback module file changed (i.e. we injected something).
        """
        if not file_group or not module_path or not preprocessed_i:
            return False
        if not module_path.exists() or not preprocessed_i.exists():
            return False
        needle = f"src/__c2r_generated/c2rust_fallback/{file_group}.rs"
        missing = self._extract_missing_names_from_cargo_error_for_file(cargo_error, file_path_substr=needle)
        if not missing:
            return False

        # Only bridge "true" symbols that appear in the pinned `.i`.
        filtered: List[str] = []
        for n in sorted(missing):
            if not re.match(r"^[A-Za-z_]\w*$", n or ""):
                continue
            if self._file_contains_token(preprocessed_i, n):
                filtered.append(n)
        if not filtered:
            return False

        bindgen_bin = self._find_bindgen_binary()
        if not bindgen_bin:
            return False

        tmp_dir = self.work_dir / ".c2r_c2rust_fallback_bridge" / file_group
        try:
            tmp_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            return False
        out_rs = tmp_dir / f"{preprocessed_i.stem}.rs"

        cmd: List[str] = [
            bindgen_bin,
            str(preprocessed_i),
            "-o",
            str(out_rs),
            "--no-layout-tests",
            "--no-doc-comments",
            "--use-core",
            "--default-enum-style=consts",
            "--no-prepend-enum-name",
            "--no-size_t-is-usize",
        ]
        for n in filtered:
            pat = rf"^{re.escape(n)}$"
            cmd.extend(["--allowlist-function", pat])
            cmd.extend(["--allowlist-var", pat])
            cmd.extend(["--allowlist-type", pat])
        cmd.append("--")
        cmd.extend(
            [
                "-x",
                "c",
                "-Wno-error",
                "-Wno-macro-redefined",
                "-Wno-builtin-macro-redefined",
                "-Wno-ignored-attributes",
            ]
        )

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=max(10, int(getattr(self, "_bindgen_extern_vars_timeout_sec", 90) or 90)),
                cwd=str(tmp_dir),
            )
        except Exception as e:
            logger.debug(f"c2rust fallback bridge: bindgen 失败: {e}")
            return False

        # Save logs for audit.
        try:
            log_path = self.manual_fix_root / f"c2rust_fallback_bridge_{file_group}_{preprocessed_i.stem}.log"
            _atomic_write_text(
                log_path,
                "\n".join(
                    [
                        f"cmd: {' '.join(cmd)}",
                        f"returncode: {proc.returncode}",
                        "---- stdout ----",
                        proc.stdout or "",
                        "---- stderr ----",
                        proc.stderr or "",
                    ]
                )
                + "\n",
            )
        except Exception:
            pass

        if proc.returncode != 0 or not out_rs.exists():
            return False
        try:
            bindgen_text = out_rs.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            bindgen_text = ""
        if not bindgen_text:
            return False

        extern_items, const_items = self._extract_bindgen_bridge_items(bindgen_text, names=set(filtered))
        if not extern_items and not const_items:
            return False
        return self._upsert_c2rust_fallback_bridge_block(
            module_path=module_path,
            extern_items=extern_items,
            const_items=const_items,
        )

    def _ensure_c2rust_fallback_module(self, *, func_info: FunctionInfo) -> Optional[Tuple[str, Path]]:
        """
        Ensure the per-file-group c2rust fallback module exists under `src/__c2r_generated/c2rust_fallback/`.
        Returns: (file_group, module_path) or None.
        """
        safe_name = (getattr(func_info, "file_name", "") or "").strip()
        if not safe_name:
            return None

        # If this module was previously proven to break compilation, do not attempt to reuse it.
        if self._is_c2rust_fallback_module_disabled(file_group=safe_name):
            return None

        cached_path = self._c2rust_fallback_modules.get(safe_name)
        if cached_path and cached_path.exists():
            return safe_name, self._c2rust_fallback_modules[safe_name]

        pre = self._locate_preprocessed_file(func_info)
        if not pre or not pre.exists():
            return None

        raw = self._run_c2rust_transpile(preprocessed_i=pre, file_group=safe_name)
        if not raw:
            return None
        processed = self._postprocess_c2rust_output_for_stable(raw)
        if not processed:
            return None

        out_dir = self.work_dir / "src" / "__c2r_generated" / "c2rust_fallback"
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            return None
        out_path = out_dir / f"{safe_name}.rs"
        _atomic_write_text(out_path, processed)

        # Snapshot compat.rs + module index so we can rollback if the module itself fails to compile.
        compat_path = self.work_dir / "src" / "compat.rs"
        mod_rs_path = out_dir / "mod.rs"
        try:
            compat_before = compat_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            compat_before = None
        try:
            mod_before = mod_rs_path.read_text(encoding="utf-8", errors="ignore") if mod_rs_path.exists() else None
        except Exception:
            mod_before = None

        # Ensure compat.rs includes this module (and refresh module index).
        rel = "__c2r_generated/c2rust_fallback/{file_group}.rs"
        self._ensure_compat_has_c2rust_fallback_mod(file_group=safe_name, rel_include_path=rel)

        # Compile-gate: if adding the fallback module breaks `cargo check`, disable it and rollback.
        if not self._compile_project():
            err = ""
            try:
                err = self._get_compile_error() or ""
            except Exception:
                err = ""

            # Best-effort retry: bridge missing symbols (E0425/E0433) from the pinned `.i` via bindgen,
            # and retry `cargo check` before disabling this module.
            bridged_ok = False
            max_bridge_retry = 0
            try:
                max_bridge_retry = int(os.environ.get("C2R_C2RUST_FALLBACK_BRIDGE_RETRY_MAX", "10"))
            except Exception:
                max_bridge_retry = 10
            for _ in range(max(0, max_bridge_retry)):
                if not err:
                    break
                changed = False
                try:
                    changed = self._try_patch_c2rust_fallback_module_with_bindgen_bridge(
                        file_group=safe_name,
                        module_path=out_path,
                        preprocessed_i=pre,
                        cargo_error=err,
                    )
                except Exception as e:
                    logger.debug(f"c2rust fallback bridge: patch失败(ignored): {e}")
                    changed = False
                if not changed:
                    break
                if self._compile_project():
                    bridged_ok = True
                    break
                try:
                    err = self._get_compile_error() or ""
                except Exception:
                    err = ""

            if not bridged_ok:
                # Still failing: disable and rollback.
                err = err or ""
                self._disable_c2rust_fallback_module(
                    file_group=safe_name,
                    reason="c2rust_fallback_module_failed_cargo_check",
                    error_msg=err,
                )
                try:
                    if compat_before is not None:
                        _atomic_write_text(compat_path, compat_before)
                except Exception:
                    pass
                try:
                    if mod_before is not None:
                        _atomic_write_text(mod_rs_path, mod_before)
                except Exception:
                    pass
                # Ensure caches don't retain this broken module.
                try:
                    self._c2rust_fallback_modules.pop(safe_name, None)
                    self._c2rust_fallback_function_index.pop(safe_name, None)
                except Exception:
                    pass
                return None

        # Best-effort function index (only cached when the module is compile-safe).
        try:
            fn_names = set(re.findall(r"\bfn\s+([A-Za-z_]\w*)\b", processed))
            self._c2rust_fallback_function_index[safe_name] = fn_names
        except Exception:
            self._c2rust_fallback_function_index[safe_name] = set()

        self._c2rust_fallback_modules[safe_name] = out_path
        return safe_name, out_path

    @staticmethod
    def _extract_param_names_from_rust_signature(sig: str) -> List[str]:
        """
        Extract parameter identifier names from a Rust function signature string (best-effort).
        """
        if not sig:
            return []
        s = sig.strip()
        # Find parameter list bounds.
        fn_pos = s.find("fn ")
        start = s.find("(", fn_pos if fn_pos != -1 else 0)
        if start < 0:
            return []
        depth = 0
        end = None
        for j in range(start, len(s)):
            if s[j] == "(":
                depth += 1
            elif s[j] == ")":
                depth -= 1
                if depth == 0:
                    end = j
                    break
        if end is None:
            return []
        params_raw = s[start + 1 : end].strip()
        if not params_raw:
            return []

        def _split_top_level_commas(src: str) -> List[str]:
            parts: List[str] = []
            buf: List[str] = []
            depth_paren = 0
            depth_angle = 0
            depth_brack = 0
            depth_brace = 0
            for ch in src:
                if ch == "(":
                    depth_paren += 1
                elif ch == ")":
                    depth_paren = max(0, depth_paren - 1)
                elif ch == "<":
                    depth_angle += 1
                elif ch == ">":
                    depth_angle = max(0, depth_angle - 1)
                elif ch == "[":
                    depth_brack += 1
                elif ch == "]":
                    depth_brack = max(0, depth_brack - 1)
                elif ch == "{":
                    depth_brace += 1
                elif ch == "}":
                    depth_brace = max(0, depth_brace - 1)
                if ch == "," and depth_paren == 0 and depth_angle == 0 and depth_brack == 0 and depth_brace == 0:
                    parts.append("".join(buf))
                    buf = []
                    continue
                buf.append(ch)
            parts.append("".join(buf))
            return [p for p in (p.strip() for p in parts) if p]

        out: List[str] = []
        for idx, item in enumerate(_split_top_level_commas(params_raw)):
            if item == "...":
                continue
            if ":" not in item:
                continue
            name_part, _ty_part = item.split(":", 1)
            name_part = (name_part or "").strip()
            # Strip common patterns like `mut x` / `ref x`.
            toks = [t for t in re.split(r"\s+", name_part) if t]
            name = ""
            for t in reversed(toks):
                if re.match(r"^(?:r#)?[A-Za-z_]\w*$", t):
                    name = t
                    break
            if not name:
                name = f"arg{idx}"
            out.append(name)
        return out

    @staticmethod
    def _extract_params_from_rust_signature(sig: str) -> List[Tuple[str, str]]:
        """
        Extract `(param_name, param_type)` pairs from a Rust function signature string (best-effort).
        """
        if not sig:
            return []
        s = sig.strip()
        fn_pos = s.find("fn ")
        start = s.find("(", fn_pos if fn_pos != -1 else 0)
        if start < 0:
            return []
        depth = 0
        end = None
        for j in range(start, len(s)):
            if s[j] == "(":
                depth += 1
            elif s[j] == ")":
                depth -= 1
                if depth == 0:
                    end = j
                    break
        if end is None:
            return []
        params_raw = s[start + 1 : end].strip()
        if not params_raw:
            return []

        def _split_top_level_commas(src: str) -> List[str]:
            parts: List[str] = []
            buf: List[str] = []
            depth_paren = 0
            depth_angle = 0
            depth_brack = 0
            depth_brace = 0
            for ch in src:
                if ch == "(":
                    depth_paren += 1
                elif ch == ")":
                    depth_paren = max(0, depth_paren - 1)
                elif ch == "<":
                    depth_angle += 1
                elif ch == ">":
                    depth_angle = max(0, depth_angle - 1)
                elif ch == "[":
                    depth_brack += 1
                elif ch == "]":
                    depth_brack = max(0, depth_brack - 1)
                elif ch == "{":
                    depth_brace += 1
                elif ch == "}":
                    depth_brace = max(0, depth_brace - 1)
                if ch == "," and depth_paren == 0 and depth_angle == 0 and depth_brack == 0 and depth_brace == 0:
                    parts.append("".join(buf))
                    buf = []
                    continue
                buf.append(ch)
            parts.append("".join(buf))
            return [p for p in (p.strip() for p in parts) if p]

        out: List[Tuple[str, str]] = []
        for idx, item in enumerate(_split_top_level_commas(params_raw)):
            if item == "...":
                continue
            if ":" not in item:
                continue
            name_part, ty_part = item.split(":", 1)
            name_part = (name_part or "").strip()
            ty_part = (ty_part or "").strip()
            toks = [t for t in re.split(r"\s+", name_part) if t]
            name = ""
            for t in reversed(toks):
                if re.match(r"^(?:r#)?[A-Za-z_]\w*$", t):
                    name = t
                    break
            if not name:
                name = f"arg{idx}"
            out.append((name, ty_part))
        return out

    def _build_c2rust_fallback_wrapper(self, *, func_info: FunctionInfo, file_group: str) -> Optional[str]:
        """
        Build a wrapper function (matching the skeleton signature) that calls into the c2rust fallback module.
        Uses `as _` casts so we don't need to parse types precisely.
        """
        sig = (getattr(func_info, "rust_signature", "") or "").strip()
        if not sig:
            return None
        fn_name = (getattr(func_info, "name", "") or "").strip()
        if not fn_name:
            return None
        params = self._extract_params_from_rust_signature(sig)
        prelude: List[str] = []
        arg_exprs: List[str] = []
        for name, ty in params:
            t = (ty or "").strip()
            # Arrays by-value cannot be cast with `as _` directly. Bridge by taking a raw pointer to a local copy.
            if t.startswith("[") and ";" in t and t.endswith("]"):
                tmp = f"__c2r_{name}"
                prelude.append(f"    let mut {tmp} = {name};")
                arg_exprs.append(f"(&mut {tmp} as *mut _) as _")
            else:
                arg_exprs.append(f"{name} as _")
        args = ", ".join(arg_exprs)

        # Determine if the signature returns unit.
        s = sig
        start = s.find("(")
        end = None
        if start != -1:
            depth = 0
            for j in range(start, len(s)):
                if s[j] == "(":
                    depth += 1
                elif s[j] == ")":
                    depth -= 1
                    if depth == 0:
                        end = j
                        break
        tail = s[end + 1 :] if end is not None else ""
        returns_unit = "->" not in tail

        callee = f"crate::compat::__c2rust_fallback::{file_group}::{fn_name}"
        if returns_unit:
            body_lines = ["{", "    // C2R: C2Rust fallback (LLM failed; wrapper keeps skeleton signature)"]
            body_lines.extend(prelude)
            body_lines.append(f"    unsafe {{ {callee}({args}); }}")
            body_lines.append("}")
            body = "\n".join(
                body_lines
            )
        else:
            body_lines = ["{", "    // C2R: C2Rust fallback (LLM failed; wrapper keeps skeleton signature)"]
            body_lines.extend(prelude)
            body_lines.append(f"    unsafe {{ {callee}({args}) as _ }}")
            body_lines.append("}")
            body = "\n".join(
                body_lines
            )
        return sig.rstrip() + " " + body + "\n"

    def _append_llm_failed_code_comment_below_function(
        self,
        *,
        func_name: str,
        func_info: FunctionInfo,
        llm_code: str,
        artifact_path: Optional[Path],
        reason: str,
    ) -> None:
        """
        Append the failed LLM output *below* the function as a block comment, for manual inspection.
        """
        if not llm_code:
            return
        rs_filename_with_prefix = "src_" + func_info.file_name + ".rs"
        rs_filename = func_info.file_name + ".rs"
        target_file = self.work_dir / "src" / rs_filename_with_prefix
        if not target_file.exists():
            target_file = self.work_dir / "src" / rs_filename
            if not target_file.exists():
                return
        try:
            code = target_file.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return

        marker = f"C2R_LLM_FAILED_OUTPUT_BEGIN func_key: {func_name}"
        if marker in code:
            return

        if rust_parser is None or RUST_LANGUAGE is None:
            return

        try:
            content_bytes = code.encode("utf-8")
            tree = rust_parser.parse(content_bytes)
            query = Query(RUST_LANGUAGE, 
                """
                (function_item
                    name: (identifier) @name
                )
                """
            )
            captures = _query_captures_compat(query, tree.root_node)
            target_node = None
            for node, cap in captures:
                if cap != "name":
                    continue
                name = content_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="ignore")
                if name == func_info.name:
                    target_node = node.parent  # function_item
                    break
            if target_node is None:
                return
            insert_pos = int(target_node.end_byte)
        except Exception:
            return

        safe_llm = (llm_code or "").replace("*/", "* /")
        saved = str(artifact_path) if artifact_path is not None else ""
        comment_lines = [
            "",
            "/* === C2R_LLM_FAILED_OUTPUT_BEGIN ===",
            f" * func_key: {func_name}",
            f" * reason: {reason}",
        ]
        if saved:
            comment_lines.append(f" * saved_translation: {saved}")
        comment_lines.append(" * ------------------------------------------------------------")
        comment_lines.append(safe_llm.rstrip("\n"))
        comment_lines.append(" * ------------------------------------------------------------")
        comment_lines.append(f" * {marker}")
        comment_lines.append(" * === C2R_LLM_FAILED_OUTPUT_END === */")
        comment = "\n".join(comment_lines) + "\n"

        try:
            new_bytes = content_bytes[:insert_pos] + comment.encode("utf-8") + content_bytes[insert_pos:]
            _atomic_write_text(target_file, new_bytes.decode("utf-8", errors="ignore"))
        except Exception:
            return

    def _ensure_extern_decls_from_preprocessed(self, functions: Dict[str, FunctionInfo]) -> int:
        """
        Ensure external C API declarations exist *before* any function translation/injection.

        This prevents E0425 explosions and avoids relying on LLM to emit local `extern "C" {}` blocks.
        Decls are generated from preprocessed `.i` files and written to `work_dir/src/compat.rs`.
        """
        if not self._enable_preprocessed_decl_hints:
            return 0
        compat_path = self.work_dir / "src" / "compat.rs"
        types_path = self.work_dir / "src" / "types.rs"
        if not compat_path.exists():
            return 0
        if not functions:
            return 0

        bindgen_bin = self._find_bindgen_binary()
        defined_types: Set[str] = set()
        try:
            if types_path.exists():
                types_rs_text = types_path.read_text(encoding="utf-8", errors="ignore")
                defined_types = self._parse_defined_type_names_from_types_rs(types_rs_text)
        except Exception:
            defined_types = set()

        internal_names: Set[str] = {fi.name for fi in functions.values() if getattr(fi, "name", None)}

        # Group by translation unit (file_name) to pick the right `.i`.
        by_file: Dict[str, List[FunctionInfo]] = defaultdict(list)
        for fi in functions.values():
            by_file[fi.file_name].append(fi)

        decls_by_name: Dict[str, str] = {}
        decl_reports: List[Dict[str, Any]] = []
        lang_by_file_group: Dict[str, str] = {}

        for file_name, fis in by_file.items():
            if not fis:
                continue
            rep = fis[0]
            lang = "c"
            try:
                src = self._locate_source_file(rep)
                if src and src.suffix.lower() in {".cc", ".cpp", ".cxx", ".c++"}:
                    lang = "c++"
            except Exception:
                lang = "c"
            lang_by_file_group[file_name] = lang
            pre_path = self._locate_preprocessed_file(rep)
            pre_idx = self._get_preprocessed_decl_prototypes(rep)
            if not pre_idx:
                continue

            called: Set[str] = set()
            for fi in fis:
                called.update(self._extract_called_identifiers_from_c(fi.c_code or ""))
            # Only predeclare external callees to avoid name conflicts with translated Rust functions.
            called.difference_update(internal_names)

            for callee in sorted(called):
                protos = pre_idx.get(callee) or []
                if not protos:
                    continue
                best = max(protos, key=len)
                converted = self._c_decl_to_rust_extern_decl(best)
                if not converted:
                    continue
                cname, decl = converted
                if cname and decl and cname not in decls_by_name:
                    decls_by_name[cname] = decl
                    decl_reports.append(
                        {
                            "name": cname,
                            "rust_decl": decl,
                            "c_prototype": best,
                            "source_i": str(pre_path) if pre_path else None,
                            "source_file_group": file_name,
                            "source_lang": lang,
                        }
                    )

        if not decls_by_name:
            return 0

        # Backfill missing typedefs/opaque types referenced by generated decls, so baseline compile
        # won't be blocked by `compat.rs` referring to unknown `crate::types::X` (e.g. StorageHookFunction).
        extra_types_by_name: Dict[str, str] = {}
        if bindgen_bin and types_path.exists():
            try:
                referenced: Set[str] = set()
                for decl in decls_by_name.values():
                    referenced.update(
                        re.findall(r"\b(?:crate::)?types::([A-Za-z_][A-Za-z0-9_]*)\b", decl or "")
                    )
                missing_types = {t for t in referenced if t and t not in defined_types}
            except Exception:
                missing_types = set()

            if missing_types:
                tmp_dir = self.work_dir / ".c2r_bindgen_extern"
                try:
                    tmp_dir.mkdir(parents=True, exist_ok=True)
                except Exception:
                    pass

                missing_by_i: Dict[Tuple[Path, str], Set[str]] = defaultdict(set)
                for rep in decl_reports:
                    try:
                        callee = rep.get("name") or ""
                        src_i = rep.get("source_i") or ""
                        if not callee or not src_i:
                            continue
                        decl = decls_by_name.get(callee) or ""
                        used = set(
                            re.findall(r"\b(?:crate::)?types::([A-Za-z_][A-Za-z0-9_]*)\b", decl)
                        )
                        used = {u for u in used if u in missing_types}
                        if not used:
                            continue
                        group = rep.get("source_file_group") or ""
                        lang = lang_by_file_group.get(group, "c")
                        missing_by_i[(Path(src_i), lang)].update(used)
                    except Exception:
                        continue

                for (pre_path, lang), need_types in sorted(
                    missing_by_i.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])
                ):
                    if not need_types:
                        continue
                    if not pre_path.exists():
                        continue
                    out_rs = tmp_dir / f"{pre_path.stem}.types.rs"
                    cmd: List[str] = [
                        bindgen_bin,
                        str(pre_path),
                        "-o",
                        str(out_rs),
                        "--no-layout-tests",
                        "--no-doc-comments",
                        "--use-core",
                        "--default-enum-style=consts",
                        "--no-prepend-enum-name",
                        "--no-size_t-is-usize",
                    ]
                    for t in sorted(need_types):
                        cmd.extend(["--allowlist-type", rf"^{re.escape(t)}$"])
                    cmd.append("--")
                    if lang == "c++":
                        cmd.extend(["-x", "c++", "-std=c++17"])
                    else:
                        cmd.extend(["-x", "c"])
                    cmd.extend(
                        [
                            "-Wno-error",
                            "-Wno-macro-redefined",
                            "-Wno-builtin-macro-redefined",
                            "-Wno-ignored-attributes",
                        ]
                    )
                    try:
                        proc = subprocess.run(
                            cmd,
                            capture_output=True,
                            text=True,
                            timeout=max(10, int(self._bindgen_extern_timeout_sec or 90)),
                        )
                    except Exception:
                        continue
                    if proc.returncode != 0 or not out_rs.exists():
                        continue
                    try:
                        bindgen_text = out_rs.read_text(encoding="utf-8", errors="ignore")
                    except Exception:
                        bindgen_text = ""
                    type_blocks, _fn_blocks = self._parse_bindgen_types_and_fns(bindgen_text)
                    for t in sorted(need_types):
                        blk = type_blocks.get(t)
                        if not blk:
                            continue
                        if t in defined_types or t in extra_types_by_name:
                            continue
                        extra_types_by_name[t] = blk

                if extra_types_by_name:
                    # We will emit these missing types into `compat.rs` (NOT `types.rs`) to keep the truth layer clean.
                    # Therefore, extern decls must NOT refer to them as `crate::types::T`.
                    moved = sorted(set(extra_types_by_name.keys()), key=len, reverse=True)
                    if moved:
                        for fn in list(decls_by_name.keys()):
                            try:
                                decl = decls_by_name.get(fn) or ""
                            except Exception:
                                decl = ""
                            if not decl:
                                continue
                            for t in moved:
                                decl = re.sub(rf"\bcrate::types::{re.escape(t)}\b", t, decl)
                                decl = re.sub(rf"\btypes::{re.escape(t)}\b", t, decl)
                            decls_by_name[fn] = decl.rstrip()

        # Make sure decls consistently reference `crate::types::X` for available type names.
        if defined_types:
            available_types = set(defined_types)
            for n in list(decls_by_name.keys()):
                try:
                    decls_by_name[n] = self._prefix_types_in_rust_fn_decl(decls_by_name[n], available_types).rstrip()
                except Exception:
                    pass

        report_path = self.manual_fix_root / "extern_decls_from_preprocessed.json"

        section_lines: List[str] = []
        section_lines.append("// === C2R_EXTERN_DECLS_BEGIN ===")
        section_lines.append("// Auto-generated extern decls (C2R step 2.55).")
        section_lines.append(f"// Source: {self.preprocessed_dir}/*.i (preprocessed translation units)")
        section_lines.append(f"// Details: {report_path}")
        section_lines.append("#[allow(improper_ctypes)]")
        section_lines.append("#[allow(non_snake_case)]")
        section_lines.append('extern "C" {')
        for name in sorted(decls_by_name.keys()):
            for ln in (decls_by_name[name] or "").splitlines():
                if not ln.strip():
                    continue
                section_lines.append(f"    {ln.strip()}")
        section_lines.append("}")
        section_lines.append("// === C2R_EXTERN_DECLS_END ===")
        section = "\n".join(section_lines) + "\n\n"

        try:
            compat_text = compat_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            compat_text = ""

        # Emit missing types into compat.rs (truth-layer safe), before extern decls.
        if extra_types_by_name:
            types_begin = "// === C2R_EXTERN_TYPES_BEGIN ==="
            types_end = "// === C2R_EXTERN_TYPES_END ==="
            type_lines: List[str] = []
            type_lines.append(types_begin)
            type_lines.append("// Auto-generated by C2R step 2.55 (preprocessed extern prelude type backfill).")
            type_lines.append("// Only includes missing type items required by extern decls below.")
            imports = self._compute_needed_crate_types_imports(type_blocks=extra_types_by_name, defined_types=defined_types)
            if imports:
                type_lines.append(f"use crate::types::{{{', '.join(imports)}}};")
                type_lines.append("")
            for name in sorted(extra_types_by_name.keys()):
                type_lines.append(extra_types_by_name[name].rstrip())
                type_lines.append("")
            type_lines.append(types_end)
            type_block = "\n".join(type_lines).rstrip() + "\n\n"
            compat_text = self._upsert_marked_block(
                compat_text,
                begin=types_begin,
                end=types_end,
                block=type_block,
                anchor="// === C2R_ACCESSOR_SHIMS_BEGIN ===",
            )

        compat_new = self._upsert_marked_block(
            compat_text,
            begin="// === C2R_EXTERN_DECLS_BEGIN ===",
            end="// === C2R_EXTERN_DECLS_END ===",
            block=section,
            anchor="// === C2R_ACCESSOR_SHIMS_BEGIN ===",
        )
        _atomic_write_text(compat_path, compat_new)

        # Save a machine-readable report for manual audit/migration to stable FFI modules.
        try:
            payload = {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "project": getattr(self, "project_name", None),
                "llm_name": getattr(self, "llm_name", None),
                "workspace_root": str(self.workspace_root),
                "preprocessed_dir": str(self.preprocessed_dir),
                "types_rs": str(types_path) if types_path.exists() else None,
                "compat_rs": str(compat_path),
                "extern_count": len(decls_by_name),
                "extra_types_count": len(extra_types_by_name),
                "extra_types": sorted(extra_types_by_name.keys()),
                "decls": sorted(decl_reports, key=lambda x: (x.get("name") or "")),
            }
            _atomic_write_text(report_path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
            self._extern_decls_report_path = report_path
        except Exception as e:
            logger.warning(f"写入 extern 声明报告失败（已忽略）: {e}")

        return len(decls_by_name)

    def _semantic_cache_file(self, func_info: FunctionInfo) -> Path:
        return self.context_cache_dir / f"{func_info.file_name}_{func_info.index}.json"

    def _read_semantic_cache(self, cache_file: Path) -> Optional[Dict]:
        if not cache_file.exists():
            return None
        try:
            return json.loads(cache_file.read_text(encoding='utf-8'))
        except Exception as exc:
            logger.debug(f"语义切片缓存读取失败 {cache_file}: {exc}")
            return None

    def _write_semantic_cache(self, cache_file: Path, data: Dict) -> None:
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
        except Exception as exc:
            logger.debug(f"语义切片缓存写入失败 {cache_file}: {exc}")

    def _load_semantic_data(self, func_info: FunctionInfo) -> Optional[Dict]:
        if func_info.semantic_data is not None:
            return func_info.semantic_data
        cache_file = self._semantic_cache_file(func_info)
        data = self._read_semantic_cache(cache_file)
        if data is None:
            source_file = self._locate_source_file(func_info)
            if not source_file:
                return None
            try:
                data = extract_semantic_slice(source_file, func_info.name)
                if data:
                    self._write_semantic_cache(cache_file, data)
            except Exception as e:
                logger.debug(f"语义切片生成失败 [{func_info.name}]: {e}")
                return None
        if data is None:
            return None
        data = self._ensure_global_definitions(data, cache_file)
        func_info.semantic_data = data
        func_info.semantic_cache_path = cache_file
        return data

    def _ensure_global_definitions(self, data: Dict, cache_file: Path) -> Dict:
        unresolved = [sym for sym in (data.get("unresolved_symbols") or []) if sym]
        if not unresolved or not self.project_src_dir or not self.project_src_dir.exists():
            return data
        existing_symbols = {item.get("symbol") for item in (data.get("global_definitions") or []) if item.get("symbol")}
        missing = [sym for sym in unresolved if sym not in existing_symbols]
        if not missing:
            return data
        try:
            decls = collect_global_declarations(self.project_src_dir, missing)
        except Exception as exc:
            logger.debug(f"全局声明探针失败: {exc}")
            return data
        if not decls:
            return data
        global_defs = data.get("global_definitions") or []
        for symbol, entries in decls.items():
            for entry in entries:
                snippet = (entry.get("code") or "").strip()
                if not snippet:
                    continue
                payload = {
                    "symbol": symbol,
                    "code": snippet
                }
                source_path = entry.get("source_path") or entry.get("path")
                if source_path:
                    payload["source_path"] = source_path
                kind = entry.get("kind")
                if kind:
                    payload["kind"] = kind
                global_defs.append(payload)
        # 去重 + 限制数量
        deduped = []
        seen = set()
        for item in global_defs:
            key = (item.get("symbol"), item.get("source_path"), item.get("code"))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        if deduped:
            max_items = 6
            data["global_definitions"] = deduped[:max_items]
            self._write_semantic_cache(cache_file, data)
        return data

    def _get_semantic_context(self, func_info: FunctionInfo, data: Optional[Dict] = None) -> str:
        if data is None:
            data = self._load_semantic_data(func_info)
        return self._format_semantic_context(data)

    def _get_global_declaration_context(self, func_info: FunctionInfo, data: Optional[Dict] = None) -> str:
        target = data if data is not None else (func_info.semantic_data or self._load_semantic_data(func_info))
        if not target:
            return ""
        return self._format_global_declarations(target)

    def _get_usage_examples_context(self, func_info: FunctionInfo, semantic_data: Optional[Dict] = None) -> Tuple[str, List[Dict[str, Any]]]:
        if not self.project_src_dir or not self.project_src_dir.exists():
            return "", []
        data = semantic_data or func_info.semantic_data or self._load_semantic_data(func_info)
        if not data:
            return "", []
        unresolved = [sym.strip() for sym in (data.get("unresolved_symbols") or []) if sym]
        if not unresolved:
            return "", []
        symbols = unresolved[:8]
        try:
            usage_map = ensure_usage_examples(
                self.project_src_dir,
                self.usage_examples_file,
                symbols,
                max_examples_per_symbol=2
            )
        except Exception as exc:
            logger.debug(f"用法示例收集失败 [{func_info.name}]: {exc}")
            return "", []
        parts: List[str] = []
        usage_stats: List[Dict[str, Any]] = []
        total_len = 0
        limit = 2500
        for symbol in symbols:
            entries = usage_map.get(symbol) or []
            if not entries:
                continue
            usage_stats.append({
                "symbol": symbol,
                "example_count": len(entries)
            })
            symbol_lines: List[str] = [f"// Symbol: {symbol}"]
            added = False
            for entry in entries[:2]:
                snippet = (entry.get("snippet") or entry.get("code") or "").strip()
                if not snippet:
                    continue
                location = entry.get("file") or entry.get("source_path")
                display_loc = None
                if location:
                    try:
                        display_loc = str(Path(location).resolve().relative_to(self.project_src_dir.resolve()))
                    except Exception:
                        display_loc = Path(location).name
                line_no = entry.get("line")
                arg_count = entry.get("arg_count")
                header = "//   usage"
                if display_loc:
                    header += f" @ {display_loc}"
                if line_no:
                    header += f":{line_no}"
                if arg_count is not None:
                    header += f" (args={arg_count})"
                block = f"{header}\n{snippet}"
                prospective_len = total_len + len(block) + 1
                if prospective_len > limit:
                    break
                symbol_lines.append(block)
                total_len = prospective_len
                added = True
            if added:
                parts.append("\n".join(symbol_lines))
            if total_len >= limit:
                break
        return "\n\n".join(parts).strip(), usage_stats

    def _type_exists_in_skeleton(self, type_name: str, file_name: str) -> bool:
        # 尝试两种文件名格式：src_xxx.rs 和 xxx.rs（向后兼容）
        target_file = self.work_dir / "src" / f"src_{file_name}.rs"
        if not target_file.exists():
            target_file = self.work_dir / "src" / f"{file_name}.rs"
            if not target_file.exists():
                return False
        try:
            content = target_file.read_text(encoding='utf-8', errors='ignore')
        except Exception:
            return False
        pattern = re.compile(rf"\b(pub\s+)?struct\s+{re.escape(type_name)}\b")
        return bool(pattern.search(content))

    def _ensure_placeholder_stub(self, func_info: FunctionInfo, placeholder_code: str, marker: Optional[str] = None) -> None:
        # 尝试两种文件名格式：src_xxx.rs 和 xxx.rs（向后兼容）
        target_file = self.work_dir / "src" / f"src_{func_info.file_name}.rs"
        if not target_file.exists():
            target_file = self.work_dir / "src" / f"{func_info.file_name}.rs"
            if not target_file.exists():
                return
        try:
            content = target_file.read_text(encoding='utf-8', errors='ignore')
        except Exception:
            return
        if marker and marker in content:
            return
        if not content.endswith('\n'):
            content += '\n'
        content = content.rstrip() + '\n\n' + placeholder_code.strip() + '\n'
        target_file.write_text(content, encoding='utf-8')

    def _infer_signature_from_cpp(self, func_info: FunctionInfo) -> Optional[Dict[str, Any]]:
        code = (func_info.c_code or "").strip()
        if not code:
            return None
        lines = [l for l in code.splitlines() if l.strip()]
        first_line = lines[0].strip() if lines else code

        # 检测析构函数（仅在“函数声明行”匹配，避免把 C 代码中的按位取反 ~0x1F 误判成 C++ 析构函数）
        # 支持两类形式：
        # - Class::~Class(...)
        # - ~Class(...)
        destructor_match = re.match(r'^\s*(?:[\w:]+::)?~\s*(\w+)\s*\(', first_line)
        if destructor_match:
            class_name = destructor_match.group(1)
            if not self._type_exists_in_skeleton(class_name, func_info.file_name):
                return {"skip": True, "reason": f"类型 {class_name} 在骨架中不存在"}
            placeholder = f"impl Drop for {class_name} {{\n    fn drop(&mut self) {{}}\n}}"
            return {
                "signature": f"impl Drop for {class_name}",
                "placeholder": placeholder,
                "marker": f"impl Drop for {class_name}",
                "kind": "drop"
            }
        # 构造函数（Class::Class 或 Class())
        ctor_scope = re.match(r'(\w+)::(\w+)\s*\(', first_line)
        if ctor_scope and ctor_scope.group(1) == ctor_scope.group(2):
            class_name = ctor_scope.group(1)
        else:
            ctor_simple = re.match(r'(\w+)\s*\(', first_line)
            class_name = ctor_simple.group(1) if ctor_simple else None
        if class_name and 'return' not in first_line:
            if not self._type_exists_in_skeleton(class_name, func_info.file_name):
                return None
            placeholder = f"impl {class_name} {{\n    pub fn new() -> Self {{\n        unimplemented!()\n    }}\n}}"
            return {
                "signature": f"impl {class_name} {{ pub fn new() -> Self }}",
                "placeholder": placeholder,
                "marker": "pub fn new() -> Self",
                "kind": "ctor"
            }
        return None

    def _mark_function_skipped(self, func_name: str, reason: str) -> None:
        print(f"  跳过 {func_name}: {reason}")
        logger.info(f"跳过函数 {func_name}: {reason}")
        self.stats["skipped"] += 1

    def _build_context_prefix(self, func_info: FunctionInfo) -> Tuple[str, Dict[str, Any]]:
        semantic_data = func_info.semantic_data or self._load_semantic_data(func_info)
        semantic_context = self._format_semantic_context(semantic_data) if semantic_data else ""
        global_context = self._format_global_declarations(semantic_data) if semantic_data else ""
        usage_context, usage_stats = self._get_usage_examples_context(func_info, semantic_data=semantic_data)
        
        # 获取调用图上下文（被调用函数的签名或已翻译代码）
        call_graph_context = self._get_call_graph_context(func_info)

        block_candidates: List[Dict[str, Any]] = []
        
        # 优先添加调用图上下文（最相关的上下文）
        if call_graph_context:
            block_candidates.append({
                "label": "Called Functions Context",
                "content": call_graph_context,
                "stats": {
                    "call_graph_context": True,
                    "context_length": len(call_graph_context)
                }
            })
        
        if semantic_context:
            block_candidates.append({
                "label": "Semantic Slice",
                "content": semantic_context,
                "stats": {
                    "helper_definitions": len(semantic_data.get("infile_definitions") or []) if semantic_data else 0,
                    "unresolved_symbols": len(semantic_data.get("unresolved_symbols") or []) if semantic_data else 0
                }
            })
        if global_context:
            block_candidates.append({
                "label": "Relevant C Declarations",
                "content": global_context,
                "stats": {
                    "global_declarations": len(semantic_data.get("global_definitions") or []) if semantic_data else 0
                }
            })
        if usage_context:
            block_candidates.append({
                "label": "Usage Examples",
                "content": usage_context,
                "stats": {
                    "usage_symbols": usage_stats
                }
            })

        selected_blocks: List[str] = []
        block_meta: List[Dict[str, Any]] = []
        used_chars = 0
        for block in block_candidates:
            content = block.get("content", "").strip()
            label = block.get("label", "unknown")
            stats = block.get("stats") or {}
            if not content:
                block_meta.append({
                    "label": label,
                    "length": 0,
                    "included": False,
                    "reason": "empty",
                    "stats": stats
                })
                continue
            block_text = f"// ===== {label} =====\n{content}"
            block_length = len(block_text)
            if used_chars + block_length > CONTEXT_PREFIX_CHAR_LIMIT:
                block_meta.append({
                    "label": label,
                    "length": block_length,
                    "included": False,
                    "reason": "budget",
                    "stats": stats
                })
                continue
            selected_blocks.append(block_text)
            used_chars += block_length
            block_meta.append({
                "label": label,
                "length": block_length,
                "included": True,
                "stats": stats
            })

        prefix = "\n\n".join(selected_blocks).strip()
        meta = {
            "context_budget": CONTEXT_PREFIX_CHAR_LIMIT,
            "context_used": used_chars,
            "context_blocks": block_meta
        }
        return prefix, meta

    def _save_context_summary(self, func_info: FunctionInfo, context_meta: Optional[Dict[str, Any]]) -> Optional[Path]:
        if not context_meta:
            return None
        try:
            summary_dir = self.llm_prompts_dir / "context_summary"
            summary_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            safe_name = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in func_info.name)
            file_path = summary_dir / f"context_{safe_name}_{timestamp}.txt"
            lines = [
                f"Function: {func_info.name}",
                f"Budget(chars): {context_meta.get('context_budget')}",
                f"Used(chars): {context_meta.get('context_used')}",
                ""
            ]
            blocks = context_meta.get("context_blocks") or []
            for block in blocks:
                status = "included" if block.get("included") else f"skipped ({block.get('reason', 'n/a')})"
                lines.append(f"- {block.get('label')}: {block.get('length', 0)} chars, {status}")
                stats = block.get("stats") or {}
                if stats:
                    lines.append(f"  stats: {json.dumps(stats, ensure_ascii=False)}")
            file_path.write_text("\n".join(lines), encoding='utf-8')
            return file_path
        except Exception as exc:
            logger.debug(f"保存上下文摘要失败 [{func_info.name}]: {exc}")
            return None

    def _format_semantic_context(self, data: Optional[Dict]) -> str:
        if not data:
            return ""
        parts: List[str] = []
        definitions = data.get("infile_definitions") or []
        if definitions:
            parts.append("// Semantic slice: helper definitions from source file")
            for item in definitions[:3]:
                name = item.get("name", "unknown")
                snippet = (item.get("code") or "").strip()
                if not snippet:
                    continue
                if len(snippet) > 1200:
                    snippet = snippet[:1200] + "\n/* ...truncated... */"
                parts.append(f"// ---- definition: {name} ----\n{snippet}")
        unresolved = data.get("unresolved_symbols") or []
        unresolved = [sym for sym in unresolved if sym]
        if unresolved:
            parts.append("// Symbols referenced but not defined in this file:")
            parts.append(", ".join(sorted(set(unresolved))[:20]))
        semantic_context = "\n".join(parts).strip()
        # C2R: 记录语义上下文长度，暂不截断，观察是否超限
        if len(semantic_context) > 4000:
            logger.info(f"[上下文统计] semantic_context 长度: {len(semantic_context)} 字符 (~{len(semantic_context)//4} tokens)")
        return semantic_context

    def _format_global_declarations(self, data: Optional[Dict]) -> str:
        if not data:
            return ""
        entries = data.get("global_definitions") or []
        if not entries:
            return ""
        parts: List[str] = []
        for entry in entries[:8]:  # C2R: 增加到 8 个全局声明
            symbol = entry.get("symbol") or "unknown"
            origin = entry.get("source_path")
            header = f"// ---- declaration: {symbol}"
            if origin:
                header += f" ({Path(origin).name})"
            header += " ----"
            snippet = (entry.get("code") or "").strip()
            if not snippet:
                continue
            if len(snippet) > 2000:  # C2R: 增加单个片段限制
                logger.info(f"[上下文统计] global snippet '{symbol}' 过长: {len(snippet)} 字符，截断到 2000")
                snippet = snippet[:2000] + "\n/* ...truncated... */"
            parts.append(f"{header}\n{snippet}")
        global_context = "\n\n".join(parts).strip()
        # C2R: 记录全局上下文长度，暂不截断
        if len(global_context) > 4000:
            logger.info(f"[上下文统计] global_context 长度: {len(global_context)} 字符 (~{len(global_context)//4} tokens)")
        return global_context
    
    def _extract_type_definitions(self, code: str) -> List[str]:
        """提取类型定义（包括常量）"""
        definitions = []
        
        try:
            tree = rust_parser.parse(bytes(code, 'utf-8'))
            
            # 查询 struct、enum、type alias 和 const
            # 注意：必须包含 const_item，否则 types.rs 中的常量定义会被截断
            query = Query(RUST_LANGUAGE, """
                (struct_item) @struct
                (enum_item) @enum
                (type_item) @type
                (const_item) @const
            """)
            
            captures = _query_captures_compat(query, tree.root_node)
            for node, _ in captures:
                # 提取完整的定义，确保不被截断
                definition = code[node.start_byte:node.end_byte].strip()
                if definition:
                    definitions.append(definition)
        except Exception as e:
            logger.debug(f"提取类型定义失败: {e}")
        
        return definitions
    
    def _detect_opaque_types(self, code: str) -> Set[str]:
        """
        通用方法：检测骨架中的不透明类型（如 type X = c_void）
        
        返回: 不透明类型名称集合
        """
        opaque_types = set()

        # fallback：tree-sitter-rust 不可用时，用正则识别 `type X = ...;`
        if rust_parser is None or RUST_LANGUAGE is None:
            pat = re.compile(r'(?m)^\s*(?:pub\s+)?type\s+(\w+)\s*=\s*([^;]+);')
            for m in pat.finditer(code or ""):
                name = m.group(1)
                value = m.group(2) or ""
                if "c_void" in value or value.strip() == "()" or "()" in value:
                    opaque_types.add(name)
            # 即使在 fallback 模式，也需要检测 struct 形式的 opaque 类型
            # (后面的代码会处理这个)

        try:
            tree = rust_parser.parse(bytes(code, 'utf-8'))

            query = Query(RUST_LANGUAGE, """
                (type_item
                    name: (type_identifier) @type_name
                    type: (_) @type_value
                )
            """)

            captures = _query_captures_compat(query, tree.root_node)
            type_name = None

            for node, capture_name in captures:
                if capture_name == 'type_name':
                    type_name = code[node.start_byte:node.end_byte]
                elif capture_name == 'type_value' and type_name:
                    type_value = code[node.start_byte:node.end_byte]
                    if 'c_void' in type_value or '()' in type_value:
                        opaque_types.add(type_name)
                    type_name = None
        except Exception as e:
            logger.debug(f"检测不透明类型失败: {e}")

        # === C2R FIX: 检测 bindgen 生成的 struct 形式 opaque 类型 ===
        # bindgen 对找不到定义的类型生成形如:
        #   /// Opaque placeholder for external type `TypeName`
        #   #[repr(C)]
        #   pub struct TypeName { _private: [u8; 0] }
        # 这些类型的字段不可访问，LLM 需要知道这一点。

        # 方法 1: 检测带有 "Opaque placeholder" 注释的 struct
        opaque_comment_pat = re.compile(
            r'///\s*Opaque placeholder[^\n]*?`(\w+)`',
            re.IGNORECASE
        )
        for m in opaque_comment_pat.finditer(code or ""):
            opaque_types.add(m.group(1))

        # 方法 2: 检测只有 _private/_opaque 字段的 struct (更通用)
        private_struct_pat = re.compile(
            r'pub struct (\w+)\s*\{\s*'
            r'(?:pub\s+)?_(?:private|opaque|unused|c2r_private|reserved):\s*\[u8;\s*0\]\s*,?\s*\}'
        )
        for m in private_struct_pat.finditer(code or ""):
            opaque_types.add(m.group(1))

        if opaque_types:
            logger.debug(f"检测到 {len(opaque_types)} 个 opaque 类型: {sorted(opaque_types)[:10]}...")

        return opaque_types

    def _extract_function_signatures(self, code: str) -> List[str]:
        """提取函数签名（不包含函数体）"""
        signatures = []
        
        try:
            tree = rust_parser.parse(bytes(code, 'utf-8'))
            
            query = Query(RUST_LANGUAGE, """
                (function_item
                    name: (identifier) @name
                    parameters: (parameters) @params
                    return_type: (type_identifier)? @ret
                ) @func
            """)
            
            captures = _query_captures_compat(query, tree.root_node)
            for node, capture_name in captures:
                if capture_name == 'func':
                    # 提取签名部分（到 { 之前）
                    func_text = code[node.start_byte:node.end_byte]
                    brace_idx = func_text.find('{')
                    if brace_idx != -1:
                        sig = func_text[:brace_idx].strip()
                        signatures.append(sig + ";")
        except Exception as e:
            logger.debug(f"提取函数签名失败: {e}")
        
        return signatures
    
    def _extract_standalone_signature_from_skeleton(self, func_name: str, func_info: FunctionInfo) -> Optional[str]:
        """
        从骨架文件中提取独立函数签名（非 trait 方法）
        
        优先返回 `pub fn xxx(...)` 格式的独立函数签名
        """
        # 尝试两种文件名格式：src_xxx.rs 和 xxx.rs（向后兼容）
        rs_filename_with_prefix = "src_" + func_info.file_name + ".rs"
        rs_filename = func_info.file_name + ".rs"
        skeleton_file = self.work_dir / "src" / rs_filename_with_prefix
        
        if not skeleton_file.exists():
            # 回退到不带前缀的格式（向后兼容）
            skeleton_file = self.work_dir / "src" / rs_filename
            if not skeleton_file.exists():
                return None
        
        try:
            with open(skeleton_file, 'r', encoding='utf-8', errors='ignore') as f:
                code = f.read()
            
            tree = rust_parser.parse(bytes(code, 'utf-8'))
            
            # 查询所有函数定义
            query = Query(RUST_LANGUAGE, """
                (function_item
                    name: (identifier) @func_name
                ) @func_def
            """)
            
            captures = _query_captures_compat(query, tree.root_node)
            
            # 从 C 代码中提取实际函数名
            actual_func_name = func_info.name
            
            # 构建多种可能的函数名变体（通用匹配策略）
            possible_names = {
                actual_func_name,
                actual_func_name.lower(),
                actual_func_name.upper(),
                # 驼峰转蛇形：GetObjectItem -> get_object_item
                re.sub(r'([A-Z])', r'_\1', actual_func_name).lower().lstrip('_'),
                # 蛇形转驼峰：get_object_item -> GetObjectItem
                ''.join(word.capitalize() for word in actual_func_name.split('_')),
                # 移除下划线：get_object_item -> getobjectitem
                actual_func_name.replace('_', '').lower(),
            }
            
            for node, capture_name in captures:
                if capture_name == 'func_name':
                    node_text = code[node.start_byte:node.end_byte]
                    
                    # 检查函数名是否匹配（支持多种命名风格）
                    if (node_text in possible_names or
                        node_text.lower() in {n.lower() for n in possible_names} or
                        node_text.replace('_', '').lower() in {n.replace('_', '').lower() for n in possible_names}):
                        
                        # 获取函数定义节点
                        func_def_node = node.parent
                        while func_def_node and func_def_node.type != 'function_item':
                            func_def_node = func_def_node.parent
                        
                        if func_def_node:
                            # 检查是否在 impl 块中（trait 方法）
                            parent = func_def_node.parent
                            if parent and parent.type == 'impl_item':
                                continue  # 跳过 trait 方法
                            
                            # 提取签名（到 { 之前）
                            func_text = code[func_def_node.start_byte:func_def_node.end_byte]
                            brace_idx = func_text.find('{')
                            if brace_idx != -1:
                                sig = func_text[:brace_idx].strip()
                                return sig
            
        except Exception as e:
            logger.debug(f"从骨架提取签名失败: {e}")
        
        return None
    
    def _process_function(
        self,
        func_name: str,
        func_info: FunctionInfo,
        skeleton_context: str
    ) -> bool:
        """
        处理单个函数：翻译 -> 注入 -> 编译 -> 修复循环

        返回: 是否成功
        """
        # 0. 【新增】优先从确定性映射文件读取签名（100%准确，避免BM25猜测）
        rust_signature = self._load_signature_from_mapping(func_name)
        if rust_signature:
            print(f"  [签名-确定性] 从映射文件获取: {rust_signature[:60]}...")
            logger.debug(f"[DEBUG] {func_name}: 从映射文件加载签名 = '{rust_signature}'")
            func_info.rust_signature = rust_signature
            # 跳过后续的签名查找步骤
        else:
            # 回退到原有的签名获取流程
            # 1. 读取 Rust 签名（优先从骨架中提取独立函数签名）
            placeholder_info: Optional[Dict[str, Any]] = None

            skeleton_sig = self._extract_standalone_signature_from_skeleton(func_name, func_info)
            if skeleton_sig:
                print(f"  [签名] 从骨架提取: {skeleton_sig[:60]}...")
            logger.debug(f"[DEBUG] {func_name}: 骨架签名提取结果 = '{skeleton_sig}'")
            if skeleton_sig:
                rust_signature = skeleton_sig
            else:
                # 方法2: 从签名匹配文件读取
                signature_file = self.signature_dir / f"{func_name}.txt"
                logger.debug(f"[DEBUG] {func_name}: 检查签名文件 {signature_file}, 存在={signature_file.exists()}")
                if signature_file.exists():
                    with open(signature_file, 'r', encoding='utf-8', errors='ignore') as f:
                        rust_signature = f.read().strip()
                    logger.debug(f"[DEBUG] {func_name}: 从签名文件读取 = '{rust_signature[:100] if rust_signature else ''}'")

                    # 如果是 trait 方法签名，转换为独立函数签名
                    if "&self" in rust_signature:
                        rust_signature = rust_signature.replace("&self, ", "").replace("&self,", "").replace("(&self)", "()")
                        if not rust_signature.startswith("pub "):
                            rust_signature = "pub " + rust_signature

            if not rust_signature:
                print(f"  [签名] 尝试从 C++ 代码推断...")
                logger.debug(f"[DEBUG] {func_name}: 尝试从 C++ 推断签名")
                placeholder_info = self._infer_signature_from_cpp(func_info)
                if placeholder_info and placeholder_info.get("signature"):
                    print(f"  [签名] 推断成功: {placeholder_info.get('signature', '')[:60]}...")
                logger.debug(f"[DEBUG] {func_name}: 推断结果 = {placeholder_info}")
                if placeholder_info:
                    if placeholder_info.get("skip"):
                        func_info.skipped = True
                        self._mark_function_skipped(func_name, placeholder_info.get("reason", "未匹配到可用类型"))
                        return False  # 返回 False 表示跳过，不是成功
                    rust_signature = placeholder_info.get("signature", "")
                    if placeholder_info.get("placeholder"):
                        self._ensure_placeholder_stub(
                            func_info,
                            placeholder_info["placeholder"],
                            placeholder_info.get("marker")
                        )

            if not rust_signature:
                func_info.skipped = True
                self._mark_function_skipped(func_name, "未找到匹配的 Rust 签名")
                return False  # 返回 False 表示跳过，不是成功

            func_info.rust_signature = rust_signature
        
        # 2. 检查是否已有翻译结果
        translated_file = self.translated_dir / f"{func_name}.txt"
        if translated_file.exists():
            with open(translated_file, 'r', encoding='utf-8', errors='ignore') as f:
                translated_content = f.read()
            
            # 尝试直接使用现有翻译
            if self._try_inject_and_compile(func_name, func_info, translated_content):
                self.stats["translated"] += 1
                return True

        # types.rs slice: extra symbols accumulated from rustc errors during repair attempts
        types_slice_extra_symbols: Set[str] = set()
        
        # 3. 翻译函数
        per_func_context = self._build_skeleton_context_for_function(
            func_info,
            skeleton_context,
            extra_types_symbols=types_slice_extra_symbols,
        )
        translated_code = self._translate_function(func_info, per_func_context)
        if not translated_code:
            return False
        
        self.stats["translated"] += 1
        
        # 注册已翻译函数（用于为后续函数提供调用上下文）
        self._register_translated_function(func_info, translated_code)
        
        # 4. 注入并编译（不回退，以便修复循环能获取正确的编译错误）
        # 注意：这里设置 rollback_on_failure=False，让翻译后的代码保留在文件中
        # 这样修复循环中 _get_compile_error() 才能获取到正确的编译错误
        if self._try_inject_and_compile(func_name, func_info, translated_code, rollback_on_failure=False):
            return True
        if getattr(func_info, "injection_failed", False):
            # 不做“删除函数/删除内容”来提高编译率：保留占位符，并把 LLM 翻译结果落盘供人工复检
            artifact = self._save_manual_fix_artifact(
                func_name,
                func_info,
                translated_code,
                reason="injection_failed",
                error_msg="",
            )
            comment = self._build_manual_fix_comment(
                func_name,
                func_info,
                reason="injection_failed",
                artifact_path=artifact,
                error_msg="",
            )
            self._inject_failure_comment(func_name, func_info, comment)
            # 在函数定义下方保留 LLM 失败输出（注释形式；便于对比/人工复检）
            try:
                self._append_llm_failed_code_comment_below_function(
                    func_name=func_name,
                    func_info=func_info,
                    llm_code=translated_code,
                    artifact_path=artifact,
                    reason="injection_failed",
                )
            except Exception:
                pass

            # 可选：使用 C2Rust 兜底填充函数体（只在能通过 cargo check 时才保留）
            if self._enable_c2rust_fallback:
                try:
                    mod_info = self._ensure_c2rust_fallback_module(func_info=func_info)
                    if mod_info:
                        file_group, _mod_path = mod_info
                        idx = self._c2rust_fallback_function_index.get(file_group)
                        if idx is None or func_info.name in idx:
                            wrapper = self._build_c2rust_fallback_wrapper(func_info=func_info, file_group=file_group)
                            if wrapper and self._try_inject_and_compile(
                                func_name, func_info, wrapper, rollback_on_failure=True
                            ):
                                try:
                                    self.stats["c2rust_fallback"] += 1
                                except Exception:
                                    pass
                                print("  ✓ C2Rust 兜底翻译成功（已填充函数体，LLM 输出已保留为注释）")
                                return True
                except Exception as e:
                    logger.debug(f"[C2Rust fallback] {func_name}: injection_failed path failed (ignored): {e}")
            print("  ❌ 无法将翻译结果注入骨架（已保留翻译工件，保留占位符）")
            return False
        
        # 5. 修复循环（带错误历史追踪）
        # 此时文件中保留的是翻译后的代码（可能有编译错误）
        error_history = []  # 记录历史错误，避免重复
        repair_attempts_history = []  # 记录每次尝试的完整错误和代码
        # 创建函数修复历史目录
        func_history_dir = self.repair_history_dir / func_name
        func_history_dir.mkdir(parents=True, exist_ok=True)
        
        for attempt in range(self.max_repair_attempts):
            print(f"  修复尝试 {attempt + 1}/{self.max_repair_attempts}...")
            
            # 获取编译错误（此时文件中是翻译后的代码，可以获取正确的错误）
            error_msg = self._get_compile_error()
            
            # 保存每次尝试的完整错误信息
            error_file = func_history_dir / f"attempt_{attempt + 1}_error.txt"
            _atomic_write_text(
                error_file,
                "".join(
                    [
                        f"函数: {func_name}\n",
                        f"文件: {func_info.file_name}\n",
                        f"尝试次数: {attempt + 1}/{self.max_repair_attempts}\n",
                        f"{'='*60}\n",
                        "完整编译错误信息:\n",
                        f"{'='*60}\n",
                        (error_msg if error_msg else "(无错误信息)"),
                    ]
                ),
            )
            
            # 保存当前尝试的翻译代码（修复前的代码）
            code_file = func_history_dir / f"attempt_{attempt + 1}_code.rs"
            _atomic_write_text(
                code_file,
                "".join(
                    [
                        f"函数: {func_name}\n",
                        f"文件: {func_info.file_name}\n",
                        f"尝试次数: {attempt + 1}/{self.max_repair_attempts}\n",
                        f"{'='*60}\n",
                        "翻译代码（修复前）:\n",
                        f"{'='*60}\n",
                        (translated_code if translated_code else "(无代码)"),
                    ]
                ),
            )

            # ------------------------------------------------------------
            # Scheme-B: deterministic C accessor shims for missing fields
            # ------------------------------------------------------------
            try:
                if self._try_fix_missing_fields_with_c_shims(error_msg):
                    # Re-compile to validate the deterministic fix.
                    if self._compile_project():
                        logger.info(f"[{func_name}] C accessor shims fixed E0609 field errors")
                        self.stats["repaired"] += 1
                        return True
                    # Refresh error message for subsequent repair logic.
                    error_msg = self._get_compile_error()
            except Exception as e:
                logger.debug(f"[{func_name}] 字段 accessor shim 修复失败: {e}")
            
            # 检查是否是重复错误（陷入死循环）
            error_key = error_msg[:500] if error_msg else ""
            if error_key in error_history[-3:] if len(error_history) >= 3 else False:
                print(f"  检测到重复错误（历史记录已传递给LLM）...")
            error_history.append(error_key)
            
            # 保存本次尝试的完整信息到历史记录（用于传递给LLM）
            repair_attempts_history.append({
                "attempt_num": attempt + 1,
                "error_msg": error_msg,  # 完整错误信息
                "code_before": translated_code,  # 修复前的代码
            })
            
            # 保存错误历史到文件
            history_file = func_history_dir / "error_history.txt"
            history_lines = [
                f"函数: {func_name}\n",
                "错误历史摘要（前500字符）:\n",
                f"{'='*60}\n",
            ]
            for i, err in enumerate(error_history, 1):
                history_lines.append(f"\n尝试 {i}:\n{err}\n")
                history_lines.append(f"{'-'*60}\n")
            _atomic_write_text(history_file, "".join(history_lines))
            
            # 请求 LLM 修复（传递完整历史记录）
            # 注意：传递所有历史记录，让 LLM 能看到之前所有的尝试
            # 第1次修复时，repair_attempts_history 已经包含了当前（第1次）的错误和代码
            # types.rs slice 扩展：根据 rustc 错误提取缺失类型/常量符号，补齐上下文后再让 LLM 修复
            try:
                types_slice_extra_symbols.update(self._extract_missing_symbols_for_types_slice(error_msg))
            except Exception:
                pass
            per_attempt_context = self._build_skeleton_context_for_function(
                func_info,
                skeleton_context,
                extra_types_symbols=types_slice_extra_symbols,
            )
            repaired_code = self._repair_function(
                func_info, translated_code, error_msg, per_attempt_context,
                attempt_num=attempt + 1,
                error_history=error_history[-3:],  # 错误摘要（用于检测重复）
                repair_attempts_history=repair_attempts_history  # 传递所有历史记录
            )
            
            # 保存修复后的代码
            if repaired_code:
                repaired_code_file = func_history_dir / f"attempt_{attempt + 1}_repaired_code.rs"
                _atomic_write_text(
                    repaired_code_file,
                    "".join(
                        [
                            f"函数: {func_name}\n",
                            f"文件: {func_info.file_name}\n",
                            f"尝试次数: {attempt + 1}/{self.max_repair_attempts}\n",
                            f"{'='*60}\n",
                            "修复后的代码:\n",
                            f"{'='*60}\n",
                            repaired_code,
                        ]
                    ),
                )
                
                # 更新历史记录，添加修复后的代码
                if len(repair_attempts_history) > 0:
                    repair_attempts_history[-1]["code_after"] = repaired_code
            
            # 注入修复后的代码并编译（不回退，以便下次循环能获取正确的错误）
            if repaired_code and self._try_inject_and_compile(func_name, func_info, repaired_code, rollback_on_failure=False):
                # 修复成功，保存成功标记
                success_file = func_history_dir / "repair_success.txt"
                _atomic_write_text(
                    success_file,
                    "".join(
                        [
                            f"函数: {func_name}\n",
                            f"成功修复于尝试: {attempt + 1}/{self.max_repair_attempts}\n",
                            f"最终代码已保存到: attempt_{attempt + 1}_repaired_code.rs\n",
                        ]
                    ),
                )
                self.stats["repaired"] += 1
                return True
            if getattr(func_info, "injection_failed", False):
                break
            
            translated_code = repaired_code or translated_code
        
        # 6. 修复失败，保存失败标记
        failure_file = func_history_dir / "repair_failed.txt"
        _atomic_write_text(
            failure_file,
            "".join(
                [
                    f"函数: {func_name}\n",
                    f"修复失败: 经过 {self.max_repair_attempts} 次尝试后仍无法修复\n",
                    "所有尝试的错误信息和代码已保存在当前目录\n",
                ]
            ),
        )

        # 6.5 保留翻译工件（避免“为编译通过删内容”导致人工无法复检）
        final_error_msg = ""
        try:
            final_error_msg = self._get_compile_error() or ""
        except Exception:
            final_error_msg = ""
        artifact = self._save_manual_fix_artifact(
            func_name,
            func_info,
            translated_code,
            reason=f"repair_failed_after_{self.max_repair_attempts}",
            error_msg=final_error_msg,
        )
        base_comment = self._build_manual_fix_comment(
            func_name,
            func_info,
            reason=f"repair_failed_after_{self.max_repair_attempts}",
            artifact_path=artifact,
            error_msg=final_error_msg,
        )

        # 回退到占位符版本（所有修复尝试都失败后才回退）
        self._rollback_function(func_name, func_info)

        # 6.6 可选：使用 C2Rust 工具兜底填充函数体（只在能通过 cargo check 时才保留）
        used_c2rust_fallback = False
        if self._enable_c2rust_fallback:
            try:
                mod_info = self._ensure_c2rust_fallback_module(func_info=func_info)
                if mod_info:
                    file_group, _mod_path = mod_info
                    # Best-effort: if we have an index and it doesn't contain the function, skip.
                    idx = self._c2rust_fallback_function_index.get(file_group)
                    if idx is None or func_info.name in idx:
                        wrapper = self._build_c2rust_fallback_wrapper(func_info=func_info, file_group=file_group)
                        if wrapper and self._try_inject_and_compile(func_name, func_info, wrapper, rollback_on_failure=True):
                            used_c2rust_fallback = True
                            try:
                                self.stats["c2rust_fallback"] += 1
                            except Exception:
                                pass
            except Exception as e:
                logger.debug(f"[C2Rust fallback] {func_name}: failed (ignored): {e}")

        # 7. 注入失败注释（默认 simple：不额外走 LLM，避免拖慢整体）
        comment = base_comment
        try:
            extra_comment = ""
            if self.failure_comment_mode in ("llm", "full"):
                extra_comment = self._generate_failure_comment(func_name, func_info, error_history)
            elif self.failure_comment_mode in ("simple", "default", ""):
                extra_comment = self._generate_default_failure_comment(func_name, func_info)
            if extra_comment:
                comment = base_comment + "\n" + extra_comment.strip() + "\n"
        except Exception as e:
            logger.debug(f"生成额外失败注释失败（忽略）: {e}")

        try:
            self._inject_failure_comment(func_name, func_info, comment)
            # 保存注释到修复历史目录
            _atomic_write_text(func_history_dir / "failure_comment.txt", comment)
        except Exception as e:
            logger.warning(f"注入失败注释时出错: {e}")

        # 7.5 在函数定义下方保留 LLM 失败输出（注释形式；便于对比/人工复检）
        try:
            self._append_llm_failed_code_comment_below_function(
                func_name=func_name,
                func_info=func_info,
                llm_code=translated_code,
                artifact_path=artifact,
                reason=f"repair_failed_after_{self.max_repair_attempts}",
            )
        except Exception:
            pass
        
        # 如果 C2Rust 兜底成功并且项目可编译，则视为本函数已“恢复成功”
        if used_c2rust_fallback:
            return True
        return False
    
    def _translate_function_only(
        self,
        func_name: str,
        func_info: FunctionInfo,
        skeleton_context: str
    ) -> Optional[str]:
        """
        仅翻译函数（不注入），用于并行翻译模式
        
        Args:
            func_name: 函数名
            func_info: 函数信息
            skeleton_context: 骨架上下文
            
        Returns:
            翻译后的代码，或 None（如果失败）
        """
        # 1. 读取 Rust 签名
        rust_signature = ""
        placeholder_info = None
        
        # 调试日志：记录签名提取过程
        logger.debug(f"[{func_name}] 开始签名提取, C函数名={func_info.name}, 文件名={func_info.file_name}")
        
        skeleton_sig = self._extract_standalone_signature_from_skeleton(func_name, func_info)
        if skeleton_sig:
            rust_signature = skeleton_sig
            logger.debug(f"[{func_name}] 从骨架提取签名成功: {skeleton_sig[:80]}...")
        else:
            logger.debug(f"[{func_name}] 骨架签名提取失败，尝试签名文件")
            signature_file = self.signature_dir / f"{func_name}.txt"
            if signature_file.exists():
                with open(signature_file, 'r', encoding='utf-8', errors='ignore') as f:
                    rust_signature = f.read().strip()
                logger.debug(f"[{func_name}] 从签名文件读取: {rust_signature[:80] if rust_signature else '(空)'}")
                if "&self" in rust_signature:
                    rust_signature = rust_signature.replace("&self, ", "").replace("&self,", "").replace("(&self)", "()")
                    if not rust_signature.startswith("pub "):
                        rust_signature = "pub " + rust_signature
            else:
                logger.debug(f"[{func_name}] 签名文件不存在: {signature_file}")
        
        if not rust_signature:
            logger.debug(f"[{func_name}] 尝试从 C++ 代码推断签名")
            placeholder_info = self._infer_signature_from_cpp(func_info)
            if placeholder_info:
                if placeholder_info.get("skip"):
                    skip_reason = placeholder_info.get("reason", "未知原因")
                    logger.warning(f"[{func_name}] 跳过: {skip_reason}")
                    print(f"  [跳过] {skip_reason}")
                    func_info.skipped = True
                    setattr(func_info, "skip_reason", skip_reason)
                    return None
                rust_signature = placeholder_info.get("signature", "")
                logger.debug(f"[{func_name}] 推断签名: {rust_signature[:80] if rust_signature else '(空)'}")
            else:
                logger.debug(f"[{func_name}] 推断签名失败")
        
        if not rust_signature:
            # 详细记录失败原因
            logger.warning(f"[{func_name}] 翻译失败: 无法获取 Rust 签名")
            logger.warning(f"  C函数名: {func_info.name}")
            logger.warning(f"  文件名: {func_info.file_name}")
            logger.warning(f"  C代码片段: {func_info.c_code[:200] if func_info.c_code else '(无)'}")
            print(f"  [失败] 无法获取 Rust 签名 (C函数: {func_info.name})")
            func_info.skipped = True
            setattr(func_info, "skip_reason", "无法获取 Rust 签名")
            return None
        
        func_info.rust_signature = rust_signature
        
        # 2. 检查是否已有翻译结果
        translated_file = self.translated_dir / f"{func_name}.txt"
        if translated_file.exists():
            with open(translated_file, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        
        # 3. 翻译函数
        per_func_context = self._build_skeleton_context_for_function(func_info, skeleton_context)
        translated_code = self._translate_function(func_info, per_func_context)
        if translated_code:
            # Rule-based rewrite: safe globals -> accessor API
            try:
                translated_code = self._rewrite_safe_globals_in_code(translated_code)
            except Exception:
                pass
            self.stats["translated"] += 1
            # 注册已翻译函数（用于为后续函数提供调用上下文）
            self._register_translated_function(func_info, translated_code)
            # 保存翻译结果以便调试
            try:
                self.translated_dir.mkdir(parents=True, exist_ok=True)
                with open(translated_file, 'w', encoding='utf-8') as f:
                    f.write(translated_code)
            except Exception as e:
                logger.debug(f"[{func_name}] 保存翻译结果失败: {e}")
        else:
            # 翻译失败，记录详细信息
            logger.warning(f"[{func_name}] LLM 翻译返回空结果")
        return translated_code
    
    def _inject_and_compile_with_repair(
        self,
        func_name: str,
        func_info: FunctionInfo,
        translated_code: str,
        skeleton_context: str
    ) -> bool:
        """
        注入代码并编译，如果失败则进行修复
        
        Args:
            func_name: 函数名
            func_info: 函数信息
            translated_code: 已翻译的代码
            skeleton_context: 骨架上下文
            
        Returns:
            是否成功
        """
        # 1. 尝试注入并编译
        if self._try_inject_and_compile(func_name, func_info, translated_code, rollback_on_failure=False):
            return True
        
        if getattr(func_info, "injection_failed", False):
            artifact = self._save_manual_fix_artifact(
                func_name,
                func_info,
                translated_code,
                reason="injection_failed",
                error_msg="",
            )
            comment = self._build_manual_fix_comment(
                func_name,
                func_info,
                reason="injection_failed",
                artifact_path=artifact,
                error_msg="",
            )
            self._inject_failure_comment(func_name, func_info, comment)
            # 在函数定义下方保留 LLM 失败输出（注释形式；便于对比/人工复检）
            try:
                self._append_llm_failed_code_comment_below_function(
                    func_name=func_name,
                    func_info=func_info,
                    llm_code=translated_code,
                    artifact_path=artifact,
                    reason="injection_failed",
                )
            except Exception:
                pass

            # 可选：使用 C2Rust 兜底填充函数体（只在能通过 cargo check 时才保留）
            if self._enable_c2rust_fallback:
                try:
                    mod_info = self._ensure_c2rust_fallback_module(func_info=func_info)
                    if mod_info:
                        file_group, _mod_path = mod_info
                        idx = self._c2rust_fallback_function_index.get(file_group)
                        if idx is None or func_info.name in idx:
                            wrapper = self._build_c2rust_fallback_wrapper(func_info=func_info, file_group=file_group)
                            if wrapper and self._try_inject_and_compile(
                                func_name, func_info, wrapper, rollback_on_failure=True
                            ):
                                try:
                                    self.stats["c2rust_fallback"] += 1
                                except Exception:
                                    pass
                                return True
                except Exception as e:
                    logger.debug(f"[C2Rust fallback] {func_name}: parallel injection_failed path failed (ignored): {e}")
            return False

        # 1.5 按需类型/结构恢复（不消耗 LLM 修复次数）
        # 目标：先把缺失类型/常量/opaque struct 字段补齐，避免让 LLM 为“编译上下文缺失”改坏函数语义
        if self.type_recovery:
            try:
                max_rounds = int(os.environ.get("C2R_TYPE_RECOVERY_ROUNDS", "4"))
            except Exception:
                max_rounds = 4

            source_file_hint = None
            try:
                source_file_hint = self._locate_source_file(func_info)
            except Exception:
                source_file_hint = None

            for round_idx in range(1, max_rounds + 1):
                error_msg = self._get_compile_error()
                if not error_msg:
                    break

                changed = False
                try:
                    changed = self.type_recovery.apply_from_compile_error(
                        error_msg,
                        source_file=source_file_hint,
                        c_code=getattr(func_info, "c_code", "") or "",
                    )
                except Exception as e:
                    logger.debug(f"type_recovery 执行失败 (round {round_idx}): {e}")
                    changed = False

                if not changed:
                    break

                # types.rs 发生变化：刷新骨架上下文与 opaque_types（后续 LLM 会看到更准确的类型）
                try:
                    skeleton_context = self._extract_skeleton_context(include_types_rs_defs=not self._enable_types_slice)
                    self._skeleton_non_types_context = skeleton_context
                    if self._enable_types_slice:
                        self._reload_types_registry()
                except Exception as e:
                    logger.debug(f"刷新 skeleton_context 失败: {e}")

                # 重新编译
                if self._compile_project():
                    return True
        
        # 2. 修复循环
        error_history = []
        repair_attempts_history = []
        func_history_dir = self.repair_history_dir / func_name
        func_history_dir.mkdir(parents=True, exist_ok=True)

        types_slice_extra_symbols: Set[str] = set()
        syntax_error_warned = False
        
        for attempt in range(self.max_repair_attempts):
            print(f"  修复尝试 {attempt + 1}/{self.max_repair_attempts}...")
            
            error_msg = self._get_compile_error()

            # 额外提示：语法错误通常没有 error[E...]，容易导致“错误码提取不到”
            # 但依然可以走正常的增量修复（让修复器优先修语法）。
            try:
                if (not syntax_error_warned) and self._looks_like_rust_syntax_error(error_msg or ""):
                    syntax_error_warned = True
                    print("  ⚠️ 检测到可能的 Rust 语法错误（无 error[E...] 错误码），将优先按语法修复尝试。")
            except Exception:
                pass
            
            # 保存错误信息
            error_file = func_history_dir / f"attempt_{attempt + 1}_error.txt"
            _atomic_write_text(
                error_file,
                "".join(
                    [
                        f"函数: {func_name}\n",
                        f"尝试次数: {attempt + 1}/{self.max_repair_attempts}\n",
                        f"{'='*60}\n",
                        (error_msg if error_msg else "(无错误信息)"),
                    ]
                ),
            )

            # Scheme-B: deterministic C accessor shims for missing fields (E0609)
            try:
                if self._try_fix_missing_fields_with_c_shims(error_msg):
                    if self._compile_project():
                        logger.info(f"[{func_name}] C accessor shims fixed E0609 field errors")
                        self.stats["repaired"] += 1
                        return True
                    error_msg = self._get_compile_error()
            except Exception as e:
                logger.debug(f"[{func_name}] 字段 accessor shim 修复失败: {e}")

            # Deterministic internal symbol resolver for E0425 (missing in-scope function calls).
            # This is a common non-LLM bug: the function exists in another module but the call site is unqualified.
            try:
                if self._try_fix_missing_internal_symbols(error_msg):
                    if self._compile_project():
                        logger.info(f"[{func_name}] internal symbol resolver fixed E0425 missing callees")
                        self.stats["repaired"] += 1
                        return True
                    error_msg = self._get_compile_error()
            except Exception as e:
                logger.debug(f"[{func_name}] internal symbol resolver 修复失败: {e}")
            
            error_key = error_msg[:500] if error_msg else ""
            if error_key in error_history[-3:] if len(error_history) >= 3 else False:
                print(f"  检测到重复错误（历史记录已传递给LLM）...")
            error_history.append(error_key)

            repair_attempts_history.append({
                "attempt_num": attempt + 1,
                "error_msg": error_msg,
                "code_before": translated_code,
            })
            
            # 请求 LLM 修复
            # types.rs slice 扩展：根据 rustc 错误补齐缺失类型/常量符号
            try:
                types_slice_extra_symbols.update(self._extract_missing_symbols_for_types_slice(error_msg))
            except Exception:
                pass
            per_attempt_context = self._build_skeleton_context_for_function(
                func_info,
                skeleton_context,
                extra_types_symbols=types_slice_extra_symbols,
            )
            repaired_code = self._repair_function(
                func_info, translated_code, error_msg, per_attempt_context,
                attempt_num=attempt + 1,
                error_history=error_history[-3:],
                repair_attempts_history=repair_attempts_history
            )
            
            if repaired_code:
                repaired_code_file = func_history_dir / f"attempt_{attempt + 1}_repaired_code.rs"
                _atomic_write_text(repaired_code_file, repaired_code)
                
                if len(repair_attempts_history) > 0:
                    repair_attempts_history[-1]["code_after"] = repaired_code
            
            if repaired_code and self._try_inject_and_compile(func_name, func_info, repaired_code, rollback_on_failure=False):
                self.stats["repaired"] += 1
                return True
            
            if getattr(func_info, "injection_failed", False):
                break
            
            translated_code = repaired_code or translated_code
        
        # 修复失败：保留翻译工件 + 回退到占位符（不为编译通过而“删内容”）
        final_error_msg = ""
        try:
            final_error_msg = self._get_compile_error() or ""
        except Exception:
            final_error_msg = ""
        artifact = self._save_manual_fix_artifact(
            func_name,
            func_info,
            translated_code,
            reason=f"repair_failed_after_{self.max_repair_attempts}",
            error_msg=final_error_msg,
        )
        comment = self._build_manual_fix_comment(
            func_name,
            func_info,
            reason=f"repair_failed_after_{self.max_repair_attempts}",
            artifact_path=artifact,
            error_msg=final_error_msg,
        )

        # 关键改进：确保回退成功，如果普通回退失败则强制恢复整个文件
        rollback_success = self._rollback_function_safe(func_name, func_info)
        if not rollback_success:
            logger.error(f"回退函数 {func_name} 失败，尝试强制恢复整个文件")
            self._force_restore_skeleton_file(func_info)

        # 尽力在占位符函数上方落下注释，方便人工定位
        try:
            self._inject_failure_comment(func_name, func_info, comment)
            _atomic_write_text(func_history_dir / "failure_comment.txt", comment)
        except Exception as e:
            logger.debug(f"注入 manual-fix 注释失败（忽略）: {e}")

        # 在函数定义下方保留 LLM 失败输出（注释形式；便于对比/人工复检）
        try:
            self._append_llm_failed_code_comment_below_function(
                func_name=func_name,
                func_info=func_info,
                llm_code=translated_code,
                artifact_path=artifact,
                reason=f"repair_failed_after_{self.max_repair_attempts}",
            )
        except Exception:
            pass

        # 可选：使用 C2Rust 工具兜底填充函数体（只在能通过 cargo check 时才保留）
        if self._enable_c2rust_fallback:
            try:
                mod_info = self._ensure_c2rust_fallback_module(func_info=func_info)
                if mod_info:
                    file_group, _mod_path = mod_info
                    idx = self._c2rust_fallback_function_index.get(file_group)
                    if idx is None or func_info.name in idx:
                        wrapper = self._build_c2rust_fallback_wrapper(func_info=func_info, file_group=file_group)
                        if wrapper and self._try_inject_and_compile(
                            func_name, func_info, wrapper, rollback_on_failure=True
                        ):
                            try:
                                self.stats["c2rust_fallback"] += 1
                            except Exception:
                                pass
                            return True
            except Exception as e:
                logger.debug(f"[C2Rust fallback] {func_name}: repair_failed path failed (ignored): {e}")

        return False

    # ===================== RQ3.3: Oracle knowledge support (C0-C6) =====================
    def _oracle_project_dir(self) -> Path:
        return Path(self._oracle_knowledge_root) / str(self.project_name)

    def _maybe_load_oracle_index(self) -> None:
        """
        Build a best-effort index for oracle reference files:
        - by function_name (record["function_name"] or record["name"])
        - by func_key (record["func_key"]) if present
        """
        if self._oracle_index_loaded:
            return
        self._oracle_index_loaded = True

        proj_dir = self._oracle_project_dir()
        if not proj_dir.is_dir():
            return

        # Prefer manifest.json if present (smaller + explicit).
        candidates: List[Path] = []
        manifest = proj_dir / "manifest.json"
        if manifest.is_file():
            try:
                data = json.loads(manifest.read_text(encoding="utf-8", errors="ignore") or "{}")
                files = data.get("files") if isinstance(data, dict) else None
                if isinstance(files, list):
                    for fn in files:
                        p = proj_dir / str(fn)
                        if p.is_file() and p.suffix.lower() == ".json":
                            candidates.append(p)
            except Exception:
                candidates = []

        # Fall back to scanning the project dir.
        if not candidates:
            try:
                for p in sorted(proj_dir.glob("*.json")):
                    if p.name == "manifest.json":
                        continue
                    candidates.append(p)
            except Exception:
                candidates = []

        for p in candidates:
            try:
                rec = json.loads(p.read_text(encoding="utf-8", errors="ignore") or "{}")
            except Exception:
                continue
            if not isinstance(rec, dict):
                continue
            fn = (rec.get("function_name") or rec.get("name") or "").strip()
            if fn:
                self._oracle_funcname_to_path.setdefault(fn, p)
            fk = (rec.get("func_key") or "").strip()
            if fk:
                self._oracle_funckey_to_path.setdefault(fk, p)

    def _load_oracle_record(self, func_info: FunctionInfo) -> Optional[Dict[str, Any]]:
        """
        Load an oracle reference record for this function.

        Supported layouts under <oracle_root>/<project>/:
        - <file_name>_<index>.json
        - <func_name>.json
        - manifest.json + arbitrary file list (each record contains function_name)
        """
        proj_dir = self._oracle_project_dir()
        if not proj_dir.is_dir():
            return None

        # 1) direct match by func_key (file_name_index)
        direct_key = f"{func_info.file_name}_{func_info.index}"
        for candidate in (
            proj_dir / f"{direct_key}.json",
            proj_dir / f"{func_info.name}.json",
        ):
            if candidate.is_file():
                try:
                    rec = json.loads(candidate.read_text(encoding="utf-8", errors="ignore") or "{}")
                    return rec if isinstance(rec, dict) else None
                except Exception:
                    return None

        # 2) index-based match (manifest or scan)
        self._maybe_load_oracle_index()
        if direct_key in self._oracle_funckey_to_path:
            p = self._oracle_funckey_to_path[direct_key]
            try:
                rec = json.loads(p.read_text(encoding="utf-8", errors="ignore") or "{}")
                return rec if isinstance(rec, dict) else None
            except Exception:
                return None

        if func_info.name in self._oracle_funcname_to_path:
            p = self._oracle_funcname_to_path[func_info.name]
            try:
                rec = json.loads(p.read_text(encoding="utf-8", errors="ignore") or "{}")
                return rec if isinstance(rec, dict) else None
            except Exception:
                return None

        return None

    @staticmethod
    def _clean_json_text_for_oracle(s: str) -> str:
        s = (s or "").strip()
        if s.startswith("```json"):
            s = s[len("```json") :]
        if s.startswith("```"):
            s = s[len("```") :]
        if s.endswith("```"):
            s = s[: -len("```")]
        return s.strip()

    def _extract_oracle_knowledge_via_llm(self, *, c_code: str, rust_code: str, task_id: str) -> List[Dict[str, Any]]:
        """
        Reuse the existing knowledge extractor prompt (vllm_parallel_knowledge_extractor.py)
        to extract API_Mapping / Partial knowledge from (C, oracle Rust).
        """
        from generate.generation import generation
        from vllm_parallel_knowledge_extractor import PROMPT_TEMPLATE, convert_llm_record_to_extracted_knowledge

        prompt = PROMPT_TEMPLATE.replace("{c_code}", c_code or "").replace("{rust_code}", rust_code or "")
        content = generation([{"role": "user", "content": prompt}])

        cleaned = self._clean_json_text_for_oracle(content)
        obj = None
        try:
            obj = json.loads(cleaned)
        except Exception:
            # Best-effort salvage: take the outermost JSON object.
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start >= 0 and end > start:
                try:
                    obj = json.loads(cleaned[start : end + 1])
                except Exception:
                    obj = None

        if not isinstance(obj, dict):
            logger.debug(f"[Oracle] knowledge extraction JSON parse failed: task={task_id}")
            return []

        # Convert to the KB schema used by downstream prompt injection.
        extracted = convert_llm_record_to_extracted_knowledge(obj, source="oracle_vllm")
        # Keep only the two channels used in RQ3.3 (ignore Full).
        out: List[Dict[str, Any]] = []
        for k in extracted:
            if not isinstance(k, dict):
                continue
            kt = (k.get("knowledge_type") or "").strip()
            if kt in ("API_Mapping", "Partial"):
                out.append(k)
        return out

    def _load_oracle_extracted_knowledge(self, func_info: FunctionInfo, *, topk: int) -> List[Dict[str, Any]]:
        """
        Load per-function oracle extracted knowledge (API_Mapping / Partial).

        Cache strategy (run-local, safe):
        - <workspace>/rag/oracle_extracted_cache/<project>/<file_name>_<index>.json
        """
        func_key = f"{func_info.file_name}_{func_info.index}"
        cache_dir = self.workspace_root / "rag" / "oracle_extracted_cache" / str(self.project_name)
        cache_path = cache_dir / f"{func_key}.json"

        # 1) cache hit
        if cache_path.is_file():
            try:
                data = json.loads(cache_path.read_text(encoding="utf-8", errors="ignore") or "[]")
                if isinstance(data, list):
                    return data
            except Exception:
                pass

        # 2) oracle record contains extracted_knowledge already
        rec = self._load_oracle_record(func_info)
        if not rec:
            return []
        ek = rec.get("extracted_knowledge")
        if isinstance(ek, list) and ek:
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps(ek, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                pass
            return ek

        # 3) optionally run extractor to derive knowledge from (C, oracle Rust)
        if not self._oracle_auto_extract:
            return []

        rust_code = (rec.get("rust_code") or "").strip()
        if not rust_code:
            return []
        c_code = (func_info.c_code or "").strip()
        if not c_code:
            c_code = (rec.get("c_code") or "").strip()
        if not c_code:
            return []

        try:
            task_id = f"{self.project_name}:{func_key}"
            extracted = self._extract_oracle_knowledge_via_llm(c_code=c_code, rust_code=rust_code, task_id=task_id)
            if extracted:
                # Best-effort cap: keep cache reasonably small even if the extractor outputs many items.
                if topk > 0:
                    # keep a bit more than topk to allow per-channel capping later
                    extracted = extracted[: max(topk * 2, topk)]
                cache_dir.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps(extracted, ensure_ascii=False, indent=2), encoding="utf-8")
            return extracted
        except Exception as e:
            logger.debug(f"[Oracle] knowledge auto-extract failed: {func_key}: {e}")
            return []
    
    def _translate_function(self, func_info: FunctionInfo, skeleton_context: str) -> Optional[str]:
        """
        两步走翻译策略 (参考 SACTOR)
        
        Phase 3.1: C to Unidiomatic Rust (语义等价优先)
          - 允许大量使用 unsafe、裸指针
          - 不需要处理复杂的生命周期和借用规则
          - 确保逻辑完全等价
        
        Phase 3.2: Unidiomatic to Idiomatic Rust (安全性重构) [可选]
          - 将 unsafe 块移除
          - 将裸指针转换为 Reference 或 Option/Result
          - 只关注语法重构，不改变业务逻辑
        """
        from generate.generation import generation
        
        print(f"  Phase 3.1 翻译...", end=" ", flush=True)
        
        # 附加多级上下文（语义切片 / 全局声明 / 用法示例）
        context_prefix, context_meta = self._build_context_prefix(func_info)
        context_summary_path = None
        if context_meta and context_meta.get("context_blocks"):
            context_summary_path = self._save_context_summary(func_info, context_meta)
        if context_prefix:
            if skeleton_context:
                skeleton_context = f"{context_prefix}\n\n{skeleton_context}"
            else:
                skeleton_context = context_prefix

        # 检查并截断骨架上下文（防止超过模型最大上下文窗口）
        if len(skeleton_context) > MAX_SKELETON_CONTEXT_CHARS:
            skeleton_context = smart_truncate_context(skeleton_context, MAX_SKELETON_CONTEXT_CHARS)

        # 检查签名是否是 trait 方法（带 &self）
        is_trait_method = "&self" in func_info.rust_signature

        # 构建不透明类型警告信息（通用，动态检测）
        opaque_type_warning = ""
        accessor_shim_hints = ""

        if self.opaque_types:
            opaque_list = ", ".join(sorted(self.opaque_types))

            # 主动生成 accessor shims（如果 C 代码访问了 opaque 类型的字段）
            generated_shims = self._generate_accessor_shims_proactively(func_info, self.opaque_types)
            if generated_shims:
                accessor_shim_hints = self._build_accessor_shim_hints(generated_shims)
                logger.info(f"[AccessorShim] 函数 {func_info.name}: 生成 {len(generated_shims)} 个 accessor shims")
                for type_name, field_name, func_name in generated_shims:
                    logger.debug(f"  - {func_name}: {type_name}.{field_name}")

            # 构建 opaque 类型警告
            opaque_type_warning = f"""
## OPAQUE TYPES DETECTED
The following types are defined as opaque (e.g., `type X = c_void`):
  {opaque_list}

CRITICAL: These are OPAQUE types - you CANNOT access their struct fields directly!
"""
            if accessor_shim_hints:
                # 如果有 accessor shims，提示使用它们
                opaque_type_warning += """
**SOLUTION**: Use the accessor shim functions provided below to access fields.
"""
            else:
                # 没有 accessor shims 时，提示返回默认值
                opaque_type_warning += """
- If C/C++ code accesses fields like `.child`, `.next`, `.string` on these types, return a safe default instead
- For pointer return types: return `std::ptr::null_mut()`
- For integer return types: return `0`
- For boolean return types: return `0` (cJSON_bool)
- For void functions: just return
"""

        # Phase 3.1: 生成语义正确但不地道的 Rust 代码（允许 unsafe）
        # 精简版提示词：聚焦最常见错误，减少 token 消耗
        system_prompt = f"""You are a C/C++ to Rust translator. Generate code that COMPILES. Unsafe is OK.

OUTPUT FORMAT (STRICT):
- Output ONLY raw Rust code
- DO NOT use ``` fences
- DO NOT add explanations, markdown, or extra text

## ⚠️ CRITICAL: AVOID THESE MISTAKES ⚠️
1. **Option<fn> MUST unwrap**: `if let Some(f) = cb {{ unsafe {{ f(args) }} }}`
2. **ALL ptr ops need unsafe**: `unsafe {{ *ptr = val; (*ptr).field; }}`
3. **OPAQUE types (c_void) have NO fields** - return defaults instead
4. **Pointer cast**: `ptr as *mut T` (NOT `ptr.cast_mut()` - doesn't exist!)
5. **DO NOT invent externs for macros/inline/external APIs**: external symbols are declared in `compat.rs` (bindgen allowlist); macro/static inline are not stable link symbols.
6. **DO NOT emit C macro names / header-only inline helper names**: input is a preprocessed `.i` (macros expanded). If you see a macro-like call, output the expanded expression or implement a local Rust helper (NOT an extern).

## ❌ NON-EXISTENT / WRONG SYNTAX - NEVER USE ❌
- `ptr.cast_mut()` → Use `ptr as *mut T`
- `ptr.cast_const()` → Use `ptr as *const T`
- `(void)var;` → This is C! Use `let _ = var;` in Rust
- `NULL` → Use `std::ptr::null()` or `std::ptr::null_mut()`

## RULES
1. Match TARGET SIGNATURE exactly (pub fn / impl method / Drop)
2. Use only skeleton types and existing `compat.rs` extern decls; DO NOT add ad-hoc `extern "C"` blocks
3. OPAQUE types: can't access fields, use ptr arithmetic or defaults
4. Wrap ALL ptr derefs in `unsafe {{ }}`
5. Function ptrs: `Option<fn>` - must unwrap before call

## NAMESPACE CONTRACT (CRITICAL)
- `crate::types` contains types + constants only (bindgen truth). Do NOT put functions there.
- External C APIs are declared in `crate::compat` and are already imported by the skeleton (`use crate::compat::*;`):
  call them as `Foo(...)` or `crate::compat::Foo(...)` (NOT `crate::types::Foo`).
- Global variables are in `crate::globals` and are imported by the skeleton (`use crate::globals::*;`):
  use `NAME` or `crate::globals::NAME` (NOT `crate::types::NAME`).
- Internal translated functions live in per-source modules (e.g. `crate::src_xxx::Func`):
  use the provided \"Internal Callees\" mapping when calling them.
- Use `::core::ffi::c_void` for void pointers (NOT `crate::types::c_void`).

## TYPE CHECKING (MUST COMPILE)
- Every call MUST satisfy the provided Rust signatures (Target signature + \"Called Rust APIs\").
- If types mismatch, add explicit casts/conversions (e.g. `as i32`, `as u32`, `as usize`, pointer casts).
- Treat `size_t/ssize_t` as `usize/isize` in Rust; cast before arithmetic/comparisons to avoid E0308.

## SIZE_T / USIZE CONVERSION (CRITICAL - E0308 FIX)
- `crate::types::size_t` is defined as `u64` by bindgen (on 64-bit systems)
- But `libc` functions (malloc, realloc, memcpy, etc.) expect `usize`
- ALWAYS cast when calling libc: `libc::malloc(size as usize)`
- ALWAYS cast strlen result: `(libc::strlen(ptr) + 1) as usize` for malloc
- When assigning to size_t field: `(value as u64)` or `(value as crate::types::size_t)`

## RAW POINTER RULES (CRITICAL - E0599 FIX)
- Raw pointers (`*mut T`, `*const T`) have NO methods like `.len()`, `.is_empty()`, etc.
- For C string length: use `libc::strlen(ptr)` (NOT `ptr.len()`)
- For array/slice length: you must track length separately or use `std::slice::from_raw_parts(ptr, len)`
- WRONG: `ptr.len()` on `*mut i8` → ERROR: no method named `len`
- CORRECT: `unsafe {{ libc::strlen(ptr) }}`

## INTERNAL FUNCTION CALLS (CRITICAL - E0425 FIX)
- When calling functions defined in OTHER source modules (e.g., `src_tree.rs`), you MUST use the full path
- Check the "Internal Callees" section in the prompt for the correct module path
- WRONG: `ZopfliCalculateEntropy(...)` → ERROR: E0425 cannot find function
- CORRECT: `crate::src_tree::ZopfliCalculateEntropy(...)`

## OUTPUT SIZE
- Keep the function body compact. Do NOT generate hundreds of unused temporary variables; every `let` must be used.

## QUICK REFERENCE
- sizeof: `std::mem::size_of::<T>()`
- malloc: `libc::malloc(n) as *mut T`
- memcpy: `std::ptr::copy_nonoverlapping(src, dst, n)`
- memset: `std::ptr::write_bytes(ptr, val, n)`
- NULL check: `ptr.is_null()`
- C enums are represented as integers in `crate::types` (bindgen constified); use integer comparisons / constants.
- Ignore value: `let _ = expr;` (NOT `(void)expr;`)

GOAL: Make it compile. Idiomatic refactoring happens in Phase 3.2."""

        # 如果签名是 trait 方法或析构函数，需要特殊处理
        target_signature = func_info.rust_signature
        
        # 检查是否是 Drop trait 实现（析构函数）
        is_drop_impl = target_signature.startswith("impl Drop for")
        
        if is_drop_impl:
            # 析构函数保持原样，不需要转换
            pass
        elif is_trait_method:
            # 移除 &self 参数
            target_signature = target_signature.replace("&self, ", "").replace("&self,", "").replace("(&self)", "()")
            # 添加 pub fn 前缀
            if not target_signature.startswith("pub "):
                target_signature = "pub " + target_signature

        # RQ3.3 / RQ3.2:
        # - default: inject predicted knowledge from RAG (API_Mapping + Partial)
        # - ablation: independently switch each channel to none/predicted/oracle
        rag_knowledge = ""
        api_mode = getattr(self, "_rq3_api_mapping_mode", "predicted")
        partial_mode = getattr(self, "_rq3_partial_idiom_mode", "predicted")

        # Fast path: keep legacy behavior 100% unchanged unless user explicitly enables ablation modes.
        if api_mode == "predicted" and partial_mode == "predicted":
            if self._enable_rag_context:
                # RAG 文件路径：rag/reranked_results/{project}/{file_name}_{index}.txt
                rag_file = self.rag_reranked_dir / f"{func_info.file_name}_{func_info.index}.txt"
                # 如果文件不存在，尝试不带index的格式（向后兼容）
                if not rag_file.exists():
                    # 兼容历史输出：有些版本可能写成 {file_name}.txt 或 {function_name}.txt
                    legacy_candidates = [
                        self.rag_reranked_dir / f"{func_info.file_name}.txt",
                        self.rag_reranked_dir / f"{func_info.name}.txt",
                    ]
                    for candidate in legacy_candidates:
                        if candidate.exists():
                            rag_file = candidate
                            break
                # Fallback: if semantic reranker (Jina) output is missing (no GPU / skipped),
                # use the BM25 (elastic_search_results) output which still contains
                # `[EXTRACTED_KNOWLEDGE] ... [/EXTRACTED_KNOWLEDGE]` blocks.
                if not rag_file.exists():
                    elastic_dir = self.workspace_root / "rag" / "elastic_search_results" / str(self.project_name)
                    elastic_candidates = [
                        elastic_dir / f"{func_info.file_name}_{func_info.index}.txt",
                        elastic_dir / f"{func_info.file_name}.txt",
                        elastic_dir / f"{func_info.name}.txt",
                    ]
                    for candidate in elastic_candidates:
                        if candidate.exists():
                            rag_file = candidate
                            break
                if rag_file.exists():
                    try:
                        with open(rag_file, 'r', encoding='utf-8', errors='ignore') as f:
                            rag_content = f.read()
                        
                        # 解析 RAG 文件（格式：C_Code, Function, Extracted_Knowledge, Unixcoder Score, 分隔符）
                        # 每个块对应一个检索到的候选代码对（按 reranker 排序）
                        rag_blocks = [b for b in rag_content.split("-" * 50) if b.strip()]
                        
                        # 配置：
                        # - 默认保持旧行为：API_Mapping=10，Partial=2
                        # - 若设置 C2R_RAG_TOPK=k，则限制“注入的知识条目数”为 k（而不是前 k 个代码对）
                        #   用于 RQ3.2 消融/敏感性实验（top-k sensitivity）
                        rag_topk = None
                        _rag_topk_raw = os.environ.get("C2R_RAG_TOPK", "").strip()
                        if _rag_topk_raw:
                            try:
                                rag_topk = int(_rag_topk_raw)
                            except Exception:
                                rag_topk = None

                        if rag_topk is not None and rag_topk > 0:
                            TOP_K_API = rag_topk
                            TOP_K_PARTIAL = rag_topk
                        else:
                            # NOTE: 这里的 TOP_K_* 表示“最多注入多少条知识”，而不是“看前多少个代码对”。
                            TOP_K_API = 10  # 最多注入的 API_Mapping 条目数
                            TOP_K_PARTIAL = 2  # 最多注入的 Partial/Idiom 条目数
                        
                        api_mappings = []
                        partial_patterns = []
                        
                        # 遍历各块，按需提取知识
                        for block_idx, block in enumerate(rag_blocks):
                            # 已达到两类知识的上限，就无需继续扫描更多代码对（保持 prompt 尺寸稳定）
                            if len(api_mappings) >= TOP_K_API and len(partial_patterns) >= TOP_K_PARTIAL:
                                break
                            
                            try:
                                # 提取 Extracted_Knowledge 部分（两种格式）：
                                # - Jina reranker: `Extracted_Knowledge: <json> ... Unixcoder Score:`
                                # - BM25 (elastic_search.py): `[EXTRACTED_KNOWLEDGE] <json> [/EXTRACTED_KNOWLEDGE]`
                                knowledge_str = ""
                                if "Extracted_Knowledge:" in block:
                                    knowledge_str = block.split("Extracted_Knowledge:")[1].split("Unixcoder Score:")[0].strip()
                                elif "[EXTRACTED_KNOWLEDGE]" in block:
                                    knowledge_str = block.split("[EXTRACTED_KNOWLEDGE]")[1].split("[/EXTRACTED_KNOWLEDGE]")[0].strip()
                                else:
                                    continue
                                if not knowledge_str:
                                    continue
                                
                                knowledge_list = json.loads(knowledge_str)
                                # 确保是列表
                                if isinstance(knowledge_list, dict):
                                    knowledge_list = [knowledge_list]
                                elif not isinstance(knowledge_list, list):
                                    continue
                                
                                # 从每个块中提取所有相关知识
                                for k in knowledge_list:
                                    k_type = k.get("knowledge_type", "Unknown")
                                    
                                    # API_Mapping: 限制“知识条目数”（而不是代码对数）
                                    if k_type == "API_Mapping" and len(api_mappings) < TOP_K_API:
                                        c_api = k.get("c_api", "").strip()
                                        rust_api = k.get("rust_api", "").strip()
                                        desc = k.get("description", "").strip()
                                        if c_api and rust_api:
                                            api_mappings.append((c_api, rust_api, desc))
                                    
                                    # Partial/Idiom: 同样限制“知识条目数”
                                    elif k_type == "Partial" and len(partial_patterns) < TOP_K_PARTIAL:
                                        c_frag = k.get("c_fragment", "").strip()
                                        rust_frag = k.get("rust_fragment", "").strip()
                                        desc = k.get("description", "").strip()
                                        if c_frag and rust_frag:
                                            partial_patterns.append((c_frag, rust_frag, desc))
                                    
                                    # 忽略 Full 和其他类型
                            
                            except (json.JSONDecodeError, IndexError, ValueError) as e:
                                logger.debug(f"解析知识失败 (块 {block_idx}): {e}")
                                continue
                        
                        # 格式化提取的知识
                        knowledge_hints = []
                        
                        # 输出 API 映射（按“知识条目数”限制）
                        if api_mappings:
                            knowledge_hints.append(f"## API Mappings (top {TOP_K_API} items from RAG)")
                            for c_api, rust_api, desc in api_mappings:
                                if desc:
                                    knowledge_hints.append(f"- `{c_api}` → `{rust_api}` ({desc})")
                                else:
                                    knowledge_hints.append(f"- `{c_api}` → `{rust_api}`")
                        
                        # 输出代码模式（按“知识条目数”限制）
                        if partial_patterns:
                            knowledge_hints.append(f"\n## Code Patterns (top {TOP_K_PARTIAL} items from RAG)")
                            for c_frag, rust_frag, desc in partial_patterns:
                                # 根据片段长度决定显示方式
                                if len(c_frag) <= 300 and len(rust_frag) <= 300:
                                    # 短片段：完整显示
                                    if desc:
                                        knowledge_hints.append(f"- C: `{c_frag}` → Rust: `{rust_frag}` ({desc})")
                                    else:
                                        knowledge_hints.append(f"- C: `{c_frag}` → Rust: `{rust_frag}`")
                                else:
                                    # 长片段：显示更多内容（保留开头150+结尾50）
                                    c_preview = c_frag[:150] + "..." + c_frag[-50:] if len(c_frag) > 200 else c_frag
                                    rust_preview = rust_frag[:150] + "..." + rust_frag[-50:] if len(rust_frag) > 200 else rust_frag
                                    if desc:
                                        knowledge_hints.append(f"- C: `{c_preview}` → Rust: `{rust_preview}` ({desc})")
                                    else:
                                        knowledge_hints.append(f"- C: `{c_preview}` → Rust: `{rust_preview}`")

                        if knowledge_hints:
                            rag_knowledge = "\n\n## Extracted Translation Knowledge (from RAG)\n" + "\n".join(knowledge_hints)
                            # C2R: 记录 RAG 知识提取统计
                            logger.info(f"[RAG统计] 函数: {func_info.name}, API映射: {len(api_mappings)}, 代码模式: {len(partial_patterns)}, 总长度: {len(rag_knowledge)} 字符")
                    except Exception as e:
                        logger.debug(f"读取 RAG 文件失败: {e}")
                else:
                    # RAG 文件不存在时的警告（仅在 debug 模式下记录，避免日志过多）
                    logger.debug(
                        "RAG 文件不存在，跳过 RAG 增强: %s (尝试路径: %s / %s / %s)",
                        rag_file,
                        self.rag_reranked_dir / f"{func_info.file_name}_{func_info.index}.txt",
                        self.rag_reranked_dir / f"{func_info.file_name}.txt",
                        self.rag_reranked_dir / f"{func_info.name}.txt",
                    )
            else:
                logger.debug("RAG prompt injection disabled (C2R_USE_RAG_CONTEXT=false); skipping.")
        else:
            # Controlled ablation path (C0-C6): per-channel = none/predicted/oracle.
            # Top-k semantics: cap by *knowledge items*, not by code-pair blocks.
            rag_topk = None
            _rag_topk_raw = os.environ.get("C2R_RAG_TOPK", "").strip()
            if _rag_topk_raw:
                try:
                    rag_topk = int(_rag_topk_raw)
                except Exception:
                    rag_topk = None

            if rag_topk is not None and rag_topk > 0:
                TOP_K_API = rag_topk
                TOP_K_PARTIAL = rag_topk
            else:
                TOP_K_API = 10
                TOP_K_PARTIAL = 2

            api_mappings: List[Tuple[str, str, str]] = []
            partial_patterns: List[Tuple[str, str, str]] = []

            need_pred_api = (api_mode == "predicted") and self._enable_rag_context
            need_pred_partial = (partial_mode == "predicted") and self._enable_rag_context

            # -------- predicted (RAG) --------
            rag_file = self.rag_reranked_dir / f"{func_info.file_name}_{func_info.index}.txt"
            if (need_pred_api or need_pred_partial):
                # legacy filename fallbacks (backward compatible)
                if not rag_file.exists():
                    for candidate in (
                        self.rag_reranked_dir / f"{func_info.file_name}.txt",
                        self.rag_reranked_dir / f"{func_info.name}.txt",
                    ):
                        if candidate.exists():
                            rag_file = candidate
                            break
                # Fallback to BM25 outputs when reranked results are missing.
                if not rag_file.exists():
                    elastic_dir = self.workspace_root / "rag" / "elastic_search_results" / str(self.project_name)
                    for candidate in (
                        elastic_dir / f"{func_info.file_name}_{func_info.index}.txt",
                        elastic_dir / f"{func_info.file_name}.txt",
                        elastic_dir / f"{func_info.name}.txt",
                    ):
                        if candidate.exists():
                            rag_file = candidate
                            break

                if rag_file.exists():
                    try:
                        rag_content = rag_file.read_text(encoding="utf-8", errors="ignore")
                        rag_blocks = [b for b in rag_content.split("-" * 50) if b.strip()]

                        for block_idx, block in enumerate(rag_blocks):
                            done_api = (not need_pred_api) or (len(api_mappings) >= TOP_K_API)
                            done_partial = (not need_pred_partial) or (len(partial_patterns) >= TOP_K_PARTIAL)
                            if done_api and done_partial:
                                break
                            try:
                                knowledge_str = ""
                                if "Extracted_Knowledge:" in block:
                                    knowledge_str = block.split("Extracted_Knowledge:")[1].split("Unixcoder Score:")[0].strip()
                                elif "[EXTRACTED_KNOWLEDGE]" in block:
                                    knowledge_str = block.split("[EXTRACTED_KNOWLEDGE]")[1].split("[/EXTRACTED_KNOWLEDGE]")[0].strip()
                                else:
                                    continue
                                if not knowledge_str:
                                    continue
                                knowledge_list = json.loads(knowledge_str)
                                if isinstance(knowledge_list, dict):
                                    knowledge_list = [knowledge_list]
                                if not isinstance(knowledge_list, list):
                                    continue
                            except Exception as e:
                                logger.debug(f"解析知识失败 (块 {block_idx}): {e}")
                                continue

                            for k in knowledge_list:
                                k_type = k.get("knowledge_type", "Unknown")
                                if k_type == "API_Mapping" and need_pred_api and len(api_mappings) < TOP_K_API:
                                    c_api = (k.get("c_api") or "").strip()
                                    rust_api = (k.get("rust_api") or "").strip()
                                    desc = (k.get("description") or "").strip()
                                    if c_api and rust_api:
                                        api_mappings.append((c_api, rust_api, desc))
                                elif k_type == "Partial" and need_pred_partial and len(partial_patterns) < TOP_K_PARTIAL:
                                    c_frag = (k.get("c_fragment") or "").strip()
                                    rust_frag = (k.get("rust_fragment") or "").strip()
                                    desc = (k.get("description") or "").strip()
                                    if c_frag and rust_frag:
                                        partial_patterns.append((c_frag, rust_frag, desc))
                    except Exception as e:
                        logger.debug(f"读取 RAG 文件失败: {e}")
                else:
                    logger.debug(
                        "RAG 文件不存在，跳过 RAG 增强: %s (尝试路径: %s / %s / %s)",
                        rag_file,
                        self.rag_reranked_dir / f"{func_info.file_name}_{func_info.index}.txt",
                        self.rag_reranked_dir / f"{func_info.file_name}.txt",
                        self.rag_reranked_dir / f"{func_info.name}.txt",
                    )
            elif (api_mode == "predicted" or partial_mode == "predicted") and (not self._enable_rag_context):
                logger.debug("[RQ3.3] predicted channel requested but C2R_USE_RAG_CONTEXT=false; skipping predicted knowledge.")

            # -------- oracle --------
            need_oracle_api = (api_mode == "oracle")
            need_oracle_partial = (partial_mode == "oracle")
            if need_oracle_api or need_oracle_partial:
                oracle_knowledge = self._load_oracle_extracted_knowledge(func_info, topk=max(TOP_K_API, TOP_K_PARTIAL))
                if oracle_knowledge:
                    if need_oracle_api:
                        for k in oracle_knowledge:
                            if len(api_mappings) >= TOP_K_API:
                                break
                            if not isinstance(k, dict):
                                continue
                            if (k.get("knowledge_type") or "") != "API_Mapping":
                                continue
                            c_api = (k.get("c_api") or "").strip()
                            rust_api = (k.get("rust_api") or "").strip()
                            desc = (k.get("description") or "").strip()
                            if c_api and rust_api:
                                api_mappings.append((c_api, rust_api, desc))
                    if need_oracle_partial:
                        for k in oracle_knowledge:
                            if len(partial_patterns) >= TOP_K_PARTIAL:
                                break
                            if not isinstance(k, dict):
                                continue
                            if (k.get("knowledge_type") or "") != "Partial":
                                continue
                            c_frag = (k.get("c_fragment") or "").strip()
                            rust_frag = (k.get("rust_fragment") or "").strip()
                            desc = (k.get("description") or "").strip()
                            if c_frag and rust_frag:
                                partial_patterns.append((c_frag, rust_frag, desc))
                else:
                    logger.debug(f"[Oracle] 未找到/未生成可用知识: {func_info.name} ({func_info.file_name}_{func_info.index})")

            # -------- format prompt injection --------
            knowledge_hints: List[str] = []

            api_src = "RAG" if api_mode == "predicted" else ("ORACLE" if api_mode == "oracle" else "")
            partial_src = "RAG" if partial_mode == "predicted" else ("ORACLE" if partial_mode == "oracle" else "")

            if api_mappings:
                knowledge_hints.append(f"## API Mappings (top {TOP_K_API} items from {api_src})")
                for c_api, rust_api, desc in api_mappings:
                    if desc:
                        knowledge_hints.append(f"- `{c_api}` → `{rust_api}` ({desc})")
                    else:
                        knowledge_hints.append(f"- `{c_api}` → `{rust_api}`")

            if partial_patterns:
                knowledge_hints.append(f"\n## Code Patterns (top {TOP_K_PARTIAL} items from {partial_src})")
                for c_frag, rust_frag, desc in partial_patterns:
                    if len(c_frag) <= 300 and len(rust_frag) <= 300:
                        if desc:
                            knowledge_hints.append(f"- C: `{c_frag}` → Rust: `{rust_frag}` ({desc})")
                        else:
                            knowledge_hints.append(f"- C: `{c_frag}` → Rust: `{rust_frag}`")
                    else:
                        c_preview = c_frag[:150] + "..." + c_frag[-50:] if len(c_frag) > 200 else c_frag
                        rust_preview = rust_frag[:150] + "..." + rust_frag[-50:] if len(rust_frag) > 200 else rust_frag
                        if desc:
                            knowledge_hints.append(f"- C: `{c_preview}` → Rust: `{rust_preview}` ({desc})")
                        else:
                            knowledge_hints.append(f"- C: `{c_preview}` → Rust: `{rust_preview}`")

            if knowledge_hints:
                sources: List[str] = []
                if api_mappings and api_src and api_src not in sources:
                    sources.append(api_src)
                if partial_patterns and partial_src and partial_src not in sources:
                    sources.append(partial_src)
                # Keep legacy wording for single-source RAG (user might diff prompts for regressions).
                if sources == ["RAG"]:
                    header = "from RAG"
                elif sources == ["ORACLE"]:
                    header = "from ORACLE"
                else:
                    header = "from " + "+".join(sources)

                rag_knowledge = "\n\n## Extracted Translation Knowledge (" + header + ")\n" + "\n".join(knowledge_hints)
                logger.info(
                    f"[Knowledge统计] 函数: {func_info.name}, api_mode={api_mode}, partial_mode={partial_mode}, "
                    f"API映射: {len(api_mappings)}, 代码模式: {len(partial_patterns)}, 总长度: {len(rag_knowledge)} 字符"
                )

        # 检查是否是析构函数
        is_drop_impl = target_signature.startswith("impl Drop for")
        
        # 构建输出要求（泛化版本）
        if is_drop_impl:
            output_requirements = f"""## Output Requirements
1. Output EXACTLY ONE Rust item: the `fn drop(&mut self) {{ ... }}` method definition
2. DO NOT output the enclosing `impl Drop for ... {{ ... }}` block (it already exists in the skeleton)
3. The method signature MUST match the target signature’s `drop` method
4. Keep output short: no long comments, no extra items, no markdown fences/explanations"""
        elif "&self" in target_signature or "&mut self" in target_signature:
            # impl 方法
            output_requirements = f"""## Output Requirements
1. Output EXACTLY ONE Rust item: the target `fn ... {{ ... }}` method definition (no enclosing `impl` block)
2. The method signature MUST match: `{target_signature}`
3. Do NOT output any extra items (no `impl`, `mod`, `extern` blocks at top-level, helper fns, or `use` statements)
4. Keep output short: no long comments, no markdown fences/explanations"""
        elif "fn new" in target_signature.lower() or "-> Self" in target_signature:
            # 构造函数
            output_requirements = f"""## Output Requirements
1. Generate the COMPLETE constructor function
2. The signature MUST match: `{target_signature}`
3. Initialize all struct fields appropriately
4. If some fields cannot be initialized, use Default::default() or safe defaults
5. Do NOT output any extra items (no helper fns/types/impl/mod/use)
6. Keep output short: no long comments, no markdown fences/explanations
7. Output ONLY the target constructor function definition"""
        else:
            # 普通独立函数
            output_requirements = f"""## Output Requirements
1. Output EXACTLY ONE Rust item: the complete target function definition
2. The function signature MUST match: `{target_signature}`
3. Do NOT output any extra items (no helper fns/types/impl/mod/use)
4. Keep output short: no long comments, no markdown fences/explanations"""

        # C2R FIX: 移除过于宽泛的 type_info_hint，避免在不必要时分散 LLM 注意力
        # 关键的类型处理指导已经在 system_prompt 的 CRITICAL RULES 中提供
        type_info_hint = ""

        # 从依赖文件提取“被调函数签名”（C-level constness），减少调用点 *mut/*const 错误
        callee_signature_hints = self._build_dependency_signature_hints(func_info)
        inline_helper_hints = self._build_preprocessed_inline_helper_hints(func_info)
        # Rust-level 被调函数签名（来自 signature_matches），进一步约束 *mut/*const/unsafe/返回类型
        callee_rust_signature_hints = self._build_called_rust_signature_hints(func_info)
        internal_callee_paths_hints = self._build_internal_callee_module_path_hints(func_info)
        typed_constants_hints = self._build_typed_constants_hints(func_info)
        c_field_access_hints = self._build_c_field_access_hints(func_info)
        try:
            c_pointer_contract_hints = self._build_c_pointer_contract_hints(func_info)
        except Exception as e:
            logger.debug(f"C pointer contract hints failed: {e}")
            c_pointer_contract_hints = ""

        user_prompt = f"""Translate the following C/C++ function to Rust.
{opaque_type_warning}{accessor_shim_hints}
## Target Rust Function Signature (MUST use this exact signature)
```rust
{target_signature}
```

## C/C++ Source Code to Translate
NOTE: The source below may be a slice from the build’s preprocessed `.i` TU (macros expanded, inline bodies visible).
```cpp
{func_info.c_code}
```
{callee_signature_hints}
{inline_helper_hints}
{callee_rust_signature_hints}
{internal_callee_paths_hints}
{typed_constants_hints}
{c_field_access_hints}
{c_pointer_contract_hints}

## Skeleton Context (available types and functions)
```rust
{skeleton_context}
```
{rag_knowledge}
{output_requirements}"""

        # 构建消息列表
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # C2R: 详细的 prompt 组成部分统计
        total_prompt_chars = len(system_prompt) + len(user_prompt)
        estimated_tokens = estimate_tokens(system_prompt) + estimate_tokens(user_prompt)

        # 记录各部分长度，方便调试
        logger.info(f"[Prompt统计] 函数: {func_info.name}")
        logger.info(f"[Prompt统计] system_prompt: {len(system_prompt)} 字符 (~{len(system_prompt)//4} tokens)")
        logger.info(f"[Prompt统计] user_prompt: {len(user_prompt)} 字符 (~{len(user_prompt)//4} tokens)")
        logger.info(f"[Prompt统计]   - skeleton_context: {len(skeleton_context)} 字符")
        logger.info(f"[Prompt统计]   - c_code: {len(func_info.c_code)} 字符")
        logger.info(f"[Prompt统计]   - rag_knowledge: {len(rag_knowledge) if rag_knowledge else 0} 字符")
        logger.info(f"[Prompt统计]   - opaque_type_warning: {len(opaque_type_warning) if opaque_type_warning else 0} 字符")
        logger.info(f"[Prompt统计]   - accessor_shim_hints: {len(accessor_shim_hints) if accessor_shim_hints else 0} 字符")
        logger.info(f"[Prompt统计] 总计: {total_prompt_chars} 字符 (~{estimated_tokens} tokens)")

        # 检查总体 prompt 长度（防止超过模型最大上下文窗口）
        # Qwen3-Coder-30B: 262144 tokens (256K)
        # 模型最大 token 约 262144，预留 16384 给输出，所以输入最多约 245000 tokens (~980000 chars)
        MAX_INPUT_TOKENS = 245000
        if estimated_tokens > MAX_INPUT_TOKENS:
            logger.error(f"[Prompt超限] 函数 {func_info.name}: ~{estimated_tokens} tokens > 最大 {MAX_INPUT_TOKENS} tokens")
            logger.error(f"[Prompt超限] 需要截断 {estimated_tokens - MAX_INPUT_TOKENS} tokens (~{(estimated_tokens - MAX_INPUT_TOKENS) * 4} 字符)")

            # 计算需要减少的字符数
            excess_tokens = estimated_tokens - MAX_INPUT_TOKENS
            chars_to_reduce = excess_tokens * 4  # 约 4 字符/token

            # 截断骨架上下文
            new_max = max(20000, len(skeleton_context) - chars_to_reduce)
            logger.info(f"[Prompt截断] skeleton_context 从 {len(skeleton_context)} 截断到 {new_max} 字符")
            skeleton_context_truncated = smart_truncate_context(skeleton_context, new_max)

            # 重新构建 user_prompt
            user_prompt = f"""Translate the following C/C++ function to Rust.
{opaque_type_warning}{accessor_shim_hints}
## Target Rust Function Signature (MUST use this exact signature)
```rust
{target_signature}
```

## C/C++ Source Code to Translate
NOTE: The source below may be a slice from the build’s preprocessed `.i` TU (macros expanded, inline bodies visible).
```cpp
{func_info.c_code}
```
{callee_signature_hints}
{inline_helper_hints}
{callee_rust_signature_hints}
{internal_callee_paths_hints}
{typed_constants_hints}
{c_field_access_hints}
{c_pointer_contract_hints}

## Skeleton Context (available types and functions) [TRUNCATED due to length]
```rust
{skeleton_context_truncated}
```
{rag_knowledge}
{output_requirements}"""
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            new_tokens = estimate_tokens(system_prompt) + estimate_tokens(user_prompt)
            logger.info(f"[Prompt截断] 截断后: ~{new_tokens} tokens")
        
        # 保存提示词
        try:
            from save_llm_prompts import save_llm_prompt, save_llm_prompt_text
            metadata = {
                "target_signature": target_signature,
                "file_name": func_info.file_name,
                "opaque_types": list(self.opaque_types) if self.opaque_types else [],
                "skeleton_context_length": len(skeleton_context),
                "c_code_length": len(func_info.c_code),
                "callee_signature_hints_length": len(callee_signature_hints) if callee_signature_hints else 0,
                "inline_helper_hints_length": len(inline_helper_hints) if inline_helper_hints else 0,
                "callee_rust_signature_hints_length": len(callee_rust_signature_hints) if callee_rust_signature_hints else 0,
                "typed_constants_hints_length": len(typed_constants_hints) if typed_constants_hints else 0,
                "c_field_access_hints_length": len(c_field_access_hints) if c_field_access_hints else 0,
                "c_pointer_contract_hints_length": len(c_pointer_contract_hints) if c_pointer_contract_hints else 0,
                "context_prefix_length": len(context_prefix) if context_prefix else 0,
                "context_meta": context_meta,
                "estimated_tokens": estimated_tokens
            }
            if context_summary_path:
                metadata["context_summary_file"] = str(context_summary_path)
            save_llm_prompt(
                messages=messages,
                project_name=self.project_name,
                llm_name=self.llm_name,
                task_type="incremental_translate",
                function_name=func_info.name,
                metadata=metadata,
                output_dir=self.llm_prompts_dir
            )
            save_llm_prompt_text(
                messages=messages,
                project_name=self.project_name,
                llm_name=self.llm_name,
                task_type="incremental_translate",
                function_name=func_info.name,
                metadata=metadata,
                output_dir=self.llm_prompts_dir
            )
        except Exception as e:
            logger.warning(f"保存提示词失败: {e}")

        try:
            # 使用信号量控制 vLLM 并发请求数
            _vllm_semaphore.acquire()
            try:
                response = generation(messages)
            finally:
                _vllm_semaphore.release()

            if isinstance(response, dict):
                response = response.get("content", "")

            print("✓")

            # 保存 LLM 原始响应（用于调试）
            try:
                response_dir = self.llm_prompts_dir / self.llm_name / "responses"
                response_dir.mkdir(parents=True, exist_ok=True)
                response_file = response_dir / f"{func_info.name}_response.txt"
                _atomic_write_text(
                    response_file,
                    "".join(
                        [
                            f"=== LLM Response for {func_info.name} ===\n",
                            f"Length: {len(response) if response else 0} characters\n",
                            f"{'='*50}\n\n",
                            (response if response else "(empty response)"),
                        ]
                    ),
                )
            except Exception as save_err:
                logger.debug(f"保存响应失败: {save_err}")
            
            # 提取代码
            extracted_code = self._extract_rust_code(response)
            
            # 如果提取失败，记录详细信息
            if not extracted_code:
                logger.warning(f"[{func_info.name}] 无法从响应中提取 Rust 代码")
                logger.warning(f"  响应长度: {len(response) if response else 0}")
                logger.warning(f"  响应开头: {response[:200] if response else '(空)'}")
            
            return extracted_code
        except Exception as e:
            print("✗")
            logger.error(f"翻译失败: {e}")
            # 检查是否是超时错误，如果是则记录到重试队列
            error_str = str(e).lower()
            is_timeout = "timeout" in error_str or "timed out" in error_str
            if is_timeout:
                self.stats["timeout_failed"] += 1
                self.timeout_failed_funcs.append(func_info.file_name)
                logger.warning(f"[{func_info.name}] LLM 超时，已加入重试队列")
            # 保存错误信息
            try:
                response_dir = self.llm_prompts_dir / self.llm_name / "errors"
                response_dir.mkdir(parents=True, exist_ok=True)
                error_file = response_dir / f"{func_info.name}_error.txt"
                _atomic_write_text(
                    error_file,
                    "".join(
                        [
                            f"=== LLM Error for {func_info.name} ===\n",
                            f"Exception: {type(e).__name__}: {e}\n",
                        ]
                    ),
                )
            except (OSError, IOError, PermissionError):
                pass
            return None
    
    def _extract_rust_code(self, response: str) -> Optional[str]:
        """从 LLM 响应中提取 Rust 代码"""
        if not response:
            return None

        patterns = [r'```rust(.*?)```', r'```Rust(.*?)```', r'```(.*?)```']

        extracted = response.strip()
        for pattern in patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                extracted = match.group(1).strip()
                break

        # Robustness: strip stray markdown fences even when the code block is unbalanced.
        # This prevents injection failures when the model outputs ```rust without a closing fence.
        cleaned_lines: List[str] = []
        for line in (extracted or "").splitlines():
            if line.strip().startswith("```"):
                continue
            cleaned_lines.append(line)
        cleaned = "\n".join(cleaned_lines).strip()
        return cleaned if cleaned else None
    
    def _refactor_to_idiomatic(self, func_info: FunctionInfo, unidiomatic_code: str) -> Optional[str]:
        """
        Phase 3.2: Unidiomatic to Idiomatic Rust (安全性重构)
        
        将 Phase 3.1 生成的 unsafe 代码重构为地道的 Safe Rust
        
        Args:
            func_info: 函数信息
            unidiomatic_code: Phase 3.1 生成的非地道代码
            
        Returns:
            重构后的地道 Rust 代码，如果失败则返回原代码
        """
        from generate.generation import generation
        
        # 检查是否需要重构（如果没有 unsafe，跳过）
        if 'unsafe' not in unidiomatic_code and '*mut' not in unidiomatic_code and '*const' not in unidiomatic_code:
            return unidiomatic_code  # 已经是安全代码，无需重构
        
        print(f"  Phase 3.2 重构...", end=" ", flush=True)
        
        # Phase 3.2: 上下文隔离 - 只使用 Rust 代码，不引入 C 代码
        # 这样 LLM 专注于 Rust->Rust 重构，避免 C 思维定势
        system_prompt = """You are a Rust expert specializing in safe code refactoring.

## PHASE 3.2: REFACTOR UNSAFE RUST TO IDIOMATIC SAFE RUST

CONTEXT ISOLATION (IMPORTANT):
- You are given ONLY Rust code (no C/C++ code)
- The business logic translation is ALREADY DONE in Phase 3.1
- Your ONLY task is: Rust syntax refactoring (unsafe → safe)
- Do NOT try to re-translate or re-think the original C logic

REFACTORING RULES:
1. Remove `unsafe` blocks where possible
2. Convert raw pointers (`*mut T`, `*const T`) to:
   - References (`&T`, `&mut T`) when lifetime is clear
   - `Option<&T>` for nullable pointers
   - `Box<T>` for owned heap allocations
3. Use Rust idioms:
   - `Option` and `Result` instead of null checks
   - Iterators instead of manual loops with index
   - `Vec` instead of raw arrays with manual management
   - `.is_null()` checks → `Option` pattern matching
4. Keep the SAME function signature - don't change the public API
5. If a refactoring is risky or unclear, KEEP the original unsafe code

PRIORITY ORDER:
1. Correctness > Safety > Idiomaticity
2. Working unsafe code > Broken safe code
3. When in doubt, preserve the original"""

        user_prompt = f"""Refactor this working Rust code to be more idiomatic and safe.

## Input: Working Unsafe Rust Code
```rust
{unidiomatic_code}
```

## Constraint: Keep This Exact Signature
```rust
{func_info.rust_signature}
```

## Task
1. Analyze the unsafe patterns used
2. Refactor to safe Rust where possible
3. Keep unsafe blocks that cannot be safely refactored
4. Output the complete refactored function

Output the refactored code in ```rust ... ``` block:"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        try:
            # 使用信号量控制 vLLM 并发请求数
            _vllm_semaphore.acquire()
            try:
                response = generation(messages)
            finally:
                _vllm_semaphore.release()

            if isinstance(response, dict):
                response = response.get("content", "")

            refactored = self._extract_rust_code(response)
            if refactored:
                print("✓")
                return refactored
            else:
                print("✗ (keeping original)")
                return unidiomatic_code
        except Exception as e:
            print(f"✗ ({e})")
            logger.warning(f"Phase 3.2 重构失败: {e}")
            return unidiomatic_code

    def _analyze_compilation_errors(self, error_msg: str) -> List[str]:
        """
        解析 cargo check 错误并生成结构化的错误分析提示
        
        返回: 错误提示列表
        """
        error_hints = []
        
        if not error_msg:
            return error_hints

        # 语法错误（通常没有 error[E...] 代码）。这种情况下“错误码解析”会提取不到，
        # 需要显式提示修复器优先按语法去修（而不是类型/借用）。
        try:
            if self._looks_like_rust_syntax_error(error_msg):
                error_hints.append("## POSSIBLE RUST SYNTAX ERROR (no error[E...] code)")
                error_hints.append("   rustc did not emit an error code. Treat this as a SYNTAX issue first.")
                error_hints.append("   Common LLM mistake: using definition syntax inside a call:")
                error_hints.append("   WRONG: `some_func(arg1: Type1, arg2: Type2)`")
                error_hints.append("   CORRECT: `some_func(arg1, arg2)`")
                error_hints.append("   Also check for missing/extra braces, commas, quotes, and delimiters.")
        except Exception:
            pass
        
        # 解析 Rust 错误代码（如 E0277, E0425, E0609 等）
        error_code_pattern = re.compile(r'error\[E(\d+)\]:\s*(.+)')
        error_codes_found = error_code_pattern.findall(error_msg)
        
        # Rust 错误代码映射（扩展版）
        rust_error_explanations = {
            "0038": ("trait is not dyn compatible / object-safe",
                     "Trait cannot be used as a trait object. Check if methods have &self parameter. All trait methods must have &self to be object-safe."),
            "0133": ("unsafe block required",
                     "CRITICAL: Raw pointer dereference requires unsafe block. Wrap ALL pointer operations in `unsafe { ... }`. Example: `unsafe { *ptr = value; }`"),
            "0185": ("method has &self in impl but not in trait",
                     "CRITICAL: The trait definition does NOT have &self, but your impl does. Check the trait definition and match the signature exactly."),
            "0277": ("trait bound not satisfied",
                     "Type doesn't implement required trait. Check if you need to add #[derive(...)] or implement the trait."),
            "0369": ("binary operation cannot be applied",
                     "Type doesn't support this operator. Add #[derive(PartialEq)] for == or !=, #[derive(PartialOrd)] for < or >."),
            "0412": ("cannot find type in this scope",
                     "Type is not defined. Do NOT invent new type definitions in the output. Prefer using existing types from the skeleton/types.rs; "
                     "otherwise treat it as an upstream types/TU gap and return safe defaults / TODOs."),
            "0425": (
                "cannot find value/function in this scope",
                "CRITICAL: Symbol not found. If it is an internal callee, use the provided 'Internal Callees' mapping "
                "and call it as `crate::<module>::Name(...)` (do NOT assume `crate::types`). "
                "Do NOT guess signatures via local `extern \"C\" { ... }` blocks. External C APIs should be declared "
                "deterministically in `compat.rs` (bindgen allowlist). If it is still missing, treat it as a TU / decl-generation gap.",
            ),
            "0432": ("unresolved import",
                     "Module or crate not found. If using `libc`, make sure it's in Cargo.toml. Or use std equivalents."),
            "0433": ("failed to resolve",
                     "Module or crate not found. Check if the crate is in Cargo.toml or use std alternatives."),
            "0449": ("visibility qualifiers not permitted here",
                     "CRITICAL: Do NOT use `pub` on methods inside trait impl blocks. Remove the `pub` keyword."),
            "0605": ("non-primitive cast",
                     "CRITICAL: Cannot cast between these types directly. For enum casts, use `as i32` first then match. "
                     "For struct casts, use transmute or proper conversion. Example: `unsafe {{ std::mem::transmute(value) }}`"),
            "0606": ("invalid cast",
                     "CRITICAL: Cannot cast reference to raw pointer of different type. Use proper pointer conversion: "
                     "`&mut x as *mut _ as *mut TargetType` or `std::ptr::addr_of_mut!(x) as *mut TargetType`"),
            "0609": ("no field on type",
                     "Cannot access struct field. If the type is opaque (defined as c_void), return a safe default instead."),
            "0308": ("mismatched types",
                     "Type mismatch. Cast with `as` or use appropriate conversion methods. For integer types use `as i32`, `as u64`, etc."),
            "0382": ("use of moved value",
                     "Value was moved. Use .clone() or restructure to avoid moving."),
            "0507": ("cannot move out of borrowed content",
                     "Trying to move from a reference. Use .clone() or work with references."),
            "0502": ("cannot borrow as mutable",
                     "Already borrowed as immutable. Restructure borrows or use interior mutability."),
            "0503": ("cannot use data mutably",
                     "Cannot mutate in this context. Check borrow rules."),
            "0596": ("cannot borrow as mutable",
                     "Value is not mutable. Add `mut` keyword: `let mut x = ...` instead of `let x = ...`"),
            "0597": ("borrowed value does not live long enough",
                     "Lifetime issue. Consider using owned types or extending lifetimes."),
            "0599": ("no method found",
                     "Method doesn't exist on type. CRITICAL: If calling `is_null()` or `as_ptr()` on integer type (usize, u64, i32), "
                     "this means the type should be a pointer but is defined as integer. Use pointer comparison: `ptr == std::ptr::null_mut()` "
                     "or cast to pointer first: `(value as *const T).is_null()`"),
            "0603": ("cannot assign to this expression",
                     "Cannot assign. Check if the target is mutable."),
            "0614": ("cannot find value in this scope",
                     "Variable not found. Check spelling and scope."),
            "0615": ("cannot find value in this scope",
                     "Variable not found. Check spelling and scope."),
            "0618": ("expected function, found Option",
                     "CRITICAL: Function pointer is wrapped in Option<fn>. You MUST unwrap before calling: "
                     "`if let Some(func) = callback {{ unsafe {{ func(args) }} }}` or use `.map()`. "
                     "NEVER call Option directly like `callback(args)`."),
            "0624": ("cannot assign to immutable field",
                     "Field is immutable. Make the struct mutable or use interior mutability."),
        }
        
        # 收集所有找到的错误代码
        found_errors = set()
        for error_code, error_desc in error_codes_found:
            found_errors.add(error_code)
            if error_code in rust_error_explanations:
                name, suggestion = rust_error_explanations[error_code]
                error_hints.append(f"## Error E{error_code}: {name}")
                error_hints.append(f"   Details: {error_desc[:300]}")
                error_hints.append(f"   Suggestion: {suggestion}")
        
        # 提取具体的错误位置和内容
        location_pattern = re.compile(r'--> ([^:]+):(\d+):(\d+)')
        locations = location_pattern.findall(error_msg)
        if locations:
            error_hints.append(f"\n## Error Locations:")
            for file_path, line, col in locations[:3]:  # 只显示前3个
                error_hints.append(f"   - {file_path}:{line}:{col}")
        
        # 通用错误模式识别（如果没有找到错误代码）
        if not found_errors:
            if "cannot find" in error_msg.lower() or "not found" in error_msg.lower():
                error_hints.append("## Missing identifier")
                error_hints.append("   Some identifier is not in scope.")
                error_hints.append("   Suggestion: Check skeleton context for definitions, or add `extern \"C\"` for C functions.")
            if "expected" in error_msg.lower() and "found" in error_msg.lower():
                error_hints.append("## Type mismatch")
                error_hints.append("   Suggestion: Use type casting with `as` or appropriate conversion.")
            if "borrow" in error_msg.lower() or "lifetime" in error_msg.lower():
                error_hints.append("## Borrow/Lifetime issue")
                error_hints.append("   Suggestion: Use raw pointers, .clone(), or restructure code.")
        
        # 检测不透明类型相关错误
        if "field" in error_msg.lower() and ("struct" in error_msg.lower() or "type" in error_msg.lower()):
            for opaque_type in self.opaque_types:
                if opaque_type.lower() in error_msg.lower():
                    error_hints.append(f"\n## CRITICAL: Opaque Type Access")
                    error_hints.append(f"   {opaque_type} is OPAQUE (defined as c_void). You CANNOT access its fields!")
                    error_hints.append("   Return safe defaults instead:")
                    error_hints.append("   - For `-> *mut T`: return `std::ptr::null_mut()`")
                    error_hints.append("   - For `-> i32` or integer: return `0`")
                    error_hints.append("   - For `-> bool`: return `false` or `0`")
                    error_hints.append("   - For `()`: just return")
                    break
        
        # 检测缺失类型错误
        if "0412" in found_errors:
            # 尝试提取缺失的类型名
            type_pattern = re.compile(r'cannot find type `(\w+)`')
            missing_types = type_pattern.findall(error_msg)
            if missing_types:
                unique_types = list(set(missing_types))
                error_hints.append(f"\n## Missing Types Detected: {', '.join(unique_types)}")
                error_hints.append("   Suggestion: Declare placeholder structs before your function:")
                for t in unique_types[:3]:
                    error_hints.append(f"   `pub struct {t};`")
        
        # 检测 E0449 错误（pub 在 trait impl 中）
        if "0449" in found_errors:
            error_hints.append("\n## CRITICAL FIX for E0449:")
            error_hints.append("   Remove ALL `pub` keywords from methods inside `impl Trait for Type` blocks!")
            error_hints.append("   WRONG: `impl Foo for Bar { pub fn method(&self) {} }`")
            error_hints.append("   CORRECT: `impl Foo for Bar { fn method(&self) {} }`")
        
        # 检测 E0185 错误（&self 不匹配）
        if "0185" in found_errors:
            error_hints.append("\n## CRITICAL FIX for E0185:")
            error_hints.append("   The trait method signature does NOT have &self, but your impl does (or vice versa).")
            error_hints.append("   Check the trait definition in the skeleton and match it EXACTLY.")
            error_hints.append("   If trait says `fn foo(arg: T)`, your impl must be `fn foo(arg: T)` (no &self)")
            error_hints.append("   If trait says `fn foo(&self, arg: T)`, your impl must be `fn foo(&self, arg: T)`")
        
        # 检测 E0038 错误（trait 不是 object-safe）
        if "0038" in found_errors:
            error_hints.append("\n## CRITICAL FIX for E0038:")
            error_hints.append("   The trait is not 'dyn compatible' (object-safe).")
            error_hints.append("   This usually means some methods don't have &self parameter.")
            error_hints.append("   Check the trait definition - all methods need &self to be used as `dyn Trait`.")

        # === C2R FIX: 添加更多错误类型的特殊处理 ===

        # 检测 E0618 错误（Option 函数指针调用）
        if "0618" in found_errors:
            error_hints.append("\n## CRITICAL FIX for E0618 (Option function pointer):")
            error_hints.append("   Function pointer is wrapped in Option<fn>. You CANNOT call it directly!")
            error_hints.append("   WRONG: `callback(arg1, arg2)`")
            error_hints.append("   CORRECT: `if let Some(func) = callback { unsafe { func(arg1, arg2) } }`")
            error_hints.append("   OR: `callback.map(|f| unsafe { f(arg1, arg2) })`")

        # 检测 E0133 错误（缺少 unsafe 块）
        if "0133" in found_errors:
            error_hints.append("\n## CRITICAL FIX for E0133 (unsafe required):")
            error_hints.append("   Raw pointer operations require unsafe block!")
            error_hints.append("   WRONG: `*ptr = value;`")
            error_hints.append("   CORRECT: `unsafe { *ptr = value; }`")
            error_hints.append("   Also wrap: extern C calls, static mut access, transmute, etc.")

        # 检测 E0605/E0606 错误（类型转换问题）
        if "0605" in found_errors or "0606" in found_errors:
            error_hints.append("\n## CRITICAL FIX for E0605/E0606 (cast errors):")
            error_hints.append("   Invalid type cast. Use proper conversion methods:")
            error_hints.append("   - Enum to i32: `my_enum as i32`")
            error_hints.append("   - i32 to Enum: `unsafe { std::mem::transmute::<i32, MyEnum>(val) }`")
            error_hints.append("   - Ref to ptr: `&mut x as *mut T`")
            error_hints.append("   - Different ptr types: `ptr as *mut T` or `ptr.cast::<T>()`")
            error_hints.append("   - For complex casts: use intermediate types or transmute")

        # 检测 E0277 错误（类型运算不兼容）
        if "0277" in found_errors:
            # 检查是否是位运算问题
            if "&" in error_msg or "|" in error_msg or "BitAnd" in error_msg or "BitOr" in error_msg:
                error_hints.append("\n## CRITICAL FIX for E0277 (bitwise operation):")
                error_hints.append("   Cannot perform bitwise operation between different types.")
                error_hints.append("   Cast both operands to same type first:")
                error_hints.append("   WRONG: `u32_val & i32_val`")
                error_hints.append("   CORRECT: `u32_val & (i32_val as u32)`")

        # 检测 E0599 错误中的 is_null/as_ptr 问题
        if "0599" in found_errors:
            error_hints.append("\n## CRITICAL FIX for E0599 (method not found on type):")

            # 检测 cast_mut/cast_const 等不存在的方法
            if "cast_mut" in error_msg.lower():
                error_hints.append("   `cast_mut()` does NOT exist on raw pointers!")
                error_hints.append("   WRONG: `ptr.cast_mut()`")
                error_hints.append("   CORRECT: `ptr as *mut T`")
            elif "cast_const" in error_msg.lower():
                error_hints.append("   `cast_const()` does NOT exist on raw pointers!")
                error_hints.append("   WRONG: `ptr.cast_const()`")
                error_hints.append("   CORRECT: `ptr as *const T`")
            elif "is_null" in error_msg.lower() or "as_ptr" in error_msg.lower():
                error_hints.append("   Calling pointer methods on non-pointer type!")
                error_hints.append("   If type is usize/u64/i32, it should be a pointer but is defined as integer.")
                error_hints.append("   Solutions:")
                error_hints.append("   - Use comparison: `value == 0` instead of `ptr.is_null()`")
                error_hints.append("   - Cast first: `(value as *const T).is_null()`")
            else:
                # 通用 E0599 修复
                error_hints.append("   Method not found. Check:")
                error_hints.append("   - Is the type correct? Check skeleton types.rs")
                error_hints.append("   - For pointer cast: use `ptr as *mut T` (not cast_mut)")
                error_hints.append("   - For array/slice methods on raw ptr: cast to slice first")
        
        # 检测函数调用语法错误
        if "expected" in error_msg.lower() and (":" in error_msg or "type" in error_msg.lower()):
            # 可能是函数调用语法错误
            if re.search(r'\w+\s*\(\s*\w+\s*:\s*\w+', error_msg):
                error_hints.append("\n## POSSIBLE SYNTAX ERROR:")
                error_hints.append("   You may be using definition syntax in a function call.")
                error_hints.append("   WRONG: `some_func(arg1: Type1, arg2: Type2)`")
                error_hints.append("   CORRECT: `some_func(arg1, arg2)`")
        
        return error_hints

    def _looks_like_rust_syntax_error(self, error_msg: str) -> bool:
        """
        Heuristic: detect Rust parser/syntax errors which often look like `error: expected ...`
        and do NOT have an `error[E1234]` code.
        """
        if not error_msg:
            return False
        if re.search(r'error\[E\d+\]:', error_msg):
            return False

        # Must have a source location into a Rust file to be a code syntax error
        has_location = bool(re.search(r'^\s*-->\s+.*\.rs:\d+:\d+', error_msg, re.MULTILINE))
        if not has_location:
            return False

        # Ignore cargo summary lines like: "error: could not compile ... due to ..."
        if re.search(r'^\s*error:\s+could not compile\b', error_msg, re.MULTILINE):
            # Still allow if there are other more specific syntax lines
            pass

        # Typical parser error keywords
        if re.search(r'^\s*error:\s*(expected|unexpected|mismatched|unterminated|unclosed)\b', error_msg, re.MULTILINE | re.IGNORECASE):
            return True
        if "expected one of" in error_msg.lower():
            return True
        if "mismatched closing delimiter" in error_msg.lower():
            return True
        if "unclosed delimiter" in error_msg.lower():
            return True
        if re.search(r'^\s*error:\s+expected .* found .*', error_msg, re.MULTILINE | re.IGNORECASE):
            return True

        return False
    
    def _get_repair_output_requirements(self, is_drop_impl: bool, is_impl_method: bool, is_constructor: bool, target_signature: str) -> str:
        """生成泛化的修复输出要求"""
        if is_drop_impl:
            return f"""1. Generate ONLY the `fn drop(&mut self) {{ ... }}` method definition
2. Do NOT output the enclosing `impl Drop for ... {{ ... }}` block (skeleton already contains it)
3. For destructors:
   - If C/C++ destructor is empty/default, use: `fn drop(&mut self) {{}}`
   - If it releases resources, translate appropriately
4. If you cannot access opaque type fields, just do nothing or return
5. DO NOT delete major logic just to make it compile; if needed, add TODO + safe defaults explicitly
6. Output ONLY raw Rust code (no ``` fences, no explanations)
7. Output EXACTLY ONE Rust item; keep it short; no extra helper items"""
        elif is_impl_method:
            return f"""1. Generate the COMPLETE method definition (WITH &self or &mut self)
2. Use EXACTLY this signature: `{target_signature}`
3. Do NOT output the enclosing `impl` block; do NOT output any extra items (no helper fns/types/mod/use)
4. If you cannot implement the logic due to opaque types, return safe defaults
5. Refer to the original C/C++ code to understand the intended behavior
6. DO NOT delete major logic just to make it compile; if needed, add TODO + safe defaults explicitly
7. Output ONLY raw Rust code (no ``` fences, no explanations); keep it short"""
        elif is_constructor:
            return f"""1. Generate the COMPLETE constructor function
2. Use EXACTLY this signature: `{target_signature}`
3. Initialize all struct fields appropriately
4. If some fields cannot be initialized, use Default::default() or safe defaults
5. Do NOT output any extra items (no helper fns/types/impl/mod/use)
6. DO NOT delete initialization logic; if needed, add TODO + safe defaults explicitly
7. Output ONLY raw Rust code (no ``` fences, no explanations); keep it short"""
        else:
            return f"""1. Generate the COMPLETE fixed standalone function
2. Use EXACTLY this signature: `{target_signature}`
3. If you cannot implement the logic due to opaque types, return safe defaults
4. Refer to the original C/C++ code to understand the intended behavior
5. Do NOT output any extra items (no helper fns/types/impl/mod/use)
6. DO NOT add &self parameter
7. DO NOT delete major logic just to make it compile; if needed, add TODO + safe defaults explicitly
8. Output ONLY raw Rust code (no ``` fences, no explanations); keep it short"""

    @staticmethod
    def _extract_target_fn_name(func_info: FunctionInfo) -> str:
        name = getattr(func_info, "name", "") or ""
        if name:
            return name
        sig = getattr(func_info, "rust_signature", "") or ""
        m = re.search(r"\bfn\s+(r#[A-Za-z_]\w*|[A-Za-z_]\w*)\b", sig)
        if not m:
            return ""
        n = m.group(1)
        return n[2:] if n.startswith("r#") else n

    @staticmethod
    def _looks_injectable_rust(snippet: str, *, func_name: str) -> bool:
        s = (snippet or "").strip()
        if not s:
            return False
        if "```" in s:
            return False
        if func_name and not re.search(rf"\bfn\s+(?:r#)?{re.escape(func_name)}\b", s):
            return False
        # Best-effort brace balance check (fast path). Tree-sitter validation happens in the injector.
        depth = 0
        for ch in s:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth < 0:
                    return False
        if depth != 0:
            return False
        if "{" not in s:
            return False
        # Heuristic: require closing brace at the end to avoid common truncation.
        if not s.endswith("}"):
            return False
        return True

    def _llm_regenerate_for_injection(self, func_info: FunctionInfo, skeleton_context: str, *, previous_output: str, attempt_num: int) -> Optional[str]:
        """
        When the injector rejects an LLM output due to structural issues (truncation/unclosed braces),
        ask LLM to re-generate a strictly-injectable single-function output.
        """
        try:
            from generate.generation import generation
        except Exception as e:
            logger.debug(f"[InjectionRetry] 无法导入 generation: {e}")
            return None

        target_signature = getattr(func_info, "rust_signature", "") or ""
        func_name = self._extract_target_fn_name(func_info) or "<unknown>"

        prev = (previous_output or "").strip()
        if len(prev) > 1500:
            prev = prev[:1500] + "\n/* ... truncated ... */"

        ctx = skeleton_context or ""
        if len(ctx) > MAX_SKELETON_CONTEXT_CHARS:
            ctx = smart_truncate_context(ctx, MAX_SKELETON_CONTEXT_CHARS)

        system_prompt = (
            "You are a C/C++ to Rust translator.\n"
            "OUTPUT FORMAT (STRICT):\n"
            "- Output ONLY ONE complete Rust function/method definition\n"
            "- NO markdown fences, NO explanations, NO extra items (no mod/impl/use/helper fns)\n"
            "- Keep it SHORT: no long comments\n"
            "- Ensure all braces are closed\n"
        )

        user_prompt = f"""Regenerate the Rust translation for injection.

The previous output was NOT injectible (likely truncated / unclosed braces). Generate again.

## Target Rust Signature (MUST match exactly)
```rust
{target_signature}
```

## C/C++ Source
```c
{func_info.c_code}
```

## Skeleton Context (types/functions available)
```rust
{ctx}
```

## Previous Output (invalid)
```rust
{prev}
```

REMINDER: Output ONLY the target function/method definition for `{func_name}`. Nothing else."""

        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

        try:
            _vllm_semaphore.acquire()
            try:
                response = generation(messages)
            finally:
                _vllm_semaphore.release()
            if isinstance(response, dict):
                response = response.get("content", "")
            return self._extract_rust_code(response or "")
        except Exception as e:
            logger.debug(f"[InjectionRetry] LLM 重试失败 (attempt={attempt_num}, fn={func_name}): {e}")
            return None
    
    def _try_inject_and_compile(
        self,
        func_name: str,
        func_info: FunctionInfo,
        translated_code: str,
        rollback_on_failure: bool = True
    ) -> bool:
        """
        尝试注入代码并编译
        
        参数:
            rollback_on_failure: 编译失败时是否回退到备份。
                                 在修复循环中设为 False，以便获取正确的编译错误。
                                 最终修复失败后再手动回退。
        
        返回: 是否编译成功
        """
        # 1. 解析翻译结果
        from auto_test_rust import read_translated_function, find_and_replace_function_signature
        
        # 提取目标函数名
        target_func_name = func_info.name
        if not target_func_name and func_info.rust_signature:
            fn_match = re.search(r'fn\s+(\w+)', func_info.rust_signature)
            if fn_match:
                target_func_name = fn_match.group(1)

        def _prepare_injection_payload(code: str) -> tuple[str, list, str]:
            """Return (translated_function, imports, translated_for_injection)."""
            try:
                _, functions, imports = read_translated_function(code)
                # 只保留与目标函数名匹配的函数，避免注入辅助函数导致重复定义
                if target_func_name and functions:
                    matched_functions = []
                    for func in functions:
                        func_name_match = re.search(r'fn\s+(\w+)', func)
                        if func_name_match and func_name_match.group(1) == target_func_name:
                            matched_functions.append(func)
                    translated_function = matched_functions[0] if matched_functions else (functions[0] if functions else code)
                else:
                    translated_function = "\n".join(functions) if functions else code
            except (re.error, IndexError, TypeError, ValueError, AttributeError) as e:
                logger.debug(f"Code extraction failed: {e}")
                translated_function = code
                imports = []

            # 1.5 后处理：清理翻译后代码中的问题
            translated_function = self._postprocess_translated_code(translated_function)
            # 1.5.1 全局变量访问重写：safe globals -> accessor API（减少 stub/unsafe 引入的编译失败）
            try:
                translated_function = self._rewrite_safe_globals_in_code(translated_function)
            except Exception:
                pass
            imports = self._filter_valid_imports(imports)

            # 1.6 兜底：LLM/修复器有时只返回“函数体片段”（没有 fn ...），导致注入逻辑直接失败。
            # 如果已有确定的 Rust 签名，则将片段包装成完整函数定义再注入。
            translated_for_injection = translated_function
            try:
                if func_info.rust_signature and translated_function:
                    full_def_pat = re.compile(
                        r'^\s*(pub(\([^)]*\))?\s+)?(unsafe\s+)?(extern\s+"C"\s+)?fn\s+\w+\s*\(',
                        re.MULTILINE,
                    )
                    if (not full_def_pat.search(translated_function)) and (
                        not translated_function.strip().startswith("impl ")
                    ):
                        body = translated_function.strip()
                        if body.startswith("{") and body.endswith("}"):
                            body = body[1:-1].strip()
                        translated_for_injection = f"{func_info.rust_signature.rstrip()} {{\n{body}\n}}\n"
            except Exception:
                translated_for_injection = translated_function

            return translated_function, imports, translated_for_injection

        translated_function, imports, translated_for_injection = _prepare_injection_payload(translated_code)

        # Injection retry: if the output is clearly truncated/unclosed, regenerate before touching files.
        if self._enable_injection_retry and self._injection_retry_max > 0:
            fn_name_for_check = (target_func_name or self._extract_target_fn_name(func_info)).strip()
            if fn_name_for_check:
                for i in range(int(self._injection_retry_max or 0)):
                    if self._looks_injectable_rust(translated_for_injection, func_name=fn_name_for_check):
                        break
                    try:
                        base_ctx = self._skeleton_non_types_context or ""
                        per_ctx = self._build_skeleton_context_for_function(func_info, base_ctx)
                    except Exception:
                        per_ctx = self._skeleton_non_types_context or ""
                    regenerated = self._llm_regenerate_for_injection(
                        func_info, per_ctx, previous_output=translated_for_injection, attempt_num=i + 1
                    )
                    if not regenerated:
                        break
                    translated_code = regenerated
                    translated_function, imports, translated_for_injection = _prepare_injection_payload(translated_code)
        
        # 2. 找到目标文件
        # 尝试两种文件名格式：src_xxx.rs 和 xxx.rs（向后兼容）
        rs_filename_with_prefix = "src_" + func_info.file_name + ".rs"
        rs_filename = func_info.file_name + ".rs"
        target_file = self.work_dir / "src" / rs_filename_with_prefix
        
        if not target_file.exists():
            # 回退到不带前缀的格式（向后兼容）
            target_file = self.work_dir / "src" / rs_filename
            if not target_file.exists():
                logger.error(f"目标文件不存在: {target_file} 或 {self.work_dir / 'src' / rs_filename_with_prefix}")
                return False
        
        # 3. 备份当前文件
        backup_content = target_file.read_text(encoding='utf-8', errors='ignore')
        
        # 4. 注入代码
        try:
            with open(target_file, 'r', encoding='utf-8', errors='ignore') as f:
                skeleton_code = f.read()
            
            # 先清理可能存在的重复签名（修复重复 pub extern "C" 问题）
            duplicate_pattern = r'(pub\s+extern\s+"C"\s+){2,}'
            skeleton_code = re.sub(duplicate_pattern, r'pub extern "C" ', skeleton_code)
            
            # 检查是否是 Drop trait 实现（析构函数）
            is_drop_impl = func_info.rust_signature.startswith("impl Drop for") if func_info.rust_signature else False
            
            if is_drop_impl:
                # 对于 Drop trait 实现，需要特殊处理
                # 提取类名
                class_name = func_info.rust_signature.split('for ')[1].split()[0] if 'for ' in func_info.rust_signature else func_name.split('_')[0].capitalize()
                
                # 查找是否已有 Drop 实现（更宽松的匹配，支持嵌套大括号）
                # 使用非贪婪匹配和递归大括号匹配
                drop_start_pattern = re.compile(
                    r'impl\s+Drop\s+for\s+' + re.escape(class_name) + r'\s*\{',
                    re.DOTALL
                )
                
                match = drop_start_pattern.search(skeleton_code)
                if match:
                    # 找到 Drop 实现开始，现在需要找到匹配的结束大括号
                    start = match.start()
                    brace_start = match.end() - 1  # 指向 {
                    brace_count = 1
                    idx = match.end()
                    while idx < len(skeleton_code) and brace_count > 0:
                        if skeleton_code[idx] == '{':
                            brace_count += 1
                        elif skeleton_code[idx] == '}':
                            brace_count -= 1
                        idx += 1
                    end = idx
                    
                    # 确保 translated_function 包含完整的 impl 块
                    if not translated_function.strip().startswith('impl'):
                        drop_impl = f"impl Drop for {class_name} {{\n    {translated_function}\n}}"
                    else:
                        drop_impl = translated_function
                    skeleton_code = skeleton_code[:start] + drop_impl + skeleton_code[end:]
                else:
                    # 添加新的 Drop 实现
                    # 确保 translated_function 包含完整的 impl 块
                    if not translated_function.strip().startswith('impl'):
                        drop_impl = f"\nimpl Drop for {class_name} {{\n    {translated_function}\n}}\n"
                    else:
                        drop_impl = f"\n{translated_function}\n"
                    skeleton_code += drop_impl
            else:
                # 使用签名匹配替换（普通函数）
                injection_success = False

                # 获取函数名
                actual_func_name = func_info.name
                if not actual_func_name:
                    fn_match = re.search(r'fn\s+(\w+)', func_info.rust_signature or "")
                    if fn_match:
                        actual_func_name = fn_match.group(1)

                # 策略0（优先）：使用位置索引进行确定性注入
                if hasattr(self, 'function_locations') and self.function_locations and actual_func_name:
                    # 尝试精确键匹配：文件名:函数名
                    file_stem = target_file.stem
                    location_key = f"{file_stem}:{actual_func_name}"
                    location = self.function_locations.get(location_key) or self.function_locations.get(actual_func_name)

                    if location and location.file_path == target_file:
                        # 提取 LLM 生成代码的函数体
                        body_match = re.search(r'fn\s+\w+\s*\([^)]*\)[^{]*(\{.*\})\s*$', translated_for_injection, re.DOTALL)
                        if body_match:
                            new_body = body_match.group(1)
                            # 使用精确字节偏移替换
                            content_bytes = skeleton_code.encode("utf-8")

                            # 重新解析获取最新位置（文件可能已被修改）
                            try:
                                tree = rust_parser.parse(content_bytes)
                                new_location = self._find_function_in_tree(tree.root_node, content_bytes, actual_func_name)
                                if new_location:
                                    # 提取原签名
                                    old_sig = content_bytes[new_location['start']:new_location['body_start']].decode("utf-8", errors="ignore")
                                    # 构建新函数
                                    new_func = old_sig + new_body
                                    # 替换
                                    new_content = (
                                        content_bytes[:new_location['start']].decode("utf-8", errors="ignore") +
                                        new_func +
                                        content_bytes[new_location['end']:].decode("utf-8", errors="ignore")
                                    )
                                    skeleton_code = new_content
                                    injection_success = True
                                    logger.info(f"位置索引注入成功: {actual_func_name}")
                            except Exception as e:
                                logger.debug(f"位置索引注入失败: {e}")

                # 策略1：完整签名匹配（后备）
                if not injection_success and func_info.rust_signature:
                    new_code, success = find_and_replace_function_signature(
                        skeleton_code, func_info.rust_signature, translated_for_injection
                    )
                    if success:
                        skeleton_code = new_code
                        injection_success = True

                # 策略2：只用函数名匹配
                if not injection_success and actual_func_name:
                    logger.debug(f"签名匹配失败，尝试函数名匹配: {actual_func_name}")
                    new_code, success = find_and_replace_function_signature(
                        skeleton_code, f"fn {actual_func_name}", translated_for_injection
                    )
                    if success:
                        skeleton_code = new_code
                        injection_success = True

                # 策略3：直接用正则表达式替换（最终后备）
                if not injection_success and actual_func_name:
                    logger.debug(f"函数名匹配也失败，尝试直接正则替换: {actual_func_name}")
                    patterns_to_try = [
                        rf'(pub\s+extern\s+"C"\s+fn\s+{re.escape(actual_func_name)}\s*\([^)]*\)[^{{]*)\{{\s*unimplemented!\(\)\s*\}}',
                        rf'(pub\s+fn\s+{re.escape(actual_func_name)}\s*\([^)]*\)[^{{]*)\{{\s*unimplemented!\(\)\s*\}}',
                        rf'(fn\s+{re.escape(actual_func_name)}\s*\([^)]*\)[^{{]*)\{{\s*unimplemented!\(\)\s*\}}',
                    ]
                    for pattern in patterns_to_try:
                        match = re.search(pattern, skeleton_code, re.DOTALL)
                        if match:
                            body_match = re.search(r'fn\s+\w+\s*\([^)]*\)[^{]*(\{.*\})\s*$', translated_for_injection, re.DOTALL)
                            if body_match:
                                new_body = body_match.group(1)
                                old_sig = match.group(1)
                                skeleton_code = skeleton_code[:match.start()] + old_sig + new_body + skeleton_code[match.end():]
                                injection_success = True
                                logger.info(f"正则替换成功: {actual_func_name}")
                                break

                if not injection_success:
                    logger.error(f"所有匹配策略失败，放弃注入: {func_name}")
                    func_info.injection_failed = True
                    self.stats["injection_failed"] += 1
                    with open(target_file, 'w', encoding='utf-8') as f:
                        f.write(backup_content)
                    return False
            
            # 添加导入
            if imports:
                from auto_test_rust import add_import_to_translated_result
                skeleton_code = add_import_to_translated_result(skeleton_code, imports)
            
            # 自动检测并添加缺失的依赖
            skeleton_code = self._auto_add_missing_dependencies(skeleton_code)
            
            with open(target_file, 'w', encoding='utf-8') as f:
                f.write(skeleton_code)
            
            # 4.5 关键修复：注入后验证文件语法是否正确
            # 使用 tree-sitter 解析。如果有严重语法错误：
            # - 在普通路径(rollback_on_failure=True)立即回退，避免污染项目
            # - 在修复路径(rollback_on_failure=False)允许继续，让 rustc 产生“语法错误”并进入增量修复
            if not self._validate_rust_syntax(skeleton_code):
                if rollback_on_failure:
                    logger.warning(f"注入后文件语法无效，立即回退: {func_name}")
                    with open(target_file, 'w', encoding='utf-8') as f:
                        f.write(backup_content)
                    return False
                logger.warning(f"注入后文件语法无效，将继续编译以获取 rustc 语法错误并尝试修复: {func_name}")
                
        except Exception as e:
            logger.error(f"注入代码失败: {e}")
            import traceback
            logger.debug(f"注入代码失败详情: {traceback.format_exc()}")
            # 回退到备份
            try:
                with open(target_file, 'w', encoding='utf-8') as f:
                    f.write(backup_content)
            except (OSError, IOError, PermissionError):
                pass
            return False
        
        # 5. 编译测试
        success = self._compile_project()

        # Deterministic extern vars/consts backfill (Step 2.56): avoid LLM guessing for missing globals.
        if not success:
            try:
                error_msg = self._get_compile_error()
                missing_vals = self._extract_missing_value_symbols_from_rustc_error(error_msg)
                if missing_vals:
                    added = self._ensure_extern_vars_from_bindgen_allowlist(missing_names=missing_vals)
                    if added:
                        success = self._compile_project()
                        if success:
                            logger.info(
                                f"[Step2.56] {func_name}: bindgen extern vars/const 补全 {added} 项后通过编译"
                            )
            except Exception as e:
                logger.debug(f"[Step2.56] {func_name}: extern vars/const 补全失败（已忽略）: {e}")
        
        if not success and rollback_on_failure:
            self._restore_function_from_backup(func_name, func_info, backup_content, target_file)
        
        return success
    
    def _auto_add_missing_dependencies(self, code: str) -> str:
        """
        自动检测代码中使用的外部 crate 并添加到 Cargo.toml
        
        目前支持检测的 crate:
        - libc: 用于 C 类型和函数
        - openssl: 用于加密功能
        - rand: 用于随机数生成
        - serde/serde_json: 用于序列化
        - tokio: 用于异步运行时
        - regex: 用于正则表达式
        - log: 用于日志
        - lazy_static: 用于静态变量
        - once_cell: 用于单次初始化
        - nix: 用于 Unix API
        """
        # 常见 crate 的检测模式和依赖声明
        crate_patterns = {
            'libc': ('use libc', 'libc::'),
            'openssl': ('use openssl', 'openssl::'),
            'rand': ('use rand', 'rand::'),
            'serde': ('use serde', 'serde::'),
            'serde_json': ('use serde_json', 'serde_json::'),
            'tokio': ('use tokio', 'tokio::'),
            'regex': ('use regex', 'regex::'),
            'log': ('use log', 'log::'),
            'lazy_static': ('lazy_static!', 'use lazy_static'),
            'once_cell': ('use once_cell', 'once_cell::'),
            'nix': ('use nix', 'nix::'),
        }
        
        # crate 的版本声明
        crate_versions = {
            'libc': 'libc = "0.2"',
            'openssl': 'openssl = "0.10"',
            'rand': 'rand = "0.8"',
            'serde': 'serde = { version = "1.0", features = ["derive"] }',
            'serde_json': 'serde_json = "1.0"',
            'tokio': 'tokio = { version = "1", features = ["full"] }',
            'regex': 'regex = "1"',
            'log': 'log = "0.4"',
            'lazy_static': 'lazy_static = "1.4"',
            'once_cell': 'once_cell = "1.18"',
            'nix': 'nix = "0.27"',
        }
        
        cargo_toml = self.work_dir / "Cargo.toml"
        if not cargo_toml.exists():
            return code
        
        try:
            cargo_content = cargo_toml.read_text(encoding='utf-8')
            modified = False
            
            for crate_name, patterns in crate_patterns.items():
                # 检查代码中是否使用了该 crate
                if any(pattern in code for pattern in patterns):
                    # 检查 Cargo.toml 中是否已有该依赖
                    if crate_name not in cargo_content:
                        version_decl = crate_versions.get(crate_name, f'{crate_name} = "*"')
                        
                        if '[dependencies]' in cargo_content:
                            cargo_content = cargo_content.replace(
                                '[dependencies]',
                                f'[dependencies]\n{version_decl}'
                            )
                        else:
                            cargo_content += f'\n[dependencies]\n{version_decl}\n'
                        
                        modified = True
                        logger.info(f"已自动添加 {crate_name} 依赖到 Cargo.toml")
            
            if modified:
                cargo_toml.write_text(cargo_content, encoding='utf-8')
                
        except Exception as e:
            logger.warning(f"自动添加依赖失败: {e}")
        
        return code
    
    def _postprocess_translated_code(self, code: str) -> str:
        """
        后处理翻译后的代码，移除常见问题
        """
        if not code:
            return code

        # 0. 修复 C 语法混入 Rust：(void)var; -> let _ = var;
        # 这是 LLM 常见错误：使用 C 的强制忽略返回值语法
        code = re.sub(r'\(void\)\s*(\w+)\s*;', r'let _ = \1;', code)
        # 同样处理 (void)(expr); 形式
        code = re.sub(r'\(void\)\s*\(([^)]+)\)\s*;', r'let _ = \1;', code)

        # 1. 移除循环类型别名
        invalid_type_patterns = [
            r'pub\s+type\s+(\w+)\s*=\s*\1\s*;',
            r'type\s+(\w+)\s*=\s*\1\s*;',
        ]
        for pattern in invalid_type_patterns:
            code = re.sub(pattern, '', code)
        
        # 2. （可选）保守兜底：修复 LLM 生成的无效函数体（把参数声明语法当作函数调用）
        #
        # 历史策略：直接将该函数体替换为 unimplemented!()，以“保证项目可编译”为第一优先级。
        # 但这会阻断“让 rustc 报语法错 -> 走增量修复”的路径，导致 still_unimplemented 增多。
        #
        # 现在默认关闭该兜底，改为：允许注入后由 rustc 产生语法错误，进入正常的增量修复流程。
        # 如需恢复旧行为，可设置：
        #   C2R_SANITIZE_INVALID_FUNCTION_BODIES=1
        try:
            sanitize_invalid = str(os.environ.get("C2R_SANITIZE_INVALID_FUNCTION_BODIES", "0")).lower() in (
                "1",
                "true",
                "yes",
            )
        except Exception:
            sanitize_invalid = False
        if sanitize_invalid:
            code = self._fix_invalid_function_bodies(code)
        
        # 3. 清理多余空行
        code = re.sub(r'\n{3,}', '\n\n', code)

        # 4. 修复 LLM 错误地将常量当作宏使用的问题
        # 例如: crate::globals::HDF_SUCCESS!() -> HDF_SUCCESS
        # 例如: crate::types::HDF_FAILURE!() -> HDF_FAILURE
        # 常见的被误用为宏的常量列表
        constants_mistaken_as_macros = [
            'HDF_SUCCESS', 'HDF_FAILURE', 'HDF_ERR_INVALID_PARAM', 'HDF_ERR_MALLOC_FAIL',
            'HDF_ERR_TIMEOUT', 'HDF_ERR_NOT_SUPPORT', 'HDF_ERR_IO', 'HDF_ERR_DEVICE_BUSY',
            'SOFTBUS_OK', 'SOFTBUS_ERR', 'SOFTBUS_INVALID_PARAM', 'SOFTBUS_NOT_IMPLEMENT',
            'SOFTBUS_MEM_ERR', 'SOFTBUS_MALLOC_ERR', 'SOFTBUS_PERMISSION_DENIED', 'SOFTBUS_NETWORK_ERR',
            'LOS_OK', 'LOS_NOK', 'LOS_ERRNO_BASE',
            'TRUE', 'FALSE',
            'EINVAL', 'ENOMEM', 'ENOENT', 'EEXIST', 'EAGAIN', 'ETIMEDOUT', 'EBUSY', 'EPERM', 'EFAULT',
            'ENODEV', 'ENOSYS', 'ERANGE', 'ENOTCONN',
        ]
        for const in constants_mistaken_as_macros:
            # 匹配 crate::globals::CONST!() 或 crate::types::CONST!()
            pattern = rf'crate::(globals|types)::{const}!\s*\(\s*\)'
            code = re.sub(pattern, const, code)
            # 也匹配简单的 CONST!() 形式（不带路径）
            pattern = rf'\b{const}!\s*\(\s*\)'
            code = re.sub(pattern, const, code)

        return code.strip()

    def _rewrite_safe_globals_in_code(self, code: str) -> str:
        """
        将对“RustMap safe globals”的直接变量访问重写为 accessor API 调用。

        依赖: `self.safe_globals`（来自 src/globals_accessors.json）

        目标：
        - 把 `g = expr;` -> `crate::globals::set_g(expr);`
        - 把 `g += expr;` -> `crate::globals::with_g(|v| { *v += expr; });`
        - 把表达式里的 `g` -> `crate::globals::get_g()`

        注意：
        - 这是“编译优先”的确定性重写，并不保证 100% 语义等价（尤其是 &g 指针逃逸场景）。
        - 如果检测到 Rust 代码中出现 `let g`/参数 `g:` 之类的局部绑定，会跳过对该名字的重写，
          避免把局部变量错误替换成全局 accessor。
        """
        if not code or not self.safe_globals:
            return code

        safe_map = self.safe_globals
        safe_names = set(safe_map.keys())
        if not safe_names:
            return code

        # 如果在当前片段中出现了同名局部绑定（shadowing），跳过该名字的重写
        shadowed: Set[str] = set()
        try:
            for n in safe_names:
                # Local bindings:
                # - `let g = ...` / `let mut g = ...`
                # - function params / patterns like `g: Type`
                # NOTE: Avoid treating `&mut g` as a local binding (it is an expression).
                pat = re.compile(rf"\blet\s+(?:mut\s+)?{re.escape(n)}\b|\b{re.escape(n)}\s*:", re.MULTILINE)
                if pat.search(code):
                    shadowed.add(n)
        except Exception:
            shadowed = set()

        def _tokenize(src: str):
            tokens = []
            i = 0
            n = len(src)
            # multi-char ops/punct (longest-first)
            multi = [
                ">>=",
                "<<=",
                "==",
                "!=",
                "<=",
                ">=",
                "+=",
                "-=",
                "*=",
                "/=",
                "%=",
                "&=",
                "|=",
                "^=",
                "&&",
                "||",
                "::",
                "->",
                "=>",
                "<<",
                ">>",
            ]

            while i < n:
                ch = src[i]
                # line comment
                if ch == "/" and i + 1 < n and src[i + 1] == "/":
                    j = src.find("\n", i)
                    if j == -1:
                        j = n
                    tokens.append(("comment", src[i:j], None))
                    i = j
                    continue
                # block comment (non-nested)
                if ch == "/" and i + 1 < n and src[i + 1] == "*":
                    j = src.find("*/", i + 2)
                    if j == -1:
                        j = n
                    else:
                        j += 2
                    tokens.append(("comment", src[i:j], None))
                    i = j
                    continue
                # string literal
                if ch == '"':
                    j = i + 1
                    esc = False
                    while j < n:
                        c = src[j]
                        if esc:
                            esc = False
                            j += 1
                            continue
                        if c == "\\":
                            esc = True
                            j += 1
                            continue
                        if c == '"':
                            j += 1
                            break
                        j += 1
                    tokens.append(("string", src[i:j], None))
                    i = j
                    continue
                # char literal
                if ch == "'":
                    j = i + 1
                    esc = False
                    while j < n:
                        c = src[j]
                        if esc:
                            esc = False
                            j += 1
                            continue
                        if c == "\\":
                            esc = True
                            j += 1
                            continue
                        if c == "'":
                            j += 1
                            break
                        j += 1
                    tokens.append(("string", src[i:j], None))
                    i = j
                    continue
                # whitespace
                if ch.isspace():
                    j = i + 1
                    while j < n and src[j].isspace():
                        j += 1
                    tokens.append(("ws", src[i:j], None))
                    i = j
                    continue
                # identifier
                if ch.isalpha() or ch == "_":
                    j = i + 1
                    while j < n and (src[j].isalnum() or src[j] == "_"):
                        j += 1
                    ident = src[i:j]
                    tokens.append(("ident", src[i:j], ident))
                    i = j
                    continue

                # multi-char operators
                matched = None
                for op in multi:
                    if src.startswith(op, i):
                        matched = op
                        break
                if matched:
                    tokens.append(("op", matched, matched))
                    i += len(matched)
                    continue

                # single-char token
                tokens.append(("op", ch, ch))
                i += 1

            return tokens

        def _next_meaningful(tokens, idx):
            j = idx + 1
            while j < len(tokens):
                if tokens[j][0] in {"ws", "comment"}:
                    j += 1
                    continue
                return j
            return None

        def _prev_meaningful(tokens, idx):
            j = idx - 1
            while j >= 0:
                if tokens[j][0] in {"ws", "comment"}:
                    j -= 1
                    continue
                return j
            return None

        toks = _tokenize(code)
        out_parts: List[str] = []
        i = 0
        assign_ops = {"=", "+=", "-=", "*=", "/=", "%=", "|=", "&=", "^=", "<<=", ">>="}

        while i < len(toks):
            kind, text, val = toks[i]
            if kind == "ident" and val in safe_names and val not in shadowed:
                # Avoid rewriting when already used as path segment: crate::globals::xxx
                prev_i = _prev_meaningful(toks, i)
                if prev_i is not None and toks[prev_i][1] in {".", "::"}:
                    out_parts.append(text)
                    i += 1
                    continue
                # If used via deref `*g` (common for pointer globals), treat as a read (`get_*()`),
                # even when the next token is `=` (i.e. `*g = ...` should become `*get_g() = ...`).
                if prev_i is not None and toks[prev_i][1] == "*":
                    get_fn = safe_map[val].get("get") or f"get_{val}"
                    out_parts.append(f"crate::globals::{get_fn}()")
                    i += 1
                    continue

                # Assignment / compound assignment handling: <name> <op>= <rhs> ;
                op_i = _next_meaningful(toks, i)
                if op_i is not None and toks[op_i][0] == "op" and toks[op_i][1] in assign_ops:
                    # Skip `let name = ...`
                    if prev_i is not None and toks[prev_i][0] == "ident" and toks[prev_i][2] == "let":
                        out_parts.append(text)
                        i += 1
                        continue
                    op = toks[op_i][1]
                    # Find statement-ending ';' not inside (), [], {} started after the operator.
                    depth_paren = 0
                    depth_brack = 0
                    depth_brace = 0
                    j = op_i + 1
                    end_i = None
                    while j < len(toks):
                        k, t, v = toks[j]
                        if k == "string" or k == "comment":
                            j += 1
                            continue
                        if t == "(":
                            depth_paren += 1
                        elif t == ")":
                            depth_paren = max(0, depth_paren - 1)
                        elif t == "[":
                            depth_brack += 1
                        elif t == "]":
                            depth_brack = max(0, depth_brack - 1)
                        elif t == "{":
                            depth_brace += 1
                        elif t == "}":
                            depth_brace = max(0, depth_brace - 1)
                        elif t == ";" and depth_paren == 0 and depth_brack == 0 and depth_brace == 0:
                            end_i = j
                            break
                        j += 1

                    if end_i is None:
                        # No semicolon; fall back to read-rewrite
                        get_fn = safe_map[val].get("get") or f"get_{val}"
                        out_parts.append(f"crate::globals::{get_fn}()")
                        i += 1
                        continue

                    rhs_text = "".join(t[1] for t in toks[op_i + 1 : end_i]).strip()
                    info = safe_map.get(val) or {}
                    set_fn = info.get("set") or f"set_{val}"
                    with_fn = info.get("with") or f"with_{val}"

                    if op == "=":
                        out_parts.append(f"crate::globals::{set_fn}({rhs_text});")
                    else:
                        out_parts.append(
                            f"crate::globals::{with_fn}(|v| {{ *v {op} {rhs_text}; }});"
                        )

                    i = end_i + 1
                    continue

                # Read access: replace identifier with get_*()
                # Skip function call: name(...)
                nxt = _next_meaningful(toks, i)
                if nxt is not None and toks[nxt][1] == "(":
                    out_parts.append(text)
                    i += 1
                    continue
                get_fn = safe_map[val].get("get") or f"get_{val}"
                out_parts.append(f"crate::globals::{get_fn}()")
                i += 1
                continue

            out_parts.append(text)
            i += 1

        rewritten = "".join(out_parts)
        return rewritten
    
    def _fix_invalid_function_bodies(self, code: str) -> str:
        """
        修复 LLM 生成的无效函数体，例如把参数声明语法当作函数调用
        """
        # 匹配函数定义，然后检查函数体是否有效
        fn_pattern = r'((?:pub\s+)?(?:unsafe\s+)?(?:extern\s+"C"\s+)?fn\s+\w+\s*\([^)]*\)\s*(?:->\s*[^{]+)?\s*)\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
        
        def check_and_fix_body(match):
            fn_signature = match.group(1)
            fn_body = match.group(2)
            
            # 检查函数体是否包含无效的参数声明语法
            # 模式：identifier: Type, 或 identifier: &Type
            invalid_body_pattern = r'^\s*\w+\s*\(\s*(?:\w+\s*:\s*[^,)]+,?\s*)+\)\s*$'
            
            # 去除空白后检查
            body_stripped = fn_body.strip()
            
            # 检查是否是无效的函数调用（参数带类型声明）
            # 例如: check_persist_policy(\n  token_id: u32,\n  policy: &Vec<PolicyInfo>,\n)
            lines = body_stripped.split('\n')
            has_invalid_syntax = False
            
            # 首先检查是否包含 extern "C" { fn ... } FFI 声明（这是合法语法）
            has_ffi_declaration = 'extern "C"' in body_stripped or "extern 'C'" in body_stripped
            
            for line in lines:
                line = line.strip()
                if not line or line.startswith('//'):
                    continue
                # 跳过 extern "C" 声明内部的内容
                if has_ffi_declaration and (line.startswith('extern') or line.startswith('fn ') or line == '}' or line == '{'):
                    continue
                # 检查是否有 "param: Type" 模式（而不是 "param: value" 或正常表达式）
                # 但要排除 extern "C" 声明内的函数签名
                if re.match(r'^[\w]+\s*:\s*[&*]?(?:mut\s+)?[\w:<>]+\s*,?\s*$', line):
                    # 确保这不是在 FFI 声明的上下文中
                    if not has_ffi_declaration:
                        has_invalid_syntax = True
                        break
                # 检查是否是只有函数调用但参数有类型声明
                if re.match(r'^\w+\s*\($', line):
                    # 排除 extern "C" { 和 unsafe { 等合法块开始
                    if not has_ffi_declaration:
                        has_invalid_syntax = True
                        break
            
            if has_invalid_syntax:
                # 尝试找到函数名
                fn_name_match = re.search(r'fn\s+(\w+)', fn_signature)
                fn_name = fn_name_match.group(1) if fn_name_match else 'unknown'
                return f'{fn_signature}{{\n        // TODO: Fix invalid function body for {fn_name}\n        unimplemented!()\n    }}'
            
            return match.group(0)
        
        # 使用更简单的方法：逐行检查函数体
        lines = code.split('\n')
        result_lines = []
        in_function = False
        function_start = -1
        brace_count = 0
        current_function_lines = []
        
        for i, line in enumerate(lines):
            if not in_function:
                # 检查是否是函数开始
                if re.search(r'(?:pub\s+)?(?:unsafe\s+)?(?:extern\s+"C"\s+)?fn\s+\w+', line) and '{' in line:
                    in_function = True
                    function_start = i
                    brace_count = line.count('{') - line.count('}')
                    current_function_lines = [line]
                    if brace_count == 0:
                        # 单行函数
                        result_lines.append(line)
                        in_function = False
                        current_function_lines = []
                else:
                    result_lines.append(line)
            else:
                current_function_lines.append(line)
                brace_count += line.count('{') - line.count('}')
                
                if brace_count <= 0:
                    # 函数结束，检查函数体
                    fn_code = '\n'.join(current_function_lines)
                    
                    # 检查是否有无效的参数声明语法
                    # 只检查函数体部分（去掉签名）
                    body_match = re.search(r'\{([\s\S]*)\}$', fn_code)
                    if body_match:
                        body = body_match.group(1)
                        # 检查是否有 "param: Type," 模式在函数调用中
                        # 但要排除 extern "C" { fn xxx(...); } 这种合法的 FFI 声明
                        invalid_call = re.search(r'\w+\s*\(\s*\n?\s*\w+\s*:\s*[&*]?[\w<>:]+', body)
                        
                        # 排除 FFI 声明的误判
                        is_ffi_declaration = False
                        if invalid_call:
                            # 检查匹配位置之前是否有 "extern" 和 "fn" 关键字
                            match_pos = invalid_call.start()
                            context_before = body[:match_pos]
                            # 如果在 extern "C" { 和 fn 之间，说明是合法的 FFI 声明
                            if re.search(r'extern\s+"C"\s*\{[^}]*fn\s*$', context_before, re.DOTALL):
                                is_ffi_declaration = True
                            # 也检查是否是函数定义（fn xxx(param: Type)）
                            elif re.search(r'fn\s+\w*$', context_before):
                                is_ffi_declaration = True
                        
                        if invalid_call and not is_ffi_declaration:
                            # 找到无效语法，替换整个函数体
                            fn_sig_match = re.match(r'([\s\S]*?)\{', fn_code)
                            if fn_sig_match:
                                fn_sig = fn_sig_match.group(1).strip()
                                indent = len(current_function_lines[0]) - len(current_function_lines[0].lstrip())
                                indent_str = ' ' * indent
                                fixed_fn = f'{indent_str}{fn_sig}{{\n{indent_str}        // TODO: Invalid function body fixed\n{indent_str}        unimplemented!()\n{indent_str}    }}'
                                result_lines.append(fixed_fn)
                            else:
                                result_lines.extend(current_function_lines)
                        else:
                            result_lines.extend(current_function_lines)
                    else:
                        result_lines.extend(current_function_lines)
                    
                    in_function = False
                    current_function_lines = []
        
        # 处理未结束的函数（不应该发生）
        if current_function_lines:
            result_lines.extend(current_function_lines)
        
        return '\n'.join(result_lines)
    
    def _validate_rust_syntax(self, code: str) -> bool:
        """
        使用 tree-sitter 验证 Rust 代码语法是否有效
        
        检测注入代码后是否破坏了文件的语法结构。
        主要检测：
        1. 语法错误节点 (ERROR)
        2. 不匹配的括号
        3. 孤立的代码片段（不在函数体内）
        
        返回:
            True: 语法基本正确，可以继续编译
            False: 有严重语法错误，应该回退
        """
        # tree-sitter-rust 不可用时：退化为“花括号平衡”检查，避免直接把所有注入都回退。
        if rust_parser is None or RUST_LANGUAGE is None:
            try:
                from auto_test_rust import _iter_rust_code_chars  # 轻量 lexer
                depth = 0
                for _, ch in _iter_rust_code_chars(code or "", 0):
                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth < 0:
                            return False
                return depth == 0
            except Exception:
                # 最保守兜底：只做朴素计数
                return (code.count("{") == code.count("}"))

        try:
            tree = rust_parser.parse(bytes(code, 'utf-8'))
            
            # 统计 ERROR 节点数量
            error_count = 0
            total_nodes = 0
            
            def count_errors(node):
                nonlocal error_count, total_nodes
                total_nodes += 1
                if node.type == "ERROR":
                    error_count += 1
                    # 记录错误位置用于调试
                    logger.debug(f"语法错误节点: 行 {node.start_point[0]+1}, 列 {node.start_point[1]}")
                for child in node.children:
                    count_errors(child)
            
            count_errors(tree.root_node)
            
            # 如果 ERROR 节点超过一定比例，认为语法严重损坏
            # 阈值：超过 5% 或超过 3 个 ERROR 节点
            error_rate = error_count / max(total_nodes, 1)
            
            if error_count > 3 or error_rate > 0.05:
                logger.warning(f"语法验证失败: {error_count} 个错误节点 ({error_rate:.1%})")
                return False
            
            # 额外检查：检测孤立的代码片段（不在函数体内的语句）
            # 这通常是注入错误导致的
            root = tree.root_node
            for child in root.children:
                # 允许的顶层节点类型
                allowed_types = {
                    'function_item', 'struct_item', 'enum_item', 'type_item',
                    'const_item', 'static_item', 'impl_item', 'trait_item',
                    'use_declaration', 'mod_item', 'extern_crate_declaration',
                    'attribute_item', 'inner_attribute_item', 'macro_invocation',
                    'macro_definition', 'line_comment', 'block_comment',
                    'extern_modifier', 'ERROR'  # ERROR 已经计数过了
                }
                if child.type not in allowed_types:
                    # 发现不允许的顶层节点（可能是孤立的代码片段）
                    snippet = code[child.start_byte:min(child.end_byte, child.start_byte+50)]
                    logger.warning(f"发现孤立的代码片段 (类型: {child.type}): {snippet}...")
                    return False
            
            return True
            
        except Exception as e:
            logger.warning(f"语法验证异常: {e}")
            # 解析异常也视为语法错误
            return False
    
    def _filter_valid_imports(self, imports: list) -> list:
        """
        过滤掉无效的导入语句
        注意：不再硬编码过滤特定项目的导入模式
        """
        if not imports:
            return imports
        
        # 只过滤明显语法错误的导入，不针对特定项目
        valid_imports = []
        for imp in imports:
            # 基本的语法检查：必须以 use 开头，以 ; 结尾
            if imp.strip().startswith('use ') and imp.strip().endswith(';'):
                valid_imports.append(imp)
            elif imp.strip():  # 非空但格式不对，跳过
                continue
        
        return valid_imports if valid_imports else imports
    
    def _compile_project(self) -> bool:
        """
        编译项目
        
        使用 RUSTFLAGS 抑制 LLM 生成代码常见的无害警告：
        - unused_imports: LLM 倾向于添加"以防万一"的 use 语句
        - dead_code: 增量翻译中间状态，很多函数尚未被调用
        
        这避免了"假阳性"失败，专注于真正的编译错误。
        """
        try:
            # 设置 RUSTFLAGS 抑制无害警告
            env = os.environ.copy()
            env["RUSTFLAGS"] = "-A unused_imports -A dead_code -A unused_variables -A unused_mut"
            
            cmd = ["cargo", "check"]
            try:
                from cargo_utils import with_cargo_jobs
                cmd = with_cargo_jobs(cmd)
            except Exception:
                pass

            result = subprocess.run(
                cmd,
                cwd=self.work_dir,
                capture_output=True,
                text=True,
                timeout=60,
                env=env
            )
            # 检查是否有实际的编译错误（不仅仅是警告）
            output = result.stdout + result.stderr
            # Cache last cargo output to avoid re-running `cargo check` just to extract errors.
            # `_get_compile_error()` will consume this output when available.
            try:
                self._last_cargo_check_output = output
                self._last_cargo_check_returncode = int(result.returncode)
            except Exception:
                pass
            
            # 如果退出码为0，肯定成功
            if result.returncode == 0:
                return True
            
            # 如果退出码非0，检查是否有实际的错误（error[E...]）
            has_error = bool(re.search(r'error\[E\d+\]:', output))
            has_finished = "Finished" in output
            
            # 如果有 Finished 且没有错误，说明只有警告，视为成功
            if has_finished and not has_error:
                return True
            
            # 否则视为失败
            return False
        except Exception as e:
            logger.error(f"编译失败: {e}")
            return False
    
    def _get_compile_error(self) -> str:
        """获取编译错误信息（过滤掉无害警告）"""
        try:
            # Prefer using the cached output from the most recent `_compile_project()` call.
            output = getattr(self, "_last_cargo_check_output", "") or ""
            if not output:
                # Fallback: run cargo check to obtain fresh errors.
                env = os.environ.copy()
                env["RUSTFLAGS"] = "-A unused_imports -A dead_code -A unused_variables -A unused_mut"

                cmd = ["cargo", "check"]
                try:
                    from cargo_utils import with_cargo_jobs

                    cmd = with_cargo_jobs(cmd)
                except Exception:
                    pass

                result = subprocess.run(
                    cmd,
                    cwd=self.work_dir,
                    capture_output=True,
                    text=True,
                    timeout=60,
                    env=env,
                )
                output = result.stdout + result.stderr
                try:
                    self._last_cargo_check_output = output
                    self._last_cargo_check_returncode = int(result.returncode)
                except Exception:
                    pass
            
            # 过滤掉警告，只保留错误
            lines = output.split('\n')
            error_lines = []
            in_error_block = False
            
            for line in lines:
                # 检测错误开始（包含无错误码的语法错误：`error: ...`）
                if re.search(r'error\[E\d+\]:', line) or re.match(r'\s*error:', line):
                    in_error_block = True
                    error_lines.append(line)
                # 检测错误块中的相关行（以 --> 或 ^ 开头，或包含 help:）
                elif in_error_block and (line.strip().startswith('-->') or 
                                         line.strip().startswith('^') or 
                                         'help:' in line or
                                         line.strip().startswith('|')):
                    error_lines.append(line)
                # 检测错误块结束（空行或新的错误/警告开始）
                elif in_error_block:
                    if line.strip() == '':
                        in_error_block = False
                    elif re.search(r'(error(\[E\d+\])?:|warning:)', line):
                        if re.search(r'error\[E\d+\]:', line) or re.match(r'\s*error:', line):
                            error_lines.append(line)
                        else:
                            in_error_block = False
            
            error_output = '\n'.join(error_lines)
            
            # 如果没有找到错误，返回完整输出（可能包含警告）
            if not error_output.strip():
                error_output = output
            
            return error_output[:5000]  # 增加长度限制，保留更多错误信息
        except Exception as e:
            return str(e)
    
    def _format_attempt_history(self, attempt: Dict, is_first: bool = False) -> str:
        """格式化单次修复尝试的历史信息"""
        prev_num = attempt.get("attempt_num", "?")
        prev_error = attempt.get("error_msg", "")
        prev_code_before = attempt.get("code_before", "")
        prev_code_after = attempt.get("code_after", "")  # 修复后的代码（如果有）
        
        # Truncate long history items to avoid prompt blow-ups; configurable via env.
        # Default: no truncation (0) to maximize repair context.
        try:
            max_chars = int(os.environ.get("C2R_REPAIR_HISTORY_ITEM_MAX_CHARS", "0") or "0")
        except Exception:
            max_chars = 0
        if max_chars < 0:
            max_chars = 0

        def _maybe_trunc(s: str) -> str:
            if not s:
                return ""
            if max_chars == 0:
                return s
            return s[:max_chars] if len(s) > max_chars else s

        prev_error_truncated = _maybe_trunc(prev_error)
        prev_code_before_truncated = _maybe_trunc(prev_code_before)
        prev_code_after_truncated = _maybe_trunc(prev_code_after)
        
        result = f"\n### Attempt {prev_num}"
        if is_first:
            result += " (Initial Translation)"
        result += ":\n"
        
        result += f"**Code tried (that failed to compile):**\n```rust\n{prev_code_before_truncated}\n```\n"
        if prev_code_after:
            result += f"**Code after repair (still failed):**\n```rust\n{prev_code_after_truncated}\n```\n"
        result += f"**Error encountered:**\n```\n{prev_error_truncated}\n```\n"
        
        return result
    
    def _repair_function(
        self,
        func_info: FunctionInfo,
        current_code: str,
        error_msg: str,
        skeleton_context: str,
        attempt_num: int = 1,
        error_history: List[str] = None,
        repair_attempts_history: List[Dict] = None
    ) -> Optional[str]:
        """修复函数 - 先尝试规则修复，失败再使用 LLM

        支持两种模式（通过 C2R_REPAIR_USE_TRANSLATION_PROMPT 环境变量控制）：
        1. 复用翻译 prompt（默认，推荐）：使用与翻译相同的完整 prompt，追加错误信息和历史记录
           - 好处：保持翻译规则一致性，LLM 不会被不同 prompt 风格困惑
        2. 专用修复 prompt：使用简化的修复 prompt
           - 好处：prompt 更短，但可能丢失重要翻译规则
        """
        from generate.generation import generation

        # 检查是否复用翻译 prompt（默认启用）
        use_translation_prompt = (os.environ.get("C2R_REPAIR_USE_TRANSLATION_PROMPT", "1") or "1").strip().lower() in (
            "1", "true", "yes", "y", "on"
        )

        # C2R: repair 阶段的骨架上下文统计和截断
        # 记录原始长度
        original_skeleton_len = len(skeleton_context) if skeleton_context else 0
        logger.info(f"[Repair统计] 函数: {func_info.name}, skeleton_context: {original_skeleton_len} 字符, use_translation_prompt: {use_translation_prompt}")

        try:
            if skeleton_context and len(skeleton_context) > MAX_SKELETON_CONTEXT_CHARS:
                logger.warning(
                    f"[Repair截断] 骨架上下文过长: {len(skeleton_context)} 字符 (~{len(skeleton_context)//4} tokens)"
                )
                logger.warning(
                    f"[Repair截断] 将截断到: {MAX_SKELETON_CONTEXT_CHARS} 字符 (~{MAX_SKELETON_CONTEXT_CHARS//4} tokens)"
                )
                skeleton_context = smart_truncate_context(skeleton_context, MAX_SKELETON_CONTEXT_CHARS)
        except Exception as e:
            logger.debug(f"repair 阶段骨架上下文截断失败: {e}")
        
        # ============================================================
        # 阶段 1: 尝试规则修复（快速、确定性）
        # ============================================================
        if RULE_FIX_AVAILABLE:
            fixed_code, fixer_name = try_rule_fix(current_code, error_msg)
            if fixed_code:
                logger.info(f"[{func_info.name}] 规则修复成功: {fixer_name}")
                return fixed_code
        
        # ============================================================
        # 阶段 2: 规则修复失败，使用 LLM 修复
        # ============================================================
        
        # 检查签名是否是 trait 方法或析构函数，转换为独立函数签名
        target_signature = func_info.rust_signature
        
        # 检查是否是 Drop trait 实现（析构函数）
        is_drop_impl = target_signature.startswith("impl Drop for")
        
        if is_drop_impl:
            # 析构函数保持原样，不需要转换
            pass
        elif "&self" in target_signature:
            # trait 方法，转换为独立函数签名
            target_signature = target_signature.replace("&self, ", "").replace("&self,", "").replace("(&self)", "()")
            if not target_signature.startswith("pub "):
                target_signature = "pub " + target_signature
        
        # 增强错误分析：解析 Rust 错误代码并提供针对性建议
        error_hints = self._analyze_compilation_errors(error_msg)
        error_hints_str = "\n".join(error_hints) if error_hints else "- Analyze the error message carefully."
        
        # 构建详细的修复历史提示（包含之前尝试的具体错误和代码）
        history_hint = ""
        if repair_attempts_history and len(repair_attempts_history) > 0:
            total_attempts = len(repair_attempts_history)
            history_hint = f"\n## Previous Repair Attempts\nYou have tried {total_attempts} time(s). Here are the attempts with their errors and code:\n"
            
            # Default: include ALL attempts to maximize repair context.
            include_all = (os.environ.get("C2R_REPAIR_HISTORY_INCLUDE_ALL", "1") or "1").strip().lower() in (
                "1",
                "true",
                "yes",
                "y",
                "on",
            )

            # Default remains compact to avoid prompt blow-ups; enable full history via env var.
            if (not include_all) and total_attempts > 3:
                # Show attempt 1 + last 2 attempts.
                first_attempt = repair_attempts_history[0]
                history_hint += self._format_attempt_history(first_attempt, is_first=True)

                skipped_count = total_attempts - 3
                history_hint += f"\n... (skipped {skipped_count} intermediate attempts; set C2R_REPAIR_HISTORY_INCLUDE_ALL=1 to include all) ...\n"

                for prev_attempt in repair_attempts_history[-2:]:
                    history_hint += self._format_attempt_history(prev_attempt)
            else:
                # Show all attempts.
                for i, prev_attempt in enumerate(repair_attempts_history):
                    history_hint += self._format_attempt_history(prev_attempt, is_first=(i == 0))
            
            # 只提供历史记录作为参考，不强制要求改变方法
            # 让 LLM 基于更多信息做出判断，而不是被迫尝试"不同的方法"
            history_hint += f"\n**Note:** The above shows your previous repair attempts and the errors they produced.\n"
            history_hint += "- Use this information to understand what approaches did NOT work\n"
            history_hint += "- Focus on fixing the SPECIFIC error in the current code\n"
            history_hint += "- If a type conversion is needed (e.g., usize vs u64), add explicit cast"

        # 构建不透明类型说明（通用）
        opaque_type_info = ""
        if self.opaque_types:
            opaque_list = ", ".join(sorted(self.opaque_types))
            opaque_type_info = f"""
## OPAQUE TYPES (DO NOT access their fields)
The following types are opaque (e.g., `type X = c_void`):
  {opaque_list}

If you need to return values for these types:
  - For `-> *mut OpaqueType`: return `std::ptr::null_mut()`
  - For `-> i32` or integer: return `0`
  - For `-> bool` or boolean: return `0`
  - For `()`: just return
"""

        # 检查是否是 Drop trait 实现（析构函数）
        is_drop_impl = target_signature.startswith("impl Drop for")
        is_impl_method = "&self" in target_signature or "&mut self" in target_signature
        is_constructor = "fn new" in target_signature.lower() or "-> Self" in target_signature
        
        # 通用语法规则提示
        syntax_rules = """
OUTPUT FORMAT (STRICT):
- Output ONLY raw Rust code
- DO NOT use ``` fences
- DO NOT add explanations, markdown, or extra text

SYNTAX RULES (CRITICAL - READ CAREFULLY):
- Function CALL syntax: `func(arg1, arg2)` - just pass variable names
- Function DEFINITION syntax: `fn func(arg1: Type, arg2: Type)` - include types
- WRONG call: `check_policy(token_id: u32, policy: &Vec<T>)` 
- CORRECT call: `check_policy(token_id, policy)`
- Function pointer fields: `Option<unsafe extern \"C\" fn(...) -> ...>` MUST unwrap:
  `if let Some(f) = cb { unsafe { f(args) } }`
- In trait impl blocks: do NOT add `pub` visibility modifier
- If trait defines `fn foo(&self, ...)`, your impl MUST have `fn foo(&self, ...)`

NAMESPACE CONTRACT (CRITICAL):
- `crate::types` contains types + constants only (bindgen truth). Do NOT put functions there.
- External C APIs are declared in `crate::compat` (generated by bindgen allowlist). Prefer calling `Foo(...)`
  (the skeleton imports `use crate::compat::*;`). Do NOT write `crate::types::Foo(...)`.
- Global variables are in `crate::globals` (skeleton imports `use crate::globals::*;`).
- Use `::core::ffi::c_void` for void pointers (NOT `crate::types::c_void`).

TYPE CHECKING (MUST COMPILE):
- Every call MUST satisfy the provided Rust signatures; add explicit casts/conversions when needed.
- Treat `size_t/ssize_t` as `usize/isize`; cast before arithmetic/comparisons to avoid E0308.
"""

        # External symbol policy: extern decls should be generated deterministically (bindgen allowlist),
        # not guessed by the LLM during repair.
        local_ffi_rule = """
6. EXTERNAL SYMBOLS (E0425 / E0422):
   - DO NOT add ad-hoc `extern \"C\" { ... }` declarations (neither inside the function nor at module level).
   - External C APIs should already be declared in `compat.rs` (generated by bindgen allowlist).
   - If a symbol is missing, it indicates a TU / decl-generation gap; do not “patch” it with guessed signatures."""

        if is_drop_impl:
            system_prompt = f"""You are a Rust expert fixing compilation errors.

CRITICAL RULES:
1. Generate a COMPLETE `impl Drop` block with `fn drop(&mut self)` method
2. OPAQUE TYPES: Types defined as `type X = c_void` are opaque - you CANNOT access their fields
3. If the error is about accessing struct fields on an opaque type, just do nothing or log
4. Keep the implementation EXACTLY as specified: `{target_signature}`
5. For destructors:
   - If C/C++ destructor is empty/default, use: `fn drop(&mut self) {{}}`
   - If it releases resources, translate appropriately
{local_ffi_rule}
{syntax_rules}"""
        elif is_impl_method:
            system_prompt = f"""You are a Rust expert fixing compilation errors.

CRITICAL RULES:
1. Generate a method inside an impl block (WITH &self or &mut self parameter as required)
2. OPAQUE TYPES: Types defined as `type X = c_void` are opaque - you CANNOT access their fields
3. If the error is about accessing struct fields on an opaque type, return a safe default value
4. Keep the method signature EXACTLY as specified: `{target_signature}`
5. This is an impl method, so &self parameter is REQUIRED
6. Implement the logic DIRECTLY in the method body, do NOT call a non-existent standalone function
{local_ffi_rule}
{syntax_rules}"""
        elif is_constructor:
            system_prompt = f"""You are a Rust expert fixing compilation errors.

CRITICAL RULES:
1. Generate a constructor function that returns Self
2. Initialize all struct fields appropriately
3. If some fields cannot be initialized, use Default::default() or safe defaults
4. Keep the signature EXACTLY as specified: `{target_signature}`
{local_ffi_rule}
{syntax_rules}"""
        else:
            system_prompt = f"""You are a Rust expert fixing compilation errors.

CRITICAL RULES:
1. Generate a STANDALONE function with `pub fn` prefix (NOT a trait method)
2. OPAQUE TYPES: Types defined as `type X = c_void` are opaque - you CANNOT access their fields
3. If the error is about accessing struct fields on an opaque type, return a safe default value
4. Keep the function signature EXACTLY as specified: `{target_signature}`
6. DO NOT add &self parameter
{local_ffi_rule}
        {syntax_rules}"""

        # 从依赖文件提取“被调函数签名”（C-level constness），减少调用点 *mut/*const 错误
        callee_signature_hints = self._build_dependency_signature_hints(func_info)
        inline_helper_hints = self._build_preprocessed_inline_helper_hints(func_info)
        callee_rust_signature_hints = self._build_called_rust_signature_hints(func_info)
        internal_callee_paths_hints = self._build_internal_callee_module_path_hints(func_info)
        typed_constants_hints = self._build_typed_constants_hints(func_info)
        c_field_access_hints = self._build_c_field_access_hints(func_info)
        try:
            c_pointer_contract_hints = self._build_c_pointer_contract_hints(func_info)
        except Exception as e:
            logger.debug(f"C pointer contract hints failed (repair): {e}")
            c_pointer_contract_hints = ""

        # ============================================================
        # 根据 use_translation_prompt 选择 prompt 构建方式
        # ============================================================
        if use_translation_prompt:
            # 复用翻译 prompt：使用与 _translate_function 相同的完整 system_prompt
            # 好处：保持翻译规则一致性，LLM 不会被不同 prompt 风格困惑
            system_prompt = f"""You are a C/C++ to Rust translator. Generate code that COMPILES. Unsafe is OK.

OUTPUT FORMAT (STRICT):
- Output ONLY raw Rust code
- DO NOT use ``` fences
- DO NOT add explanations, markdown, or extra text

## ⚠️ CRITICAL: AVOID THESE MISTAKES ⚠️
1. **Option<fn> MUST unwrap**: `if let Some(f) = cb {{ unsafe {{ f(args) }} }}`
2. **ALL ptr ops need unsafe**: `unsafe {{ *ptr = val; (*ptr).field; }}`
3. **OPAQUE types (c_void) have NO fields** - return defaults instead
4. **Pointer cast**: `ptr as *mut T` (NOT `ptr.cast_mut()` - doesn't exist!)
5. **DO NOT invent externs for macros/inline/external APIs**: external symbols are declared in `compat.rs` (bindgen allowlist); macro/static inline are not stable link symbols.
6. **DO NOT emit C macro names / header-only inline helper names**: input is a preprocessed `.i` (macros expanded). If you see a macro-like call, output the expanded expression or implement a local Rust helper (NOT an extern).

## ❌ NON-EXISTENT / WRONG SYNTAX - NEVER USE ❌
- `ptr.cast_mut()` → Use `ptr as *mut T`
- `ptr.cast_const()` → Use `ptr as *const T`
- `(void)var;` → This is C! Use `let _ = var;` in Rust
- `NULL` → Use `std::ptr::null()` or `std::ptr::null_mut()`

## RULES
1. Match TARGET SIGNATURE exactly (pub fn / impl method / Drop)
2. Use only skeleton types and existing `compat.rs` extern decls; DO NOT add ad-hoc `extern "C"` blocks
3. OPAQUE types: can't access fields, use ptr arithmetic or defaults
4. Wrap ALL ptr derefs in `unsafe {{ }}`
5. Function ptrs: `Option<fn>` - must unwrap before call

## NAMESPACE CONTRACT (CRITICAL)
- `crate::types` contains types + constants only (bindgen truth). Do NOT put functions there.
- External C APIs are declared in `crate::compat` and are already imported by the skeleton (`use crate::compat::*;`):
  call them as `Foo(...)` or `crate::compat::Foo(...)` (NOT `crate::types::Foo`).
- Global variables are in `crate::globals` and are imported by the skeleton (`use crate::globals::*;`):
  use `NAME` or `crate::globals::NAME` (NOT `crate::types::NAME`).
- Internal translated functions live in per-source modules (e.g. `crate::src_xxx::Func`):
  use the provided "Internal Callees" mapping when calling them.
- Use `::core::ffi::c_void` for void pointers (NOT `crate::types::c_void`).

## TYPE CHECKING (MUST COMPILE)
- Every call MUST satisfy the provided Rust signatures (Target signature + "Called Rust APIs").
- If types mismatch, add explicit casts/conversions (e.g. `as i32`, `as u32`, `as usize`, pointer casts).
- Treat `size_t/ssize_t` as `usize/isize` in Rust; cast before arithmetic/comparisons to avoid E0308.

## SIZE_T / USIZE CONVERSION (CRITICAL - E0308 FIX)
- `crate::types::size_t` is defined as `u64` by bindgen (on 64-bit systems)
- But `libc` functions (malloc, realloc, memcpy, etc.) expect `usize`
- ALWAYS cast when calling libc: `libc::malloc(size as usize)`
- ALWAYS cast strlen result: `(libc::strlen(ptr) + 1) as usize` for malloc
- When assigning to size_t field: `(value as u64)` or `(value as crate::types::size_t)`

## RAW POINTER RULES (CRITICAL - E0599 FIX)
- Raw pointers (`*mut T`, `*const T`) have NO methods like `.len()`, `.is_empty()`, etc.
- For C string length: use `libc::strlen(ptr)` (NOT `ptr.len()`)
- For array/slice length: you must track length separately or use `std::slice::from_raw_parts(ptr, len)`
- WRONG: `ptr.len()` on `*mut i8` → ERROR: no method named `len`
- CORRECT: `unsafe {{ libc::strlen(ptr) }}`

## INTERNAL FUNCTION CALLS (CRITICAL - E0425 FIX)
- When calling functions defined in OTHER source modules (e.g., `src_tree.rs`), you MUST use the full path
- Check the "Internal Callees" section in the prompt for the correct module path
- WRONG: `ZopfliCalculateEntropy(...)` → ERROR: E0425 cannot find function
- CORRECT: `crate::src_tree::ZopfliCalculateEntropy(...)`

## OUTPUT SIZE
- Keep the function body compact. Do NOT generate hundreds of unused temporary variables; every `let` must be used.

## QUICK REFERENCE
- sizeof: `std::mem::size_of::<T>()`
- malloc: `libc::malloc(n) as *mut T`
- memcpy: `std::ptr::copy_nonoverlapping(src, dst, n)`
- memset: `std::ptr::write_bytes(ptr, val, n)`
- NULL check: `ptr.is_null()`
- C enums are represented as integers in `crate::types` (bindgen constified); use integer comparisons / constants.
- Ignore value: `let _ = expr;` (NOT `(void)expr;`)

GOAL: Make it compile. Idiomatic refactoring happens in Phase 3.2."""

            # 构建复用翻译 prompt 的 user_prompt
            # 使用翻译的结构，但追加当前错误代码、编译错误、历史记录
            user_prompt = f"""Translate the following C/C++ function to Rust.
{opaque_type_info}
## Target Rust Function Signature (MUST use this exact signature)
```rust
{target_signature}
```

## C/C++ Source Code to Translate
NOTE: The source below may be a slice from the build's preprocessed `.i` TU (macros expanded, inline bodies visible).
```cpp
{func_info.c_code}
```
{callee_signature_hints}
{inline_helper_hints}
{callee_rust_signature_hints}
{internal_callee_paths_hints}
{typed_constants_hints}
{c_field_access_hints}
{c_pointer_contract_hints}

## Skeleton Context (available types and functions)
```rust
{skeleton_context}
```

## Output Requirements
1. Output EXACTLY ONE Rust item: the complete target function definition
2. The function signature MUST match: `{target_signature}`
3. Do NOT output any extra items (no helper fns/types/impl/mod/use)
4. Keep output short: no long comments, no markdown fences/explanations

## ⚠️ REPAIR CONTEXT (Previous Attempt Had Errors)
The following code was generated in a previous attempt but has compilation errors:
```rust
{current_code}
```

### Compilation Error
```
{error_msg}
```

### Error Analysis
{error_hints_str}
{history_hint}

**Focus on fixing the specific errors above while keeping the translation correct.**
"""

        else:
            # 使用原有的专用修复 prompt
            user_prompt = f"""Fix the following Rust code that has compilation errors.

## Original C/C++ Code (for reference)
```cpp
{func_info.c_code}
```

## Current Code (has errors)
```rust
{current_code}
```

## Compilation Error
```
{error_msg}
```

## Error Analysis
{error_hints_str}
{history_hint}

## IMPORTANT: Missing Symbols (E0425 / E0422)
If you see errors like "cannot find function `RAND_bytes`" / missing const / missing type:
- Do NOT add guessed `extern "C"` declarations or no-op macro stubs to “make it compile”.
- These should be fixed by build-context/TU selection and regenerating `compat.rs` via bindgen allowlist.
- Treat them as deterministic closure gaps (report), and focus fixes on other local compile errors.

## REQUIRED Function Signature (MUST use exactly this)
```rust
{target_signature}
```
{callee_signature_hints}
{inline_helper_hints}
{callee_rust_signature_hints}
{internal_callee_paths_hints}
{typed_constants_hints}
{c_field_access_hints}
{c_pointer_contract_hints}
{opaque_type_info}
## Available Context (from skeleton)
```rust
{skeleton_context}
```

## Type Information Notes
- Output must be EXACTLY ONE Rust item (the target function/method).
- Do NOT define new helper types/aliases/imports/modules/extern blocks in the output.
- If a type is missing (E0412), treat it as an upstream types/TU truth gap; return safe defaults or keep TODOs, but do NOT invent new type definitions here.
- Use `::core::ffi::c_void` for void pointers; external C APIs come from `compat.rs`; globals come from `globals.rs`.

## Output Requirements
{self._get_repair_output_requirements(is_drop_impl, is_impl_method, is_constructor, target_signature)}
"""

        # 构建消息列表
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # 保存提示词
        try:
            from save_llm_prompts import save_llm_prompt, save_llm_prompt_text
            metadata = {
                "target_signature": target_signature,
                "file_name": func_info.file_name,
                "attempt_num": attempt_num,
                "error_msg_preview": error_msg[:500],
                "opaque_types": list(self.opaque_types) if self.opaque_types else [],
                "callee_signature_hints_length": len(callee_signature_hints) if callee_signature_hints else 0,
                "inline_helper_hints_length": len(inline_helper_hints) if inline_helper_hints else 0,
                "callee_rust_signature_hints_length": len(callee_rust_signature_hints) if callee_rust_signature_hints else 0,
                "internal_callee_paths_hints_length": len(internal_callee_paths_hints) if internal_callee_paths_hints else 0,
                "typed_constants_hints_length": len(typed_constants_hints) if typed_constants_hints else 0,
                "c_field_access_hints_length": len(c_field_access_hints) if c_field_access_hints else 0,
                "c_pointer_contract_hints_length": len(c_pointer_contract_hints) if c_pointer_contract_hints else 0,
                "error_history_count": len(error_history) if error_history else 0
            }
            save_llm_prompt(
                messages=messages,
                project_name=self.project_name,
                llm_name=self.llm_name,
                task_type="incremental_repair",
                function_name=f"{func_info.name}_attempt_{attempt_num}",
                metadata=metadata,
                output_dir=self.llm_prompts_dir
            )
            save_llm_prompt_text(
                messages=messages,
                project_name=self.project_name,
                llm_name=self.llm_name,
                task_type="incremental_repair",
                function_name=f"{func_info.name}_attempt_{attempt_num}",
                metadata=metadata,
                output_dir=self.llm_prompts_dir
            )
        except Exception as e:
            logger.warning(f"保存提示词失败: {e}")

        try:
            # 使用信号量控制 vLLM 并发请求数
            _vllm_semaphore.acquire()
            try:
                response = generation(messages)
            finally:
                _vllm_semaphore.release()

            if isinstance(response, dict):
                response = response.get("content", "")

            print("✓")
            return self._extract_rust_code(response)
        except Exception as e:
            print("✗")
            logger.error(f"修复失败: {e}")
            return None

    def _extract_function_block(self, source_content: str, func_info: FunctionInfo) -> Optional[str]:
        """
        从给定源码中提取与目标函数匹配的代码块。
        优先使用完整签名，失败时退化到函数名匹配。
        """
        if not source_content:
            return None

        # 关键修复：使用与注入相同的 lexer 逻辑提取函数块（避免字符串里有 '{' 导致截断）
        try:
            from auto_test_rust import extract_function_block as _extract_block
        except Exception:
            _extract_block = None

        signature = (func_info.rust_signature or "").strip()
        actual_func_name = func_info.name
        if not actual_func_name and signature:
            fn_match = re.search(r'fn\s+(\w+)', signature)
            if fn_match:
                actual_func_name = fn_match.group(1)

        if _extract_block and actual_func_name:
            if signature:
                block = _extract_block(source_content, signature, actual_func_name)
                if block:
                    return block
            # fallback: name-only
            return _extract_block(source_content, f"fn {actual_func_name}", actual_func_name)

        return None

    def _restore_function_from_backup(
        self,
        func_name: str,
        func_info: FunctionInfo,
        backup_content: str,
        target_file: Path
    ):
        """
        编译失败后，只回滚当前函数：优先使用备份文本恢复，失败时退化到骨架。
        """
        from auto_test_rust import find_and_replace_function_signature

        try:
            current_content = target_file.read_text(encoding='utf-8', errors='ignore')
        except FileNotFoundError:
            logger.warning(f"无法回滚函数 {func_name}，目标文件不存在: {target_file}")
            return

        backup_block = self._extract_function_block(backup_content, func_info)
        if backup_block:
            match_key = func_info.rust_signature or (f"fn {func_info.name}" if func_info.name else None)
            new_content: Optional[str] = None
            success = False
            if match_key:
                new_content, success = find_and_replace_function_signature(current_content, match_key, backup_block)
            if not success and func_info.name:
                new_content, success = find_and_replace_function_signature(
                    current_content,
                    f"fn {func_info.name}",
                    backup_block
                )
            if success and new_content is not None:
                with open(target_file, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                logger.info(f"已回退函数 {func_name} 到注入前版本（备份恢复）")
                return

        logger.warning(f"无法从备份精确恢复函数 {func_name}，改用骨架占位符")
        self._rollback_function(func_name, func_info)

    def _rollback_function(self, func_name: str, func_info: FunctionInfo):
        """
        回退函数到 skeleton 的原始占位符状态（unimplemented!() 等）
        
        当所有修复尝试都失败后调用此方法，将函数恢复到骨架中的占位符版本
        注意：不再整个文件回退，而是只回退失败的函数
        """
        from auto_test_rust import find_and_replace_function_signature
        
        # 尝试两种文件名格式：src_filename.rs 和 filename.rs
        rs_filename_with_prefix = "src_" + func_info.file_name + ".rs"
        rs_filename_without_prefix = func_info.file_name + ".rs"
        
        # 确定 target_file
        target_file_with_prefix = self.work_dir / "src" / rs_filename_with_prefix
        target_file_without_prefix = self.work_dir / "src" / rs_filename_without_prefix
        
        if target_file_with_prefix.exists():
            target_file = target_file_with_prefix
            rs_filename = rs_filename_with_prefix
        elif target_file_without_prefix.exists():
            target_file = target_file_without_prefix
            rs_filename = rs_filename_without_prefix
        else:
            logger.warning(f"回退失败：目标文件不存在 {target_file_with_prefix} 或 {target_file_without_prefix}")
            return
        
        # 确定 skeleton_file
        skeleton_file = self.skeleton_dir / "src" / rs_filename
        
        if not skeleton_file.exists():
            logger.warning(f"回退失败：骨架文件不存在 {skeleton_file}")
            return
        
        try:
            # 读取骨架文件（包含占位符版本）
            skeleton_content = skeleton_file.read_text(encoding='utf-8', errors='ignore')
            
            # 读取当前工作文件
            current_content = target_file.read_text(encoding='utf-8', errors='ignore')
            
            # 先清理可能存在的重复签名（修复重复 pub extern "C" 问题）
            duplicate_pattern = r'(pub\s+extern\s+"C"\s+){2,}'
            current_content = re.sub(duplicate_pattern, r'pub extern "C" ', current_content)
            
            # 从骨架中提取该函数的占位符版本
            placeholder_func = self._extract_function_block(skeleton_content, func_info)
            
            # 如果找到占位符版本，进行替换
            if placeholder_func:
                new_content, success = find_and_replace_function_signature(
                    current_content, func_info.rust_signature or f"fn {func_info.name}", placeholder_func
                )
                if success:
                    with open(target_file, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                    logger.info(f"已回退函数 {func_name} 到占位符版本")
                    return
                else:
                    # 尝试使用函数名进行替换
                    actual_func_name = func_info.name
                    if not actual_func_name and func_info.rust_signature:
                        fn_match = re.search(r'fn\s+(\w+)', func_info.rust_signature)
                        if fn_match:
                            actual_func_name = fn_match.group(1)
                    
                    if actual_func_name:
                        new_content, success = find_and_replace_function_signature(
                            current_content, f"fn {actual_func_name}", placeholder_func
                        )
                        if success:
                            with open(target_file, 'w', encoding='utf-8') as f:
                                f.write(new_content)
                            logger.info(f"已回退函数 {func_name} 到占位符版本（使用函数名匹配）")
                            return
            
            # 如果所有策略都失败，记录警告但不覆盖整个文件
            # 这样可以保留其他已成功翻译的函数
            logger.warning(f"无法单独回退函数 {func_name}，保留当前状态（不覆盖整个文件）")
            # 注意：不再用骨架文件覆盖整个文件，以避免破坏其他已翻译的函数
            
        except Exception as e:
            logger.error(f"回退函数 {func_name} 失败: {e}")
    
    def _rollback_function_safe(self, func_name: str, func_info: FunctionInfo) -> bool:
        """
        安全回退函数到占位符版本，返回是否成功
        
        与 _rollback_function 的区别：
        1. 返回布尔值表示是否成功
        2. 回退后验证文件语法是否正确
        3. 如果验证失败，尝试强制恢复
        """
        from auto_test_rust import find_and_replace_function_signature
        
        # 尝试两种文件名格式：src_xxx.rs 和 xxx.rs（向后兼容）
        rs_filename_with_prefix = "src_" + func_info.file_name + ".rs"
        rs_filename = func_info.file_name + ".rs"
        target_file = self.work_dir / "src" / rs_filename_with_prefix
        skeleton_file = self.skeleton_dir / "src" / rs_filename_with_prefix
        
        if not target_file.exists():
            target_file = self.work_dir / "src" / rs_filename
        if not skeleton_file.exists():
            skeleton_file = self.skeleton_dir / "src" / rs_filename
        
        if not target_file.exists() or not skeleton_file.exists():
            logger.warning(f"回退失败：文件不存在 {target_file} 或 {skeleton_file}")
            return False
        
        try:
            # 读取骨架文件（包含占位符版本）
            skeleton_content = skeleton_file.read_text(encoding='utf-8', errors='ignore')
            
            # 读取当前工作文件
            current_content = target_file.read_text(encoding='utf-8', errors='ignore')
            backup_content = current_content  # 备份当前内容
            
            # 先清理可能存在的重复签名
            duplicate_pattern = r'(pub\s+extern\s+"C"\s+){2,}'
            current_content = re.sub(duplicate_pattern, r'pub extern "C" ', current_content)
            
            # 从骨架中提取该函数的占位符版本
            placeholder_func = self._extract_function_block(skeleton_content, func_info)
            
            if not placeholder_func:
                logger.warning(f"无法从骨架提取函数 {func_name} 的占位符版本")
                return False
            
            # 尝试替换
            new_content, success = find_and_replace_function_signature(
                current_content, func_info.rust_signature or f"fn {func_info.name}", placeholder_func
            )
            
            if not success:
                # 尝试使用函数名匹配
                actual_func_name = func_info.name
                if not actual_func_name and func_info.rust_signature:
                    fn_match = re.search(r'fn\s+(\w+)', func_info.rust_signature)
                    if fn_match:
                        actual_func_name = fn_match.group(1)
                
                if actual_func_name:
                    new_content, success = find_and_replace_function_signature(
                        current_content, f"fn {actual_func_name}", placeholder_func
                    )
            
            if not success:
                logger.warning(f"无法单独回退函数 {func_name}")
                return False
            
            # 验证替换后的语法
            if not self._validate_rust_syntax(new_content):
                logger.warning(f"回退后文件语法无效，保留原内容: {func_name}")
                with open(target_file, 'w', encoding='utf-8') as f:
                    f.write(backup_content)
                return False
            
            # 写入回退后的内容
            with open(target_file, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            logger.info(f"已回退函数 {func_name} 到占位符版本")
            return True
            
        except Exception as e:
            logger.error(f"安全回退函数 {func_name} 失败: {e}")
            return False
    
    def _force_restore_skeleton_file(self, func_info: FunctionInfo):
        """
        强制恢复整个骨架文件
        
        当单个函数回退失败时，强制从骨架复制整个文件
        这会丢失该文件中其他已翻译的函数，但确保文件语法正确
        """
        # 尝试两种文件名格式：src_xxx.rs 和 xxx.rs（向后兼容）
        rs_filename_with_prefix = "src_" + func_info.file_name + ".rs"
        rs_filename = func_info.file_name + ".rs"
        target_file = self.work_dir / "src" / rs_filename_with_prefix
        skeleton_file = self.skeleton_dir / "src" / rs_filename_with_prefix
        
        if not target_file.exists():
            target_file = self.work_dir / "src" / rs_filename
        if not skeleton_file.exists():
            skeleton_file = self.skeleton_dir / "src" / rs_filename
        
        if not skeleton_file.exists():
            logger.error(f"骨架文件不存在: {skeleton_file}")
            return
        
        try:
            import shutil
            # 先备份当前文件（避免“恢复骨架”导致已翻译内容丢失）
            try:
                if target_file.exists():
                    backup_dir = self.manual_fix_root / "_forced_restore"
                    backup_dir.mkdir(parents=True, exist_ok=True)
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    backup_path = backup_dir / f"{target_file.name}.{ts}.bak.rs"
                    shutil.copy2(target_file, backup_path)
            except Exception as e:
                logger.debug(f"备份当前文件失败（继续强制恢复骨架）: {e}")
            shutil.copy2(skeleton_file, target_file)
            logger.warning(f"已强制恢复整个骨架文件: {rs_filename}")
        except Exception as e:
            logger.error(f"强制恢复骨架文件失败: {e}")

    def _save_manual_fix_artifact(
        self,
        func_name: str,
        func_info: FunctionInfo,
        rust_code: str,
        reason: str,
        error_msg: str = "",
    ) -> Optional[Path]:
        """
        将“未能自动注入/未能自动修复”的翻译结果落盘，供人工复检。
        - 不把内容塞进源码文件（避免源码膨胀），而是写到 repair_history/_manual_fix/
        - 返回保存的 Rust 代码路径（可能为 None）
        """
        try:
            entry_dir = self.manual_fix_root / func_name
            entry_dir.mkdir(parents=True, exist_ok=True)

            code_path = entry_dir / "translated_rust.rs"
            code_path.write_text(rust_code or "", encoding="utf-8", errors="ignore")

            meta = {
                "generated_at": datetime.now().isoformat(),
                "project": self.project_name,
                "llm": self.llm_name,
                "func_key": func_name,
                "c_func": getattr(func_info, "name", None),
                "rust_signature": getattr(func_info, "rust_signature", None),
                "rust_file": f"{getattr(func_info, 'file_name', '')}.rs" if getattr(func_info, "file_name", None) else None,
                "reason": reason,
                "error_truncated": (error_msg or "")[:4000],
                "artifact_dir": str(entry_dir),
                "rust_code_path": str(code_path),
            }

            (entry_dir / "meta.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2),
                encoding="utf-8",
                errors="ignore",
            )

            # 追加到 manifest（jsonl，便于并行/增量追加）
            with open(self.manual_fix_manifest_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(meta, ensure_ascii=False) + "\n")

            return code_path
        except Exception as e:
            logger.debug(f"保存 manual fix 工件失败: {e}")
            return None

    def _build_manual_fix_comment(
        self,
        func_name: str,
        func_info: FunctionInfo,
        reason: str,
        artifact_path: Optional[Path],
        error_msg: str = "",
    ) -> str:
        """
        生成“人工修复提示”注释。
        注意：必须是纯 Rust 注释（//），避免破坏语法。
        """
        c_first_line = ""
        try:
            if getattr(func_info, "c_code", None):
                c_first_line = str(func_info.c_code).splitlines()[0][:200]
        except Exception:
            c_first_line = ""

        parts: List[str] = [
            "// === C2R MANUAL FIX REQUIRED ===",
            f"// reason: {reason}",
            f"// func_key: {func_name}",
            f"// c_function: {getattr(func_info, 'name', '')}",
        ]
        if getattr(func_info, "file_name", None):
            parts.append(f"// rust_file: {getattr(func_info, 'file_name')}.rs")
        if getattr(func_info, "rust_signature", None):
            sig = str(func_info.rust_signature).replace("\n", " ")
            parts.append(f"// rust_signature: {sig[:220]}{'...' if len(sig) > 220 else ''}")
        if c_first_line:
            parts.append(f"// c_first_line: {c_first_line}")
        if artifact_path is not None:
            parts.append(f"// saved_translation: {artifact_path}")
        if error_msg:
            parts.append("// last_error_truncated:")
            for line in (error_msg.strip().splitlines()[:8] if error_msg else []):
                parts.append(f"//   {line[:240]}")
        parts.append("// =================================")
        return "\n".join(parts) + "\n"
    
    def _generate_failure_comment(self, func_name: str, func_info: FunctionInfo, error_history: List[str]) -> str:
        """
        当所有修复尝试都失败时，使用 LLM 生成注释解释函数内容，方便后续人工修复
        
        返回: 生成的注释字符串
        """
        from generate.generation import generation
        
        print(f"  生成失败注释...", end=" ", flush=True)
        
        # 构建精简的错误摘要
        error_summary = ""
        if error_history:
            # 只取最后一个错误的前500字符
            last_error = error_history[-1] if error_history else ""
            error_summary = f"Last compilation error (truncated): {last_error[:500]}"
        
        system_prompt = """You are a code documentation expert. Generate helpful comments for a function that failed to translate from C/C++ to Rust.

Your task:
1. Explain what the original C/C++ function does (based on the code)
2. Explain why the translation might have failed (based on the error)
3. Provide hints for manual implementation

Output ONLY the comment block in Rust format (using // or /* */)."""

        user_prompt = f"""Generate a helpful comment for this C/C++ function that failed to translate to Rust.

## Original C/C++ Code
```cpp
{func_info.c_code}
```

## Target Rust Signature
```rust
{func_info.rust_signature}
```

## {error_summary}

Generate a comment block explaining:
1. What this function does
2. Key logic and data structures used
3. Potential challenges for Rust translation
4. Hints for manual implementation

Output format (Rust comment):
```rust
// TODO: Manual implementation needed
// Function: <function_name>
// Purpose: <brief description>
// ...
```"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        try:
            # 使用信号量控制 vLLM 并发请求数
            _vllm_semaphore.acquire()
            try:
                response = generation(messages)
            finally:
                _vllm_semaphore.release()

            if isinstance(response, dict):
                response = response.get("content", "")

            # 提取注释内容
            comment = self._extract_rust_code(response)
            if comment:
                # 确保注释格式正确
                if not comment.strip().startswith("//") and not comment.strip().startswith("/*"):
                    comment = f"// TODO: Manual implementation needed for {func_name}\n" + \
                              f"// Original C/C++ function: {func_info.name}\n" + \
                              f"// {comment}"
                print("✓")
                return comment
            else:
                print("✗ (using default)")
                return self._generate_default_failure_comment(func_name, func_info)
        except Exception as e:
            print(f"✗ ({e})")
            logger.warning(f"生成失败注释失败: {e}")
            return self._generate_default_failure_comment(func_name, func_info)
    
    def _generate_default_failure_comment(self, func_name: str, func_info: FunctionInfo) -> str:
        """生成默认的失败注释"""
        return f"""// TODO: Manual implementation needed
// Function: {func_name}
// Original C/C++ function: {func_info.name}
// File: {func_info.file_name}.rs
// 
// The automatic translation failed after multiple repair attempts.
// Please review the original C/C++ code and implement manually.
//
// Original C/C++ signature (first line):
// {func_info.c_code.split(chr(10))[0] if func_info.c_code else 'N/A'}"""
    
    def _inject_failure_comment(self, func_name: str, func_info: FunctionInfo, comment: str):
        """
        将失败注释注入到骨架文件中对应函数的上方
        """
        # 尝试两种文件名格式：src_xxx.rs 和 xxx.rs（向后兼容）
        rs_filename_with_prefix = "src_" + func_info.file_name + ".rs"
        rs_filename = func_info.file_name + ".rs"
        target_file = self.work_dir / "src" / rs_filename_with_prefix
        
        if not target_file.exists():
            target_file = self.work_dir / "src" / rs_filename
            if not target_file.exists():
                logger.warning(f"无法注入注释，文件不存在: {target_file} 或 {self.work_dir / 'src' / rs_filename_with_prefix}")
                return
        
        try:
            with open(target_file, 'r', encoding='utf-8', errors='ignore') as f:
                code = f.read()

            # 防重复：同一 func_key 的注释只注入一次（便于增量重跑）
            marker = f"func_key: {func_name}"
            if marker in code and "C2R MANUAL FIX REQUIRED" in code:
                return
            
            # 先清理可能存在的重复签名（修复重复 pub extern "C" 问题）
            # 匹配模式: pub extern "C" pub extern "C" ... (重复多次)
            duplicate_pattern = r'(pub\s+extern\s+"C"\s+){2,}'
            code = re.sub(duplicate_pattern, r'pub extern "C" ', code)
            
            # 查找函数定义位置（更精确的匹配）
            # 匹配: pub extern "C" fn func_name 或 fn func_name
            func_pattern = re.compile(
                r'(pub\s+extern\s+"C"\s+)?fn\s+' + re.escape(func_info.name) + r'\s*[<(]',
                re.MULTILINE
            )
            
            match = func_pattern.search(code)
            if match:
                # 在函数定义前插入注释
                insert_pos = match.start()
                # 确保注释以换行结尾
                if not comment.endswith('\n'):
                    comment += '\n'
                new_code = code[:insert_pos] + comment + code[insert_pos:]
                
                with open(target_file, 'w', encoding='utf-8') as f:
                    f.write(new_code)
                
                logger.info(f"已为 {func_name} 注入失败注释")
            else:
                logger.debug(f"未找到函数定义，无法注入注释: {func_name}")
        except Exception as e:
            logger.warning(f"注入失败注释失败: {e}")
    
    def _fill_trait_impl_methods(self):
        """
        通用方法：自动填充 trait impl 方法
        
        如果对应的独立函数已翻译，让 trait 方法调用独立函数
        这样可以减少 unimplemented!() 的数量
        """
        if rust_parser is None or RUST_LANGUAGE is None:
            return
        src_dir = self.work_dir / "src"
        if not src_dir.exists():
            return
        
        for rs_file in src_dir.glob("*.rs"):
            try:
                with open(rs_file, 'r', encoding='utf-8', errors='ignore') as f:
                    code = f.read()
                
                tree = rust_parser.parse(bytes(code, 'utf-8'))
                
                # 查找所有 impl 块中的方法
                query = Query(RUST_LANGUAGE, """
                    (impl_item
                        (type_identifier) @trait_name
                        (declaration_list
                            (function_item
                                name: (identifier) @method_name
                                body: (block) @method_body
                            )
                        )
                    )
                """)
                
                captures = _query_captures_compat(query, tree.root_node)
                method_info = {}  # method_name -> (body_node, trait_name)
                
                for node, capture_name in captures:
                    if capture_name == 'method_name':
                        method_name = code[node.start_byte:node.end_byte]
                        # 查找对应的 body
                        method_node = node.parent
                        while method_node and method_node.type != 'function_item':
                            method_node = method_node.parent
                        
                        if method_node:
                            body_node = None
                            for child in method_node.children:
                                if child.type == 'block':
                                    body_node = child
                                    break
                            
                            if body_node:
                                body_text = code[body_node.start_byte:body_node.end_byte]
                                # 检查是否是 unimplemented!()
                                if 'unimplemented!' in body_text:
                                    method_info[method_name] = (body_node, method_node)
                
                # 查找对应的独立函数（一次只处理一个方法，避免偏移量问题）
                modified = False
                max_iterations = len(method_info)  # 最多处理所有方法
                iteration = 0
                
                while method_info and iteration < max_iterations:
                    iteration += 1
                    # 重新解析代码（每次修改后都需要重新解析）
                    try:
                        current_tree = rust_parser.parse(bytes(code, 'utf-8'))
                    except (AttributeError, TypeError, ValueError):
                        break

                    # 重新查找所有 unimplemented!() 的 trait 方法
                    impl_query = Query(RUST_LANGUAGE, """
                        (impl_item
                            (declaration_list
                                (function_item
                                    name: (identifier) @method_name
                                    body: (block) @method_body
                                )
                            )
                        )
                    """)
                    
                    impl_captures = _query_captures_compat(impl_query, current_tree.root_node)
                    current_methods = {}  # method_name -> body_node
                    
                    for node, capture_name in impl_captures:
                        if capture_name == 'method_name':
                            method_name = code[node.start_byte:node.end_byte]
                            # 查找对应的 body
                            method_func_node = node.parent
                            while method_func_node and method_func_node.type != 'function_item':
                                method_func_node = method_func_node.parent
                            
                            if method_func_node:
                                for child in method_func_node.children:
                                    if child.type == 'block':
                                        body_text = code[child.start_byte:child.end_byte]
                                        if 'unimplemented!' in body_text:
                                            current_methods[method_name] = child
                                        break
                    
                    if not current_methods:
                        break  # 没有更多需要处理的方法
                    
                    # 查找对应的独立函数
                    standalone_query = Query(RUST_LANGUAGE, """
                        (function_item
                            name: (identifier) @func_name
                        ) @func_def
                    """)
                    
                    standalone_captures = _query_captures_compat(standalone_query, current_tree.root_node)
                    found_standalone = False
                    
                    for method_name, body_node in current_methods.items():
                        for node, capture_name in standalone_captures:
                            if capture_name == 'func_name':
                                func_name_text = code[node.start_byte:node.end_byte]
                                if func_name_text == method_name:
                                    # 找到对应的独立函数，检查是否不在 impl 块中
                                    func_def_node = node.parent
                                    while func_def_node and func_def_node.type != 'function_item':
                                        func_def_node = func_def_node.parent
                                    
                                    if func_def_node:
                                        parent = func_def_node.parent
                                        if parent and parent.type != 'impl_item':
                                            # 找到了对应的独立函数
                                            # 提取方法参数（去掉 &self）
                                            method_func_node = body_node.parent
                                            while method_func_node and method_func_node.type != 'function_item':
                                                method_func_node = method_func_node.parent
                                            
                                            if method_func_node:
                                                params_node = None
                                                for child in method_func_node.children:
                                                    if child.type == 'parameters':
                                                        params_node = child
                                                        break
                                                
                                                if params_node:
                                                    params_text = code[params_node.start_byte:params_node.end_byte]
                                                    # 移除 &self 参数
                                                    params_text = re.sub(r'&self\s*,?\s*', '', params_text).strip()
                                                    if params_text.startswith('(') and params_text.endswith(')'):
                                                        params_text = params_text[1:-1]
                                                    
                                                    # 构建调用语句
                                                    call_args = params_text if params_text else ""
                                                    new_body = f" {{\n        {method_name}({call_args})\n    }}"
                                                    
                                                    # 替换 body
                                                    code = (code[:body_node.start_byte] + 
                                                           new_body + 
                                                           code[body_node.end_byte:])
                                                    modified = True
                                                    found_standalone = True
                                                    break
                        
                        if found_standalone:
                            break  # 一次只处理一个方法
                    
                    if not found_standalone:
                        break  # 没有找到对应的独立函数，停止处理
                
                if modified:
                    # 备份原文件
                    backup_code = code
                    # 重新解析（因为代码已修改）
                    try:
                        tree = rust_parser.parse(bytes(code, 'utf-8'))
                    except:
                        # 解析失败，跳过
                        continue
                    
                    with open(rs_file, 'w', encoding='utf-8') as f:
                        f.write(code)
                    
                    # 重新编译验证
                    if self._compile_project():
                        print(f"  ✓ 已填充 {rs_file.name} 中的 trait impl 方法")
                    else:
                        # 回退
                        with open(rs_file, 'w', encoding='utf-8') as f:
                            f.write(backup_code)
                            
            except Exception as e:
                logger.debug(f"填充 trait impl 方法失败 ({rs_file.name}): {e}")
    
    def _count_unimplemented_functions(self):
        """
        统计仍然是 unimplemented!() 占位符的函数数量
        
        这是"功能完整性"检查：
        - 编译成功不等于功能完整
        - 如果函数体仍是 unimplemented!()，说明翻译实际上失败了
        """
        unimplemented_count = 0
        src_dir = self.work_dir / "src"
        
        if not src_dir.exists():
            return
        
        for rs_file in src_dir.glob("*.rs"):
            try:
                content = rs_file.read_text(encoding='utf-8', errors='ignore')
                # 统计 unimplemented!() 的数量
                unimplemented_count += len(re.findall(r'unimplemented!\s*\(\s*\)', content))
            except Exception as e:
                logger.warning(f"读取文件 {rs_file} 失败: {e}")
        
        self.stats["still_unimplemented"] = unimplemented_count
        
        if unimplemented_count > 0:
            print(f"  ⚠️ 功能完整性检查: {unimplemented_count} 个函数仍是 unimplemented!() 占位符")
    
    def _save_translation_stats(self):
        """保存翻译统计信息到 incremental_work 目录"""
        try:
            import json
            # 确保工作目录存在
            self.work_dir.mkdir(parents=True, exist_ok=True)
            stats_file = self.work_dir / "translation_stats.json"
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(self.stats, f, ensure_ascii=False, indent=2)
            print(f"  ✓ 统计信息已保存: {stats_file}")
            logger.info(f"统计信息已保存: {stats_file}")
        except Exception as e:
            print(f"  ⚠ 警告: 保存统计信息失败: {e}")
            logger.warning(f"保存统计信息失败: {e}")
    
    def _save_final_project(self):
        """保存最终项目"""
        print(f"  开始保存最终项目...")
        print(f"  工作目录: {self.work_dir}")
        print(f"  目标目录: {self.final_project_dir}")
        logger.info(f"开始保存最终项目: {self.work_dir} -> {self.final_project_dir}")
        
        # 1. 验证工作目录是否存在且不为空
        if not self.work_dir.exists():
            print(f"  ❌ 错误: 工作目录不存在: {self.work_dir}")
            logger.error(f"工作目录不存在: {self.work_dir}")
            return False
        
        # 检查工作目录是否为空
        src_dir = self.work_dir / "src"
        if not src_dir.exists():
            print(f"  ❌ 错误: 工作目录中缺少 src/ 目录: {self.work_dir}")
            logger.error(f"工作目录中缺少 src/ 目录: {self.work_dir}")
            return False
        
        rs_files = list(src_dir.glob("*.rs"))
        if not rs_files:
            print(f"  ❌ 错误: 工作目录中没有 .rs 文件: {self.work_dir}")
            logger.error(f"工作目录中没有 .rs 文件: {self.work_dir}")
            return False
        
        print(f"  找到 {len(rs_files)} 个 .rs 文件")
        
        # 2. 确保目标目录的父目录存在
        self.final_project_dir.parent.mkdir(parents=True, exist_ok=True)
        print(f"  目标目录父目录: {self.final_project_dir.parent}")
        
        # 3. 使用临时目录进行安全的复制操作
        temp_final_dir = self.final_project_dir.parent / f"{self.final_project_dir.name}.tmp"
        print(f"  临时目录: {temp_final_dir}")
        
        try:
            # 先复制到临时目录
            if temp_final_dir.exists():
                print(f"  清理旧的临时目录...")
                shutil.rmtree(temp_final_dir)
            print(f"  复制工作目录到临时目录...")
            shutil.copytree(self.work_dir, temp_final_dir)
            
            # 4. 验证复制是否成功（检查关键文件）
            temp_src_dir = temp_final_dir / "src"
            temp_rs_files = list(temp_src_dir.glob("*.rs"))
            if not temp_src_dir.exists() or not temp_rs_files:
                print(f"  ❌ 错误: 复制后验证失败，临时目录中没有有效的 .rs 文件")
                logger.error(f"复制后验证失败: {temp_final_dir}")
                if temp_final_dir.exists():
                    shutil.rmtree(temp_final_dir)
                return False
            
            print(f"  复制验证成功: {len(temp_rs_files)} 个 .rs 文件")
            
            # 5. 如果复制成功，再删除旧目录并重命名
            if self.final_project_dir.exists():
                # 备份旧目录（可选，用于调试）
                backup_dir = self.final_project_dir.parent / f"{self.final_project_dir.name}.backup"
                if backup_dir.exists():
                    shutil.rmtree(backup_dir)
                print(f"  备份旧目录到: {backup_dir}")
                shutil.move(str(self.final_project_dir), str(backup_dir))
                logger.info(f"已备份旧目录到: {backup_dir}")
            
            # 将临时目录重命名为最终目录
            print(f"  将临时目录重命名为最终目录...")
            shutil.move(str(temp_final_dir), str(self.final_project_dir))
            print(f"  ✓ 目录重命名成功")
            
        except Exception as e:
            print(f"  ❌ 保存最终项目失败: {e}")
            logger.error(f"保存最终项目失败: {e}", exc_info=True)
            # 清理临时目录
            if temp_final_dir.exists():
                try:
                    shutil.rmtree(temp_final_dir)
                except:
                    pass
            return False
        
        # 6. 验证最终目录确实存在
        if not self.final_project_dir.exists():
            print(f"  ❌ 错误: 最终项目目录创建失败")
            logger.error(f"最终项目目录创建失败: {self.final_project_dir}")
            return False
        
        print(f"  ✓ 最终项目目录已创建: {self.final_project_dir}")
        
        # 7. 最终编译测试（在保存后的最终目录中编译）
        print(f"  开始最终编译测试...")
        original_work_dir = self.work_dir
        final_compile_error = ""
        final_rule_fixer = ""
        try:
            self.work_dir = self.final_project_dir
            success = self._compile_project()

            # 如果最终编译失败，尽量给出可解释的错误信息，并做一次确定性的规则兜底修复
            if not success:
                final_compile_error = self._get_compile_error()

                # Tier-0: 确定性规则修复（例如 E0005 常量模式导致 let 绑定失败）
                if RULE_FIX_AVAILABLE and final_compile_error:
                    # 最多尝试 3 轮，避免卡死
                    for _ in range(3):
                        # 从错误信息中提取第一个报错文件路径（相对 work_dir）
                        m = re.search(r'-->\s+([^\s:]+):\d+:\d+', final_compile_error)
                        if not m:
                            break
                        rel_path = m.group(1)
                        file_path = Path(rel_path)
                        if not file_path.is_absolute():
                            file_path = self.work_dir / rel_path
                        if not file_path.exists() or not file_path.is_file():
                            break

                        try:
                            code = file_path.read_text(encoding="utf-8", errors="ignore")
                        except Exception:
                            break

                        fixed, fixer_name = try_rule_fix(code, final_compile_error)
                        if not fixed:
                            break

                        try:
                            file_path.write_text(fixed, encoding="utf-8")
                            final_rule_fixer = fixer_name or ""
                            logger.info(f"最终编译规则修复成功: {fixer_name} ({rel_path})")
                        except Exception:
                            break

                        # 重新编译验证
                        if self._compile_project():
                            success = True
                            final_compile_error = ""
                            break

                        final_compile_error = self._get_compile_error()
        finally:
            self.work_dir = original_work_dir
        
        # 8. 保存结果文件
        try:
            result_file = self.final_project_dir / "build_result.txt"
            with open(result_file, 'w', encoding='utf-8') as f:
                f.write("Success\n" if success else "Fail\n")
                f.write(f"\n统计信息:\n")
                f.write(f"  总函数数: {self.stats['total']}\n")
                f.write(f"  翻译成功: {self.stats['translated']}\n")
                f.write(f"  编译通过: {self.stats['compiled']}\n")
                f.write(f"  修复成功: {self.stats['repaired']}\n")
                f.write(f"  最终失败: {self.stats['failed']}\n")
                if final_rule_fixer:
                    f.write(f"\n最终规则修复: {final_rule_fixer}\n")
                if (not success) and final_compile_error:
                    f.write("\n最终编译错误 (cargo check):\n")
                    f.write(final_compile_error)
                    if not final_compile_error.endswith("\n"):
                        f.write("\n")
                f.write(f"\n保存时间: {self.final_project_dir.stat().st_mtime}\n")
            print(f"  ✓ 结果文件已保存: {result_file}")
        except Exception as e:
            print(f"  ⚠ 警告: 保存结果文件失败: {e}")
            logger.warning(f"保存结果文件失败: {e}")
        
        print(f"  最终项目: {self.final_project_dir}")
        print(f"  编译结果: {'成功' if success else '失败'}")
        logger.info(f"最终项目保存完成: {self.final_project_dir}, 编译结果: {'成功' if success else '失败'}")
        
        return True


def main():
    """主函数"""
    # 强制行缓冲，确保日志实时输出
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
        
    if len(sys.argv) < 3:
        print("用法: python3 incremental_translate.py <project_name> <llm_name> [max_repair_attempts]")
        sys.exit(1)
    
    project_name = sys.argv[1]
    llm_name = sys.argv[2]
    max_repair_attempts = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    
    translator = IncrementalTranslator(
        project_name=project_name,
        llm_name=llm_name,
        max_repair_attempts=max_repair_attempts
    )
    
    result = translator.run()
    # result: 2=完全成功, 1=部分成功, 0=失败
    # 退出码:
    # - Truth-mode: 仅完全成功视为成功（保证输出无占位符/无跳过/无注入失败）
    # - Default: 完全成功或部分成功都返回 0（便于批量跑完后再汇总）
    if translator.truth_mode:
        # Some suites (e.g. OSS smoke runs) prefer "compile success = success" so the batch run can
        # finish and report pass rates, even if placeholders remain.
        strict_exit = (os.environ.get("C2R_TRUTH_STRICT_EXIT", "1") or "1").strip().lower() not in (
            "0",
            "false",
            "no",
        )
        sys.exit(0 if (result == 2 or (not strict_exit and result >= 1)) else 1)
    sys.exit(0 if result >= 1 else 1)


if __name__ == "__main__":
    main()
