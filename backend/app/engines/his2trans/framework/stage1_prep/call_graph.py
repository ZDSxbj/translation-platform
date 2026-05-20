#!/usr/bin/env python3
"""
函数调用关系提取模块

基于 Tree-sitter 提取 C/C++ 代码中的函数调用关系，支持：
1. 构建调用图（Call Graph）
2. 拓扑排序确定翻译顺序
3. 为 LLM 提供调用上下文
4. 支持并行分析

使用示例：
    from call_graph import CallGraphBuilder
    
    builder = CallGraphBuilder()
    builder.analyze_directory(Path("src/"))
    
    # 获取翻译顺序
    order = builder.get_translation_order()
    
    # 获取函数的调用上下文
    context = builder.get_context_for_function("main", depth=2)
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import threading

# Tree-sitter
try:
    from tree_sitter import Language, Parser
    import tree_sitter_cpp as tscpp
    
    # 兼容不同版本 API
    try:
        CPP_LANGUAGE = Language(tscpp.language(), "cpp")
    except TypeError:
        CPP_LANGUAGE = Language(tscpp.language())
    
    TREESITTER_AVAILABLE = True
except ImportError:
    TREESITTER_AVAILABLE = False
    CPP_LANGUAGE = None

logger = logging.getLogger(__name__)

# 源文件扩展名
SOURCE_EXTENSIONS = {'.c', '.cc', '.cpp', '.cxx', '.h', '.hh', '.hpp', '.hxx'}


def generate_function_uid(file_path: str, start_line: int, name: str, mangled_name: str = None) -> str:
    """
    生成唯一函数ID (function_uid)

    格式: rel_path:start_line:name
    如果有 mangled_name (C++ libclang 可以获取)，优先使用: rel_path:start_line:mangled_name

    这样可以区分：
    - 不同文件中的同名函数
    - 同一文件中不同位置的同名 static 函数
    - C++ 重载函数（通过 mangled_name）

    Args:
        file_path: 文件路径（绝对或相对）
        start_line: 函数起始行号
        name: 函数名
        mangled_name: C++ mangled name（可选，libclang 可以获取）

    Returns:
        唯一函数ID字符串
    """
    # 使用相对路径或文件名，避免绝对路径带来的环境依赖
    if file_path:
        # 尝试提取相对路径或文件名
        path = Path(file_path)
        # 如果路径过长，只保留文件名和最多2级父目录
        parts = path.parts
        if len(parts) > 3:
            rel_path = str(Path(*parts[-3:]))
        else:
            rel_path = str(path)
    else:
        rel_path = "unknown"

    # 优先使用 mangled_name（C++ 唯一标识）
    identifier = mangled_name if mangled_name else name

    return f"{rel_path}:{start_line}:{identifier}"


@dataclass
class FunctionInfo:
    """函数信息"""
    name: str
    file_path: str
    start_line: int
    end_line: int
    signature: str = ""
    body: str = ""
    callees: Set[str] = field(default_factory=set)  # 调用的函数 (uid)
    callers: Set[str] = field(default_factory=set)  # 被谁调用 (uid)
    global_vars: Set[str] = field(default_factory=set)  # 使用的全局变量
    type_refs: Set[str] = field(default_factory=set)  # 引用的类型
    uid: str = ""  # 唯一函数ID (file:line:name)
    mangled_name: str = ""  # C++ mangled name (libclang 可获取)

    def __post_init__(self):
        """初始化后自动生成 uid（如果未提供）"""
        if not self.uid:
            self.uid = generate_function_uid(
                self.file_path, self.start_line, self.name, self.mangled_name
            )


@dataclass
class CallGraphStats:
    """调用图统计信息"""
    total_functions: int = 0
    total_calls: int = 0
    total_files: int = 0
    max_depth: int = 0
    cyclic_deps: int = 0
    leaf_functions: int = 0  # 不调用其他函数的函数
    root_functions: int = 0  # 不被调用的函数


class CallGraphBuilder:
    """
    函数调用图构建器
    
    使用 Tree-sitter 解析 C/C++ 源码，提取函数定义和调用关系
    """
    
    def __init__(self, max_workers: int = 4):
        """
        初始化调用图构建器
        
        Args:
            max_workers: 并行分析的最大线程数
        """
        if not TREESITTER_AVAILABLE:
            raise RuntimeError("Tree-sitter 不可用，请安装 tree_sitter_cpp")
        
        self.max_workers = max_workers

        # 函数信息: {uid -> FunctionInfo}
        # 使用 uid (file:line:name) 作为主键，避免函数名冲突
        self.functions: Dict[str, FunctionInfo] = {}

        # 函数名索引: {name -> [uid...]}
        # 用于按名查找/模糊上下文匹配
        self.name_index: Dict[str, List[str]] = defaultdict(list)

        # 调用图: {caller_uid -> callee_uid集合}
        self.call_graph: Dict[str, Set[str]] = defaultdict(set)

        # 反向调用图: {callee_uid -> caller_uid集合}
        self.reverse_call_graph: Dict[str, Set[str]] = defaultdict(set)

        # 全局变量: {变量名 -> 定义位置}
        self.global_vars: Dict[str, str] = {}

        # 类型定义: {类型名 -> 定义位置}
        self.type_defs: Dict[str, str] = {}

        # 已分析的文件
        self.analyzed_files: Set[str] = set()

        # 线程锁
        self._lock = threading.Lock()

        # 翻译顺序缓存
        self._translation_order: Optional[List[str]] = None

        # 统计信息
        self.stats = CallGraphStats()
    
    def _create_parser(self) -> Parser:
        """创建线程局部的解析器"""
        parser = Parser()
        try:
            parser.set_language(CPP_LANGUAGE)
        except Exception:
            parser = Parser(CPP_LANGUAGE)
        return parser
    
    def analyze_file(self, file_path: Path) -> Dict[str, FunctionInfo]:
        """
        分析单个文件，提取函数定义和调用关系
        
        Args:
            file_path: 源文件路径
            
        Returns:
            该文件中的函数信息字典
        """
        file_path = Path(file_path)
        if not file_path.exists():
            logger.warning(f"文件不存在: {file_path}")
            return {}
        
        if file_path.suffix.lower() not in SOURCE_EXTENSIONS:
            return {}
        
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
        except Exception as e:
            logger.warning(f"读取文件失败 {file_path}: {e}")
            return {}
        
        return self._analyze_content(content, str(file_path))
    
    def _analyze_content(self, content: str, file_path: str) -> Dict[str, FunctionInfo]:
        """
        分析源码内容
        
        Args:
            content: 源码内容
            file_path: 文件路径（用于记录）
            
        Returns:
            函数信息字典
        """
        parser = self._create_parser()
        
        # 规范化换行符
        content = content.replace('\r\n', '\n').replace('\r', '\n')
        source_bytes = bytes(content, 'utf-8')

        try:
            tree = parser.parse(source_bytes)
        except Exception as e:
            logger.warning(f"解析文件失败 {file_path}: {e}")
            return {}

        # 存储当前文件的函数: {uid -> FunctionInfo}
        file_functions: Dict[str, FunctionInfo] = {}

        # uid -> name 的映射，用于比较调用关系
        uid_to_name: Dict[str, str] = {}

        # 当前正在分析的函数 uid（用于提取调用关系）
        current_function_uid: Optional[str] = None
        current_function_name: Optional[str] = None
        
        def extract_text(node) -> str:
            """安全地从节点提取文本"""
            try:
                return source_bytes[node.start_byte:node.end_byte].decode('utf-8', errors='ignore')
            except:
                return ""
        
        def find_function_name(node) -> Optional[str]:
            """递归查找函数名"""
            if node.type == "identifier":
                return extract_text(node)
            if node.type == "field_identifier":
                return extract_text(node)
            if node.type == "qualified_identifier":
                # C++ 限定名: ClassName::methodName
                parts = []
                for child in node.children:
                    if child.type in ("identifier", "type_identifier"):
                        parts.append(extract_text(child))
                return "::".join(parts) if parts else None
            
            # 递归查找
            for child in node.children:
                if child.type in ("function_declarator", "identifier", "field_identifier", 
                                  "qualified_identifier", "destructor_name"):
                    result = find_function_name(child)
                    if result:
                        return result
            return None
        
        def traverse(node):
            """遍历 AST"""
            nonlocal current_function_uid, current_function_name

            # 函数定义
            if node.type == "function_definition":
                func_name = find_function_name(node)
                if func_name:
                    # 提取签名（去掉函数体）
                    signature = ""
                    body = ""
                    for child in node.children:
                        if child.type == "compound_statement":
                            body = extract_text(child)
                        else:
                            signature += extract_text(child) + " "

                    func_info = FunctionInfo(
                        name=func_name,
                        file_path=file_path,
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        signature=signature.strip(),
                        body=body
                    )
                    # 使用 uid 作为键
                    file_functions[func_info.uid] = func_info
                    uid_to_name[func_info.uid] = func_name

                    # 设置当前函数，用于提取调用关系
                    old_uid = current_function_uid
                    old_name = current_function_name
                    current_function_uid = func_info.uid
                    current_function_name = func_name

                    # 遍历函数体
                    for child in node.children:
                        if child.type == "compound_statement":
                            traverse(child)

                    current_function_uid = old_uid
                    current_function_name = old_name
                    return

            # 函数调用
            elif node.type == "call_expression":
                if current_function_uid:
                    callee = self._extract_callee_name(node, source_bytes)
                    # 比较被调用函数名和当前函数名（避免自调用记录）
                    if callee and callee != current_function_name:
                        if current_function_uid in file_functions:
                            # 暂时存储 callee 名字，后续在更新全局数据时解析为 uid
                            file_functions[current_function_uid].callees.add(callee)
            
            # 全局变量声明（顶层 declaration）
            elif node.type == "declaration" and node.parent and node.parent.type == "translation_unit":
                for child in node.children:
                    if child.type == "init_declarator":
                        for sub in child.children:
                            if sub.type == "identifier":
                                var_name = extract_text(sub)
                                with self._lock:
                                    self.global_vars[var_name] = file_path
            
            # 类型定义
            elif node.type in ("struct_specifier", "enum_specifier", "union_specifier"):
                for child in node.children:
                    if child.type == "type_identifier":
                        type_name = extract_text(child)
                        with self._lock:
                            self.type_defs[type_name] = file_path
            
            elif node.type == "type_definition":
                # typedef
                for child in node.children:
                    if child.type == "type_identifier":
                        type_name = extract_text(child)
                        with self._lock:
                            self.type_defs[type_name] = file_path
            
            # 递归遍历子节点
            for child in node.children:
                traverse(child)

        traverse(tree.root_node)

        # 更新全局数据
        with self._lock:
            self.analyzed_files.add(file_path)

            # 第一步：添加所有函数到 functions 和 name_index
            for uid, func_info in file_functions.items():
                self.functions[uid] = func_info
                # 更新函数名索引
                if func_info.name not in self.name_index:
                    self.name_index[func_info.name] = []
                if uid not in self.name_index[func_info.name]:
                    self.name_index[func_info.name].append(uid)

            # 第二步：处理调用关系（callee 名字 -> callee uid）
            for caller_uid, func_info in file_functions.items():
                # 将 callees 从 “名字集合” 升级为 “uid 集合”
                resolved_callees: Set[str] = set()
                for callee_name in func_info.callees:
                    # 尝试解析 callee_name 为 uid
                    callee_uids = self._resolve_callee_uid(callee_name, file_path)
                    for callee_uid in callee_uids:
                        self.call_graph[caller_uid].add(callee_uid)
                        self.reverse_call_graph[callee_uid].add(caller_uid)
                        resolved_callees.add(callee_uid)
                func_info.callees = resolved_callees

            # 清除翻译顺序缓存
            self._translation_order = None

        return file_functions

    def _resolve_callee_uid(self, callee_name: str, caller_file: str) -> List[str]:
        """
        解析被调用函数名到 uid

        策略：
        1. 优先查找同一文件中的同名函数
        2. 如果找不到，查找 name_index 中的所有匹配
        3. 如果都找不到，生成一个占位 uid（external:0:name）

        Args:
            callee_name: 被调用函数名
            caller_file: 调用者所在文件

        Returns:
            可能的 callee uid 列表
        """
        # 检查 name_index 中是否有匹配
        if callee_name in self.name_index:
            matching_uids = self.name_index[callee_name]

            # 优先返回同文件的函数
            same_file_uids = [
                uid for uid in matching_uids
                if uid in self.functions and self.functions[uid].file_path == caller_file
            ]
            if same_file_uids:
                return same_file_uids

            # 否则返回所有匹配的 uid
            if matching_uids:
                return matching_uids

        # 未找到定义，生成外部函数占位 uid
        # 格式: external:0:name（表示未解析的外部函数）
        return [f"external:0:{callee_name}"]
    
    def _extract_callee_name(self, node, source_bytes: bytes) -> Optional[str]:
        """
        从调用表达式提取被调用函数名
        
        支持:
        - func()
        - obj.method()
        - obj->method()
        - ClassName::staticMethod()
        - (*func_ptr)()
        """
        def extract_text(n) -> str:
            try:
                return source_bytes[n.start_byte:n.end_byte].decode('utf-8', errors='ignore')
            except:
                return ""
        
        for child in node.children:
            # 简单函数调用: func()
            if child.type == "identifier":
                return extract_text(child)
            
            # 成员函数调用: obj.method() 或 obj->method()
            elif child.type == "field_expression":
                for sub in child.children:
                    if sub.type == "field_identifier":
                        return extract_text(sub)
            
            # 限定调用: ClassName::method()
            elif child.type == "qualified_identifier":
                parts = []
                for sub in child.children:
                    if sub.type in ("identifier", "type_identifier"):
                        parts.append(extract_text(sub))
                return "::".join(parts) if parts else None
            
            # 函数指针调用: (*func_ptr)()
            elif child.type == "parenthesized_expression":
                for sub in child.children:
                    if sub.type == "pointer_expression":
                        for inner in sub.children:
                            if inner.type == "identifier":
                                return extract_text(inner)
        
        return None
    
    def analyze_directory(self, directory: Path, recursive: bool = True):
        """
        分析目录中的所有源文件（并行）
        
        Args:
            directory: 目录路径
            recursive: 是否递归分析子目录
        """
        directory = Path(directory)
        if not directory.exists():
            logger.warning(f"目录不存在: {directory}")
            return
        
        # 收集所有源文件
        if recursive:
            files = [f for f in directory.rglob("*") if f.suffix.lower() in SOURCE_EXTENSIONS]
        else:
            files = [f for f in directory.iterdir() if f.suffix.lower() in SOURCE_EXTENSIONS]
        
        if not files:
            logger.warning(f"未找到源文件: {directory}")
            return
        
        logger.info(f"分析 {len(files)} 个源文件...")
        
        # 并行分析
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.analyze_file, f): f for f in files}
            
            for future in as_completed(futures):
                file_path = futures[future]
                try:
                    future.result()
                except Exception as e:
                    logger.warning(f"分析文件失败 {file_path}: {e}")
        
        # 更新统计信息
        self._update_stats()
        
        logger.info(f"分析完成: {self.stats.total_functions} 个函数, {self.stats.total_calls} 条调用关系")
    
    def analyze_files(self, files: List[Path]):
        """
        分析指定的文件列表（并行）
        
        Args:
            files: 文件路径列表
        """
        if not files:
            return
        
        logger.info(f"分析 {len(files)} 个源文件...")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.analyze_file, f): f for f in files}
            
            for future in as_completed(futures):
                file_path = futures[future]
                try:
                    future.result()
                except Exception as e:
                    logger.warning(f"分析文件失败 {file_path}: {e}")
        
        self._update_stats()
    
    def _update_stats(self):
        """更新统计信息"""
        self.stats.total_functions = len(self.functions)
        self.stats.total_calls = sum(len(callees) for callees in self.call_graph.values())
        self.stats.total_files = len(self.analyzed_files)
        
        # 计算叶子函数和根函数
        self.stats.leaf_functions = sum(1 for f in self.functions if not self.call_graph.get(f))
        self.stats.root_functions = sum(1 for f in self.functions if not self.reverse_call_graph.get(f))
        
        # 检测循环依赖
        self.stats.cyclic_deps = self._count_cycles()
    
    def _count_cycles(self) -> int:
        """使用 Tarjan 算法计数强连通分量中的循环"""
        index_counter = [0]
        stack = []
        lowlinks = {}
        index_map = {}
        on_stack = {}
        sccs = []
        
        def strongconnect(node):
            index_map[node] = index_counter[0]
            lowlinks[node] = index_counter[0]
            index_counter[0] += 1
            stack.append(node)
            on_stack[node] = True
            
            for neighbor in self.call_graph.get(node, []):
                if neighbor not in self.functions:
                    continue
                if neighbor not in index_map:
                    strongconnect(neighbor)
                    lowlinks[node] = min(lowlinks[node], lowlinks[neighbor])
                elif on_stack.get(neighbor, False):
                    lowlinks[node] = min(lowlinks[node], index_map[neighbor])
            
            if lowlinks[node] == index_map[node]:
                scc = []
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    scc.append(w)
                    if w == node:
                        break
                if len(scc) > 1:
                    sccs.append(scc)
        
        for func in self.functions:
            if func not in index_map:
                strongconnect(func)
        
        return len(sccs)
    
    def get_translation_order(self) -> List[str]:
        """
        获取推荐的翻译顺序（拓扑排序）
        
        被调用的函数优先翻译，以便 LLM 可以参考已翻译的依赖
        
        Returns:
            函数名列表，按推荐翻译顺序排列
        """
        if self._translation_order is not None:
            return self._translation_order
        
        # 只考虑已定义的函数
        defined_functions = set(self.functions.keys())
        
        # 计算入度（被调用次数，仅统计已定义的函数）
        in_degree: Dict[str, int] = {f: 0 for f in defined_functions}
        
        for caller, callees in self.call_graph.items():
            if caller not in defined_functions:
                continue
            for callee in callees:
                if callee in in_degree:
                    in_degree[callee] += 1
        
        # Kahn 算法拓扑排序
        queue = [f for f, d in in_degree.items() if d == 0]
        result = []
        
        while queue:
            # 按字母顺序稳定排序
            queue.sort()
            func = queue.pop(0)
            result.append(func)
            
            for callee in self.call_graph.get(func, []):
                if callee in in_degree:
                    in_degree[callee] -= 1
                    if in_degree[callee] == 0:
                        queue.append(callee)
        
        # 处理循环依赖（剩余节点按字母顺序添加）
        remaining = sorted([f for f in defined_functions if f not in result])
        result.extend(remaining)
        
        # 反转顺序：被调用者优先
        self._translation_order = list(reversed(result))
        return self._translation_order
    
    def get_context_for_function(self, func_name: str, depth: int = 1, 
                                  include_body: bool = False) -> Dict[str, str]:
        """
        获取函数的调用上下文
        
        Args:
            func_name: 函数名
            depth: 查找深度（1=直接调用，2=间接调用）
            include_body: 是否包含函数体
            
        Returns:
            {函数名 -> 签名或完整代码} 字典
        """
        context = {}
        visited = {func_name}
        current = {func_name}
        
        for _ in range(depth):
            next_level = set()
            for f in current:
                callees = self.call_graph.get(f, set())
                for callee in callees:
                    if callee not in visited and callee in self.functions:
                        next_level.add(callee)
                        visited.add(callee)
            
            for callee in next_level:
                func_info = self.functions[callee]
                if include_body:
                    context[callee] = f"{func_info.signature}\n{func_info.body}"
                else:
                    context[callee] = func_info.signature
            
            current = next_level
        
        return context
    
    def get_callers_context(self, func_name: str, depth: int = 1) -> Dict[str, str]:
        """
        获取调用者上下文（谁调用了这个函数）
        
        Args:
            func_name: 函数名
            depth: 查找深度
            
        Returns:
            {函数名 -> 签名} 字典
        """
        context = {}
        visited = {func_name}
        current = {func_name}
        
        for _ in range(depth):
            next_level = set()
            for f in current:
                callers = self.reverse_call_graph.get(f, set())
                for caller in callers:
                    if caller not in visited and caller in self.functions:
                        next_level.add(caller)
                        visited.add(caller)
            
            for caller in next_level:
                func_info = self.functions[caller]
                context[caller] = func_info.signature
            
            current = next_level
        
        return context
    
    def get_function_info(self, func_name: str) -> Optional[FunctionInfo]:
        """获取函数信息（支持 uid 或函数名）"""
        uids = self.resolve_uids(func_name)
        if not uids:
            return None
        return self.functions.get(uids[0])

    def resolve_uids(self, func_identifier: str, file_hint: Optional[str] = None) -> List[str]:
        """
        将输入（uid 或函数名）解析为 uid 列表。

        - 如果 func_identifier 已经是 uid，直接返回 [uid]
        - 如果是函数名，则从 name_index 查找对应的 uid 列表
        - 如果提供 file_hint，则尝试按文件提示进行更精确的 disambiguation
        """
        if not func_identifier:
            return []

        # 1) 已是 uid
        if func_identifier in self.functions:
            return [func_identifier]

        # 2) 作为函数名，从 name_index 查找
        uids = list(self.name_index.get(func_identifier, []))
        if not uids:
            # 兜底：线性扫描（极少数情况下 name_index 可能不完整）
            uids = [uid for uid, info in self.functions.items() if info.name == func_identifier]

        if not uids or not file_hint:
            return uids

        # 3) 按 file_hint 过滤
        filtered = []
        for uid in uids:
            info = self.functions.get(uid)
            if not info:
                continue
            if self._file_path_matches_hint(info.file_path, file_hint):
                filtered.append(uid)
        if filtered and len(uids) > 1 and len(filtered) == 1:
            logger.debug(
                "resolve_uids: disambiguated '%s' with file_hint='%s' -> %s",
                func_identifier,
                file_hint,
                filtered[0],
            )
        elif not filtered and len(uids) > 1:
            logger.debug(
                "resolve_uids: ambiguous '%s' with file_hint='%s' (candidates=%d)",
                func_identifier,
                file_hint,
                len(uids),
            )
        return filtered or uids

    @staticmethod
    def _file_path_matches_hint(file_path: str, file_hint: str) -> bool:
        """
        file_hint 通常来自 safe_module_name（如 src_linux_ipc）。
        这里用启发式匹配，兼容：
        - file_path 本身就是 safe_name
        - file_path 是绝对路径 / 相对路径（将其转换为“safe-like”字符串做 suffix 匹配）
        """
        if not file_path or not file_hint:
            return False

        if file_path == file_hint:
            return True

        # 将 file_path 去掉扩展名并转换成 safe-like
        try:
            p = Path(file_path)
            no_ext = str(p.with_suffix(""))
        except Exception:
            no_ext = str(file_path)
        safe_like = no_ext.replace("/", "_").replace("\\", "_")
        safe_like = re.sub(r"[^a-zA-Z0-9_]", "_", safe_like)

        return safe_like.endswith(file_hint) or (file_hint in safe_like)
    
    def get_dependency_chain(self, func_name: str, max_depth: int = 10) -> List[List[str]]:
        """
        获取函数的依赖链（调用路径）
        
        Args:
            func_name: 函数名
            max_depth: 最大深度
            
        Returns:
            所有依赖链的列表
        """
        chains = []
        
        def dfs(current: str, path: List[str], depth: int):
            if depth > max_depth:
                return
            
            callees = self.call_graph.get(current, set())
            if not callees:
                if len(path) > 1:
                    chains.append(path.copy())
                return
            
            for callee in callees:
                if callee in path:  # 避免循环
                    chains.append(path + [callee, "...循环"])
                    continue
                
                if callee in self.functions:
                    path.append(callee)
                    dfs(callee, path, depth + 1)
                    path.pop()
        
        dfs(func_name, [func_name], 0)
        return chains
    
    def to_dict(self) -> Dict:
        """
        导出为字典格式

        新格式 schema (v2)：
        - functions: {uid: {name, file, start_line, end_line, signature, callees, uid}}
        - call_graph: {caller_uid: [callee_uid, ...]}
        - reverse_call_graph: {callee_uid: [caller_uid, ...]}
        - name_index: {name: [uid, ...]}
        - schema_version: "2.0"
        """
        return {
            "schema_version": "2.0",  # 标识新 schema
            "functions": {
                uid: {
                    "name": info.name,
                    "file": info.file_path,
                    "start_line": info.start_line,
                    "end_line": info.end_line,
                    "signature": info.signature,
                    "callees": list(info.callees),
                    "uid": uid,
                    "mangled_name": info.mangled_name,
                }
                for uid, info in self.functions.items()
            },
            "call_graph": {k: list(v) for k, v in self.call_graph.items()},
            "reverse_call_graph": {k: list(v) for k, v in self.reverse_call_graph.items()},
            "name_index": dict(self.name_index),  # {name: [uid...]}
            "global_vars": self.global_vars,
            "type_defs": self.type_defs,
            "translation_order": self.get_translation_order(),
            "stats": {
                "total_functions": self.stats.total_functions,
                "total_calls": self.stats.total_calls,
                "total_files": self.stats.total_files,
                "cyclic_deps": self.stats.cyclic_deps,
                "leaf_functions": self.stats.leaf_functions,
                "root_functions": self.stats.root_functions,
            }
        }
    
    def save(self, output_path: Path):
        """保存调用图到 JSON 文件"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

        logger.info(f"调用图已保存: {output_path}")

        # 保存后自动执行重复名检查
        self.duplicate_name_check()

    def duplicate_name_check(self) -> Dict[str, List[str]]:
        """
        检查调用图中的同名函数，打印统计信息

        返回: {函数名: [对应的uid列表...]}，只返回有多个uid的函数名
        """
        duplicates = {}
        for name, uids in self.name_index.items():
            if len(uids) > 1:
                duplicates[name] = uids

        if duplicates:
            logger.info(f"同名函数统计（{len(duplicates)} 个函数名对应多个定义）:")
            for name, uids in sorted(duplicates.items(), key=lambda x: -len(x[1]))[:10]:
                logger.info(f"  '{name}': {len(uids)} 个实例")
                for uid in uids[:3]:
                    if uid in self.functions:
                        info = self.functions[uid]
                        logger.info(f"    - {uid} ({info.file_path}:{info.start_line})")
                if len(uids) > 3:
                    logger.info(f"    ... 还有 {len(uids) - 3} 个")
            if len(duplicates) > 10:
                logger.info(f"  ... 还有 {len(duplicates) - 10} 个同名函数")

            # 验证图中节点是否正确分开
            node_check_passed = True
            for name, uids in duplicates.items():
                for uid in uids:
                    if uid not in self.functions:
                        logger.warning(f"  警告: uid '{uid}' 在 functions 中未找到")
                        node_check_passed = False

            if node_check_passed:
                logger.info("验证通过: 所有同名函数在图中保持独立节点")
        else:
            logger.info("无同名函数冲突")

        return duplicates

    def verify_uid_integrity(self) -> bool:
        """
        验证调用图的 uid 完整性

        检查:
        1. functions 中的所有 uid 是否在 name_index 中有对应
        2. call_graph 中的所有 caller/callee uid 是否在 functions 中存在
        3. 是否存在重复的 uid

        返回: True 如果验证通过，False 否则
        """
        all_valid = True

        # 检查 functions 中的 uid 是否在 name_index 中
        for uid, func_info in self.functions.items():
            if func_info.name not in self.name_index:
                logger.warning(f"uid '{uid}' 的函数名 '{func_info.name}' 不在 name_index 中")
                all_valid = False
            elif uid not in self.name_index[func_info.name]:
                logger.warning(f"uid '{uid}' 不在 name_index['{func_info.name}'] 中")
                all_valid = False

        # 检查 call_graph 中的 uid 是否有效
        external_count = 0
        for caller_uid, callee_uids in self.call_graph.items():
            if not caller_uid.startswith("external:") and caller_uid not in self.functions:
                logger.warning(f"调用图中的 caller_uid '{caller_uid}' 不在 functions 中")
                all_valid = False
            for callee_uid in callee_uids:
                if callee_uid.startswith("external:"):
                    external_count += 1
                elif callee_uid not in self.functions:
                    logger.warning(f"调用图中的 callee_uid '{callee_uid}' 不在 functions 中")
                    all_valid = False

        if external_count > 0:
            logger.info(f"外部函数引用数: {external_count}")

        if all_valid:
            logger.info("uid 完整性验证通过")
        else:
            logger.warning("uid 完整性验证失败")

        return all_valid
    
    @classmethod
    def load(cls, input_path: Path) -> "CallGraphBuilder":
        """
        从 JSON 文件加载调用图

        支持两种 schema：
        - v2.0 (新): 使用 uid 作为主键
        - v1.0 (旧): 使用函数名作为主键（带警告）
        """
        input_path = Path(input_path)

        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        builder = cls()

        # 检测 schema 版本
        schema_version = data.get("schema_version", "1.0")

        if schema_version == "2.0":
            # 新格式: uid 作为主键
            for uid, info in data.get("functions", {}).items():
                builder.functions[uid] = FunctionInfo(
                    name=info.get("name", ""),
                    file_path=info.get("file", ""),
                    start_line=info.get("start_line", 0),
                    end_line=info.get("end_line", 0),
                    signature=info.get("signature", ""),
                    callees=set(info.get("callees", [])),
                    uid=uid,
                    mangled_name=info.get("mangled_name", ""),
                )

            # 恢复 name_index
            for name, uids in data.get("name_index", {}).items():
                builder.name_index[name] = list(uids)

            # 恢复调用图（已经是 uid 格式）
            for caller, callees in data.get("call_graph", {}).items():
                builder.call_graph[caller] = set(callees)
            for callee, callers in data.get("reverse_call_graph", {}).items():
                builder.reverse_call_graph[callee] = set(callers)
        else:
            # 旧格式: 函数名作为主键
            logger.warning(
                f"检测到旧版调用图格式 (schema v1.0)，存在函数名冲突风险。"
                f"建议重新生成调用图以使用新的 uid schema。"
            )

            # 检测并报告潜在的名字冲突
            name_to_files: Dict[str, List[str]] = defaultdict(list)
            for name, info in data.get("functions", {}).items():
                file_path = info.get("file", "unknown")
                name_to_files[name].append(file_path)

            conflicts = {name: files for name, files in name_to_files.items() if len(files) > 1}
            if conflicts:
                logger.warning(
                    f"发现 {len(conflicts)} 个潜在函数名冲突（同名不同文件），"
                    f"降级处理可能导致上下文错误："
                )
                for name, files in list(conflicts.items())[:5]:
                    logger.warning(f"  - '{name}' 出现在: {files}")
                if len(conflicts) > 5:
                    logger.warning(f"  ... 还有 {len(conflicts) - 5} 个冲突")

            # 降级处理：为每个函数生成 uid
            for name, info in data.get("functions", {}).items():
                file_path = info.get("file", "")
                start_line = info.get("start_line", 0)

                # 生成 uid
                uid = generate_function_uid(file_path, start_line, name)

                builder.functions[uid] = FunctionInfo(
                    name=name,
                    file_path=file_path,
                    start_line=start_line,
                    end_line=info.get("end_line", 0),
                    signature=info.get("signature", ""),
                    callees=set(info.get("callees", [])),
                    uid=uid,
                )

                # 更新 name_index
                if name not in builder.name_index:
                    builder.name_index[name] = []
                builder.name_index[name].append(uid)

            # 重新构建调用图（从函数名转换为 uid）
            for caller_name, callees in data.get("call_graph", {}).items():
                # 找到 caller 对应的 uid
                caller_uids = builder.name_index.get(caller_name, [])
                for caller_uid in caller_uids:
                    caller_file = builder.functions[caller_uid].file_path if caller_uid in builder.functions else ""
                    for callee_name in callees:
                        # 解析 callee_name 为 uid
                        callee_uids = builder._resolve_callee_uid(callee_name, caller_file)
                        for callee_uid in callee_uids:
                            builder.call_graph[caller_uid].add(callee_uid)
                            builder.reverse_call_graph[callee_uid].add(caller_uid)

        # 恢复全局变量和类型
        builder.global_vars = data.get("global_vars", {})
        builder.type_defs = data.get("type_defs", {})

        builder._update_stats()

        return builder
    
    def generate_dot(self, output_path: Path, max_nodes: int = 100):
        """
        生成 Graphviz DOT 格式的调用图
        
        Args:
            output_path: 输出文件路径
            max_nodes: 最大节点数（避免图太大）
        """
        output_path = Path(output_path)
        
        # 选择最重要的节点
        nodes = set()
        edges = []
        
        # 按调用次数排序
        call_counts = defaultdict(int)
        for caller, callees in self.call_graph.items():
            for callee in callees:
                call_counts[callee] += 1
        
        # 选择最常被调用的函数
        top_functions = sorted(call_counts.keys(), key=lambda x: -call_counts[x])[:max_nodes]
        nodes.update(top_functions)
        
        # 添加这些函数的调用者
        for func in list(nodes):
            for caller in self.reverse_call_graph.get(func, []):
                if caller in self.functions:
                    nodes.add(caller)
                    edges.append((caller, func))
        
        # 生成 DOT
        lines = ["digraph CallGraph {"]
        lines.append('  rankdir="LR";')
        lines.append('  node [shape=box, style=filled, fillcolor=lightblue];')
        
        for node in nodes:
            call_count = call_counts.get(node, 0)
            # 被调用越多，颜色越深
            color = "lightblue" if call_count < 3 else "lightyellow" if call_count < 10 else "orange"
            lines.append(f'  "{node}" [fillcolor={color}];')
        
        for caller, callee in edges:
            lines.append(f'  "{caller}" -> "{callee}";')
        
        lines.append("}")
        
        output_path.write_text("\n".join(lines), encoding='utf-8')
        logger.info(f"DOT 文件已生成: {output_path}")
    
    def print_summary(self):
        """打印调用图摘要"""
        print("\n" + "=" * 60)
        print("📊 调用图分析摘要")
        print("=" * 60)
        print(f"  总函数数: {self.stats.total_functions}")
        print(f"  总调用数: {self.stats.total_calls}")
        print(f"  分析文件数: {self.stats.total_files}")
        print(f"  循环依赖数: {self.stats.cyclic_deps}")
        print(f"  叶子函数数: {self.stats.leaf_functions} (不调用其他函数)")
        print(f"  根函数数: {self.stats.root_functions} (不被调用)")
        print("=" * 60)
        
        # 打印最常被调用的函数
        if self.reverse_call_graph:
            print("\n🔥 最常被调用的函数 (Top 10):")
            call_counts = {f: len(callers) for f, callers in self.reverse_call_graph.items()}
            top_called = sorted(call_counts.items(), key=lambda x: -x[1])[:10]
            for func, count in top_called:
                print(f"  - {func}: {count} 次")
        
        print()


class LLMContextProvider:
    """
    为 LLM 提供翻译上下文
    
    基于调用图为 LLM 翻译提供相关函数的签名和定义
    """
    
    def __init__(self, call_graph: CallGraphBuilder, max_context_tokens: int = 4000):
        """
        初始化上下文提供器
        
        Args:
            call_graph: 调用图构建器
            max_context_tokens: 最大上下文 token 数（估算）
        """
        self.call_graph = call_graph
        self.max_context_tokens = max_context_tokens
        
        # 已翻译函数的 Rust 代码缓存
        # - by_uid: 精确匹配（schema v2.0 推荐）
        # - by_name: 向后兼容 / 无法 disambiguate 的情况
        self.translated_by_uid: Dict[str, str] = {}
        self.translated_by_name: Dict[str, str] = {}
    
    def register_translated(self, func_identifier: str, rust_code: str, file_hint: Optional[str] = None):
        """注册已翻译的函数（支持 uid 或函数名）"""
        if not func_identifier:
            return

        uids = self.call_graph.resolve_uids(func_identifier, file_hint=file_hint)

        # 能唯一定位到 uid 时，优先写入 by_uid 并同步 by_name
        if len(uids) == 1 and uids[0] in self.call_graph.functions:
            uid = uids[0]
            self.translated_by_uid[uid] = rust_code
            name = self.call_graph.functions[uid].name
            if name:
                self.translated_by_name[name] = rust_code
            return

        # 否则仅按名字缓存（避免错误关联到错误 uid）
        self.translated_by_name[func_identifier] = rust_code
    
    def get_context_for_translation(self, func_identifier: str, file_hint: Optional[str] = None) -> str:
        """
        获取用于翻译的上下文
        
        优先提供：
        1. 被调用函数的已翻译 Rust 代码
        2. 被调用函数的 C 签名
        3. 调用者函数的签名（了解使用方式）
        
        Args:
            func_identifier: 要翻译的函数 uid 或函数名
            file_hint: 可选的文件提示（safe_module_name），用于 disambiguation
            
        Returns:
            上下文字符串
        """
        context_parts = []
        current_tokens = 0
        
        uids = self.call_graph.resolve_uids(func_identifier, file_hint=file_hint)
        if not uids:
            return ""
        # 目前只选取一个最匹配的 uid（避免上下文爆炸）
        uid = uids[0]
        func_info = self.call_graph.functions.get(uid)
        if not func_info:
            return ""
        
        # 1. 添加被调用函数的上下文
        callees = self.call_graph.call_graph.get(uid, set())
        
        # 分类：已翻译 vs 未翻译
        translated_callees = []
        untranslated_callees = []
        
        for callee in callees:
            if callee in self.translated_by_uid:
                translated_callees.append(callee)
                continue

            callee_info = self.call_graph.functions.get(callee)
            callee_name = callee_info.name if callee_info else self._extract_external_name(callee)
            if callee_name and callee_name in self.translated_by_name:
                translated_callees.append(callee)
            elif callee in self.call_graph.functions:
                untranslated_callees.append(callee)
        
        # 1.1 添加已翻译的 Rust 代码
        if translated_callees:
            context_parts.append("// 已翻译的被调用函数 (Rust):")
            for callee in translated_callees:
                rust_code = self.translated_by_uid.get(callee)
                if rust_code is None:
                    callee_info = self.call_graph.functions.get(callee)
                    callee_name = callee_info.name if callee_info else self._extract_external_name(callee)
                    rust_code = self.translated_by_name.get(callee_name) if callee_name else None
                if not rust_code:
                    continue
                # 只添加签名，不添加完整实现（节省 token）
                signature = self._extract_rust_signature(rust_code)
                if signature:
                    callee_info = self.call_graph.functions.get(callee)
                    label = callee_info.name if callee_info and callee_info.name else callee
                    context_parts.append(f"// {label}:")
                    context_parts.append(signature)
                    current_tokens += self._estimate_tokens(signature)
                    
                    if current_tokens > self.max_context_tokens * 0.5:
                        break
        
        # 1.2 添加未翻译的 C 签名
        if untranslated_callees and current_tokens < self.max_context_tokens * 0.7:
            context_parts.append("\n// 被调用函数 (C 签名，尚未翻译):")
            for callee in untranslated_callees:
                callee_info = self.call_graph.functions.get(callee)
                if callee_info and callee_info.signature:
                    context_parts.append(f"// {callee_info.name or callee}:")
                    context_parts.append(f"// {callee_info.signature}")
                    current_tokens += self._estimate_tokens(callee_info.signature)
                    
                    if current_tokens > self.max_context_tokens * 0.7:
                        break
        
        # 2. 添加调用者上下文（了解使用方式）
        if current_tokens < self.max_context_tokens * 0.9:
            callers = self.call_graph.reverse_call_graph.get(uid, set())
            if callers:
                context_parts.append("\n// 调用者函数 (如何使用此函数):")
                for caller in list(callers)[:3]:  # 最多 3 个
                    caller_info = self.call_graph.functions.get(caller)
                    if caller_info and caller_info.signature:
                        context_parts.append(f"// {caller_info.name or caller}:")
                        context_parts.append(f"// {caller_info.signature}")
                        current_tokens += self._estimate_tokens(caller_info.signature)
                        
                        if current_tokens > self.max_context_tokens:
                            break
        
        return "\n".join(context_parts)

    @staticmethod
    def _extract_external_name(uid: str) -> Optional[str]:
        """从 external:0:name 形式的 uid 提取 name"""
        if not uid:
            return None
        if uid.startswith("external:"):
            parts = uid.split(":", 2)
            if len(parts) == 3:
                return parts[2]
        return None
    
    def _extract_rust_signature(self, rust_code: str) -> Optional[str]:
        """从 Rust 代码提取函数签名"""
        lines = rust_code.split('\n')
        signature_lines = []
        
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('pub') or stripped.startswith('fn ') or stripped.startswith('pub fn'):
                signature_lines.append(line)
                if '{' in line:
                    # 单行签名
                    idx = line.index('{')
                    return line[:idx].strip()
                continue
            
            if signature_lines:
                signature_lines.append(line)
                if '{' in line:
                    # 多行签名
                    full_sig = ' '.join(l.strip() for l in signature_lines)
                    idx = full_sig.index('{')
                    return full_sig[:idx].strip()
        
        return None
    
    def _estimate_tokens(self, text: str) -> int:
        """估算 token 数（简单估算：每 4 个字符约 1 个 token）"""
        return len(text) // 4 + 1
    
    def get_translation_order(self) -> List[str]:
        """获取推荐的翻译顺序"""
        return self.call_graph.get_translation_order()


# 命令行接口
if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="C/C++ 函数调用图分析工具")
    parser.add_argument("path", type=Path, help="源码目录或文件路径")
    parser.add_argument("-o", "--output", type=Path, help="输出 JSON 文件路径")
    parser.add_argument("--dot", type=Path, help="输出 DOT 文件路径")
    parser.add_argument("-j", "--jobs", type=int, default=4, help="并行线程数")
    parser.add_argument("-v", "--verbose", action="store_true", help="详细输出")
    
    args = parser.parse_args()
    
    # 设置日志
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # 构建调用图
    builder = CallGraphBuilder(max_workers=args.jobs)
    
    path = args.path
    if path.is_file():
        builder.analyze_file(path)
    else:
        builder.analyze_directory(path)
    
    # 打印摘要
    builder.print_summary()
    
    # 打印翻译顺序
    order = builder.get_translation_order()
    print("\n📋 推荐翻译顺序 (前 20 个):")
    for i, func in enumerate(order[:20], 1):
        info = builder.get_function_info(func)
        callees = len(info.callees) if info else 0
        print(f"  {i:2}. {func} (调用 {callees} 个函数)")
    
    if len(order) > 20:
        print(f"  ... 还有 {len(order) - 20} 个函数")
    
    # 保存输出
    if args.output:
        builder.save(args.output)
    
    if args.dot:
        builder.generate_dot(args.dot)
