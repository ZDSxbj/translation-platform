#!/usr/bin/env python3
"""
增强型预处理模块 (Enhanced Preprocessing)

基于 EvoC2Rust 论文的预处理策略：
1. gcc -fpreprocessed -dD -E：去注释但保留宏定义
2. clang-format：规范化代码格式，提高 Tree-sitter 解析准确率
3. AST 健康检查：检测 Tree-sitter ERROR 节点，预警解析问题

参考文献:
- EvoC2Rust: A Skeleton-guided Framework for Project-Level C-to-Rust Translation
- Rustine: Preprocessing Phase
"""

import os
import re
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass
import logging

# Tree-sitter 支持
try:
    from tree_sitter import Language, Parser
    import tree_sitter_cpp as tscpp
    
    # 兼容不同版本的 API（tree-sitter >= 0.22 需要 Language(..., name)）
    try:
        CPP_LANGUAGE = Language(tscpp.language(), "cpp")
    except TypeError:
        CPP_LANGUAGE = Language(tscpp.language())

    # 兼容不同版本的 Parser API
    _parser = Parser()
    try:
        _parser.set_language(CPP_LANGUAGE)
    except Exception:
        _parser = Parser(CPP_LANGUAGE)
    TREESITTER_AVAILABLE = True
except ImportError:
    TREESITTER_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ASTHealthReport:
    """AST 健康检查报告"""
    total_nodes: int = 0
    error_nodes: int = 0
    error_rate: float = 0.0  # 错误节点占比
    error_locations: List[Tuple[int, int]] = None  # [(line, column), ...]
    is_healthy: bool = True
    recommendation: str = ""
    
    def __post_init__(self):
        if self.error_locations is None:
            self.error_locations = []


@dataclass
class PreprocessingResult:
    """预处理结果"""
    success: bool
    content: str
    method: str  # "gcc_fpreprocessed", "gcc_E", "clang_format", "raw"
    health_report: Optional[ASTHealthReport] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class EnhancedPreprocessor:
    """
    增强型预处理器
    
    处理流程:
    1. 尝试 gcc -fpreprocessed -dD -E (去注释保留宏)
    2. 可选：clang-format 格式化
    3. AST 健康检查
    4. 根据健康检查结果决定是否降级
    """
    
    def __init__(
        self,
        include_dirs: List[Path] = None,
        use_clang_format: bool = False,
        error_threshold: float = 0.05,  # 5% 错误率阈值
        temp_dir: Path = None
    ):
        """
        初始化预处理器
        
        Args:
            include_dirs: 头文件搜索目录
            use_clang_format: 是否使用 clang-format 格式化
            error_threshold: AST 错误率阈值，超过此值触发警告
            temp_dir: 临时文件目录
        """
        self.include_dirs = include_dirs or []
        self.use_clang_format = use_clang_format
        self.error_threshold = error_threshold
        self.temp_dir = temp_dir or Path(tempfile.gettempdir()) / "c2rust_preprocess"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # 检测工具可用性
        self.gcc_available = self._check_tool("gcc")
        self.clang_format_available = self._check_tool("clang-format")
        
        # 缓存
        self._cache: Dict[str, PreprocessingResult] = {}
    
    def _check_tool(self, tool: str) -> bool:
        """检查工具是否可用"""
        try:
            result = subprocess.run(
                [tool, "--version"],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False
    
    def preprocess(
        self,
        source_file: Path,
        extra_includes: List[Path] = None,
        force_method: str = None
    ) -> PreprocessingResult:
        """
        预处理源文件
        
        Args:
            source_file: 源文件路径
            extra_includes: 额外的 include 路径
            force_method: 强制使用特定方法 ("gcc_fpreprocessed", "gcc_E", "raw")
        
        Returns:
            PreprocessingResult: 预处理结果
        """
        cache_key = str(source_file)
        if cache_key in self._cache and not force_method:
            return self._cache[cache_key]
        
        # 读取原始内容
        try:
            raw_content = source_file.read_text(encoding='utf-8', errors='ignore')
        except Exception as e:
            return PreprocessingResult(
                success=False,
                content="",
                method="error",
                warnings=[f"读取文件失败: {e}"]
            )
        
        # 预处理尝试链
        result = None
        
        # 方法 1: gcc -fpreprocessed -dD -E (最佳：去注释保留宏定义)
        if force_method in (None, "gcc_fpreprocessed") and self.gcc_available:
            result = self._preprocess_gcc_fpreprocessed(
                source_file, raw_content, extra_includes
            )
            if result.success:
                logger.info(f"预处理成功 (gcc -fpreprocessed): {source_file.name}")
        
        # 方法 2: 普通 gcc -E (回退：完全展开宏)
        if (not result or not result.success) and self.gcc_available:
            if force_method in (None, "gcc_E"):
                result = self._preprocess_gcc_E(source_file, raw_content, extra_includes)
                if result.success:
                    logger.info(f"预处理成功 (gcc -E): {source_file.name}")
        
        # 方法 3: 原始内容 + 基本清理
        if not result or not result.success:
            result = self._preprocess_basic(raw_content)
            logger.warning(f"使用原始内容 (基本清理): {source_file.name}")
        
        # 可选：clang-format 格式化
        if self.use_clang_format and self.clang_format_available and result.success:
            formatted = self._apply_clang_format(result.content)
            if formatted:
                result.content = formatted
                result.method += "+clang_format"
        
        # AST 健康检查
        if TREESITTER_AVAILABLE:
            health_report = self.check_ast_health(result.content)
            result.health_report = health_report
            
            if not health_report.is_healthy:
                result.warnings.append(health_report.recommendation)
        
        self._cache[cache_key] = result
        return result
    
    def _preprocess_gcc_fpreprocessed(
        self,
        source_file: Path,
        raw_content: str,
        extra_includes: List[Path] = None
    ) -> PreprocessingResult:
        """
        使用 gcc -fpreprocessed -dD -E 预处理
        
        特点：
        - -fpreprocessed: 告诉 gcc 输入已经预处理过，只做语法检查
        - -dD: 保留 #define 指令
        - 结合使用：去除注释但保留宏定义
        
        这是 EvoC2Rust 使用的方法，比单纯的 -E 更干净
        """
        try:
            # 创建临时文件（gcc 需要文件输入）
            temp_file = self.temp_dir / f"preprocess_{source_file.name}"
            
            # 先处理续行符 (行尾反斜杠)
            cleaned = raw_content.replace("\\\n", "")
            temp_file.write_text(cleaned, encoding='utf-8')
            
            # 构建命令
            cmd = ["gcc", "-fpreprocessed", "-dD", "-E"]
            
            # 添加 include 路径
            all_includes = set(self.include_dirs)
            if extra_includes:
                all_includes.update(extra_includes)
            all_includes.add(source_file.parent)
            
            for inc_dir in all_includes:
                cmd.extend(["-I", str(inc_dir)])
            
            cmd.append(str(temp_file))
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=30
            )
            
            if result.returncode == 0:
                content = result.stdout.decode('utf-8', errors='ignore')
                return PreprocessingResult(
                    success=True,
                    content=content,
                    method="gcc_fpreprocessed"
                )
            else:
                stderr = result.stderr.decode('utf-8', errors='ignore')
                return PreprocessingResult(
                    success=False,
                    content=raw_content,
                    method="gcc_fpreprocessed_failed",
                    warnings=[f"gcc -fpreprocessed 失败: {stderr[:200]}"]
                )
                
        except subprocess.TimeoutExpired:
            return PreprocessingResult(
                success=False,
                content=raw_content,
                method="gcc_fpreprocessed_timeout",
                warnings=["预处理超时"]
            )
        except Exception as e:
            return PreprocessingResult(
                success=False,
                content=raw_content,
                method="gcc_fpreprocessed_error",
                warnings=[f"预处理异常: {e}"]
            )
        finally:
            # 清理临时文件
            if 'temp_file' in locals() and temp_file.exists():
                try:
                    temp_file.unlink()
                except:
                    pass
    
    def _preprocess_gcc_E(
        self,
        source_file: Path,
        raw_content: str,
        extra_includes: List[Path] = None
    ) -> PreprocessingResult:
        """
        使用标准 gcc -E -P 预处理（完全展开宏）
        """
        try:
            cmd = ["gcc", "-E", "-P"]
            
            # 添加 include 路径
            all_includes = set(self.include_dirs)
            if extra_includes:
                all_includes.update(extra_includes)
            all_includes.add(source_file.parent)
            
            for inc_dir in all_includes:
                cmd.extend(["-I", str(inc_dir)])
            
            # 常用宏定义避免错误
            cmd.extend([
                "-D__attribute__(x)=",
                "-D__extension__=",
                "-D__restrict=",
                "-D__inline=inline",
                "-D__inline__=inline",
                "-D__asm__(x)=",
                "-D__volatile__=",
                "-D__builtin_va_list=void*",
            ])
            
            cmd.append(str(source_file))
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=30
            )
            
            if result.returncode == 0:
                content = result.stdout.decode('utf-8', errors='ignore')
                return PreprocessingResult(
                    success=True,
                    content=content,
                    method="gcc_E"
                )
            else:
                return PreprocessingResult(
                    success=False,
                    content=raw_content,
                    method="gcc_E_failed"
                )
                
        except Exception as e:
            return PreprocessingResult(
                success=False,
                content=raw_content,
                method="gcc_E_error",
                warnings=[f"gcc -E 异常: {e}"]
            )
    
    def _preprocess_basic(self, content: str) -> PreprocessingResult:
        """
        基本预处理：清理换行符和 extern "C" 包装
        """
        # 统一换行符
        processed = content.replace('\r\n', '\n').replace('\r', '\n')
        
        # 移除 #ifdef __cplusplus extern "C" 包装
        patterns = [
            r'#ifdef\s+__cplusplus\s*\n\s*extern\s+"C"\s*\{\s*\n\s*#endif',
            r'#ifdef\s+__cplusplus\s*\n\s*\}\s*\n\s*#endif',
            r'#ifdef\s+__cplusplus\s+extern\s+"C"\s*\{\s+#endif',
            r'#ifdef\s+__cplusplus\s+\}\s+#endif',
        ]
        
        for pattern in patterns:
            processed = re.sub(pattern, '', processed)
        
        return PreprocessingResult(
            success=True,
            content=processed,
            method="basic"
        )
    
    def _apply_clang_format(self, content: str) -> Optional[str]:
        """
        使用 clang-format 格式化代码
        
        格式化的好处：
        - Tree-sitter 对格式良好的代码解析更准确
        - 宏展开后的长行会被拆分
        - 统一的代码风格减少边界 bug
        """
        try:
            # 创建临时文件
            temp_file = self.temp_dir / "format_temp.c"
            temp_file.write_text(content, encoding='utf-8')
            
            # 使用 Microsoft 风格（与 EvoC2Rust 一致）
            result = subprocess.run(
                ["clang-format", "-style=Microsoft", str(temp_file)],
                capture_output=True,
                timeout=30
            )
            
            if result.returncode == 0:
                return result.stdout.decode('utf-8', errors='ignore')
            
        except Exception as e:
            logger.debug(f"clang-format 失败: {e}")
        finally:
            if 'temp_file' in locals() and temp_file.exists():
                try:
                    temp_file.unlink()
                except:
                    pass
        
        return None
    
    def check_ast_health(
        self,
        content: str,
        error_threshold: float = None
    ) -> ASTHealthReport:
        """
        检查 AST 健康度
        
        遍历 Tree-sitter AST，统计 ERROR 节点数量。
        如果错误率超过阈值，说明解析失败，应该降级处理。
        
        Args:
            content: 源代码内容
            error_threshold: 错误率阈值 (默认使用实例设置)
        
        Returns:
            ASTHealthReport: 健康检查报告
        """
        if not TREESITTER_AVAILABLE:
            return ASTHealthReport(is_healthy=True, recommendation="Tree-sitter 不可用，跳过健康检查")
        
        threshold = error_threshold or self.error_threshold
        
        try:
            source_bytes = bytes(content, 'utf-8')
            tree = _parser.parse(source_bytes)
            
            # 统计节点
            total_nodes = 0
            error_nodes = 0
            error_locations = []
            
            # 使用游标遍历（比递归更高效）
            cursor = tree.walk()
            
            visited_children = False
            while True:
                if not visited_children:
                    node = cursor.node
                    total_nodes += 1
                    
                    if node.type == 'ERROR':
                        error_nodes += 1
                        error_locations.append((
                            node.start_point[0] + 1,  # 行号从1开始
                            node.start_point[1]
                        ))
                
                if visited_children:
                    if cursor.goto_next_sibling():
                        visited_children = False
                    elif cursor.goto_parent():
                        visited_children = True
                    else:
                        break
                else:
                    if cursor.goto_first_child():
                        visited_children = False
                    else:
                        visited_children = True
            
            # 计算错误率
            error_rate = error_nodes / total_nodes if total_nodes > 0 else 0.0
            is_healthy = error_rate <= threshold
            
            # 生成建议
            recommendation = ""
            if not is_healthy:
                if error_rate > 0.2:  # 超过 20%
                    recommendation = (
                        f"⚠️ 严重：AST 错误率 {error_rate:.1%} ({error_nodes}/{total_nodes})。"
                        f"建议：1) 尝试更激进的预处理；2) 仅提取顶层函数；3) 考虑人工干预"
                    )
                elif error_rate > 0.1:  # 超过 10%
                    recommendation = (
                        f"⚠️ 警告：AST 错误率 {error_rate:.1%} ({error_nodes}/{total_nodes})。"
                        f"部分函数可能提取失败，建议检查输出"
                    )
                else:
                    recommendation = (
                        f"⚠️ 注意：AST 错误率 {error_rate:.1%} ({error_nodes}/{total_nodes})。"
                        f"可能影响少量函数提取"
                    )
            
            return ASTHealthReport(
                total_nodes=total_nodes,
                error_nodes=error_nodes,
                error_rate=error_rate,
                error_locations=error_locations[:10],  # 只保留前10个位置
                is_healthy=is_healthy,
                recommendation=recommendation
            )
            
        except Exception as e:
            return ASTHealthReport(
                is_healthy=False,
                recommendation=f"AST 健康检查异常: {e}"
            )
    
    def preprocess_string(self, content: str) -> PreprocessingResult:
        """
        直接预处理字符串内容（不读取文件）
        
        用于已经加载到内存的源代码
        """
        # 基本清理
        result = self._preprocess_basic(content)
        
        # 可选：clang-format
        if self.use_clang_format and self.clang_format_available:
            formatted = self._apply_clang_format(result.content)
            if formatted:
                result.content = formatted
                result.method += "+clang_format"
        
        # AST 健康检查
        if TREESITTER_AVAILABLE:
            health_report = self.check_ast_health(result.content)
            result.health_report = health_report
            
            if not health_report.is_healthy:
                result.warnings.append(health_report.recommendation)
        
        return result
    
    def cleanup(self):
        """清理临时文件"""
        try:
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir, ignore_errors=True)
        except:
            pass


# ============================================================================
# 递归函数声明器查找工具 (Recursive Declarator Search)
# ============================================================================

def find_function_declarator_recursive(node) -> Tuple[bool, Optional[str], Optional[object]]:
    """
    递归查找函数声明器
    
    参考 EvoC2Rust 的 has_function_declarator 实现
    能处理复杂的嵌套情况：
    - Type* func()  -> pointer_declarator > function_declarator
    - Type& func()  -> reference_declarator > function_declarator  
    - Type** func() -> pointer_declarator > pointer_declarator > function_declarator
    - Type(*func)() -> 函数指针
    
    Returns:
        (found, function_name, declarator_node)
    """
    if node.type == "function_declarator":
        # 找到函数声明器，提取函数名
        name = _find_identifier_in_declarator(node)
        return True, name, node
    
    # 递归搜索子节点
    for child in node.children:
        found, name, decl_node = find_function_declarator_recursive(child)
        if found:
            return found, name, decl_node
    
    return False, None, None


def _find_identifier_in_declarator(declarator_node) -> Optional[str]:
    """
    在声明器中查找标识符（函数名）
    
    处理情况：
    - function_declarator > identifier
    - function_declarator > parenthesized_declarator > identifier
    - function_declarator > field_identifier (类成员)
    - function_declarator > qualified_identifier (命名空间::函数)
    """
    for child in declarator_node.children:
        if child.type == "identifier":
            return child.text.decode('utf-8', errors='ignore') if hasattr(child, 'text') else None
        elif child.type == "field_identifier":
            return child.text.decode('utf-8', errors='ignore') if hasattr(child, 'text') else None
        elif child.type == "qualified_identifier":
            # 提取 ClassName::methodName 格式
            return child.text.decode('utf-8', errors='ignore') if hasattr(child, 'text') else None
        elif child.type in ("parenthesized_declarator", "pointer_declarator", "reference_declarator"):
            # 递归查找
            result = _find_identifier_in_declarator(child)
            if result:
                return result
    
    return None


def find_all_declarator_wrappers(node) -> List[str]:
    """
    查找声明器的所有包装层
    
    返回从外到内的包装类型列表，如：
    - Type* func() -> ["pointer_declarator"]
    - Type& func() -> ["reference_declarator"]
    - Type** func() -> ["pointer_declarator", "pointer_declarator"]
    """
    wrappers = []
    current = node
    
    while current:
        if current.type in ("pointer_declarator", "reference_declarator"):
            wrappers.append(current.type)
            # 向下找
            for child in current.children:
                if child.type in ("pointer_declarator", "reference_declarator", "function_declarator"):
                    current = child
                    break
            else:
                break
        elif current.type == "function_declarator":
            break
        else:
            break
    
    return wrappers


def extract_return_type_modifiers(node, source_bytes: bytes) -> Tuple[str, List[str]]:
    """
    从函数定义节点提取返回类型和修饰符
    
    Args:
        node: function_definition 节点
        source_bytes: 源代码字节
    
    Returns:
        (base_return_type, modifiers_list)
        modifiers_list 包含 "const", "*", "&" 等
    """
    base_type = ""
    modifiers = []
    
    for child in node.children:
        # 类型说明符
        if child.type in ("type_identifier", "primitive_type", "sized_type_specifier"):
            base_type = source_bytes[child.start_byte:child.end_byte].decode('utf-8', errors='ignore')
        
        # const 修饰符
        elif child.type == "type_qualifier" and child.text == b"const":
            modifiers.append("const")
        
        # 指针
        elif child.type == "pointer_declarator":
            modifiers.append("*")
            # 检查是否有更多层
            wrappers = find_all_declarator_wrappers(child)
            for w in wrappers:
                if w == "pointer_declarator":
                    modifiers.append("*")
                elif w == "reference_declarator":
                    modifiers.append("&")
        
        # 引用
        elif child.type == "reference_declarator":
            modifiers.append("&")
    
    return base_type, modifiers


# ============================================================================
# 便捷函数
# ============================================================================

def preprocess_source_enhanced(
    source_file: Path,
    include_dirs: List[Path] = None,
    use_clang_format: bool = False
) -> PreprocessingResult:
    """
    便捷函数：增强预处理单个源文件
    """
    preprocessor = EnhancedPreprocessor(
        include_dirs=include_dirs,
        use_clang_format=use_clang_format
    )
    return preprocessor.preprocess(source_file)


def check_source_health(content: str, threshold: float = 0.05) -> ASTHealthReport:
    """
    便捷函数：检查源代码 AST 健康度
    """
    preprocessor = EnhancedPreprocessor(error_threshold=threshold)
    return preprocessor.check_ast_health(content)


if __name__ == "__main__":
    # 测试
    import sys
    
    if len(sys.argv) > 1:
        test_file = Path(sys.argv[1])
        if test_file.exists():
            print(f"测试文件: {test_file}")
            
            preprocessor = EnhancedPreprocessor(use_clang_format=True)
            result = preprocessor.preprocess(test_file)
            
            print(f"\n预处理结果:")
            print(f"  成功: {result.success}")
            print(f"  方法: {result.method}")
            print(f"  内容长度: {len(result.content)} 字节")
            
            if result.health_report:
                hr = result.health_report
                print(f"\nAST 健康检查:")
                print(f"  总节点数: {hr.total_nodes}")
                print(f"  错误节点: {hr.error_nodes}")
                print(f"  错误率: {hr.error_rate:.2%}")
                print(f"  健康: {'✅' if hr.is_healthy else '❌'}")
                if hr.recommendation:
                    print(f"  建议: {hr.recommendation}")
            
            if result.warnings:
                print(f"\n警告:")
                for w in result.warnings:
                    print(f"  - {w}")
        else:
            print(f"文件不存在: {test_file}")
    else:
        print("用法: python preprocessing.py <源文件路径>")
