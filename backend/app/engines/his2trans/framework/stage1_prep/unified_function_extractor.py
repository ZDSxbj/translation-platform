#!/usr/bin/env python3
"""
统一函数提取器 (Unified Function Extractor)

基于 libclang 实现，作为函数提取和骨架生成的"唯一真理来源 (Single Source of Truth)"。

功能：
1. 使用 libclang 准确解析 C/C++ 代码
2. 支持 compile_commands.json 获取正确的编译参数
3. 正确处理宏、条件编译、属性等复杂情况
4. 输出 JSON 格式的函数列表

使用方法：
    # 提取单个文件
    python unified_function_extractor.py --file source.c --output functions.json
    
    # 提取整个项目（使用 compile_commands.json）
    python unified_function_extractor.py --project /path/to/project --output functions.json
    
    # 对比测试模式
    python unified_function_extractor.py --compare --project /path/to/project

作者: AI Assistant
日期: 2024-12
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 尝试导入 clang
try:
    import clang.cindex as ci
    from clang.cindex import CursorKind, TypeKind
    
    # 配置 libclang 库路径（pip install libclang 安装的位置）
    import os
    import sys
    
    # 允许“较新的 Python bindings + 较旧的系统 libclang”在缺少非关键符号时继续工作。
    # 当前环境里 pip 安装的是 clang 21.x，但系统只有 libclang 14；若不关闭严格检查，
    # bindings 会在加载阶段因缺少新符号（例如 clang_CXXMethod_isDeleted）直接失败。
    try:
        ci.Config.set_compatibility_check(False)
    except Exception:
        pass

    # 尝试多个可能的路径（优先使用系统 clang 14）
    possible_libclang_files = [
        # 系统 clang 14（推荐，与 clang==14.0 Python bindings 兼容）
        '/usr/lib/llvm-14/lib/libclang.so.1',
        '/usr/lib/llvm-14/lib/libclang-14.so.1',
        '/usr/lib/llvm-14/lib/libclang.so',
        # 其他系统版本
        '/usr/lib/llvm-15/lib/libclang.so.1',
        '/usr/lib/llvm-16/lib/libclang.so.1',
        '/usr/lib/x86_64-linux-gnu/libclang-14.so.1',
        '/usr/lib/x86_64-linux-gnu/libclang.so.1',
        # pip 安装的版本（可能版本不兼容）
        os.path.join(sys.prefix, 'lib', 'python' + '.'.join(map(str, sys.version_info[:2])), 
                     'site-packages', 'clang', 'native', 'libclang.so'),
    ]
    
    libclang_found = False
    for libclang_path in possible_libclang_files:
        if os.path.exists(libclang_path):
            ci.Config.set_library_file(libclang_path)
            logger.info(f"使用 libclang: {libclang_path}")
            libclang_found = True
            break
    
    if not libclang_found:
        logger.warning("未找到 libclang.so，请安装 clang: sudo apt install libclang-14-dev")
    
    LIBCLANG_AVAILABLE = True
except ImportError:
    LIBCLANG_AVAILABLE = False
    logger.warning("libclang 未安装，请运行: pip install libclang clang")


@dataclass
class FileCoverageInfo:
    """单文件覆盖率信息"""
    file_path: str                      # 文件路径
    expected_count: int                 # tree-sitter 预期函数数
    libclang_count: int                 # libclang 提取函数数
    diagnostic_errors: int              # 解析错误数
    skipped_location_mismatch: int      # 因位置不匹配跳过的函数数
    coverage_ratio: float               # 覆盖率 (libclang_count / expected_count)
    missing_functions: List[str]        # 缺失的函数名列表

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


@dataclass
class CoverageReport:
    """全局覆盖率报告"""
    total_files: int                    # 总文件数
    total_expected: int                 # tree-sitter 预期总函数数
    total_libclang: int                 # libclang 提取总函数数
    total_diagnostic_errors: int        # 总解析错误数
    total_skipped_location: int         # 总跳过函数数（位置不匹配）
    overall_coverage: float             # 总覆盖率
    files_with_low_coverage: List[str]  # 覆盖率低于阈值的文件
    per_file_coverage: List[FileCoverageInfo]  # 每文件覆盖率

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'total_files': self.total_files,
            'total_expected': self.total_expected,
            'total_libclang': self.total_libclang,
            'total_diagnostic_errors': self.total_diagnostic_errors,
            'total_skipped_location': self.total_skipped_location,
            'overall_coverage': self.overall_coverage,
            'files_with_low_coverage': self.files_with_low_coverage,
            'per_file_coverage': [f.to_dict() for f in self.per_file_coverage],
        }


@dataclass
class UnifiedFunction:
    """统一的函数数据结构"""
    name: str                           # 函数名
    source_file: str                    # 所在源文件（相对路径）
    source_file_abs: str                # 所在源文件（绝对路径）
    c_signature: str                    # C 签名（完整）
    return_type: str                    # 返回类型
    parameters: List[Tuple[str, str]]   # [(参数名, 参数类型), ...]
    body: str                           # 函数体（可选）
    is_static: bool                     # 是否静态函数
    is_inline: bool                     # 是否内联函数
    attributes: List[str]               # 属性列表（如 __attribute__((interrupt))）
    start_line: int                     # 起始行号
    end_line: int                       # 结束行号
    mangled_name: str                   # C++ 修饰名（如果有）
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UnifiedFunction':
        """从字典创建"""
        # 处理 parameters 的元组转换
        if 'parameters' in data:
            data['parameters'] = [tuple(p) if isinstance(p, list) else p for p in data['parameters']]
        return cls(**data)


@dataclass
class UnifiedGlobalVar:
    """统一的全局变量数据结构"""
    name: str                           # 变量名
    source_file: str                    # 所在源文件（相对路径）
    source_file_abs: str                # 所在源文件（绝对路径）
    c_type: str                         # C 类型
    c_declaration: str                  # 完整的 C 声明
    is_static: bool                     # 是否静态变量
    is_const: bool                      # 是否 const
    is_extern: bool                     # 是否 extern 声明
    is_array: bool                      # 是否数组
    array_size: Optional[int]           # 数组大小（如果是数组）
    initial_value: Optional[str]        # 初始值（如果有）
    start_line: int                     # 起始行号
    end_line: int                       # 结束行号
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UnifiedGlobalVar':
        """从字典创建"""
        return cls(**data)


@dataclass 
class UnifiedDependency:
    """统一的函数依赖数据结构"""
    caller: str                         # 调用者函数名
    caller_file: str                    # 调用者所在文件
    callees: List[str]                  # 被调用的函数列表
    external_calls: List[str]           # 外部调用（不在项目中的函数）
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UnifiedDependency':
        """从字典创建"""
        return cls(**data)


class CompileCommandsLoader:
    """compile_commands.json 加载器（增强版）

    支持：
    1. 精确文件匹配
    2. 同目录文件参数继承（fallback）
    3. 全局 include 路径合并
    4. 额外 include 路径（从 skeleton_builder.include_dirs 传入）
    5. 自动检测 C/C++ 文件类型
    """

    # C++ 文件扩展名
    CPP_EXTENSIONS = {'.cpp', '.cc', '.cxx', '.hpp', '.hh', '.hxx', '.C'}

    def __init__(self, compile_commands_path: Optional[Path] = None,
                 extra_include_dirs: Optional[List[str]] = None):
        self.commands: Dict[str, Dict] = {}
        self.dir_commands: Dict[str, Dict] = {}  # 按目录索引：{目录路径: 第一个文件的编译参数}
        self.basename_index: Dict[str, List[str]] = {}  # 按文件名索引：{basename: [abs_path, ...]}
        self.all_include_dirs: List[str] = []     # 所有 include 路径
        self.extra_include_dirs: List[str] = extra_include_dirs or []  # 额外 include 路径

        # 默认参数（C 语言，会根据文件扩展名自动切换）
        self.default_args_c = [
            '-x', 'c',
            '-std=c11',
            '-D__GNUC__=4',
            '-D__GNUC_MINOR__=2',
            '-D__GNUC_PATCHLEVEL__=1',
        ]
        self.default_args_cpp = [
            '-x', 'c++',
            '-std=c++14',
            '-D__GNUC__=4',
            '-D__GNUC_MINOR__=2',
            '-D__GNUC_PATCHLEVEL__=1',
        ]

        if compile_commands_path and compile_commands_path.exists():
            self._load(compile_commands_path)

    def _is_cpp_file(self, file_path: Path) -> bool:
        """判断文件是否是 C++ 文件"""
        return file_path.suffix.lower() in self.CPP_EXTENSIONS

    def _get_default_args(self, source_file: Path) -> List[str]:
        """根据文件类型获取默认参数"""
        if self._is_cpp_file(source_file):
            return self.default_args_cpp.copy()
        return self.default_args_c.copy()

    def _load(self, path: Path):
        """加载 compile_commands.json"""
        try:
            logger.info(f"开始加载 compile_commands.json: {path}")
            with open(path, 'r', encoding='utf-8') as f:
                commands_list = json.load(f)

            logger.info(f"compile_commands.json 已读取，条目数: {len(commands_list)}")

            # 如果已经传入了大量 extra_include_dirs（例如来自 CompileCommandsParser 的全量 include_dirs），
            # 则无需再次逐条解析 compile_commands 来收集 include 路径（这一步会非常慢）。
            collect_include_dirs = not bool(self.extra_include_dirs)
            include_dirs_set = set()

            for i, entry in enumerate(commands_list):
                file_path = entry.get('file', '')
                if file_path:
                    # 规范化路径：compile_commands.json 的 file 往往是相对 entry['directory'] 的相对路径
                    # 性能关键：compile_commands 可能有 10w 级条目；Path.resolve() 逐条调用会非常慢。
                    # 这里用 os.path 做纯字符串归一化（绝对路径 + normpath），足够用于匹配。
                    entry_dir = entry.get('directory', '') or '.'
                    if not os.path.isabs(entry_dir):
                        entry_dir = os.path.abspath(entry_dir)

                    if os.path.isabs(file_path):
                        abs_path = os.path.normpath(file_path)
                    else:
                        abs_path = os.path.normpath(os.path.join(entry_dir, file_path))
                    abs_path = os.path.abspath(abs_path)

                    self.commands[abs_path] = entry
                    self.basename_index.setdefault(os.path.basename(abs_path), []).append(abs_path)

                    # 按目录索引（用于 fallback）
                    dir_path = os.path.dirname(abs_path)
                    if dir_path not in self.dir_commands:
                        self.dir_commands[dir_path] = entry

                    # 收集所有 include 路径（可选：当未提供 extra_include_dirs 时才做）
                    if collect_include_dirs:
                        command = entry.get('command', '') or ' '.join(entry.get('arguments', []))
                        for inc in self._extract_include_dirs(command, entry.get('directory', '')):
                            include_dirs_set.add(inc)

                # 进度日志（避免“看起来卡住”）
                if (i + 1) % 50000 == 0:
                    logger.info(f"compile_commands 载入进度: {i + 1}/{len(commands_list)}")

            if collect_include_dirs:
                self.all_include_dirs = sorted(include_dirs_set)
            else:
                self.all_include_dirs = []

            if collect_include_dirs:
                logger.info(f"加载了 {len(self.commands)} 条编译命令, {len(self.all_include_dirs)} 个 include 路径")
            else:
                logger.info(
                    f"加载了 {len(self.commands)} 条编译命令（跳过 include 路径提取，extra_include_dirs={len(self.extra_include_dirs)}）"
                )
        except Exception as e:
            logger.warning(f"加载 compile_commands.json 失败: {e}")

    def _extract_include_dirs(self, command: str, directory: str) -> List[str]:
        """从编译命令中提取 include 路径"""
        import shlex
        include_dirs = []

        try:
            parts = shlex.split(command)
        except ValueError:
            parts = command.split()

        skip_next = False
        for i, part in enumerate(parts):
            if skip_next:
                skip_next = False
                continue

            def _add_include_path(p: str):
                if not p:
                    return
                inc = p
                if not Path(inc).is_absolute():
                    inc = str(Path(directory) / inc)
                include_dirs.append(inc)

            if part == '-I' and i + 1 < len(parts):
                _add_include_path(parts[i + 1])
                skip_next = True
                continue
            if part.startswith('-I') and part != '-I':
                _add_include_path(part[2:])
                continue

            if part == '-isystem' and i + 1 < len(parts):
                _add_include_path(parts[i + 1])
                skip_next = True
                continue
            if part.startswith('-isystem') and part != '-isystem':
                _add_include_path(part[len('-isystem'):])
                continue

        return include_dirs

    def _suffix_match_len(self, a: Path, b: Path) -> int:
        """计算两个路径的公共后缀（按 path parts）长度"""
        try:
            a_parts = a.parts
            b_parts = b.parts
        except Exception:
            return 0
        i = 1
        cnt = 0
        while i <= len(a_parts) and i <= len(b_parts):
            if a_parts[-i] != b_parts[-i]:
                break
            cnt += 1
            i += 1
        return cnt

    def _split_entry_command(self, entry: Dict) -> List[str]:
        import shlex

        if isinstance(entry.get('arguments'), list) and entry.get('arguments'):
            return list(entry.get('arguments'))
        command = entry.get('command', '') or ''
        if not command:
            return []
        try:
            return shlex.split(command)
        except ValueError:
            return command.split()

    def _normalize_path_like(self, value: str, directory: str) -> str:
        """将相对路径按 directory 归一为绝对路径（best-effort）"""
        if not value:
            return value
        try:
            p = Path(value)
            if not p.is_absolute():
                p = Path(directory) / p
            return str(p.resolve(strict=False))
        except Exception:
            return value

    def _filter_clang_args(self, parts: List[str], directory: str, source_file_abs: str) -> List[str]:
        """
        从 compile_commands 的 command/arguments 中提取适用于 libclang 的 clang 参数。

        关键要求：
        - 尽量保留 target/sysroot/isystem/-include/-D 等“会影响类型解析”的参数
        - 丢弃 -o/-c/-M* 等与解析无关且可能干扰的参数
        - 将相对路径按 entry['directory'] 归一为绝对路径
        """
        if not parts:
            return []

        # 1) 去掉前缀的 wrapper/编译器：保留从第一个 '-' 开始的参数序列
        i0 = 0
        while i0 < len(parts) and not parts[i0].startswith('-'):
            i0 += 1
        parts = parts[i0:]

        skip_with_arg = {
            '-o', '-MF', '-MT', '-MQ',
        }
        skip_alone = {
            '-c',
            '-MD', '-MMD', '-MP', '-MM', '-M',
        }
        path_flag_next = {
            '-I', '-isystem', '-iquote', '-idirafter',
            '-isysroot', '--sysroot',
            '-include', '-imacros',
            '--gcc-toolchain', '-resource-dir',
        }

        out: List[str] = []
        idx = 0
        while idx < len(parts):
            p = parts[idx]

            # 2) 跳过输出/依赖生成相关参数
            if p in skip_with_arg:
                idx += 2
                continue
            if p in skip_alone:
                idx += 1
                continue

            # 3) 跳过输入源文件路径（Index.parse 会单独传 file）
            if p == source_file_abs:
                idx += 1
                continue
            try:
                if not Path(p).is_absolute():
                    maybe = str((Path(directory) / p).resolve(strict=False))
                    if maybe == source_file_abs:
                        idx += 1
                        continue
            except Exception:
                pass

            # 4) 跳过 -Werror（libclang 解析只需要类型信息，避免把 warning 变 error）
            if p == '-Werror' or p.startswith('-Werror='):
                idx += 1
                continue

            # 4.5) 跳过 sanitizer 相关参数（不影响类型解析，且经常引用不存在的 ignorelist/blocklist 引发噪声错误）
            if p.startswith('-fsanitize') or p.startswith('-fno-sanitize'):
                idx += 1
                continue

            # 5) 处理 path 参数（独立 token）
            if p in path_flag_next and idx + 1 < len(parts):
                v = parts[idx + 1]
                v = self._normalize_path_like(v, directory)
                out.extend([p, v])
                idx += 2
                continue

            # 6) 处理 path 参数（拼接形式）
            if p.startswith('-I') and p != '-I':
                out.extend(['-I', self._normalize_path_like(p[2:], directory)])
                idx += 1
                continue
            if p.startswith('-isystem') and p != '-isystem':
                out.extend(['-isystem', self._normalize_path_like(p[len('-isystem'):], directory)])
                idx += 1
                continue
            if p.startswith('--sysroot='):
                out.append(f"--sysroot={self._normalize_path_like(p.split('=',1)[1], directory)}")
                idx += 1
                continue
            if p.startswith('-isysroot') and p != '-isysroot':
                out.extend(['-isysroot', self._normalize_path_like(p[len('-isysroot'):], directory)])
                idx += 1
                continue

            # 7) 其他参数：原样保留
            out.append(p)
            idx += 1

        return out

    def _extract_existing_include_dirs(self, args: List[str]) -> set:
        """从 args 中抽取已有 include dir（用于去重追加 extra_include_dirs）"""
        import re
        existing = set()
        i = 0
        while i < len(args):
            a = args[i]
            if a in ('-I', '-isystem', '-iquote', '-idirafter') and i + 1 < len(args):
                existing.add(args[i + 1])
                i += 2
                continue
            if a.startswith('-I') and a != '-I':
                existing.add(a[2:])
                i += 1
                continue
            if a.startswith('-isystem') and a != '-isystem':
                existing.add(a[len('-isystem'):])
                i += 1
                continue
            i += 1
        # 归一化（去掉重复的 // 或尾随 /）
        normalized = set()
        for p in existing:
            try:
                normalized.add(re.sub(r'/+$', '', p))
            except Exception:
                normalized.add(p)
        return normalized

    def get_args_for_file(self, source_file: Path) -> List[str]:
        """
        获取文件的编译参数

        查找顺序：
        1. 精确匹配：文件路径完全匹配
        2. 目录匹配：同目录下其他文件的参数（fallback）
        3. 全局 include + extra_include_dirs：合并所有已知 include 路径
        4. 默认参数 + extra_include_dirs：使用预设的默认参数（根据文件类型自动选择 C/C++）
        """
        abs_path = str(source_file.resolve())

        # 1. 精确匹配
        if abs_path in self.commands:
            entry = self.commands[abs_path]
            command = entry.get('command', '') or ' '.join(entry.get('arguments', []))
            parts = self._split_entry_command(entry)
            args = self._filter_clang_args(parts, entry.get('directory', ''), abs_path)
            # 防御：当 extra_include_dirs 非常多（例如几千个）时，
            # 逐个追加会导致解析极慢且 args 过长；精确匹配时通常不需要追加。
            if self.extra_include_dirs and len(self.extra_include_dirs) <= 200:
                existing_dirs = self._extract_existing_include_dirs(args)
                for inc_dir in self.extra_include_dirs:
                    inc_dir_norm = inc_dir.rstrip('/')
                    if inc_dir_norm not in existing_dirs:
                        args.extend(['-I', inc_dir])
                        existing_dirs.add(inc_dir_norm)
            return args

        # 2. 目录匹配（fallback）
        dir_path = str(source_file.resolve().parent)
        if dir_path in self.dir_commands:
            entry = self.dir_commands[dir_path]
            command = entry.get('command', '') or ' '.join(entry.get('arguments', []))
            parts = self._split_entry_command(entry)
            args = self._filter_clang_args(parts, entry.get('directory', ''), abs_path)
            if self.extra_include_dirs and len(self.extra_include_dirs) <= 200:
                existing_dirs = self._extract_existing_include_dirs(args)
                for inc_dir in self.extra_include_dirs:
                    inc_dir_norm = inc_dir.rstrip('/')
                    if inc_dir_norm not in existing_dirs:
                        args.extend(['-I', inc_dir])
                        existing_dirs.add(inc_dir_norm)
            logger.debug(f"使用目录 fallback 参数: {source_file.name}")
            return args

        # 2.5 文件名/后缀匹配（用于 self_contained_modules / workspace 拷贝路径）
        basename = source_file.name
        candidate_paths = self.basename_index.get(basename, [])
        if candidate_paths:
            src_p = Path(abs_path)
            best = None
            best_len = -1
            for c in candidate_paths:
                try:
                    c_p = Path(c)
                except Exception:
                    continue
                l = self._suffix_match_len(src_p, c_p)
                if l > best_len:
                    best_len = l
                    best = c
            # 若候选很多且只有文件名匹配（len=1），认为不可靠，继续走全局 include fallback
            if best and (len(candidate_paths) == 1 or best_len >= 2):
                entry = self.commands.get(best)
                if entry:
                    parts = self._split_entry_command(entry)
                    args = self._filter_clang_args(parts, entry.get('directory', ''), best)
                    logger.info(f"使用 suffix 匹配的 compile_commands 条目: {basename} (suffix_len={best_len})")
                    return args

        # 3. 全局 include 路径 + extra_include_dirs + 默认参数
        combined_includes = list(self.all_include_dirs) + list(self.extra_include_dirs)
        if combined_includes:
            args = self._get_default_args(source_file)  # 根据文件类型选择 C/C++
            seen = set()
            for inc_dir in combined_includes[:100]:  # 限制数量避免命令行过长
                if inc_dir not in seen:
                    args.append(f'-I{inc_dir}')
                    seen.add(inc_dir)
            logger.debug(f"使用全局/extra include 路径: {source_file.name} ({len(seen)} dirs)")
            return args

        # 4. 默认参数（根据文件类型）
        return self._get_default_args(source_file)

    # NOTE: _parse_command 已被 _filter_clang_args/_split_entry_command 替代，
    # 这里保留位置以避免历史引用（如有）。请勿再新增调用。


class UnifiedFunctionExtractor:
    """统一函数提取器（基于 libclang）"""

    # 支持的 C++ 函数类型 CursorKind
    CPP_FUNCTION_KINDS = [
        CursorKind.FUNCTION_DECL,          # C 函数和 C++ 自由函数
        CursorKind.CXX_METHOD,             # C++ 成员方法
        CursorKind.CONSTRUCTOR,            # C++ 构造函数
        CursorKind.DESTRUCTOR,             # C++ 析构函数
        CursorKind.CONVERSION_FUNCTION,    # C++ 类型转换运算符
        CursorKind.FUNCTION_TEMPLATE,      # C++ 函数模板
    ] if LIBCLANG_AVAILABLE else []

    def __init__(self,
                 compile_commands_path: Optional[Path] = None,
                 project_root: Optional[Path] = None,
                 coverage_threshold: float = 0.6,
                 include_headers: bool = False,
                 extra_include_dirs: Optional[List[str]] = None):
        """
        初始化提取器

        Args:
            compile_commands_path: compile_commands.json 的路径
            project_root: 项目根目录（用于计算相对路径）
            coverage_threshold: 覆盖率阈值，低于此值会触发警告（默认 0.6）
            include_headers: 是否处理头文件中的函数定义（默认 False）
            extra_include_dirs: 额外的 include 路径（从 skeleton_builder.include_dirs 传入）
        """
        if not LIBCLANG_AVAILABLE:
            raise RuntimeError("libclang 未安装，请运行: pip install libclang clang")

        self.index = ci.Index.create()
        self.compile_loader = CompileCommandsLoader(
            compile_commands_path,
            extra_include_dirs=extra_include_dirs
        )
        self.project_root = project_root or Path.cwd()
        self.coverage_threshold = coverage_threshold
        self.include_headers = include_headers

        # 统计信息
        self.stats = {
            'total_files': 0,
            'total_functions': 0,
            'failed_files': 0,
            'parse_errors': 0,
        }

        # 覆盖率跟踪
        self._coverage_info: List[FileCoverageInfo] = []
        self._skipped_location_count = 0  # 当前文件因位置不匹配跳过的函数数

        # 诊断信息（用于定位 build context / include / macro 问题）
        # key: 相对 project_root 的文件路径（或 basename），value: 诊断列表
        self.diagnostics_by_file: Dict[str, List[Dict[str, Any]]] = {}
    
    def extract_from_file(self, source_file: Path, include_body: bool = False) -> List[UnifiedFunction]:
        """
        从单个文件提取所有函数

        Args:
            source_file: 源文件路径
            include_body: 是否包含函数体

        Returns:
            函数列表
        """
        functions = []
        source_file = Path(source_file).resolve()

        # 重置当前文件的跳过计数
        self._skipped_location_count = 0
        diagnostic_errors = 0

        if not source_file.exists():
            logger.warning(f"文件不存在: {source_file}")
            return functions

        # 获取编译参数
        args = self.compile_loader.get_args_for_file(source_file)

        # 添加通用参数
        args.extend([
            '-ferror-limit=0',      # 不限制错误数量
            '-w',                    # 禁用警告
            # OpenHarmony/LiteOS 常用宏定义，将大写关键字宏转换为标准 C 关键字
            '-DSTATIC=static',      # OpenHarmony 使用 STATIC 替代 static
            '-DINLINE=inline',      # OpenHarmony 使用 INLINE 替代 inline
            '-DVOID=void',          # OpenHarmony 使用 VOID 替代 void（某些情况）
            '-DCONST=const',        # OpenHarmony 使用 CONST 替代 const
            '-DVOLATILE=volatile',  # OpenHarmony 使用 VOLATILE 替代 volatile
            '-D__attribute__(x)=',  # 忽略 GCC 特定属性（如果未定义）
            # LiteOS 段属性宏（用于指定代码段，不影响函数签名解析）
            '-DLITE_OS_SEC_TEXT=',
            '-DLITE_OS_SEC_TEXT_MINOR=',
            '-DLITE_OS_SEC_TEXT_INIT=',
            '-DLITE_OS_SEC_DATA=',
            '-DLITE_OS_SEC_DATA_MINOR=',
            '-DLITE_OS_SEC_DATA_INIT=',
            '-DLITE_OS_SEC_BSS=',
            '-DLITE_OS_SEC_BSS_MINOR=',
            '-DLITE_OS_SEC_BSS_INIT=',
            '-DLITE_OS_SEC_RODATA=',
            '-DLITE_OS_SEC_ITCM=',
            '-DLITE_OS_SEC_DTCM=',
            # LiteOS 其他常用修饰符宏
            '-DLOSCFG_DEBUG_TOOLS=0',
            '-DLOSCFG_KERNEL_TRACE=0',
        ])

        try:
            # 解析文件
            tu = self.index.parse(
                str(source_file),
                args=args,
                options=ci.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD
            )

            # 检查解析错误
            errors = [d for d in tu.diagnostics if d.severity >= ci.Diagnostic.Error]
            diagnostic_errors = len(errors)
            if errors:
                self.stats['parse_errors'] += len(errors)
                # 记录前若干条错误详情（避免输出过大）
                try:
                    try:
                        rel_key = str(source_file.relative_to(self.project_root))
                    except ValueError:
                        rel_key = source_file.name

                    err_details: List[Dict[str, Any]] = []
                    for d in errors[:50]:
                        loc = getattr(d, "location", None)
                        loc_file = None
                        loc_line = None
                        loc_col = None
                        try:
                            if loc and getattr(loc, "file", None):
                                loc_file = loc.file.name
                                loc_line = loc.line
                                loc_col = loc.column
                        except Exception:
                            pass
                        err_details.append({
                            "severity": int(getattr(d, "severity", 0)),
                            "spelling": getattr(d, "spelling", ""),
                            "location": {
                                "file": loc_file,
                                "line": loc_line,
                                "column": loc_col,
                            },
                        })
                    self.diagnostics_by_file[rel_key] = err_details
                except Exception:
                    # 诊断收集失败不应影响主流程
                    pass

                for err in errors[:3]:  # 默认仅在日志里提示前3条
                    logger.debug(f"libclang 解析错误 {source_file.name}: {err.spelling}")

            # 读取源文件内容（用于提取函数体）
            source_content = None
            if include_body:
                try:
                    with open(source_file, 'r', encoding='utf-8', errors='ignore') as f:
                        source_content = f.read()
                        source_lines = source_content.split('\n')
                except Exception as e:
                    logger.debug(f"读取源文件失败: {e}")

            # 遍历 AST
            for cursor in tu.cursor.get_children():
                self._visit_cursor(cursor, source_file, functions, source_content, include_body)

            self.stats['total_functions'] += len(functions)

        except Exception as e:
            logger.error(f"解析文件失败 {source_file}: {e}")
            self.stats['failed_files'] += 1

        # 记录当前文件的跳过计数（供后续覆盖率计算使用）
        self._last_file_skipped = self._skipped_location_count
        self._last_file_errors = diagnostic_errors

        return functions

    def save_diagnostics_report(self, output_path: Path):
        """保存 libclang diagnostics（按文件）到 JSON，便于后续精确定位问题。"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.diagnostics_by_file, f, indent=2, ensure_ascii=False)
        logger.info(f"libclang diagnostics 已保存: {output_path}")
    
    def _visit_cursor(self, cursor, source_file: Path, functions: List[UnifiedFunction],
                      source_content: Optional[str], include_body: bool):
        """递归访问 AST 节点"""
        # 位置检查：默认只处理当前文件中的定义（跳过头文件中的）
        # 如果启用 include_headers，则放宽限制
        if cursor.location.file:
            cursor_file = Path(cursor.location.file.name).resolve()
            if cursor_file != source_file:
                # 根据 include_headers 决定是否跳过
                if not self.include_headers:
                    # 记录跳过的函数（用于覆盖率统计）
                    if cursor.kind in self.CPP_FUNCTION_KINDS and cursor.is_definition():
                        self._skipped_location_count += 1
                    return

        # 处理函数定义（C 函数和 C++ 方法）
        if cursor.kind in self.CPP_FUNCTION_KINDS:
            if cursor.is_definition():
                func = self._parse_function(cursor, source_file, source_content, include_body)
                if func:
                    functions.append(func)

        # 递归处理子节点
        for child in cursor.get_children():
            self._visit_cursor(child, source_file, functions, source_content, include_body)
    
    def _parse_function(self, cursor, source_file: Path, 
                        source_content: Optional[str], include_body: bool) -> Optional[UnifiedFunction]:
        """解析函数定义"""
        try:
            # 函数名
            func_name = cursor.spelling
            if not func_name:
                return None
            
            # 检查是否是有效的 C 标识符
            import re
            if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', func_name):
                logger.debug(f"跳过非法函数名: {func_name}")
                return None
            
            # 返回类型
            return_type = cursor.result_type.spelling
            
            # 参数列表
            parameters = []
            for arg in cursor.get_arguments():
                param_name = arg.spelling or f'arg_{len(parameters)}'
                param_type = arg.type.spelling
                parameters.append((param_name, param_type))
            
            # 检查修饰符
            is_static = cursor.storage_class == ci.StorageClass.STATIC
            is_inline = cursor.is_inline_function() if hasattr(cursor, 'is_inline_function') else False
            
            # 提取属性
            attributes = self._extract_attributes(cursor)
            
            # 位置信息
            start_line = cursor.extent.start.line
            end_line = cursor.extent.end.line
            
            # 构建 C 签名
            param_str = ', '.join([f'{ptype} {pname}' for pname, ptype in parameters])
            c_signature = f"{return_type} {func_name}({param_str})"
            if is_static:
                c_signature = f"static {c_signature}"
            if is_inline:
                c_signature = f"inline {c_signature}"
            
            # 提取函数体（可选）
            body = ""
            if include_body and source_content:
                try:
                    lines = source_content.split('\n')
                    body_lines = lines[start_line - 1:end_line]
                    body = '\n'.join(body_lines)
                except Exception:
                    pass
            
            # 计算相对路径
            try:
                rel_path = source_file.relative_to(self.project_root)
            except ValueError:
                rel_path = source_file.name
            
            # C++ 修饰名
            mangled_name = cursor.mangled_name if hasattr(cursor, 'mangled_name') else func_name
            
            return UnifiedFunction(
                name=func_name,
                source_file=str(rel_path),
                source_file_abs=str(source_file),
                c_signature=c_signature,
                return_type=return_type,
                parameters=parameters,
                body=body,
                is_static=is_static,
                is_inline=is_inline,
                attributes=attributes,
                start_line=start_line,
                end_line=end_line,
                mangled_name=mangled_name
            )
            
        except Exception as e:
            logger.debug(f"解析函数失败: {e}")
            return None
    
    def _extract_attributes(self, cursor) -> List[str]:
        """提取函数属性"""
        attributes = []
        
        try:
            # 遍历子节点查找属性
            for child in cursor.get_children():
                if child.kind in [CursorKind.ANNOTATE_ATTR, 
                                  CursorKind.VISIBILITY_ATTR,
                                  CursorKind.UNEXPOSED_ATTR]:
                    attr_text = child.spelling
                    if attr_text:
                        attributes.append(attr_text)
            
            # 从 raw 注释中提取 __attribute__
            if cursor.raw_comment:
                import re
                attr_matches = re.findall(r'__attribute__\s*\(\([^)]+\)\)', cursor.raw_comment)
                attributes.extend(attr_matches)
                
        except Exception as e:
            logger.debug(f"提取属性失败: {e}")
        
        return attributes
    
    def extract_from_project(self, 
                             source_files: List[Path], 
                             include_body: bool = False,
                             max_workers: int = 4) -> List[UnifiedFunction]:
        """
        从项目中提取所有函数（并行处理）
        
        Args:
            source_files: 源文件列表
            include_body: 是否包含函数体
            max_workers: 最大并行数
            
        Returns:
            所有函数列表
        """
        all_functions = []
        self.stats['total_files'] = len(source_files)
        
        logger.info(f"开始提取 {len(source_files)} 个文件...")
        
        # 串行处理（libclang 在多线程下可能有问题）
        for i, source_file in enumerate(source_files):
            if (i + 1) % 10 == 0:
                logger.info(f"进度: {i + 1}/{len(source_files)}")
            
            functions = self.extract_from_file(source_file, include_body)
            all_functions.extend(functions)
        
        logger.info(f"提取完成: {len(all_functions)} 个函数")
        return all_functions
    
    def get_stats(self) -> Dict[str, int]:
        """获取统计信息"""
        return self.stats.copy()

    def _count_functions_treesitter(self, source_file: Path) -> Tuple[int, List[str]]:
        """
        使用 tree-sitter 快速计数文件中的函数定义（作为预期值）

        Args:
            source_file: 源文件路径

        Returns:
            (函数数量, 函数名列表)
        """
        try:
            # 延迟导入 tree-sitter（避免强依赖）
            from tree_sitter import Language, Parser
            import tree_sitter_cpp as tscpp

            try:
                cpp_lang = Language(tscpp.language(), "cpp")
            except TypeError:
                cpp_lang = Language(tscpp.language())
            ts_parser = Parser()
            try:
                ts_parser.set_language(cpp_lang)
            except Exception:
                ts_parser = Parser(cpp_lang)

            with open(source_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            tree = ts_parser.parse(bytes(content, 'utf-8'))

            func_names = []

            def find_functions(node):
                """递归查找函数定义"""
                if node.type == 'function_definition':
                    # 查找函数名
                    for child in node.children:
                        if child.type == 'function_declarator':
                            for sub in child.children:
                                if sub.type == 'identifier':
                                    func_names.append(content[sub.start_byte:sub.end_byte])
                                    break
                            break
                        elif child.type == 'identifier':
                            func_names.append(content[child.start_byte:child.end_byte])
                            break

                for child in node.children:
                    find_functions(child)

            find_functions(tree.root_node)
            return len(func_names), func_names

        except Exception as e:
            logger.debug(f"tree-sitter 计数失败 {source_file}: {e}")
            return 0, []

    def extract_with_coverage(self, source_file: Path, include_body: bool = False
                             ) -> Tuple[List[UnifiedFunction], FileCoverageInfo]:
        """
        提取函数并返回覆盖率信息

        Args:
            source_file: 源文件路径
            include_body: 是否包含函数体

        Returns:
            (函数列表, 覆盖率信息)
        """
        source_file = Path(source_file).resolve()

        # 使用 tree-sitter 获取预期函数数
        expected_count, expected_names = self._count_functions_treesitter(source_file)

        # 使用 libclang 提取
        functions = self.extract_from_file(source_file, include_body)
        libclang_count = len(functions)
        libclang_names = {f.name for f in functions}

        # 计算覆盖率
        coverage_ratio = libclang_count / expected_count if expected_count > 0 else 1.0

        # 找出缺失的函数
        missing = [name for name in expected_names if name not in libclang_names]

        # 构建覆盖率信息
        try:
            rel_path = source_file.relative_to(self.project_root)
        except ValueError:
            rel_path = source_file.name

        coverage_info = FileCoverageInfo(
            file_path=str(rel_path),
            expected_count=expected_count,
            libclang_count=libclang_count,
            diagnostic_errors=getattr(self, '_last_file_errors', 0),
            skipped_location_mismatch=getattr(self, '_last_file_skipped', 0),
            coverage_ratio=coverage_ratio,
            missing_functions=missing
        )

        # 添加到全局覆盖率列表
        self._coverage_info.append(coverage_info)

        return functions, coverage_info

    def extract_from_project_with_coverage(self, source_files: List[Path],
                                           include_body: bool = False
                                          ) -> Tuple[List[UnifiedFunction], CoverageReport]:
        """
        从项目中提取所有函数并生成覆盖率报告

        Args:
            source_files: 源文件列表
            include_body: 是否包含函数体

        Returns:
            (所有函数列表, 覆盖率报告)
        """
        all_functions = []
        self.stats['total_files'] = len(source_files)
        self._coverage_info = []  # 重置覆盖率列表

        logger.info(f"开始提取 {len(source_files)} 个文件（带覆盖率跟踪）...")

        for i, source_file in enumerate(source_files):
            if (i + 1) % 10 == 0:
                logger.info(f"进度: {i + 1}/{len(source_files)}")

            functions, _ = self.extract_with_coverage(source_file, include_body)
            all_functions.extend(functions)

        # 生成覆盖率报告
        report = self.generate_coverage_report()

        logger.info(f"提取完成: {len(all_functions)} 个函数, 覆盖率: {report.overall_coverage:.1%}")
        return all_functions, report

    def generate_coverage_report(self, coverage_threshold: float = None) -> CoverageReport:
        """
        生成覆盖率报告

        Args:
            coverage_threshold: 覆盖率阈值（默认使用初始化时设置的值）

        Returns:
            覆盖率报告
        """
        if coverage_threshold is None:
            coverage_threshold = self.coverage_threshold

        total_expected = sum(c.expected_count for c in self._coverage_info)
        total_libclang = sum(c.libclang_count for c in self._coverage_info)
        total_errors = sum(c.diagnostic_errors for c in self._coverage_info)
        total_skipped = sum(c.skipped_location_mismatch for c in self._coverage_info)

        overall_coverage = total_libclang / total_expected if total_expected > 0 else 1.0

        # 找出低覆盖率文件
        low_coverage_files = [
            c.file_path for c in self._coverage_info
            if c.coverage_ratio < coverage_threshold and c.expected_count > 0
        ]

        return CoverageReport(
            total_files=len(self._coverage_info),
            total_expected=total_expected,
            total_libclang=total_libclang,
            total_diagnostic_errors=total_errors,
            total_skipped_location=total_skipped,
            overall_coverage=overall_coverage,
            files_with_low_coverage=low_coverage_files,
            per_file_coverage=self._coverage_info
        )

    def save_coverage_report(self, output_path: Path, coverage_threshold: float = None):
        """
        保存覆盖率报告到 JSON 文件

        Args:
            output_path: 输出文件路径
            coverage_threshold: 覆盖率阈值
        """
        report = self.generate_coverage_report(coverage_threshold)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)

        logger.info(f"覆盖率报告已保存: {output_path}")
        logger.info(f"  总覆盖率: {report.overall_coverage:.1%} ({report.total_libclang}/{report.total_expected})")

        if report.files_with_low_coverage:
            logger.warning(f"  低覆盖率文件数: {len(report.files_with_low_coverage)}")
            for f in report.files_with_low_coverage[:5]:
                logger.warning(f"    - {f}")
            if len(report.files_with_low_coverage) > 5:
                logger.warning(f"    ... 还有 {len(report.files_with_low_coverage) - 5} 个")

        return report

    def should_fallback_to_treesitter(self, coverage_threshold: float = None) -> bool:
        """
        根据覆盖率判断是否应该回退到 tree-sitter

        Args:
            coverage_threshold: 覆盖率阈值（默认使用初始化时设置的值）

        Returns:
            True 如果应该回退到 tree-sitter
        """
        if coverage_threshold is None:
            coverage_threshold = self.coverage_threshold

        if not self._coverage_info:
            return False

        report = self.generate_coverage_report()

        # 条件1：总覆盖率低于阈值
        if report.overall_coverage < coverage_threshold:
            logger.warning(
                f"libclang 覆盖率过低 ({report.overall_coverage:.1%} < {coverage_threshold:.0%})，"
                f"建议回退到 tree-sitter"
            )
            return True

        # 条件2：解析错误过多（超过总函数数的 50%）
        if report.total_diagnostic_errors > report.total_expected * 0.5:
            logger.warning(
                f"libclang 解析错误过多 ({report.total_diagnostic_errors})，"
                f"建议回退到 tree-sitter"
            )
            return True

        return False
    
    def extract_global_variables(self, source_file: Path) -> List['UnifiedGlobalVar']:
        """
        从单个文件提取所有全局变量
        
        Args:
            source_file: 源文件路径
            
        Returns:
            全局变量列表
        """
        global_vars = []
        source_file = Path(source_file).resolve()
        
        if not source_file.exists():
            logger.warning(f"文件不存在: {source_file}")
            return global_vars
        
        # 获取编译参数
        args = self.compile_loader.get_args_for_file(source_file)
        args.extend(['-ferror-limit=0', '-w'])
        
        try:
            # 解析文件
            tu = self.index.parse(
                str(source_file),
                args=args,
                options=ci.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD
            )
            
            # 读取源文件内容（用于提取初始值）
            try:
                with open(source_file, 'r', encoding='utf-8', errors='ignore') as f:
                    source_content = f.read()
                    source_lines = source_content.split('\n')
            except Exception:
                source_lines = []
            
            # 遍历顶层节点查找全局变量
            for cursor in tu.cursor.get_children():
                self._visit_global_var(cursor, source_file, global_vars, source_lines)
            
        except Exception as e:
            logger.error(f"解析文件失败 {source_file}: {e}")
        
        return global_vars
    
    def _visit_global_var(self, cursor, source_file: Path, global_vars: List['UnifiedGlobalVar'], 
                          source_lines: List[str]):
        """访问全局变量节点"""
        # 只处理当前文件中的定义
        if cursor.location.file:
            cursor_file = Path(cursor.location.file.name).resolve()
            if cursor_file != source_file:
                return
        
        # 处理变量声明
        if cursor.kind == CursorKind.VAR_DECL:
            # 只处理文件作用域的变量（全局变量）
            if cursor.semantic_parent and cursor.semantic_parent.kind == CursorKind.TRANSLATION_UNIT:
                var = self._parse_global_var(cursor, source_file, source_lines)
                if var:
                    global_vars.append(var)
    
    def _parse_global_var(self, cursor, source_file: Path, 
                          source_lines: List[str]) -> Optional['UnifiedGlobalVar']:
        """解析全局变量"""
        try:
            var_name = cursor.spelling
            if not var_name:
                return None
            
            # 获取类型信息
            var_type = cursor.type.spelling
            
            # 检查修饰符
            is_static = cursor.storage_class == ci.StorageClass.STATIC
            is_extern = cursor.storage_class == ci.StorageClass.EXTERN
            is_const = cursor.type.is_const_qualified()
            
            # 检查是否是数组
            is_array = cursor.type.kind == TypeKind.CONSTANTARRAY
            array_size = None
            if is_array:
                array_size = cursor.type.get_array_size()
            
            # 位置信息
            start_line = cursor.extent.start.line
            end_line = cursor.extent.end.line
            
            # 提取初始值（从源代码行）
            initial_value = None
            if source_lines and start_line > 0:
                try:
                    decl_lines = source_lines[start_line - 1:end_line]
                    decl_text = '\n'.join(decl_lines)
                    if '=' in decl_text:
                        initial_value = decl_text.split('=', 1)[1].strip().rstrip(';').strip()
                except Exception:
                    pass
            
            # 构建完整声明
            c_declaration = var_type + ' ' + var_name
            if is_static:
                c_declaration = 'static ' + c_declaration
            if is_extern:
                c_declaration = 'extern ' + c_declaration
            if array_size:
                c_declaration = c_declaration.replace(var_name, f'{var_name}[{array_size}]')
            if initial_value:
                c_declaration += ' = ' + initial_value
            c_declaration += ';'
            
            # 计算相对路径
            try:
                rel_path = source_file.relative_to(self.project_root)
            except ValueError:
                rel_path = source_file.name
            
            return UnifiedGlobalVar(
                name=var_name,
                source_file=str(rel_path),
                source_file_abs=str(source_file),
                c_type=var_type,
                c_declaration=c_declaration,
                is_static=is_static,
                is_const=is_const,
                is_extern=is_extern,
                is_array=is_array,
                array_size=array_size,
                initial_value=initial_value,
                start_line=start_line,
                end_line=end_line
            )
            
        except Exception as e:
            logger.debug(f"解析全局变量失败: {e}")
            return None
    
    def extract_dependencies(self, functions: List[UnifiedFunction], 
                            source_files: List[Path]) -> List['UnifiedDependency']:
        """
        分析函数调用依赖关系
        
        Args:
            functions: 函数列表
            source_files: 源文件列表
            
        Returns:
            依赖关系列表
        """
        dependencies = []
        
        # 构建函数名集合（用于判断是否是内部调用）
        all_func_names = {f.name for f in functions}
        
        for source_file in source_files:
            source_file = Path(source_file).resolve()
            
            # 获取编译参数
            args = self.compile_loader.get_args_for_file(source_file)
            args.extend(['-ferror-limit=0', '-w'])
            
            try:
                tu = self.index.parse(
                    str(source_file),
                    args=args,
                    options=ci.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD
                )
                
                # 遍历函数定义，分析调用
                for cursor in tu.cursor.get_children():
                    self._analyze_function_calls(cursor, source_file, all_func_names, dependencies)
                    
            except Exception as e:
                logger.debug(f"分析依赖失败 {source_file}: {e}")
        
        return dependencies
    
    def _analyze_function_calls(self, cursor, source_file: Path, 
                                all_func_names: set, dependencies: List['UnifiedDependency']):
        """分析函数中的调用"""
        # 只处理当前文件
        if cursor.location.file:
            cursor_file = Path(cursor.location.file.name).resolve()
            if cursor_file != source_file:
                return
        
        # 处理函数定义
        if cursor.kind == CursorKind.FUNCTION_DECL and cursor.is_definition():
            func_name = cursor.spelling
            if not func_name:
                return
            
            callees = []
            external_calls = []
            
            # 遍历函数体查找调用
            for child in cursor.walk_preorder():
                if child.kind == CursorKind.CALL_EXPR:
                    called_name = child.spelling
                    if called_name:
                        if called_name in all_func_names:
                            if called_name not in callees:
                                callees.append(called_name)
                        else:
                            if called_name not in external_calls:
                                external_calls.append(called_name)
            
            if callees or external_calls:
                try:
                    rel_path = source_file.relative_to(self.project_root)
                except ValueError:
                    rel_path = source_file.name
                
                dependencies.append(UnifiedDependency(
                    caller=func_name,
                    caller_file=str(rel_path),
                    callees=callees,
                    external_calls=external_calls
                ))
        
        # 递归处理子节点（不进入函数体）
        if cursor.kind != CursorKind.FUNCTION_DECL:
            for child in cursor.get_children():
                self._analyze_function_calls(child, source_file, all_func_names, dependencies)
    
    def extract_all(self, source_files: List[Path], include_body: bool = True) -> Dict[str, Any]:
        """
        完整提取：函数、全局变量、依赖关系
        
        Args:
            source_files: 源文件列表
            include_body: 是否包含函数体
            
        Returns:
            包含所有信息的字典
        """
        logger.info(f"开始完整提取 {len(source_files)} 个文件...")
        
        # 1. 提取函数
        functions = self.extract_from_project(source_files, include_body)
        
        # 2. 提取全局变量
        global_vars = []
        for source_file in source_files:
            vars_in_file = self.extract_global_variables(source_file)
            global_vars.extend(vars_in_file)
        logger.info(f"提取了 {len(global_vars)} 个全局变量")
        
        # 3. 分析依赖关系
        dependencies = self.extract_dependencies(functions, source_files)
        logger.info(f"分析了 {len(dependencies)} 个函数的依赖关系")
        
        return {
            'version': '2.0',
            'extractor': 'libclang',
            'total_functions': len(functions),
            'total_global_vars': len(global_vars),
            'total_dependencies': len(dependencies),
            'functions': [f.to_dict() for f in functions],
            'global_vars': [v.to_dict() for v in global_vars],
            'dependencies': [d.to_dict() for d in dependencies],
            'stats': self.get_stats()
        }


def find_source_files(project_root: Path, extensions: List[str] = None,
                      include_headers: bool = False) -> List[Path]:
    """
    查找项目中的所有源文件

    Args:
        project_root: 项目根目录
        extensions: 源文件扩展名列表（默认 ['.c', '.cpp', '.cc', '.cxx']）
        include_headers: 是否包含头文件（默认 False）

    Returns:
        源文件路径列表
    """
    if extensions is None:
        extensions = ['.c', '.cpp', '.cc', '.cxx']
        if include_headers:
            extensions.extend(['.h', '.hpp', '.hh', '.hxx'])

    source_files = []
    for ext in extensions:
        source_files.extend(project_root.rglob(f'*{ext}'))

    # 过滤掉测试文件和构建目录
    filtered = []
    # 使用更精确的模式（目录名而非子串匹配）
    exclude_dirs = {'test', 'tests', 'build', 'cmake-build', '__pycache__', '.git', 'node_modules'}

    for f in source_files:
        # 检查路径的每个目录部分
        path_parts = set(p.lower() for p in f.parts)
        if not path_parts.intersection(exclude_dirs):
            filtered.append(f)

    return sorted(filtered)


def save_functions_json(functions: List[UnifiedFunction], output_path: Path):
    """保存函数列表到 JSON 文件"""
    data = {
        'version': '1.0',
        'extractor': 'libclang',
        'total_functions': len(functions),
        'functions': [f.to_dict() for f in functions]
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"已保存 {len(functions)} 个函数到 {output_path}")


def save_complete_json(data: Dict[str, Any], output_path: Path):
    """保存完整提取结果到 JSON 文件（包含函数、全局变量、依赖关系）"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"已保存完整提取结果到 {output_path}")
    logger.info(f"  - 函数: {data.get('total_functions', 0)}")
    logger.info(f"  - 全局变量: {data.get('total_global_vars', 0)}")
    logger.info(f"  - 依赖关系: {data.get('total_dependencies', 0)}")


def load_functions_json(json_path: Path) -> List[UnifiedFunction]:
    """从 JSON 文件加载函数列表"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    functions = [UnifiedFunction.from_dict(f) for f in data['functions']]
    logger.info(f"从 {json_path} 加载了 {len(functions)} 个函数")
    return functions


def compare_with_treesitter(libclang_functions: List[UnifiedFunction], 
                            treesitter_functions: List[str]) -> Dict[str, Any]:
    """对比 libclang 和 tree-sitter 的提取结果"""
    libclang_names = {f.name for f in libclang_functions}
    treesitter_names = set(treesitter_functions)
    
    only_libclang = libclang_names - treesitter_names
    only_treesitter = treesitter_names - libclang_names
    common = libclang_names & treesitter_names
    
    return {
        'libclang_count': len(libclang_names),
        'treesitter_count': len(treesitter_names),
        'common_count': len(common),
        'only_libclang': sorted(only_libclang),
        'only_treesitter': sorted(only_treesitter),
        'libclang_found_more': len(only_libclang) > len(only_treesitter)
    }


def generate_extracted_files(functions: List[UnifiedFunction], output_dir: Path, include_body: bool = True):
    """
    从函数列表生成 extracted/functions/*.txt 文件
    
    与 get_dependencies.py 的输出格式兼容
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 按源文件分组
    files_funcs = {}
    for func in functions:
        source_file = func.source_file
        if source_file not in files_funcs:
            files_funcs[source_file] = []
        files_funcs[source_file].append(func)
    
    file_count = 0
    for source_file, funcs in files_funcs.items():
        # 生成文件名前缀（与现有格式兼容）
        base_name = Path(source_file).stem
        prefix = f"src_{base_name}"
        
        for i, func in enumerate(funcs, 1):
            output_file = output_dir / f"{prefix}_{i}.txt"
            
            # 如果有函数体，使用它；否则生成签名
            if include_body and func.body:
                content = func.body
            else:
                # 生成函数定义（签名 + 空函数体）
                param_str = ', '.join([f'{ptype} {pname}' for pname, ptype in func.parameters])
                content = f"{func.return_type} {func.name}({param_str})\n{{\n    // Function body\n}}"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            file_count += 1
    
    logger.info(f"生成了 {file_count} 个函数文件到 {output_dir}")
    return file_count


def generate_skeleton_signatures(functions: List[UnifiedFunction]) -> Dict[str, List[dict]]:
    """
    从函数列表生成骨架签名数据
    
    返回按源文件分组的签名数据，供 skeleton_builder.py 使用
    """
    # 按源文件分组
    files_sigs = {}
    for func in functions:
        source_file = func.source_file
        if source_file not in files_sigs:
            files_sigs[source_file] = []
        
        files_sigs[source_file].append({
            'name': func.name,
            'c_signature': func.c_signature,
            'return_type': func.return_type,
            'parameters': func.parameters,
            'is_static': func.is_static,
            'start_line': func.start_line,
            'end_line': func.end_line,
        })
    
    return files_sigs


def main():
    parser = argparse.ArgumentParser(
        description='统一函数提取器（基于 libclang）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
    # 提取单个文件
    python unified_function_extractor.py --file source.c --output functions.json
    
    # 提取整个项目
    python unified_function_extractor.py --project /path/to/project --output functions.json
    
    # 使用 compile_commands.json
    python unified_function_extractor.py --project /path/to/project \\
        --compile-commands /path/to/compile_commands.json --output functions.json
    
    # 生成与现有流程兼容的文件
    python unified_function_extractor.py --project /path/to/project \\
        --generate-extracted /path/to/extracted/functions
    
    # 对比模式
    python unified_function_extractor.py --compare --project /path/to/project
        """
    )
    
    parser.add_argument('--file', type=Path, help='单个源文件路径')
    parser.add_argument('--project', type=Path, help='项目根目录')
    parser.add_argument('--compile-commands', type=Path, help='compile_commands.json 路径')
    parser.add_argument('--output', '-o', type=Path, default=Path('functions.json'), help='输出 JSON 文件路径')
    parser.add_argument('--include-body', action='store_true', help='包含函数体')
    parser.add_argument('--generate-extracted', type=Path, help='生成与 get_dependencies.py 兼容的函数文件目录')
    parser.add_argument('--compare', action='store_true', help='对比模式（与 tree-sitter 对比）')
    parser.add_argument('--extract-all', action='store_true', help='完整提取模式（函数+全局变量+依赖关系）')
    parser.add_argument('--verbose', '-v', action='store_true', help='详细输出')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if not LIBCLANG_AVAILABLE:
        logger.error("libclang 未安装，请运行: pip install libclang clang")
        sys.exit(1)
    
    # 确定项目根目录
    project_root = args.project or (args.file.parent if args.file else Path.cwd())
    
    # 查找 compile_commands.json
    compile_commands_path = args.compile_commands
    if not compile_commands_path:
        # 尝试自动查找
        possible_paths = [
            project_root / 'compile_commands.json',
            project_root / 'build' / 'compile_commands.json',
        ]
        for p in possible_paths:
            if p.exists():
                compile_commands_path = p
                logger.info(f"自动找到 compile_commands.json: {p}")
                break
    
    # 创建提取器
    extractor = UnifiedFunctionExtractor(
        compile_commands_path=compile_commands_path,
        project_root=project_root
    )
    
    # 提取函数
    functions = []
    complete_data = None
    
    if args.file:
        # 单文件模式
        functions = extractor.extract_from_file(args.file, args.include_body)
    elif args.project:
        # 项目模式
        source_files = find_source_files(args.project)
        
        if args.extract_all:
            # 完整提取模式
            complete_data = extractor.extract_all(source_files, args.include_body)
            # 从完整数据中恢复函数列表对象（用于后续处理）
            functions = [UnifiedFunction.from_dict(f) for f in complete_data['functions']]
        else:
            functions = extractor.extract_from_project(source_files, args.include_body)
    else:
        parser.print_help()
        sys.exit(1)
    
    # 保存结果
    if args.extract_all and complete_data:
        save_complete_json(complete_data, args.output)
    else:
        save_functions_json(functions, args.output)
    
    # 如果指定了 --generate-extracted，生成与 get_dependencies.py 兼容的文件
    if args.generate_extracted:
        generate_extracted_files(functions, args.generate_extracted, include_body=args.include_body)
    
    # 打印统计信息
    stats = extractor.get_stats()
    print(f"\n{'='*60}")
    print(f"统计信息:")
    print(f"  总文件数: {stats['total_files']}")
    print(f"  总函数数: {stats['total_functions']}")
    print(f"  失败文件: {stats['failed_files']}")
    print(f"  解析错误: {stats['parse_errors']}")
    
    if complete_data:
        print(f"  全局变量数: {complete_data.get('total_global_vars', 0)}")
        print(f"  依赖关系数: {complete_data.get('total_dependencies', 0)}")
    print(f"{'='*60}")
    
    # 显示函数列表预览
    print(f"\n函数列表预览（前 20 个）:")
    for func in functions[:20]:
        static_mark = '[static] ' if func.is_static else ''
        print(f"  {static_mark}{func.name} @ {func.source_file}:{func.start_line}")
    
    if len(functions) > 20:
        print(f"  ... 还有 {len(functions) - 20} 个函数")


if __name__ == '__main__':
    main()
