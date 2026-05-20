#!/usr/bin/env python3
"""
可配置的类型/常量预定义

此文件包含针对特定平台（如 OpenHarmony/LiteOS）的类型和常量定义。
系统具备自适应学习能力：当发现新的类型/常量后会自动添加到此文件。

用法：
    from config.predefines import PredefineManager
    
    # 获取单例
    predefines = PredefineManager.get_instance()
    
    # 获取所有类型定义
    types = predefines.get_all_types()
    
    # 获取所有常量定义
    constants = predefines.get_all_constants()
    
    # 检查类型是否存在
    if predefines.has_type("HdfDeviceObject"):
        ...
    
    # 自适应学习：发现新类型后自动添加
    predefines.learn_type("NewType", "#[repr(C)]\\npub struct NewType { _opaque: [u8; 0] }")
"""

import json
import logging
import re
import os
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field
from datetime import datetime
import threading

logger = logging.getLogger(__name__)

# =========================================================================
# 配置文件路径
# =========================================================================
# 注意：数据保存到 .cache/learned_data/ 目录，不会被意外删除
# 旧路径保留作为回退
CONFIG_DIR = Path(__file__).parent
PROJECT_ROOT = CONFIG_DIR.parent  # c2-rust_framework 根目录
_cache_root_env = os.environ.get("C2R_CACHE_ROOT", "").strip()
if _cache_root_env:
    CACHE_DIR = Path(_cache_root_env).expanduser().resolve() / "learned_data"
else:
    CACHE_DIR = PROJECT_ROOT / ".cache" / "learned_data"

# 确保目录存在
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# 新的存储路径（在 .cache 下，更安全）
LEARNED_TYPES_PATH = CACHE_DIR / "learned_types.json"
LEARNED_CONSTANTS_PATH = CACHE_DIR / "learned_constants.json"

# 旧路径（兼容性，如果新路径不存在会尝试从这里迁移）
OLD_LEARNED_TYPES_PATH = CONFIG_DIR / "learned_types.json"
OLD_LEARNED_CONSTANTS_PATH = CONFIG_DIR / "learned_constants.json"

# =========================================================================
# C 源码搜索器 - 在 C 代码中搜索类型/常量的原始定义
# =========================================================================

class CSourceSearcher:
    """
    在 C 源码中搜索类型和常量的原始定义
    
    ★★★ 优先使用 compile_commands.json 进行精准搜索 ★★★
    
    搜索策略（按优先级）：
    1. compile_commands.json 中的源文件
    2. compile_commands.json 中的 include 路径
    3. 回退：指定目录的广度搜索（仅当前两种方法失败时）
    
    这种方式比盲目搜索整个目录树更高效、更精准
    """
    
    def __init__(
        self, 
        source_dirs: List[Path], 
        include_dirs: List[Path] = None,
        compile_commands_parser=None  # CompileCommandsParser 实例
    ):
        self.source_dirs = [Path(d) for d in source_dirs]
        self.include_dirs = [Path(d) for d in (include_dirs or [])]
        self._file_cache: Dict[str, str] = {}
        
        # ★ 新增：compile_commands.json 支持
        self._cc_parser = compile_commands_parser
        self._cc_source_files: List[str] = []  # 编译数据库中的所有源文件
        self._cc_include_dirs: List[Path] = []  # 编译数据库中的所有 include 目录
        
        # 如果提供了 compile_commands parser，提取文件列表
        if self._cc_parser:
            self._init_from_compile_commands()
    
    def _init_from_compile_commands(self):
        """从 compile_commands.json 初始化文件列表"""
        if not self._cc_parser or not self._cc_parser.compile_db:
            return
        
        logger.info(f"[CSourceSearcher] 从 compile_commands.json 初始化...")
        
        # 提取所有源文件路径
        seen_files = set()
        for entry in self._cc_parser.compile_db:
            file_path = entry.get('file', '')
            if file_path and file_path not in seen_files:
                seen_files.add(file_path)
                self._cc_source_files.append(file_path)
        
        # 提取所有 include 目录
        try:
            self._cc_include_dirs = list(self._cc_parser.get_all_include_dirs())
        except Exception as e:
            logger.warning(f"获取 include 目录失败: {e}")
            self._cc_include_dirs = []
        
        logger.info(f"[CSourceSearcher] 从 compile_commands 获取: "
                   f"{len(self._cc_source_files)} 源文件, "
                   f"{len(self._cc_include_dirs)} include 目录")
    
    def _get_file_content(self, file_path: str) -> Optional[str]:
        """获取文件内容（带缓存）"""
        if file_path not in self._file_cache:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    self._file_cache[file_path] = f.read()
            except Exception:
                return None
        return self._file_cache[file_path]
    
    def _get_type_patterns(self, type_name: str) -> List[Tuple[str, str]]:
        """获取类型搜索的正则模式"""
        return [
            # struct 定义
            (rf'\bstruct\s+{re.escape(type_name)}\s*\{{[^}}]*\}}', "struct definition"),
            # typedef struct
            (rf'\btypedef\s+struct\s+\w*\s*\{{[^}}]*\}}\s*{re.escape(type_name)}\s*;', "typedef struct"),
            # typedef 其他类型
            (rf'\btypedef\s+[^;]+\s+{re.escape(type_name)}\s*;', "typedef"),
            # enum 定义
            (rf'\benum\s+{re.escape(type_name)}\s*\{{[^}}]*\}}', "enum definition"),
            # 前向声明
            (rf'\bstruct\s+{re.escape(type_name)}\s*;', "forward declaration"),
        ]
    
    def _search_in_file(self, file_path: str, patterns: List[Tuple[str, str]]) -> Optional[Dict]:
        """在单个文件中搜索匹配的模式"""
        content = self._get_file_content(file_path)
        if not content:
            return None
        
        for pattern, pattern_type in patterns:
            match = re.search(pattern, content, re.MULTILINE | re.DOTALL)
            if match:
                line_number = content[:match.start()].count('\n') + 1
                
                # 提取代码片段
                lines = content.split('\n')
                start = max(0, line_number - 3)
                end = min(len(lines), line_number + 5)
                snippet = '\n'.join(lines[start:end])
                
                return {
                    'found': True,
                    'file': file_path,
                    'line': line_number,
                    'definition': match.group(0).strip(),
                    'pattern_type': pattern_type,
                    'snippet': snippet,
                    'notes': [f"找到 {pattern_type} 在 {Path(file_path).name}:{line_number}"]
                }
        return None
    
    def find_type_definition(self, type_name: str) -> Optional[Dict]:
        """
        搜索类型的定义
        
        ★★★ 优先使用 compile_commands.json 中的文件和路径 ★★★
        
        搜索顺序：
        1. compile_commands.json 中的源文件
        2. compile_commands.json 中的 include 目录下的头文件
        3. 回退：指定 source_dirs 和 include_dirs 的广度搜索
        
        Returns:
            {
                'found': True/False,
                'file': 文件路径,
                'line': 行号,
                'definition': 定义内容,
                'notes': 备注列表
            }
        """
        patterns = self._get_type_patterns(type_name)
        
        # ★★★ 策略 1: 在 compile_commands.json 的源文件中搜索 ★★★
        if self._cc_source_files:
            logger.debug(f"[搜索类型] '{type_name}' - 策略1: compile_commands 源文件 ({len(self._cc_source_files)} 个)")
            
            for file_path in self._cc_source_files:
                if not Path(file_path).exists():
                    continue
                # 只搜索头文件（类型定义通常在头文件中）
                if not file_path.endswith(('.h', '.hpp', '.hxx')):
                    continue
                
                result = self._search_in_file(file_path, patterns)
                if result:
                    result['search_strategy'] = 'compile_commands_source'
                    logger.info(f"[搜索类型] ✓ 在 compile_commands 源文件中找到 '{type_name}'")
                    return result
        
        # ★★★ 策略 2: 在 compile_commands.json 的 include 目录中搜索 ★★★
        if self._cc_include_dirs:
            logger.debug(f"[搜索类型] '{type_name}' - 策略2: compile_commands include 目录 ({len(self._cc_include_dirs)} 个)")
            
            # 只搜索 include 目录的直接子文件（不递归，更快）
            for include_dir in self._cc_include_dirs:
                # compile_commands 解析可能会混入头文件路径（如 -include xxx.h），需要过滤非目录项
                if not include_dir.exists() or not include_dir.is_dir():
                    continue
                
                try:
                    for file_path in include_dir.iterdir():
                        if not file_path.is_file():
                            continue
                        if file_path.suffix not in ['.h', '.hpp', '.hxx']:
                            continue
                        
                        result = self._search_in_file(str(file_path), patterns)
                        if result:
                            result['search_strategy'] = 'compile_commands_include'
                            logger.info(f"[搜索类型] ✓ 在 compile_commands include 目录中找到 '{type_name}'")
                            return result
                except (PermissionError, NotADirectoryError, OSError):
                    continue
        
        # ★★★ 策略 3: 回退到传统广度搜索 ★★★
        # 只有在 compile_commands 搜索失败时才使用这种慢速方法
        if self.source_dirs or self.include_dirs:
            logger.debug(f"[搜索类型] '{type_name}' - 策略3: 回退到广度搜索")
            
            all_dirs = self.source_dirs + self.include_dirs
            
            for source_dir in all_dirs:
                if not source_dir.exists() or not source_dir.is_dir():
                    continue
                
                # 限制搜索深度和数量，避免太慢
                file_count = 0
                max_files = 5000  # 最多搜索 5000 个文件
                
                for file_path in source_dir.rglob('*'):
                    file_count += 1
                    if file_count > max_files:
                        logger.warning(f"[搜索类型] 广度搜索达到限制 ({max_files} 文件)，停止搜索")
                        break
                    
                    if file_path.suffix not in ['.c', '.cpp', '.h', '.hpp', '.cc', '.hxx']:
                        continue
                    
                    result = self._search_in_file(str(file_path), patterns)
                    if result:
                        result['search_strategy'] = 'fallback_rglob'
                        logger.info(f"[搜索类型] ✓ 在广度搜索中找到 '{type_name}'")
                        return result
        
        logger.debug(f"[搜索类型] ✗ 未找到 '{type_name}'")
        return {'found': False, 'notes': [f"未能在 C 源码中找到类型 '{type_name}' 的定义"]}
    
    def _get_constant_patterns(self, const_name: str) -> List[Tuple[str, str]]:
        """获取常量搜索的正则模式"""
        return [
            # #define 常量
            (rf'#\s*define\s+{re.escape(const_name)}\s+[^\n]+', "macro definition"),
            # const 变量
            (rf'\bconst\s+\w+\s+{re.escape(const_name)}\s*=\s*[^;]+;', "const variable"),
            # enum 成员 (NAME = value)
            (rf'\b{re.escape(const_name)}\s*=\s*-?\d+', "enum member"),
        ]
    
    def _search_constant_in_file(self, file_path: str, patterns: List[Tuple[str, str]]) -> Optional[Dict]:
        """在单个文件中搜索常量匹配的模式"""
        content = self._get_file_content(file_path)
        if not content:
            return None
        
        for pattern, pattern_type in patterns:
            match = re.search(pattern, content, re.MULTILINE)
            if match:
                line_number = content[:match.start()].count('\n') + 1
                
                lines = content.split('\n')
                start = max(0, line_number - 2)
                end = min(len(lines), line_number + 3)
                snippet = '\n'.join(lines[start:end])
                
                return {
                    'found': True,
                    'file': file_path,
                    'line': line_number,
                    'definition': match.group(0).strip(),
                    'pattern_type': pattern_type,
                    'snippet': snippet,
                    'notes': [f"找到 {pattern_type} 在 {Path(file_path).name}:{line_number}"]
                }
        return None
    
    def find_constant_definition(self, const_name: str) -> Optional[Dict]:
        """
        搜索常量的定义
        
        ★★★ 优先使用 compile_commands.json 中的文件和路径 ★★★
        
        搜索顺序：
        1. compile_commands.json 中的源文件
        2. compile_commands.json 中的 include 目录下的头文件
        3. 回退：指定 source_dirs 和 include_dirs 的广度搜索
        
        Returns:
            {
                'found': True/False,
                'file': 文件路径,
                'line': 行号,
                'definition': 定义内容,
                'notes': 备注列表
            }
        """
        patterns = self._get_constant_patterns(const_name)
        
        # ★★★ 策略 1: 在 compile_commands.json 的源文件中搜索 ★★★
        if self._cc_source_files:
            logger.debug(f"[搜索常量] '{const_name}' - 策略1: compile_commands 源文件")
            
            for file_path in self._cc_source_files:
                if not Path(file_path).exists():
                    continue
                # 常量定义通常在头文件中
                if not file_path.endswith(('.h', '.hpp', '.hxx')):
                    continue
                
                result = self._search_constant_in_file(file_path, patterns)
                if result:
                    result['search_strategy'] = 'compile_commands_source'
                    logger.info(f"[搜索常量] ✓ 在 compile_commands 源文件中找到 '{const_name}'")
                    return result
        
        # ★★★ 策略 2: 在 compile_commands.json 的 include 目录中搜索 ★★★
        if self._cc_include_dirs:
            logger.debug(f"[搜索常量] '{const_name}' - 策略2: compile_commands include 目录")
            
            for include_dir in self._cc_include_dirs:
                if not include_dir.exists() or not include_dir.is_dir():
                    continue
                
                try:
                    for file_path in include_dir.iterdir():
                        if not file_path.is_file():
                            continue
                        if file_path.suffix not in ['.h', '.hpp', '.hxx']:
                            continue
                        
                        result = self._search_constant_in_file(str(file_path), patterns)
                        if result:
                            result['search_strategy'] = 'compile_commands_include'
                            logger.info(f"[搜索常量] ✓ 在 compile_commands include 目录中找到 '{const_name}'")
                            return result
                except (PermissionError, NotADirectoryError, OSError):
                    continue
        
        # ★★★ 策略 3: 回退到传统广度搜索 ★★★
        if self.source_dirs or self.include_dirs:
            logger.debug(f"[搜索常量] '{const_name}' - 策略3: 回退到广度搜索")
            
            all_dirs = self.source_dirs + self.include_dirs
            
            for source_dir in all_dirs:
                if not source_dir.exists() or not source_dir.is_dir():
                    continue
                
                file_count = 0
                max_files = 5000
                
                for file_path in source_dir.rglob('*'):
                    file_count += 1
                    if file_count > max_files:
                        logger.warning(f"[搜索常量] 广度搜索达到限制 ({max_files} 文件)，停止搜索")
                        break
                    
                    if file_path.suffix not in ['.c', '.cpp', '.h', '.hpp', '.cc', '.hxx']:
                        continue
                    
                    result = self._search_constant_in_file(str(file_path), patterns)
                    if result:
                        result['search_strategy'] = 'fallback_rglob'
                        logger.info(f"[搜索常量] ✓ 在广度搜索中找到 '{const_name}'")
                        return result
        
        logger.debug(f"[搜索常量] ✗ 未找到 '{const_name}'")
        return {'found': False, 'notes': [f"未能在 C 源码中找到常量 '{const_name}' 的定义"]}


# =========================================================================
# 内置预定义 - 通用 POSIX/C 标准类型（保持不变）
# =========================================================================
POSIX_TYPES = {
    # pthread 类型
    'pthread_mutex_t': '#[repr(C)]\npub struct pthread_mutex_t { _opaque: [u8; 40] }',
    'pthread_cond_t': '#[repr(C)]\npub struct pthread_cond_t { _opaque: [u8; 48] }',
    'pthread_rwlock_t': '#[repr(C)]\npub struct pthread_rwlock_t { _opaque: [u8; 56] }',
    'pthread_once_t': 'pub type pthread_once_t = i32;',
    'pthread_attr_t': '#[repr(C)]\npub struct pthread_attr_t { _opaque: [u8; 56] }',
    'pthread_t': 'pub type pthread_t = usize;',
    
    # 原子类型
    'atomic_bool': 'pub type atomic_bool = u8;',
    'atomic_int': 'pub type atomic_int = i32;',
    'atomic_uint': 'pub type atomic_uint = u32;',
    'atomic_long': 'pub type atomic_long = i64;',
    'atomic_ulong': 'pub type atomic_ulong = u64;',
    
    # Socket/Network 类型
    'sockaddr': '#[repr(C)]\npub struct sockaddr { pub sa_family: u16, pub sa_data: [c_char; 14] }',
    'sockaddr_in': '#[repr(C)]\npub struct sockaddr_in { pub sin_family: u16, pub sin_port: u16, pub sin_addr: in_addr, pub sin_zero: [u8; 8] }',
    'sockaddr_in6': '#[repr(C)]\npub struct sockaddr_in6 { pub sin6_family: u16, pub sin6_port: u16, pub sin6_flowinfo: u32, pub sin6_addr: in6_addr, pub sin6_scope_id: u32 }',
    'in_addr': '#[repr(C)]\npub struct in_addr { pub s_addr: u32 }',
    'in6_addr': '#[repr(C)]\npub struct in6_addr { pub s6_addr: [u8; 16] }',
    'sa_family_t': 'pub type sa_family_t = u16;',
    'socklen_t': 'pub type socklen_t = u32;',
    
    # 文件/IO 类型
    'file': '#[repr(C)]\npub struct file { _opaque: [u8; 0] }',
    'FILE': '#[repr(C)]\npub struct FILE { _opaque: [u8; 0] }',
    'iovec': '#[repr(C)]\npub struct iovec { pub iov_base: *mut c_void, pub iov_len: usize }',
    'stat': '#[repr(C)]\npub struct stat { _opaque: [u8; 144] }',
    'off_t': 'pub type off_t = i64;',
    'mode_t': 'pub type mode_t = u32;',
    'dev_t': 'pub type dev_t = u64;',
    'ino_t': 'pub type ino_t = u64;',
    'nlink_t': 'pub type nlink_t = u64;',
    'uid_t': 'pub type uid_t = u32;',
    'gid_t': 'pub type gid_t = u32;',
    'blksize_t': 'pub type blksize_t = i64;',
    'blkcnt_t': 'pub type blkcnt_t = i64;',
    'time_t': 'pub type time_t = i64;',
    
    # 时间类型
    'timespec': '#[repr(C)]\npub struct timespec { pub tv_sec: time_t, pub tv_nsec: c_long }',
    'timeval': '#[repr(C)]\npub struct timeval { pub tv_sec: time_t, pub tv_usec: c_long }',
    
    # 网络接口类型 (lwIP)
    'netif': '#[repr(C)]\npub struct netif { _opaque: [u8; 0] }',
    'pbuf': '#[repr(C)]\npub struct pbuf { _opaque: [u8; 0] }',
    
    # 信号量类型
    'sem_t': '#[repr(C)]\npub struct sem_t { _opaque: [u8; 32] }',
}

POSIX_CONSTANTS = [
    # POSIX 常量
    # pthread_mutex_t 在不同平台/头文件下布局可能不同（bindgen 生成字段名也可能不同），
    # 用 struct literal 初始化会导致 E0560（字段不存在）。骨架阶段只要求可编译，统一用 0 初始化。
    ("PTHREAD_MUTEX_INITIALIZER", "pthread_mutex_t", "unsafe { ::core::mem::zeroed() }"),
    ("PTHREAD_COND_INITIALIZER", "pthread_cond_t", "unsafe { ::core::mem::zeroed() }"),
    ("PTHREAD_RWLOCK_INITIALIZER", "pthread_rwlock_t", "unsafe { ::core::mem::zeroed() }"),
    ("PTHREAD_ONCE_INIT", "pthread_once_t", "unsafe { ::core::mem::zeroed() }"),

    # Some code paths (or macro expansions) may refer to the internal `__PTHREAD_*` names.
    # Keep them as aliases to reduce bindgen/translation churn across libc variants.
    # Internal musl/libc names: define them directly to avoid type-mismatch across boards/profiles.
    # (Some profiles generate PTHREAD_* as numeric constants; others don't generate them at all.)
    ("__PTHREAD_MUTEX_INITIALIZER", "pthread_mutex_t", "unsafe { ::core::mem::zeroed() }"),
    ("__PTHREAD_COND_INITIALIZER", "pthread_cond_t", "unsafe { ::core::mem::zeroed() }"),
    ("__PTHREAD_RWLOCK_INITIALIZER", "pthread_rwlock_t", "unsafe { ::core::mem::zeroed() }"),
    ("__PTHREAD_ONCE_INIT", "pthread_once_t", "unsafe { ::core::mem::zeroed() }"),
    
    # errno 常量
    ("EINVAL", "i32", "22"),
    ("ENOMEM", "i32", "12"),
    ("ENOENT", "i32", "2"),
    ("EEXIST", "i32", "17"),
    ("EBUSY", "i32", "16"),
    ("EAGAIN", "i32", "11"),
    ("ETIMEDOUT", "i32", "110"),
    ("ENODEV", "i32", "19"),
    ("EFAULT", "i32", "14"),
    ("ENOSYS", "i32", "38"),
    ("ERANGE", "i32", "34"),
    ("ENOTCONN", "i32", "107"),
    
    # 布尔和状态
    ("TRUE", "i32", "1"),
    ("FALSE", "i32", "0"),
]

# =========================================================================
# 平台特定预定义 - OpenHarmony/LiteOS/HDF（可选启用）
# =========================================================================
OHOS_TYPES = {
    # HDF 驱动框架类型
    'HdfDeviceObject': '#[repr(C)]\npub struct HdfDeviceObject { _opaque: [u8; 0] }',
    'HdfSBuf': '#[repr(C)]\npub struct HdfSBuf { _opaque: [u8; 0] }',
    'HdfDriverEntry': '#[repr(C)]\npub struct HdfDriverEntry { _opaque: [u8; 0] }',
    'DeviceResourceNode': '#[repr(C)]\npub struct DeviceResourceNode { _opaque: [u8; 0] }',
    
    # 音频类型
    'AudioFormat': '#[repr(C)]\npub struct AudioFormat { _opaque: [u8; 0] }',
    'AudioPortPin': '#[repr(C)]\npub struct AudioPortPin { _opaque: [u8; 0] }',
    'AudioPort': '#[repr(C)]\npub struct AudioPort { _opaque: [u8; 0] }',
    'AudioPortCapability': '#[repr(C)]\npub struct AudioPortCapability { _opaque: [u8; 0] }',
    'AudioAdapter': '#[repr(C)]\npub struct AudioAdapter { _opaque: [u8; 0] }',
    'EffectConfigDescriptor': '#[repr(C)]\npub struct EffectConfigDescriptor { _opaque: [u8; 0] }',
    
    # 加密安全类型
    'HksBlob': '#[repr(C)]\npub struct HksBlob { pub size: u32, pub data: *mut u8 }',
    
    # 进程/ELF 类型
    'ELFLoadInfo': '#[repr(C)]\npub struct ELFLoadInfo { _opaque: [u8; 0] }',
    
    # SoftBus 类型
    'CommonScvId': '#[repr(C)]\npub struct CommonScvId { _opaque: [u8; 0] }',
    'SubscribeInfo': '#[repr(C)]\npub struct SubscribeInfo { _opaque: [u8; 0] }',
    'TcpConnInfoNode': '#[repr(C)]\npub struct TcpConnInfoNode { _opaque: [u8; 0] }',
    'LwipMonitorReportInfo': '#[repr(C)]\npub struct LwipMonitorReportInfo { _opaque: [u8; 0] }',
}

OHOS_CONSTANTS = [
    # SoftBus 常量
    ("SOFTBUS_OK", "i32", "0"),
    ("SOFTBUS_ERR", "i32", "-1"),
    ("SOFTBUS_NOT_IMPLEMENT", "i32", "-2"),
    ("SOFTBUS_INVALID_PARAM", "i32", "-3"),
    ("SOFTBUS_MEM_ERR", "i32", "-4"),
    ("SOFTBUS_MALLOC_ERR", "i32", "-5"),
    ("SOFTBUS_PERMISSION_DENIED", "i32", "-6"),
    ("SOFTBUS_NETWORK_ERR", "i32", "-7"),
    
    # HDF 常量
    ("HDF_SUCCESS", "i32", "0"),
    ("HDF_FAILURE", "i32", "-1"),
    ("HDF_ERR_INVALID_PARAM", "i32", "-2"),
    ("HDF_ERR_MALLOC_FAIL", "i32", "-3"),
    ("HDF_ERR_TIMEOUT", "i32", "-4"),
    ("HDF_ERR_NOT_SUPPORT", "i32", "-10"),
    ("HDF_ERR_IO", "i32", "-8"),
    ("HDF_ERR_DEVICE_BUSY", "i32", "-22"),
    
    # LiteOS 常量
    ("LOS_OK", "i32", "0"),
    ("LOS_NOK", "i32", "-1"),
    ("LOS_ERRNO_BASE", "u32", "0x02000000"),
    
    # Audio 常量
    ("AUDIO_FORMAT_TYPE_PCM_16_BIT", "i32", "1"),
    ("AUDIO_FORMAT_TYPE_PCM_8_BIT", "i32", "0"),
    ("AUDIO_FORMAT_TYPE_PCM_24_BIT", "i32", "2"),
    ("AUDIO_FORMAT_TYPE_PCM_32_BIT", "i32", "3"),
]

# 常量前缀规则（用于自动推断类型）
CONSTANT_PREFIX_RULES = {
    'HDF_': ('i32', '0'),
    'LOS_': ('u32', '0'),
    'SOFTBUS_': ('i32', '0'),
    'AUDIO_': ('i32', '0'),
    'E': ('i32', '0'),  # errno 常量
}

# =========================================================================
# PredefineManager - 自适应预定义管理器
# =========================================================================

@dataclass
class LearnedType:
    """学习到的类型信息"""
    name: str
    definition: str
    source: str = "auto_learned"  # auto_learned, user_added, compilation_error
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    usage_count: int = 0
    
    # 诊断信息 - 记录为什么使用这个定义
    is_placeholder: bool = False              # 是否是占位符（而非真实定义）
    c_source_file: str = ""                   # 原始 C 定义的文件位置
    c_source_line: int = 0                    # 原始 C 定义的行号
    original_c_definition: str = ""           # 原始 C 定义内容
    failure_reason: str = ""                  # 如果是占位符，为什么无法获取真实定义
    diagnostic_notes: List[str] = field(default_factory=list)  # 诊断备注


@dataclass
class LearnedConstant:
    """学习到的常量信息"""
    name: str
    rust_type: str
    value: str
    source: str = "auto_learned"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    usage_count: int = 0
    
    # 诊断信息 - 记录为什么使用这个值
    is_placeholder: bool = False              # 值是否是占位符（如 0）
    c_source_file: str = ""                   # 原始 C 定义的文件位置
    c_source_line: int = 0                    # 原始 C 定义的行号
    original_c_definition: str = ""           # 原始 C 定义内容（如 #define XXX 123）
    failure_reason: str = ""                  # 如果是占位符，为什么无法获取真实值
    diagnostic_notes: List[str] = field(default_factory=list)  # 诊断备注


class PredefineManager:
    """
    自适应预定义管理器
    
    功能：
    1. 管理内置类型/常量（POSIX + 可选的 OHOS）
    2. 管理学习到的类型/常量
    3. 自动持久化学习结果
    4. 线程安全
    """
    
    _instance = None
    _lock = threading.Lock()
    
    @classmethod
    def get_instance(cls, enable_ohos: bool = True) -> 'PredefineManager':
        """获取单例实例"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(enable_ohos)
            return cls._instance
    
    @classmethod
    def reset_instance(cls):
        """重置单例（仅用于测试）"""
        with cls._lock:
            cls._instance = None
    
    def __init__(self, enable_ohos: bool = True):
        """
        初始化预定义管理器
        
        Args:
            enable_ohos: 是否启用 OpenHarmony 特定预定义
        """
        self.enable_ohos = enable_ohos
        
        # 内置类型/常量
        self._builtin_types: Dict[str, str] = dict(POSIX_TYPES)
        self._builtin_constants: List[Tuple[str, str, str]] = list(POSIX_CONSTANTS)
        
        if enable_ohos:
            self._builtin_types.update(OHOS_TYPES)
            self._builtin_constants.extend(OHOS_CONSTANTS)
        
        # 学习到的类型/常量
        self._learned_types: Dict[str, LearnedType] = {}
        self._learned_constants: Dict[str, LearnedConstant] = {}
        
        # 加载持久化的学习结果
        self._load_learned()
        
        logger.info(f"PredefineManager 初始化完成: "
                   f"{len(self._builtin_types)} 内置类型, "
                   f"{len(self._learned_types)} 学习类型, "
                   f"OHOS={'启用' if enable_ohos else '禁用'}")
    
    # =========================================================================
    # 类型操作
    # =========================================================================
    
    def get_all_types(self) -> Dict[str, str]:
        """获取所有类型定义（内置 + 学习）"""
        result = dict(self._builtin_types)
        for name, learned in self._learned_types.items():
            result[name] = learned.definition
        return result
    
    def has_type(self, name: str) -> bool:
        """检查类型是否存在"""
        return name in self._builtin_types or name in self._learned_types
    
    def get_type(self, name: str) -> Optional[str]:
        """获取类型定义"""
        if name in self._builtin_types:
            return self._builtin_types[name]
        if name in self._learned_types:
            self._learned_types[name].usage_count += 1
            return self._learned_types[name].definition
        return None
    
    def learn_type(
        self, 
        name: str, 
        definition: str, 
        source: str = "auto_learned",
        save: bool = True,
        # 诊断信息参数
        is_placeholder: bool = False,
        c_source_file: str = "",
        c_source_line: int = 0,
        original_c_definition: str = "",
        failure_reason: str = "",
        diagnostic_notes: List[str] = None
    ) -> bool:
        """
        学习新类型
        
        Args:
            name: 类型名称
            definition: Rust 类型定义
            source: 来源（auto_learned, compilation_error, user_added）
            save: 是否立即保存到文件
            is_placeholder: 是否是占位符（而非真实定义）
            c_source_file: 原始 C 定义的文件位置
            c_source_line: 原始 C 定义的行号
            original_c_definition: 原始 C 定义内容
            failure_reason: 如果是占位符，为什么无法获取真实定义
            diagnostic_notes: 诊断备注
            
        Returns:
            是否成功添加（已存在的不会覆盖）
        """
        # 跳过已存在的类型
        if self.has_type(name):
            logger.debug(f"类型 '{name}' 已存在，跳过学习")
            return False
        
        # 验证类型名称
        if not self._is_valid_type_name(name):
            logger.warning(f"无效的类型名称: '{name}'")
            return False
        
        self._learned_types[name] = LearnedType(
            name=name,
            definition=definition,
            source=source,
            is_placeholder=is_placeholder,
            c_source_file=c_source_file,
            c_source_line=c_source_line,
            original_c_definition=original_c_definition,
            failure_reason=failure_reason,
            diagnostic_notes=diagnostic_notes or []
        )
        
        # 日志输出
        if is_placeholder:
            logger.warning(f"⚠️ 学习占位符类型: {name} (原因: {failure_reason})")
        else:
            logger.info(f"✓ 学习真实类型: {name} (来源: {source})")
        
        if save:
            self._save_learned_types()
        
        return True
    
    def generate_opaque_type(
        self, 
        name: str, 
        c_definition: str = "",
        estimated_size: int = 0
    ) -> str:
        """
        生成不透明类型定义
        
        ★★★ 增强版：保留原始 C 定义作为注释，并估算大小 ★★★
        
        Args:
            name: 类型名称
            c_definition: 原始 C 定义（可选，会作为注释保留）
            estimated_size: 估算的大小（字节），0 表示未知
            
        Returns:
            Rust 类型定义字符串
        """
        # 估算大小：如果没有提供，根据 C 定义尝试估算
        if estimated_size == 0 and c_definition:
            estimated_size = self._estimate_struct_size(c_definition)
        
        # 使用合理的默认大小（128 字节，适用于大多数结构体）
        if estimated_size == 0:
            estimated_size = 128
        
        lines = []
        lines.append("#[repr(C)]")
        
        # 添加原始 C 定义作为注释（对 LLM 后续修复很有帮助）
        if c_definition:
            lines.append("// Original C Definition:")
            for line in c_definition.strip().split('\n')[:15]:  # 最多显示 15 行
                lines.append(f"// {line.rstrip()}")
            if c_definition.count('\n') > 15:
                lines.append("// ... (truncated)")
        
        lines.append(f"pub struct {name} {{ _opaque: [u8; {estimated_size}] }}")
        
        return '\n'.join(lines)
    
    def _estimate_struct_size(self, c_definition: str) -> int:
        """
        根据 C 结构体定义估算大小
        
        这是一个简单的启发式方法：
        - 计算字段数量
        - 根据字段类型估算每个字段的大小
        - 考虑对齐
        
        Args:
            c_definition: C 结构体定义
            
        Returns:
            估算的大小（字节），如果无法估算返回 0
        """
        if not c_definition:
            return 0
        
        # 常见类型的大小（字节）
        type_sizes = {
            'char': 1, 'unsigned char': 1, 'uint8_t': 1, 'int8_t': 1,
            'short': 2, 'unsigned short': 2, 'uint16_t': 2, 'int16_t': 2,
            'int': 4, 'unsigned int': 4, 'uint32_t': 4, 'int32_t': 4, 'float': 4,
            'long': 8, 'unsigned long': 8, 'uint64_t': 8, 'int64_t': 8, 'double': 8,
            'size_t': 8, 'void*': 8, 'char*': 8, 'int*': 8,  # 指针大小（64位）
        }
        
        total_size = 0
        
        # 提取结构体内部的字段
        match = re.search(r'\{([^}]+)\}', c_definition, re.DOTALL)
        if not match:
            return 0
        
        body = match.group(1)
        
        # 简单解析每个字段
        # 格式：type field_name; 或 type *field_name;
        for line in body.split(';'):
            line = line.strip()
            if not line:
                continue
            
            # 检查是否是指针
            if '*' in line:
                total_size += 8  # 指针大小
                continue
            
            # 检查数组
            array_match = re.search(r'\[(\d+)\]', line)
            multiplier = 1
            if array_match:
                multiplier = int(array_match.group(1))
            
            # 尝试匹配类型
            field_size = 8  # 默认大小（保守估计）
            for type_name, size in type_sizes.items():
                if type_name in line:
                    field_size = size
                    break
            
            total_size += field_size * multiplier
        
        # 对齐到 8 字节边界
        if total_size > 0:
            total_size = ((total_size + 7) // 8) * 8
        
        # 最小返回 8 字节，最大限制 4096 字节
        return max(8, min(total_size, 4096))
    
    # =========================================================================
    # 常量操作
    # =========================================================================
    
    def get_all_constants(self) -> List[Tuple[str, str, str]]:
        """获取所有常量定义（内置 + 学习）"""
        result = list(self._builtin_constants)
        for name, learned in self._learned_constants.items():
            result.append((name, learned.rust_type, learned.value))
        return result
    
    def has_constant(self, name: str) -> bool:
        """检查常量是否存在"""
        for const_name, _, _ in self._builtin_constants:
            if const_name == name:
                return True
        return name in self._learned_constants
    
    def get_constant(self, name: str) -> Optional[Tuple[str, str]]:
        """获取常量信息 (type, value)"""
        for const_name, rust_type, value in self._builtin_constants:
            if const_name == name:
                return (rust_type, value)
        if name in self._learned_constants:
            learned = self._learned_constants[name]
            learned.usage_count += 1
            return (learned.rust_type, learned.value)
        return None
    
    def learn_constant(
        self, 
        name: str, 
        rust_type: str,
        value: str,
        source: str = "auto_learned",
        save: bool = True,
        # 诊断信息参数
        is_placeholder: bool = False,
        c_source_file: str = "",
        c_source_line: int = 0,
        original_c_definition: str = "",
        failure_reason: str = "",
        diagnostic_notes: List[str] = None
    ) -> bool:
        """
        学习新常量
        
        Args:
            name: 常量名称
            rust_type: Rust 类型
            value: 常量值
            source: 来源
            save: 是否立即保存
            is_placeholder: 值是否是占位符（如 0）
            c_source_file: 原始 C 定义的文件位置
            c_source_line: 原始 C 定义的行号
            original_c_definition: 原始 C 定义内容
            failure_reason: 如果是占位符，为什么无法获取真实值
            diagnostic_notes: 诊断备注
            
        Returns:
            是否成功添加
        """
        if self.has_constant(name):
            logger.debug(f"常量 '{name}' 已存在，跳过学习")
            return False
        
        self._learned_constants[name] = LearnedConstant(
            name=name,
            rust_type=rust_type,
            value=value,
            source=source,
            is_placeholder=is_placeholder,
            c_source_file=c_source_file,
            c_source_line=c_source_line,
            original_c_definition=original_c_definition,
            failure_reason=failure_reason,
            diagnostic_notes=diagnostic_notes or []
        )
        
        # 日志输出
        if is_placeholder:
            logger.warning(f"⚠️ 学习占位符常量: {name} = {value} (原因: {failure_reason})")
        else:
            logger.info(f"✓ 学习真实常量: {name}: {rust_type} = {value} (来源: {source})")
        
        if save:
            self._save_learned_constants()
        
        return True
    
    def infer_constant_type(self, name: str) -> Tuple[str, str]:
        """
        根据常量名称前缀推断类型
        
        Returns:
            (rust_type, default_value)
        """
        for prefix, (rust_type, default_value) in CONSTANT_PREFIX_RULES.items():
            if name.startswith(prefix):
                return (rust_type, default_value)
        return ('i32', '0')  # 默认
    
    # =========================================================================
    # 自适应学习 - 从编译错误中学习（带诊断分析）
    # =========================================================================
    
    def learn_from_compilation_errors(
        self, 
        error_output: str,
        types_content: str = "",
        c_source_dirs: List[Path] = None,
        include_dirs: List[Path] = None,
        compile_commands_parser=None  # ★ 新增：优先使用 compile_commands.json
    ) -> Tuple[int, int]:
        """
        从编译错误中学习缺失的类型和常量
        
        ★★★ 增强版：会尝试在 C 源码中搜索原始定义 ★★★
        - 优先使用 compile_commands.json 进行精准搜索
        - 如果找到真实值：使用真实值
        - 如果找不到：使用占位符，但记录失败原因
        
        Args:
            error_output: cargo check 的错误输出
            types_content: 当前 types.rs 内容（用于避免重复）
            c_source_dirs: C 源码目录（用于搜索原始定义）
            include_dirs: 头文件目录
            compile_commands_parser: CompileCommandsParser 实例（优先级最高）
            
        Returns:
            (learned_types_count, learned_constants_count)
        """
        import re
        
        learned_types = 0
        learned_constants = 0
        
        # 创建 C 源码搜索器
        # ★★★ 优先使用 compile_commands.json 进行精准搜索 ★★★
        c_searcher = None
        if compile_commands_parser or c_source_dirs:
            c_searcher = CSourceSearcher(
                c_source_dirs or [], 
                include_dirs or [],
                compile_commands_parser=compile_commands_parser
            )
        
        # 匹配 "cannot find type `TypeName`"
        type_pattern = r"cannot find type `(\w+)`"
        for match in re.finditer(type_pattern, error_output):
            type_name = match.group(1)
            
            # 尝试在 C 源码中搜索原始定义
            c_result = None
            if c_searcher:
                c_result = c_searcher.find_type_definition(type_name)
            
            if c_result and c_result.get('found'):
                # 找到了原始定义 - 尝试转换
                rust_def, is_placeholder, failure_reason = self._convert_c_type_to_rust(
                    type_name, c_result
                )
                if self.learn_type(
                    type_name, 
                    rust_def,
                    source="compilation_error",
                    save=False,
                    is_placeholder=is_placeholder,
                    c_source_file=c_result.get('file', ''),
                    c_source_line=c_result.get('line', 0),
                    original_c_definition=c_result.get('definition', ''),
                    failure_reason=failure_reason,
                    diagnostic_notes=c_result.get('notes', [])
                ):
                    learned_types += 1
            else:
                # 没找到 - 使用占位符并记录原因
                if self.learn_type(
                    type_name, 
                    self.generate_opaque_type(type_name),
                    source="compilation_error",
                    save=False,
                    is_placeholder=True,
                    failure_reason="未能在 C 源码中找到原始定义",
                    diagnostic_notes=[
                        "可能原因：1) 来自未包含的头文件",
                        "         2) 是宏定义展开后的类型",
                        "         3) 是条件编译排除的代码",
                        f"请手动检查类型 '{type_name}' 的真实定义"
                    ]
                ):
                    learned_types += 1
        
        # 匹配 "cannot find value `CONST_NAME`"
        const_pattern = r"cannot find value `(\w+)`"
        for match in re.finditer(const_pattern, error_output):
            const_name = match.group(1)
            
            # 尝试在 C 源码中搜索原始定义
            c_result = None
            if c_searcher:
                c_result = c_searcher.find_constant_definition(const_name)
            
            if c_result and c_result.get('found'):
                # 找到了原始定义 - 提取值
                rust_type, value, is_placeholder, failure_reason = self._extract_constant_value(
                    const_name, c_result
                )
                if self.learn_constant(
                    const_name,
                    rust_type,
                    value,
                    source="compilation_error",
                    save=False,
                    is_placeholder=is_placeholder,
                    c_source_file=c_result.get('file', ''),
                    c_source_line=c_result.get('line', 0),
                    original_c_definition=c_result.get('definition', ''),
                    failure_reason=failure_reason,
                    diagnostic_notes=c_result.get('notes', [])
                ):
                    learned_constants += 1
            else:
                # 没找到 - 使用占位符并记录原因
                rust_type, default_value = self.infer_constant_type(const_name)
                if self.learn_constant(
                    const_name,
                    rust_type,
                    default_value,
                    source="compilation_error",
                    save=False,
                    is_placeholder=True,
                    failure_reason="未能在 C 源码中找到原始定义",
                    diagnostic_notes=[
                        "可能原因：1) 来自未包含的头文件",
                        "         2) 是 #define 宏定义",
                        "         3) 是条件编译排除的代码",
                        f"请手动检查常量 '{const_name}' 的真实值"
                    ]
                ):
                    learned_constants += 1
        
        # 批量保存
        if learned_types > 0:
            self._save_learned_types()
        if learned_constants > 0:
            self._save_learned_constants()
        
        if learned_types > 0 or learned_constants > 0:
            logger.info(f"从编译错误中学习: {learned_types} 个类型, {learned_constants} 个常量")
        
        return (learned_types, learned_constants)
    
    def _convert_c_type_to_rust(
        self, 
        type_name: str, 
        c_result: Dict
    ) -> Tuple[str, bool, str]:
        """
        尝试将 C 类型转换为 Rust
        
        ★★★ 增强版：保留原始 C 定义作为注释 ★★★
        
        Returns:
            (rust_definition, is_placeholder, failure_reason)
        """
        definition = c_result.get('definition', '')
        
        # 情况 1: 前向声明 - 可以安全使用占位符
        if re.match(rf'\bstruct\s+{re.escape(type_name)}\s*;$', definition.strip()):
            return (
                self.generate_opaque_type(type_name, definition),
                False,  # 这不是"占位符"，前向声明就是不透明类型
                ""
            )
        
        # 情况 2: 简单 typedef
        typedef_simple = re.match(
            r'typedef\s+(int|char|short|long|unsigned\s+\w+|size_t|ssize_t|'
            r'int8_t|int16_t|int32_t|int64_t|uint8_t|uint16_t|uint32_t|uint64_t)\s+(\w+)\s*;',
            definition
        )
        if typedef_simple:
            c_type = typedef_simple.group(1)
            rust_type = self._c_primitive_to_rust(c_type)
            return (f"pub type {type_name} = {rust_type};", False, "")
        
        # 情况 3: 复杂结构体 - 保留原始定义作为注释
        if 'struct' in definition and '{' in definition:
            notes = c_result.get('notes', [])
            notes.append(f"原始定义: {definition[:200]}")
            c_result['notes'] = notes
            return (
                # ★★★ 增强：传入原始 C 定义，会作为注释保留 ★★★
                self.generate_opaque_type(type_name, definition),
                True,
                "找到原始定义但需要手动转换"
            )
        
        # 情况 4: enum 定义 - 也保留原始定义
        if 'enum' in definition and '{' in definition:
            return (
                self.generate_opaque_type(type_name, definition),
                True,
                "找到 enum 定义但需要手动转换"
            )
        
        # 情况 5: 其他情况
        return (
            self.generate_opaque_type(type_name, definition if definition else ""),
            True,
            "无法自动转换，需要手动处理"
        )
    
    def _extract_constant_value(
        self, 
        const_name: str, 
        c_result: Dict
    ) -> Tuple[str, str, bool, str]:
        """
        从 C 定义中提取常量值
        
        Returns:
            (rust_type, value, is_placeholder, failure_reason)
        """
        definition = c_result.get('definition', '')
        
        # 情况 1: #define NAME value
        define_match = re.search(
            rf'#\s*define\s+{re.escape(const_name)}\s+(-?\d+|0x[0-9a-fA-F]+)',
            definition
        )
        if define_match:
            value = define_match.group(1)
            rust_type = 'u32' if value.startswith('0x') else 'i32'
            return (rust_type, value, False, "")
        
        # 情况 2: #define NAME (expression) - 包含括号
        define_expr_match = re.search(
            rf'#\s*define\s+{re.escape(const_name)}\s+\((-?\d+)\)',
            definition
        )
        if define_expr_match:
            value = define_expr_match.group(1)
            return ('i32', value, False, "")
        
        # 情况 3: const int NAME = value;
        const_match = re.search(
            rf'{re.escape(const_name)}\s*=\s*(-?\d+|0x[0-9a-fA-F]+)',
            definition
        )
        if const_match:
            value = const_match.group(1)
            rust_type = 'u32' if value.startswith('0x') else 'i32'
            return (rust_type, value, False, "")
        
        # 情况 4: 枚举成员 NAME = value
        # 已在 const_match 中处理
        
        # 情况 5: 复杂宏表达式 - 无法自动提取
        if '#define' in definition:
            # 定义中包含其他宏或表达式
            notes = c_result.get('notes', [])
            notes.append(f"原始定义: {definition[:200]}")
            c_result['notes'] = notes
            rust_type, default_value = self.infer_constant_type(const_name)
            return (
                rust_type, 
                default_value, 
                True, 
                "宏定义包含复杂表达式，无法自动提取值"
            )
        
        # 情况 6: 其他情况
        rust_type, default_value = self.infer_constant_type(const_name)
        return (rust_type, default_value, True, "无法解析值")
    
    def _c_primitive_to_rust(self, c_type: str) -> str:
        """C 原始类型到 Rust 类型的映射"""
        mapping = {
            'int': 'c_int',
            'char': 'c_char',
            'short': 'i16',
            'long': 'c_long',
            'unsigned int': 'u32',
            'unsigned char': 'u8',
            'unsigned short': 'u16',
            'unsigned long': 'u64',
            'size_t': 'usize',
            'ssize_t': 'isize',
            'int8_t': 'i8',
            'int16_t': 'i16',
            'int32_t': 'i32',
            'int64_t': 'i64',
            'uint8_t': 'u8',
            'uint16_t': 'u16',
            'uint32_t': 'u32',
            'uint64_t': 'u64',
        }
        return mapping.get(c_type.strip(), 'c_int')
    
    # =========================================================================
    # 持久化
    # =========================================================================
    
    def _load_learned(self):
        """加载持久化的学习结果"""
        # 加载类型
        if LEARNED_TYPES_PATH.exists():
            try:
                with open(LEARNED_TYPES_PATH, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for item in data.get('types', []):
                        self._learned_types[item['name']] = LearnedType(
                            name=item['name'],
                            definition=item['definition'],
                            source=item.get('source', 'loaded'),
                            timestamp=item.get('timestamp', ''),
                            usage_count=item.get('usage_count', 0),
                            # 诊断信息
                            is_placeholder=item.get('is_placeholder', False),
                            c_source_file=item.get('c_source_file', ''),
                            c_source_line=item.get('c_source_line', 0),
                            original_c_definition=item.get('original_c_definition', ''),
                            failure_reason=item.get('failure_reason', ''),
                            diagnostic_notes=item.get('diagnostic_notes', [])
                        )
                # 统计占位符数量
                placeholder_count = sum(1 for t in self._learned_types.values() if t.is_placeholder)
                logger.info(f"加载 {len(self._learned_types)} 个学习类型 (其中 {placeholder_count} 个是占位符)")
            except Exception as e:
                logger.warning(f"加载学习类型失败: {e}")
        
        # 加载常量
        if LEARNED_CONSTANTS_PATH.exists():
            try:
                with open(LEARNED_CONSTANTS_PATH, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for item in data.get('constants', []):
                        self._learned_constants[item['name']] = LearnedConstant(
                            name=item['name'],
                            rust_type=item['rust_type'],
                            value=item['value'],
                            source=item.get('source', 'loaded'),
                            timestamp=item.get('timestamp', ''),
                            usage_count=item.get('usage_count', 0),
                            # 诊断信息
                            is_placeholder=item.get('is_placeholder', False),
                            c_source_file=item.get('c_source_file', ''),
                            c_source_line=item.get('c_source_line', 0),
                            original_c_definition=item.get('original_c_definition', ''),
                            failure_reason=item.get('failure_reason', ''),
                            diagnostic_notes=item.get('diagnostic_notes', [])
                        )
                placeholder_count = sum(1 for c in self._learned_constants.values() if c.is_placeholder)
                logger.info(f"加载 {len(self._learned_constants)} 个学习常量 (其中 {placeholder_count} 个是占位符)")
            except Exception as e:
                logger.warning(f"加载学习常量失败: {e}")
    
    def _save_learned_types(self):
        """保存学习的类型到文件（包含诊断信息）"""
        try:
            LEARNED_TYPES_PATH.parent.mkdir(parents=True, exist_ok=True)
            
            # 分类统计
            placeholder_count = sum(1 for t in self._learned_types.values() if t.is_placeholder)
            real_count = len(self._learned_types) - placeholder_count
            
            data = {
                'version': '2.0',  # 版本升级，包含诊断信息
                'updated': datetime.now().isoformat(),
                'statistics': {
                    'total': len(self._learned_types),
                    'real_definitions': real_count,
                    'placeholders': placeholder_count
                },
                'types': [
                    {
                        'name': t.name,
                        'definition': t.definition,
                        'source': t.source,
                        'timestamp': t.timestamp,
                        'usage_count': t.usage_count,
                        # 诊断信息
                        'is_placeholder': t.is_placeholder,
                        'c_source_file': t.c_source_file,
                        'c_source_line': t.c_source_line,
                        'original_c_definition': t.original_c_definition[:500] if t.original_c_definition else '',
                        'failure_reason': t.failure_reason,
                        'diagnostic_notes': t.diagnostic_notes
                    }
                    for t in self._learned_types.values()
                ]
            }
            with open(LEARNED_TYPES_PATH, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.debug(f"保存 {len(self._learned_types)} 个学习类型 (真实: {real_count}, 占位符: {placeholder_count})")
        except Exception as e:
            logger.error(f"保存学习类型失败: {e}")
    
    def _save_learned_constants(self):
        """保存学习的常量到文件（包含诊断信息）"""
        try:
            LEARNED_CONSTANTS_PATH.parent.mkdir(parents=True, exist_ok=True)
            
            # 分类统计
            placeholder_count = sum(1 for c in self._learned_constants.values() if c.is_placeholder)
            real_count = len(self._learned_constants) - placeholder_count
            
            data = {
                'version': '2.0',  # 版本升级，包含诊断信息
                'updated': datetime.now().isoformat(),
                'statistics': {
                    'total': len(self._learned_constants),
                    'real_values': real_count,
                    'placeholders': placeholder_count
                },
                'constants': [
                    {
                        'name': c.name,
                        'rust_type': c.rust_type,
                        'value': c.value,
                        'source': c.source,
                        'timestamp': c.timestamp,
                        'usage_count': c.usage_count,
                        # 诊断信息
                        'is_placeholder': c.is_placeholder,
                        'c_source_file': c.c_source_file,
                        'c_source_line': c.c_source_line,
                        'original_c_definition': c.original_c_definition[:200] if c.original_c_definition else '',
                        'failure_reason': c.failure_reason,
                        'diagnostic_notes': c.diagnostic_notes
                    }
                    for c in self._learned_constants.values()
                ]
            }
            with open(LEARNED_CONSTANTS_PATH, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.debug(f"保存 {len(self._learned_constants)} 个学习常量 (真实值: {real_count}, 占位符: {placeholder_count})")
        except Exception as e:
            logger.error(f"保存学习常量失败: {e}")
    
    # =========================================================================
    # 辅助方法
    # =========================================================================
    
    def _is_valid_type_name(self, name: str) -> bool:
        """验证类型名称是否有效"""
        import re
        # 类型名必须是有效的标识符
        if not re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', name):
            return False
        # 排除关键字
        keywords = {'fn', 'let', 'mut', 'pub', 'struct', 'enum', 'type', 'impl', 'trait', 'use', 'mod'}
        if name.lower() in keywords:
            return False
        return True
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        # 占位符统计
        placeholder_types = [t for t in self._learned_types.values() if t.is_placeholder]
        placeholder_constants = [c for c in self._learned_constants.values() if c.is_placeholder]
        
        return {
            'builtin_types': len(self._builtin_types),
            'learned_types': len(self._learned_types),
            'learned_types_real': len(self._learned_types) - len(placeholder_types),
            'learned_types_placeholder': len(placeholder_types),
            'builtin_constants': len(self._builtin_constants),
            'learned_constants': len(self._learned_constants),
            'learned_constants_real': len(self._learned_constants) - len(placeholder_constants),
            'learned_constants_placeholder': len(placeholder_constants),
            'ohos_enabled': self.enable_ohos,
            'most_used_types': sorted(
                [(t.name, t.usage_count) for t in self._learned_types.values()],
                key=lambda x: x[1],
                reverse=True
            )[:10],
            'most_used_constants': sorted(
                [(c.name, c.usage_count) for c in self._learned_constants.values()],
                key=lambda x: x[1],
                reverse=True
            )[:10],
        }
    
    def get_placeholder_items(self) -> Dict[str, List]:
        """
        获取所有使用占位符的项（需要人工审查）
        
        Returns:
            {
                'types': [LearnedType, ...],
                'constants': [LearnedConstant, ...]
            }
        """
        return {
            'types': [t for t in self._learned_types.values() if t.is_placeholder],
            'constants': [c for c in self._learned_constants.values() if c.is_placeholder]
        }
    
    def generate_diagnostic_report(self, output_path: Optional[Path] = None) -> str:
        """
        生成诊断报告
        
        显示所有占位符项及其失败原因
        
        Args:
            output_path: 可选，保存报告的路径
            
        Returns:
            格式化的报告文本
        """
        lines = [
            "=" * 70,
            "预定义管理器诊断报告",
            f"生成时间: {datetime.now().isoformat()}",
            "=" * 70,
            "",
        ]
        
        stats = self.get_statistics()
        lines.append("📊 统计信息")
        lines.append(f"  内置类型: {stats['builtin_types']}")
        lines.append(f"  学习类型: {stats['learned_types']} (真实定义: {stats['learned_types_real']}, 占位符: {stats['learned_types_placeholder']})")
        lines.append(f"  内置常量: {stats['builtin_constants']}")
        lines.append(f"  学习常量: {stats['learned_constants']} (真实值: {stats['learned_constants_real']}, 占位符: {stats['learned_constants_placeholder']})")
        lines.append("")
        
        placeholders = self.get_placeholder_items()
        
        if placeholders['types']:
            lines.append("-" * 70)
            lines.append(f"⚠️ 需要审查的类型 ({len(placeholders['types'])} 个)")
            lines.append("-" * 70)
            
            for t in placeholders['types']:
                lines.append(f"\n[类型] {t.name}")
                lines.append(f"  当前定义: {t.definition[:80]}...")
                lines.append(f"  失败原因: {t.failure_reason}")
                if t.c_source_file:
                    lines.append(f"  C 源码位置: {t.c_source_file}:{t.c_source_line}")
                if t.original_c_definition:
                    lines.append(f"  原始 C 定义: {t.original_c_definition[:100]}...")
                if t.diagnostic_notes:
                    lines.append("  诊断备注:")
                    for note in t.diagnostic_notes:
                        lines.append(f"    - {note}")
        
        if placeholders['constants']:
            lines.append("")
            lines.append("-" * 70)
            lines.append(f"⚠️ 需要审查的常量 ({len(placeholders['constants'])} 个)")
            lines.append("-" * 70)
            
            for c in placeholders['constants']:
                lines.append(f"\n[常量] {c.name}")
                lines.append(f"  当前值: {c.rust_type} = {c.value}")
                lines.append(f"  失败原因: {c.failure_reason}")
                if c.c_source_file:
                    lines.append(f"  C 源码位置: {c.c_source_file}:{c.c_source_line}")
                if c.original_c_definition:
                    lines.append(f"  原始 C 定义: {c.original_c_definition[:100]}...")
                if c.diagnostic_notes:
                    lines.append("  诊断备注:")
                    for note in c.diagnostic_notes:
                        lines.append(f"    - {note}")
        
        if not placeholders['types'] and not placeholders['constants']:
            lines.append("✅ 所有学习的类型和常量都有真实定义，无需审查")
        
        report = '\n'.join(lines)
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"诊断报告已保存到: {output_path}")
        
        return report
    
    def export_rust_types(self) -> str:
        """导出所有类型为 Rust 代码"""
        lines = [
            "// Auto-generated type definitions",
            "// Includes: POSIX types" + (" + OpenHarmony types" if self.enable_ohos else ""),
            "",
            "use std::ffi::{c_void, c_char, c_long};",
            "",
        ]
        
        for name, definition in sorted(self.get_all_types().items()):
            lines.append(definition)
            lines.append("")
        
        return "\n".join(lines)
    
    def export_rust_constants(self) -> str:
        """导出所有常量为 Rust 代码"""
        lines = [
            "// Auto-generated constant definitions",
            "",
        ]
        
        for name, rust_type, value in sorted(self.get_all_constants(), key=lambda x: x[0]):
            lines.append(f"pub const {name}: {rust_type} = {value};")
        
        return "\n".join(lines)


# =========================================================================
# 便捷函数
# =========================================================================

def get_predefine_manager(enable_ohos: bool = True) -> PredefineManager:
    """获取预定义管理器单例"""
    return PredefineManager.get_instance(enable_ohos)


def learn_type_from_error(type_name: str, definition: str = None) -> bool:
    """从错误中学习类型"""
    manager = get_predefine_manager()
    if definition is None:
        definition = manager.generate_opaque_type(type_name)
    return manager.learn_type(type_name, definition, source="compilation_error")


def learn_constant_from_error(const_name: str, rust_type: str = None, value: str = None) -> bool:
    """从错误中学习常量"""
    manager = get_predefine_manager()
    if rust_type is None or value is None:
        rust_type, value = manager.infer_constant_type(const_name)
    return manager.learn_constant(const_name, rust_type, value, source="compilation_error")


# =========================================================================
# 测试代码
# =========================================================================

if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.DEBUG)
    
    # 测试预定义管理器
    manager = get_predefine_manager(enable_ohos=True)
    
    print("=" * 60)
    print("预定义管理器统计")
    print("=" * 60)
    stats = manager.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    print("测试学习功能")
    print("=" * 60)
    
    # 测试学习新类型
    success = manager.learn_type("MyNewType", manager.generate_opaque_type("MyNewType"))
    print(f"学习 MyNewType: {'成功' if success else '失败（已存在）'}")
    
    # 测试学习新常量
    success = manager.learn_constant("MY_NEW_CONST", "i32", "42")
    print(f"学习 MY_NEW_CONST: {'成功' if success else '失败（已存在）'}")
    
    # 测试从编译错误学习
    fake_errors = """
    error[E0412]: cannot find type `UnknownStruct` in this scope
    error[E0425]: cannot find value `HDF_ERR_UNKNOWN` in this scope
    """
    learned = manager.learn_from_compilation_errors(fake_errors)
    print(f"从编译错误学习: {learned[0]} 类型, {learned[1]} 常量")
    
    print("\n" + "=" * 60)
    print("最终统计")
    print("=" * 60)
    stats = manager.get_statistics()
    print(f"  总类型数: {stats['builtin_types'] + stats['learned_types']}")
    print(f"  总常量数: {stats['builtin_constants'] + stats['learned_constants']}")
