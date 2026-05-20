#!/usr/bin/env python3
"""
compile_commands.json 解析器

从 compile_commands.json 提取源文件的头文件搜索路径（-I 参数）
用于优化 bindgen 和预处理过程

增强功能 (2025-12-05):
- 全局头文件索引 (Global Header Indexing)
- 智能寻路：当 bindgen 报错找不到头文件时，自动从索引中查找
- 基于 Rustine 论文的 Build-Database Guided Discovery 方法
"""

import json
import re
import hashlib
import pickle
import shlex
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Set, Tuple, Any, Iterable
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
import logging
import os

logger = logging.getLogger(__name__)


# ========== 预处理上下文选择策略 ==========

class ContextSelectionStrategy(Enum):
    """上下文选择策略"""
    ACTIVE = "active"      # 用户指定目标 board/mode
    BEST = "best"          # 选择产生函数最多/宏最完整的编译命令
    UNION = "union"        # 多条命令取并集（需要 #[cfg] 管理）


@dataclass
class PreprocessingContext:
    """预处理上下文：一个源文件的编译配置"""
    source_file: Path
    entry: Dict                          # compile_commands.json 中的条目
    preprocessed_file: Optional[Path] = None  # 预处理后的 .i 文件路径
    function_count: int = 0              # 预处理后的函数数量（用于 best 策略）
    macro_count: int = 0                 # 宏定义数量（用于 best 策略）
    line_mapping: Dict[int, Tuple[str, int]] = field(default_factory=dict)  # .i 文件行号 -> (原文件, 原行号)
    error: Optional[str] = None          # 预处理错误信息
    # Proxy-TU: when `source_file` has no compile_commands entry, we may reuse a nearby TU's flags.
    proxy_used: bool = False
    proxy_entry_file: Optional[str] = None
    proxy_reason: Optional[str] = None
    # Best-effort diagnostics when clang -E fails.
    auto_resolve: Optional[Dict[str, Any]] = None


class CompileCommandsParser:
    """编译数据库解析器"""
    
    # 类级别的缓存目录
    _cache_dir = None
    
    @classmethod
    def _get_cache_dir(cls) -> Path:
        """获取缓存目录
        
        默认缓存位置：项目根目录下的 .cache/compile_commands/
        支持通过环境变量 C2R_CACHE_ROOT 将缓存放到某次运行目录下，便于实验隔离与清理：
          C2R_CACHE_ROOT=<run>/intermediate/cache
        """
        if cls._cache_dir is None:
            cache_root_env = os.environ.get("C2R_CACHE_ROOT", "").strip()
            if cache_root_env:
                cls._cache_dir = Path(cache_root_env).expanduser().resolve() / "compile_commands"
            else:
                # 获取 framework 根目录
                project_root = Path(__file__).resolve().parent.parent
                cls._cache_dir = project_root / ".cache" / "compile_commands"
            cls._cache_dir.mkdir(parents=True, exist_ok=True)
        return cls._cache_dir
    
    def __init__(self, compile_db_path: Path, ohos_root: Path = None):
        """
        初始化解析器
        
        Args:
            compile_db_path: compile_commands.json 的路径
            ohos_root: OpenHarmony 源码根目录（用于路径规范化）
        """
        self.compile_db_path = Path(compile_db_path)
        self.ohos_root = Path(ohos_root) if ohos_root else None
        self.compile_db = None
        self.file_index = {}  # 文件名 -> 条目列表的索引
        self._all_include_dirs_cache = None  # 缓存所有 include 路径
        
        # ========== 增强: 全局头文件索引 (Global Header Indexing) ==========
        # 用于快速查找头文件的真实路径
        # 格式: { "filename.h": ["/path/to/dir1", "/path/to/dir2"] }
        self._header_index = None  # 懒加载，避免启动太慢
        self._header_index_built = False
        # Missing-header resolution cache (per compile_db): header -> {success: str|None, candidates: [str,...]}
        self._header_resolution_cache: Dict[str, Dict[str, Any]] = {}
        self._header_resolution_cache_loaded: bool = False
        self._project_build_include_dirs_cache: Dict[str, List[Path]] = {}
        
        self._load_database()

    def _get_openharmony_source_root(self) -> Optional[Path]:
        """优先返回用户配置的 OpenHarmony 全量源码根目录。"""
        candidates = [
            os.environ.get("OPENHARMONY_SOURCE_ROOT", "").strip(),
            os.environ.get("OHOS_SOURCE_ROOT", "").strip(),
        ]
        if self.ohos_root:
            candidates.append(str(self.ohos_root))

        for raw in candidates:
            if not raw:
                continue
            try:
                p = Path(raw).expanduser().resolve()
            except Exception:
                p = Path(raw).expanduser()
            if p.exists():
                return p
        return None

    def _resolve_entry_directory(self, entry: Dict) -> Path:
        """
        Resolve entry['directory'] to an absolute path.

        Why:
        - For portability, some open-source layouts keep compile_commands entries with
          relative "directory" (e.g. ".").
        - Such paths should be interpreted relative to the compile_commands.json location,
          not the current process working directory.
        """
        base_dir = Path(entry.get("directory", ".") or ".")
        if base_dir.is_absolute():
            try:
                return base_dir.resolve()
            except Exception:
                return base_dir
        try:
            return (self.compile_db_path.parent / base_dir).resolve()
        except Exception:
            return (self.compile_db_path.parent / base_dir)

    def _get_header_resolution_cache_file(self) -> Path:
        """Per-compile_db cache for missing header resolution results (small JSON; avoids rewriting huge pickle cache)."""
        cache_key = hashlib.md5(str(self.compile_db_path.resolve()).encode()).hexdigest()
        return self._get_cache_dir() / f"{cache_key}.header_resolutions.json"

    def _load_header_resolution_cache(self) -> None:
        if self._header_resolution_cache_loaded:
            return
        self._header_resolution_cache_loaded = True
        p = self._get_header_resolution_cache_file()
        if not p.exists():
            self._header_resolution_cache = {}
            return
        try:
            data = json.loads(p.read_text(encoding="utf-8", errors="ignore") or "")
        except Exception:
            self._header_resolution_cache = {}
            return
        if not isinstance(data, dict):
            self._header_resolution_cache = {}
            return
        if str(data.get("compile_db_path") or "") != str(self.compile_db_path.resolve()):
            self._header_resolution_cache = {}
            return
        resolved = data.get("resolved")
        if not isinstance(resolved, dict):
            self._header_resolution_cache = {}
            return
        out: Dict[str, Dict[str, Any]] = {}
        for k, v in resolved.items():
            if not isinstance(k, str) or not k.strip():
                continue
            if not isinstance(v, dict):
                continue
            entry: Dict[str, Any] = {}
            success = v.get("success")
            if isinstance(success, str) and success.strip():
                entry["success"] = success.strip()
            candidates = v.get("candidates")
            if isinstance(candidates, list):
                entry["candidates"] = [str(x) for x in candidates if isinstance(x, str) and x.strip()]
            if entry:
                out[k] = entry
        self._header_resolution_cache = out

    def _save_header_resolution_cache(self) -> None:
        try:
            p = self._get_header_resolution_cache_file()
            payload = {
                "version": 1,
                "compile_db_path": str(self.compile_db_path.resolve()),
                "resolved": self._header_resolution_cache,
            }
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except Exception as e:
            logger.debug(f"保存 header resolution cache 失败: {e}")

    def _get_openharmony_search_root(self) -> Optional[Path]:
        """
        Return the preferred search root for missing headers.

        Requirement: search should be rooted at the `OpenHarmony/` directory when available.
        """
        source_root = self._get_openharmony_source_root()
        if not source_root:
            return None
        try:
            root = source_root.resolve()
        except Exception:
            root = source_root
        if root.name != "OpenHarmony":
            try:
                cand = (root / "OpenHarmony").resolve()
                if cand.exists():
                    return cand
            except Exception:
                pass
        return root

    def _find_project_root_for_source(self, source_file: Optional[Path]) -> Optional[Path]:
        """基于 source_file 推断当前被处理的项目根目录。"""
        if not source_file:
            return None
        try:
            cur = Path(source_file).resolve()
        except Exception:
            cur = Path(source_file)
        if cur.is_file():
            cur = cur.parent

        for parent in [cur, *cur.parents]:
            if (parent / "original_path.txt").exists():
                return parent
            if (parent / "BUILD.gn").exists():
                return parent
        return None

    def _parse_build_gn_include_dirs(self, build_gn_path: Path, project_root: Path) -> List[Path]:
        """从 BUILD.gn 中提取 include_dirs，并解析成本地绝对路径。"""
        cache_key = str(build_gn_path.resolve())
        cached = self._project_build_include_dirs_cache.get(cache_key)
        if cached is not None:
            return list(cached)

        include_dirs: List[Path] = []
        openharmony_root = self._get_openharmony_search_root()
        try:
            text = build_gn_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            logger.debug(f"读取 BUILD.gn 失败: {build_gn_path}: {e}")
            self._project_build_include_dirs_cache[cache_key] = []
            return []

        blocks = re.findall(r'include_dirs\s*=\s*\[(.*?)\]', text, re.DOTALL)
        seen: Set[str] = set()

        def _add_dir(p: Path) -> None:
            try:
                resolved = p.resolve()
            except Exception:
                resolved = p
            key = str(resolved)
            if key in seen or not resolved.exists() or not resolved.is_dir():
                return
            seen.add(key)
            include_dirs.append(resolved)

        for block in blocks:
            for raw in re.findall(r'"([^"]+)"', block):
                item = raw.strip()
                if not item:
                    continue
                if item.startswith("//"):
                    if openharmony_root:
                        _add_dir(openharmony_root / item[2:])
                    continue
                if item.startswith("/"):
                    _add_dir(Path(item))
                    continue
                _add_dir(project_root / item)

        self._project_build_include_dirs_cache[cache_key] = list(include_dirs)
        return list(include_dirs)

    def _get_project_missing_header_include_dirs(self, source_file: Optional[Path]) -> List[Path]:
        """补充 compile_commands 之外、但对当前项目有效的 include 目录。"""
        project_root = self._find_project_root_for_source(source_file)
        if not project_root:
            return []

        build_gn_path = project_root / "BUILD.gn"
        dirs: List[Path] = []
        seen: Set[str] = set()

        def _add_dir(p: Path) -> None:
            try:
                resolved = p.resolve()
            except Exception:
                resolved = p
            key = str(resolved)
            if key in seen or not resolved.exists() or not resolved.is_dir():
                return
            seen.add(key)
            dirs.append(resolved)

        # 当前项目目录本身也作为兜底 include 根目录。
        _add_dir(project_root)
        _add_dir(project_root / "include")

        if build_gn_path.exists():
            for inc in self._parse_build_gn_include_dirs(build_gn_path, project_root):
                _add_dir(inc)

        return dirs

    def _infer_preferred_subpaths_for_source(self, source_file: Optional[Path]) -> List[str]:
        """
        为当前源文件推断更可信的 OHOS 子树提示路径。

        对于 `data/ohos/source_projects/*` 里的搬运项目，优先使用 `original_path.txt`
        把本地相对路径映射回原始 OpenHarmony 子树。
        """
        if not source_file:
            return []

        hints: List[str] = []
        seen: Set[str] = set()

        def _add_hint(p: Path) -> None:
            raw = str(p).replace("\\", "/").strip().strip("/")
            if not raw or raw in seen:
                return
            seen.add(raw)
            hints.append(raw)

        search_root = self._get_openharmony_search_root()
        try:
            src_resolved = Path(source_file).resolve()
        except Exception:
            src_resolved = Path(source_file)

        if search_root:
            try:
                rel = src_resolved.relative_to(search_root)
                parts = list(rel.parts)
                for n in range(min(len(parts), 6), 1, -1):
                    _add_hint(Path(*parts[:n]))
            except Exception:
                pass

        project_root = self._find_project_root_for_source(src_resolved)
        if project_root:
            original_path_file = project_root / "original_path.txt"
            if original_path_file.exists():
                try:
                    original_rel = Path(original_path_file.read_text(encoding="utf-8", errors="ignore").strip())
                except Exception:
                    original_rel = None
                if original_rel:
                    try:
                        rel_in_project = src_resolved.relative_to(project_root)
                    except Exception:
                        rel_in_project = None
                    combined = original_rel / rel_in_project.parent if rel_in_project else original_rel
                    parts = list(combined.parts)
                    for n in range(min(len(parts), 8), 1, -1):
                        _add_hint(Path(*parts[:n]))

        return hints
    
    def _get_cache_file(self) -> Path:
        """获取缓存文件路径"""
        # 使用文件路径的哈希值作为缓存文件名
        cache_key = hashlib.md5(str(self.compile_db_path.resolve()).encode()).hexdigest()
        return self._get_cache_dir() / f"{cache_key}.cache"
    
    def _is_cache_valid(self, cache_file: Path) -> bool:
        """检查缓存是否有效"""
        if not cache_file.exists():
            return False
        
        if not self.compile_db_path.exists():
            return False
        
        try:
            # 检查源文件修改时间
            source_mtime = self.compile_db_path.stat().st_mtime
            cache_mtime = cache_file.stat().st_mtime
            
            # 如果源文件比缓存新，缓存无效
            if source_mtime > cache_mtime:
                return False
            
            return True
        except Exception:
            return False
    
    def _load_cache(self, cache_file: Path) -> Optional[Dict]:
        """从缓存文件加载数据"""
        try:
            print(f"    [缓存] 正在加载缓存: {cache_file.name}")
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            # 验证缓存数据格式
            if not isinstance(cache_data, dict):
                return None
            
            required_keys = ['compile_db', 'file_index', 'compile_db_path']
            if not all(key in cache_data for key in required_keys):
                return None
            
            # 验证缓存是否对应同一个文件
            if cache_data['compile_db_path'] != str(self.compile_db_path.resolve()):
                return None
            
            print(f"    ✓ 缓存加载成功")
            return cache_data
        except Exception as e:
            logger.debug(f"加载缓存失败: {e}")
            return None
    
    def _save_cache(self, cache_file: Path, data: Dict):
        """保存数据到缓存文件"""
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            print(f"    ✓ 缓存已保存: {cache_file.name}")
        except Exception as e:
            logger.warning(f"保存缓存失败: {e}")
    
    def _load_database(self):
        """加载编译数据库（优先使用缓存）"""
        print(f"  [加载] 正在加载 compile_commands.json: {self.compile_db_path}")
        if not self.compile_db_path.exists():
            logger.warning(f"compile_commands.json 不存在: {self.compile_db_path}")
            print(f"  ✗ 文件不存在")
            self.compile_db = []
            return
        
        # 尝试从缓存加载
        cache_file = self._get_cache_file()
        if self._is_cache_valid(cache_file):
            cache_data = self._load_cache(cache_file)
            if cache_data:
                self.compile_db = cache_data['compile_db']
                self.file_index = cache_data['file_index']
                print(f"  ✓ 从缓存加载: {len(self.compile_db)} 个编译条目")
                logger.info(f"从缓存加载 compile_commands.json: {len(self.compile_db)} 个条目")
                return
        
        # 缓存无效或不存在，重新加载
        print(f"    [缓存] 缓存无效或不存在，重新解析...")
        try:
            # 检查文件大小
            file_size = self.compile_db_path.stat().st_size
            print(f"    文件大小: {file_size / (1024*1024):.2f} MB")
            
            print(f"    正在读取 JSON 数据...")
            with open(self.compile_db_path, 'r', encoding='utf-8') as f:
                self.compile_db = json.load(f)
            
            print(f"  ✓ 成功加载: {len(self.compile_db)} 个编译条目")
            logger.info(f"加载 compile_commands.json: {len(self.compile_db)} 个条目")
            
            # 建立文件名索引（加速查找）
            print(f"    正在建立文件名索引...")
            self._build_file_index()
            print(f"  ✓ 索引建立完成: {len(self.file_index)} 个唯一文件名")
            
            # 保存到缓存（基础数据，include 路径会在 get_all_include_dirs 时添加）
            cache_data = {
                'compile_db': self.compile_db,
                'file_index': self.file_index,
                'compile_db_path': str(self.compile_db_path.resolve())
            }
            self._save_cache(cache_file, cache_data)
            
        except Exception as e:
            logger.error(f"加载 compile_commands.json 失败: {e}")
            print(f"  ✗ 加载失败: {e}")
            import traceback
            traceback.print_exc()
            self.compile_db = []
    
    def _build_file_index(self):
        """建立文件名到条目的索引"""
        self.file_index = {}
        for entry in self.compile_db:
            file_path = entry.get('file', '')
            if not file_path:
                continue
            
            file_name = Path(file_path).name
            if file_name not in self.file_index:
                self.file_index[file_name] = []
            self.file_index[file_name].append(entry)
    
    def get_includes_for_file(
        self, 
        source_file: Path,
        normalize_paths: bool = True
    ) -> List[Path]:
        """
        获取源文件的头文件搜索路径（-I 参数）
        
        Args:
            source_file: 源文件路径（绝对路径或相对路径）
            normalize_paths: 是否规范化路径（确保路径存在）
        
        Returns:
            头文件搜索路径列表（Path 对象）
        """
        if not self.compile_db:
            return []
        
        source_file = Path(source_file).resolve()
        source_file_str = str(source_file)
        source_file_name = source_file.name
        
        # 查找匹配的条目
        matched_entry = self._find_matching_entry(source_file, source_file_str, source_file_name)
        
        if not matched_entry:
            logger.debug(f"未找到编译命令: {source_file_name}")
            return []
        
        # 提取 -I 参数
        command = matched_entry.get('command', '')
        if not command:
            return []
        
        # 使用正则表达式提取所有 -I 路径
        # 匹配模式: -I/path 或 -I /path
        include_pattern = r'-I\s*([^\s]+)'
        include_paths = re.findall(include_pattern, command)
        
        # 转换为 Path 对象并规范化
        result = []
        base_dir = self._resolve_entry_directory(matched_entry)
        
        for inc_path_str in include_paths:
            inc_path = self._normalize_path(inc_path_str, base_dir)
            
            # 如果 normalize_paths=True，只返回存在的路径
            if normalize_paths:
                if inc_path.exists():
                    result.append(inc_path)
            else:
                result.append(inc_path)
        
        return result

    def get_clang_flags_for_file(
        self,
        source_file: Path,
        normalize_paths: bool = True,
    ) -> List[str]:
        """
        获取某个源文件对应的 clang 参数（用于 bindgen/clang -E），并尽量保持参数顺序。

        与 get_includes_for_file 的区别：
        - 不仅包含 -I，还会包含 -isystem/-D/-U/--sysroot/-include 等与解析头文件/宏相关的参数
        - 保留 compile_commands 中的原始顺序（对头文件选择很关键）

        Args:
            source_file: 源文件路径（绝对路径或相对路径）
            normalize_paths: 是否规范化路径（确保路径存在、相对路径按 entry.directory/ohos_root 解析）

        Returns:
            clang 参数列表（可直接拼接到 bindgen 的 `--` 之后）
        """
        if not self.compile_db:
            return []

        source_file = Path(source_file).resolve()
        source_file_str = str(source_file)
        source_file_name = source_file.name

        matched_entry = self._find_matching_entry(source_file, source_file_str, source_file_name)
        if not matched_entry:
            logger.debug(f"未找到编译命令: {source_file_name}")
            return []

        return self.get_clang_flags_for_entry(matched_entry, normalize_paths=normalize_paths)

    def get_clang_flags_for_entry(
        self,
        entry: Dict,
        normalize_paths: bool = True,
    ) -> List[str]:
        """
        从 compile_commands.json 的某个 entry 中提取 clang 参数（用于 bindgen/clang -E），并尽量保持参数顺序。

        说明：
        - 这是 preprocess-first 的关键：同一源文件可能在 compile_commands 里有多条 entry（不同宏/参数），
          上下文选择后必须严格使用“被选中的 entry”的参数，而不能再根据文件名/路径重新匹配 entry。
        """
        if not entry:
            return []

        args: List[str] = []
        if isinstance(entry.get("arguments"), list) and entry.get("arguments"):
            args = list(entry.get("arguments"))
        else:
            command = entry.get("command", "") or ""
            if not command:
                return []
            try:
                args = shlex.split(command, posix=True)
            except Exception:
                # 极端情况下保底；不会很准确，但比直接失败好
                args = command.split()

        base_dir = self._resolve_entry_directory(entry)

        def _fallback_sysroot_path(sysroot_path: Path) -> Optional[Path]:
            """
            Some OHOS compile_commands entries point to a product-scoped sysroot like:
              out/<out_dir>/<product>/sysroot
            which may not exist unless you run further build steps (ninja/actions).

            For preprocessing/bindgen we can often fall back to a board-scoped sysroot that *does* exist,
            e.g.:
              out/<out_dir>/obj/third_party/musl
              out/<out_dir>/sdk-native/os-irrelevant/sysroot
              out/<out_dir>/NOTICE_FILES/ndk/sysroot
            """
            try:
                parts = sysroot_path.parts
                if "out" not in parts:
                    return None
                idx = parts.index("out")
                if idx + 1 >= len(parts):
                    return None
                out_root = Path(*parts[: idx + 2])  # .../out/<out_dir>

                candidates = [
                    out_root / "obj" / "third_party" / "musl",
                    out_root / "sdk-native" / "os-irrelevant" / "sysroot",
                    out_root / "NOTICE_FILES" / "ndk" / "sysroot",
                ]
                for cand in candidates:
                    if not cand.exists():
                        continue
                    # Prefer a sysroot that actually contains headers.
                    if (cand / "usr" / "include").exists() or (cand / "include").exists():
                        return cand
                # Final fallback: use OpenHarmony SDK sysroot (better than host /usr/include).
                # This helps when a profile was generated with `--build-only-gn` and product sysroot
                # wasn't materialized under out/<out_dir>/... yet.
                if self.ohos_root:
                    sdk_root = (self.ohos_root / "prebuilts" / "ohos-sdk" / "linux").resolve()
                    if sdk_root.exists():
                        versions = []
                        for p in sdk_root.iterdir():
                            if p.is_dir() and p.name.isdigit():
                                versions.append(p)
                        versions.sort(key=lambda p: int(p.name), reverse=True)
                        for v in versions:
                            cand = v / "native" / "sysroot"
                            if cand.exists() and (cand / "usr" / "include").exists():
                                return cand
            except Exception:
                return None
            return None

        def _norm_path(p: str) -> Optional[str]:
            if not p:
                return None
            try:
                normalized = self._normalize_path(p, base_dir)
                if normalize_paths and not normalized.exists():
                    return None
                return str(normalized)
            except Exception:
                return None

        clang_flags: List[str] = []

        i = 0
        while i < len(args):
            a = args[i]

            # -I /path  or  -I/path
            if a == "-I":
                if i + 1 < len(args):
                    p = _norm_path(args[i + 1])
                    if p:
                        clang_flags.extend(["-I", p])
                    i += 2
                    continue
            if a.startswith("-I") and a != "-I":
                p = _norm_path(a[2:])
                if p:
                    clang_flags.extend(["-I", p])
                i += 1
                continue

            # -isystem /path
            if a == "-isystem":
                if i + 1 < len(args):
                    p = _norm_path(args[i + 1])
                    if p:
                        clang_flags.extend(["-isystem", p])
                    i += 2
                    continue
            if a.startswith("-isystem") and a != "-isystem":
                p = _norm_path(a[len("-isystem"):])
                if p:
                    clang_flags.extend(["-isystem", p])
                i += 1
                continue

            # -D NAME or -DNAME
            if a == "-D":
                if i + 1 < len(args):
                    clang_flags.append(f"-D{args[i + 1]}")
                    i += 2
                    continue
            if a.startswith("-D") and a != "-D":
                clang_flags.append(a)
                i += 1
                continue

            # -U NAME or -UNAME
            if a == "-U":
                if i + 1 < len(args):
                    clang_flags.append(f"-U{args[i + 1]}")
                    i += 2
                    continue
            if a.startswith("-U") and a != "-U":
                clang_flags.append(a)
                i += 1
                continue

            # -include file
            if a == "-include":
                if i + 1 < len(args):
                    p = _norm_path(args[i + 1])
                    if p:
                        clang_flags.extend(["-include", p])
                    i += 2
                    continue

            # -imacros file
            if a == "-imacros":
                if i + 1 < len(args):
                    p = _norm_path(args[i + 1])
                    if p:
                        clang_flags.extend(["-imacros", p])
                    i += 2
                    continue

            # --sysroot=...  or  --sysroot ...
            if a.startswith("--sysroot="):
                raw = a.split("=", 1)[1]
                try:
                    normalized = self._normalize_path(raw, base_dir)
                except Exception:
                    normalized = None
                if normalized is not None:
                    if normalize_paths and not normalized.exists():
                        fb = _fallback_sysroot_path(normalized)
                        if fb:
                            clang_flags.append(f"--sysroot={str(fb)}")
                    else:
                        clang_flags.append(f"--sysroot={str(normalized)}")
                i += 1
                continue
            if a == "--sysroot":
                if i + 1 < len(args):
                    raw = args[i + 1]
                    try:
                        normalized = self._normalize_path(raw, base_dir)
                    except Exception:
                        normalized = None
                    if normalized is not None:
                        if normalize_paths and not normalized.exists():
                            fb = _fallback_sysroot_path(normalized)
                            if fb:
                                clang_flags.append(f"--sysroot={str(fb)}")
                        else:
                            clang_flags.append(f"--sysroot={str(normalized)}")
                    i += 2
                    continue

            # -isysroot /path
            if a == "-isysroot":
                if i + 1 < len(args):
                    raw = args[i + 1]
                    try:
                        normalized = self._normalize_path(raw, base_dir)
                    except Exception:
                        normalized = None
                    if normalized is not None:
                        if normalize_paths and not normalized.exists():
                            fb = _fallback_sysroot_path(normalized)
                            if fb:
                                clang_flags.extend(["-isysroot", str(fb)])
                        else:
                            clang_flags.extend(["-isysroot", str(normalized)])
                    i += 2
                    continue

            # -target triple
            if a == "-target":
                if i + 1 < len(args):
                    clang_flags.extend(["-target", args[i + 1]])
                    i += 2
                    continue
            # --target triple  or  --target=triple (clang also supports this form; OHOS compile_commands uses it)
            if a == "--target":
                if i + 1 < len(args):
                    clang_flags.append(f"--target={args[i + 1]}")
                    i += 2
                    continue
            if a.startswith("--target="):
                clang_flags.append(a)
                i += 1
                continue

            # Common arch/CPU flags that affect builtin macros/types (helpful for preprocessing/bindgen)
            # Keep them verbatim (no path normalization needed).
            if a in ("-mthumb", "-marm"):
                clang_flags.append(a)
                i += 1
                continue
            if a.startswith((
                "-march=", "-mcpu=", "-mfpu=", "-mfloat-abi=", "-mtune=", "-mabi=",
                "-mno-", "-mno_", "-msoft-float", "-mhard-float",
            )):
                clang_flags.append(a)
                i += 1
                continue

            # 其余参数忽略（-O/-g/-c/-o/-M* 等对 bindgen 解析意义不大且可能引入噪声）
            i += 1

        return clang_flags

    def _resolve_entry_file_path(self, entry: Dict) -> Optional[Path]:
        """
        Resolve compile_commands entry['file'] to an absolute path (best-effort).

        Notes:
        - compile_commands.json entries often store file paths relative to entry['directory'].
        - We resolve relative paths against entry['directory'] first, then against ohos_root as a fallback.
        """
        entry_file = entry.get("file") or ""
        if not entry_file:
            return None

        entry_dir = self._resolve_entry_directory(entry)
        rel = Path(entry_file)
        path = rel
        if not path.is_absolute():
            path = entry_dir / rel
            # Fallback: some databases store paths relative to OpenHarmony root.
            if not path.exists() and self.ohos_root:
                path = self.ohos_root / rel

        try:
            return path.resolve(strict=False)
        except Exception:
            return path

    def get_entry_for_file_with_reason(self, source_file: Path) -> Tuple[Optional[Dict], Dict[str, Any]]:
        """
        Get the matched compile_commands entry for a source file, plus a small match-reason dict.

        This is primarily for diagnostics (e.g. bindgen failures) so logs can show whether the
        match was exact-path, suffix-path, or a fallback-by-filename candidate.
        """
        if not self.compile_db:
            return None, {"reason": "empty_compile_db", "candidates": 0, "suffix_len": None}

        source_file = Path(source_file).resolve()
        source_file_str = str(source_file)
        source_file_name = source_file.name
        return self._find_matching_entry_with_info(source_file, source_file_str, source_file_name)

    def _find_matching_entry(
        self, 
        source_file: Path, 
        source_file_str: str,
        source_file_name: str
    ) -> Optional[Dict]:
        """查找匹配的编译数据库条目"""
        entry, _info = self._find_matching_entry_with_info(source_file, source_file_str, source_file_name)
        return entry

    def _find_matching_entry_with_info(
        self,
        source_file: Path,
        source_file_str: str,
        source_file_name: str,
    ) -> Tuple[Optional[Dict], Dict[str, Any]]:
        """查找匹配的编译数据库条目（带诊断信息）"""

        info: Dict[str, Any] = {"reason": "none", "candidates": 0, "suffix_len": None}

        def _return(entry: Optional[Dict], reason: str, suffix_len: Optional[int] = None) -> Tuple[Optional[Dict], Dict[str, Any]]:
            info["reason"] = reason
            info["suffix_len"] = suffix_len
            if entry:
                info["matched_entry_file"] = entry.get("file")
                info["matched_entry_directory"] = entry.get("directory")
            return entry, info

        def _entry_score(entry: Dict) -> Tuple[int, int, int]:
            """
            Heuristic scoring for choosing between multiple compile_commands entries for the same file.

            Common case in OpenHarmony:
            - one entry is host toolchain (clang_x64)
            - another is target toolchain (arm-linux-ohos, with --sysroot)
            For bindgen/preprocess we usually want the target one.
            """
            directory = (entry.get("directory") or "").replace("\\", "/")
            args = entry.get("arguments")
            if isinstance(args, list) and args:
                text = " ".join(str(a) for a in args)
            else:
                text = str(entry.get("command") or "")

            text_norm = text.replace("\\", "/")

            has_target = ("--target=" in text_norm) or (" -target " in f" {text_norm} ")
            is_ohos_target = has_target and ("ohos" in text_norm)
            has_sysroot = ("--sysroot=" in text_norm) or (" --sysroot " in f" {text_norm} ") or (" -isysroot " in f" {text_norm} ")

            host_hint = (
                "/clang_x64/" in directory
                or "/clang_x64/" in text_norm
                or "x86_64-linux-gnu" in text_norm
                or "--target=x86_64" in text_norm
            )

            return (
                1 if is_ohos_target else 0,
                1 if has_sysroot else 0,
                0 if host_hint else 1,
            )

        def _pick_best_entry(entries: List[Dict], *, reason: str, suffix_len: Optional[int] = None) -> Tuple[Optional[Dict], Dict[str, Any]]:
            if not entries:
                return _return(None, reason, suffix_len=suffix_len)
            if len(entries) == 1:
                return _return(entries[0], reason, suffix_len=suffix_len)
            scored = [(_entry_score(e), e) for e in entries]
            scored.sort(key=lambda x: x[0], reverse=True)
            best_score, best_entry = scored[0]
            info["multi_entry_match"] = True
            info["multi_entry_count"] = len(entries)
            info["picked_score"] = list(best_score)
            # keep only a small sample for diagnostics
            info["picked_score_samples"] = [list(s) for s, _e in scored[:5]]
            return _return(best_entry, reason, suffix_len=suffix_len)

        # 策略1: 文件名匹配（使用索引）+ 结合 entry.directory 进行绝对路径对齐
        if source_file_name in self.file_index:
            candidates = self.file_index[source_file_name]
            info["candidates"] = len(candidates)
            
            # 优先匹配：路径相似度高的
            exact_matches: List[Dict] = []
            for entry in candidates:
                entry_path = self._resolve_entry_file_path(entry)
                if entry_path and entry_path == source_file:
                    exact_matches.append(entry)
            if exact_matches:
                return _pick_best_entry(exact_matches, reason="filename_index_exact_path", suffix_len=None)
            
            # 策略1.2: 末尾路径片段匹配（降低同名文件碰撞误选概率）
            source_parts = source_file.parts
            for suffix_len in (4, 3, 2):
                if len(source_parts) < suffix_len:
                    continue
                suffix = source_parts[-suffix_len:]
                suffix_matches: List[Dict] = []
                for entry in candidates:
                    entry_path = self._resolve_entry_file_path(entry)
                    if not entry_path:
                        continue
                    parts = entry_path.parts
                    if len(parts) >= suffix_len and parts[-suffix_len:] == suffix:
                        suffix_matches.append(entry)
                if suffix_matches:
                    return _pick_best_entry(suffix_matches, reason="filename_index_suffix_path", suffix_len=suffix_len)

            # 如果没有精确匹配，返回第一个候选（可能是相对路径/同名碰撞）
            if candidates:
                # Still prefer target/sysroot style entries (reduces host/target mixups).
                return _pick_best_entry(list(candidates), reason="filename_index_first_candidate", suffix_len=None)
            return _return(None, "filename_index_no_candidates")
        
        # 策略2: 无索引命中时的保底扫描（通常不会走到这里）
        for entry in self.compile_db:
            entry_path = self._resolve_entry_file_path(entry)
            if entry_path and str(entry_path) == source_file_str:
                return _return(entry, "full_scan_exact_str")

        return _return(None, "no_match")
    
    def _normalize_path(self, path_str: str, base_dir: Path) -> Path:
        """
        规范化路径
        
        - 如果是绝对路径，直接使用
        - 如果是相对路径，相对于 base_dir
        - 如果路径不存在且提供了 ohos_root，尝试相对于 ohos_root
        """
        path = Path(path_str)
        
        # 如果是绝对路径，直接使用
        if path.is_absolute():
            return path.resolve()
        
        # 如果是相对路径，相对于 base_dir
        result = (base_dir / path).resolve()
        
        # 如果路径不存在且提供了 ohos_root，尝试相对于 ohos_root
        if not result.exists() and self.ohos_root:
            result = (self.ohos_root / path).resolve()
        
        return result
    
    def get_all_include_dirs(self, source_files: List[Path] = None) -> Set[Path]:
        """
        获取所有源文件的 include 路径（去重）
        
        优化：
        1. 使用内存缓存，避免重复计算
        2. 直接从 compile_db 中提取所有 -I 路径，避免逐个文件查找
        
        Args:
            source_files: 源文件列表（已废弃，保留以兼容旧代码）
        
        Returns:
            所有 include 路径的集合
        """
        # 如果已经缓存，直接返回
        if self._all_include_dirs_cache is not None:
            return self._all_include_dirs_cache
        
        # 尝试从缓存文件加载
        cache_file = self._get_cache_file()
        if cache_file.exists():
            try:
                cache_data = self._load_cache(cache_file)
                if cache_data and 'all_include_dirs' in cache_data:
                    self._all_include_dirs_cache = set(Path(p) for p in cache_data['all_include_dirs'])
                    print(f"    ✓ 从缓存加载 include 路径: {len(self._all_include_dirs_cache)} 个")
                    return self._all_include_dirs_cache
            except Exception as e:
                logger.debug(f"从缓存加载 include 路径失败: {e}")
        
        # 缓存未命中，重新计算
        all_includes = set()
        
        # 优化：直接从 compile_db 中提取所有 -I 路径
        # 这比逐个文件查找快得多
        include_pattern = r'-I\s*([^\s]+)'
        
        print(f"    [提取] 从 compile_db 提取所有 include 路径...")
        processed_entries = 0
        for entry in self.compile_db:
            processed_entries += 1
            if processed_entries > 0 and processed_entries % 10000 == 0:
                print(f"      已处理 {processed_entries}/{len(self.compile_db)} 个条目...")
            
            command = entry.get('command', '')
            if not command:
                continue
            
            # 提取所有 -I 路径
            include_paths = re.findall(include_pattern, command)
            base_dir = self._resolve_entry_directory(entry)
            
            for inc_path_str in include_paths:
                inc_path = self._normalize_path(inc_path_str, base_dir)
                if inc_path.exists():
                    all_includes.add(inc_path)
        
        print(f"    ✓ 从 {len(self.compile_db)} 个条目中提取了 {len(all_includes)} 个唯一 include 路径")
        
        # 保存到内存缓存
        self._all_include_dirs_cache = all_includes
        
        # 保存到文件缓存（更新现有缓存文件或创建新缓存）
        try:
            cache_data = self._load_cache(cache_file)
            if not cache_data:
                # 如果缓存文件不存在，创建新的缓存数据
                cache_data = {
                    'compile_db': self.compile_db,
                    'file_index': self.file_index,
                    'compile_db_path': str(self.compile_db_path.resolve())
                }
            # 添加 include 路径
            cache_data['all_include_dirs'] = [str(p) for p in all_includes]
            self._save_cache(cache_file, cache_data)
        except Exception as e:
            logger.debug(f"保存 include 路径到缓存失败: {e}")
        
        return all_includes

    def find_first_source_file_containing(self, subpath: str) -> Optional[Path]:
        """
        在编译数据库中查找第一个路径包含 subpath 的源文件。

        典型用途：某些 SelfContained 模块在当前产品配置下并不参与编译，
        因而 compile_commands.json 里没有它们自己的 .c 条目。
        这时仍可选取同一子系统（例如 kernel/liteos_a）的任意一个 TU 作为“编译上下文代理”，
        以获得更接近真实构建的 include 顺序与宏定义。

        Args:
            subpath: 形如 "kernel/liteos_a" 的子路径片段（使用 '/' 分隔）

        Returns:
            匹配到的源文件路径（Path）或 None
        """
        if not self.compile_db:
            return None

        needle = (subpath or "").replace("\\", "/")
        if not needle:
            return None

        # Memoize because this can be called repeatedly during bindgen retries / proxy TU selection.
        try:
            cache = getattr(self, "_first_source_file_containing_cache", None)
            if cache is None:
                cache = {}
                setattr(self, "_first_source_file_containing_cache", cache)
            if needle in cache:
                return cache[needle]
        except Exception:
            cache = None

        best: Optional[Path] = None
        best_exists = False

        for entry in self.compile_db:
            entry_path = self._resolve_entry_file_path(entry)
            if not entry_path:
                continue

            file_norm = str(entry_path).replace("\\", "/")
            if needle not in file_norm:
                continue

            # Only return C/C++ source files.
            if entry_path.suffix.lower() not in {".c", ".cc", ".cpp", ".cxx"}:
                continue

            # Prefer existing files to avoid misleading proxy paths and to support downstream readers.
            exists = False
            try:
                exists = entry_path.exists()
            except Exception:
                exists = False

            if best is None:
                best = entry_path
                best_exists = exists
                if best_exists:
                    # Fast path: the first existing match is usually good enough.
                    break
                continue

            if exists and not best_exists:
                best = entry_path
                best_exists = True
                break

        if cache is not None:
            try:
                cache[needle] = best
            except Exception:
                pass

        return best
    
    def has_file(self, source_file: Path) -> bool:
        """检查源文件是否在编译数据库中"""
        source_file = Path(source_file).resolve()
        source_file_str = str(source_file)
        source_file_name = source_file.name
        
        # 精确匹配
        for entry in self.compile_db:
            if entry.get('file') == source_file_str:
                return True
        
        # 文件名匹配
        if source_file_name in self.file_index:
            return True
        
        return False
    
    # =========================================================================
    # 全局头文件索引与智能寻路 (Global Header Indexing & Smart Discovery)
    # 基于 Rustine 论文的 Build-Database Guided Discovery 方法
    # =========================================================================
    
    def find_header_path(
        self,
        header_name: str,
        *,
        preferred_subpaths: Optional[List[str]] = None,
        preferred_include_dirs: Optional[Iterable[Path]] = None,
        source_file: Optional[Path] = None,
    ) -> Optional[str]:
        """
        在所有已知的 include 目录中查找特定的头文件
        
        这是解决 bindgen "file not found" 错误的核心方法。
        它不是造假（Mock），而是利用构建数据库找到头文件的真实路径。
        
        Args:
            header_name: 头文件名（可能包含路径，如 "core/hdf_device_desc.h"）
        
        Returns:
            头文件所在的目录路径 (str) 或 None
        """
        if not self._all_include_dirs_cache:
            self.get_all_include_dirs()

        # Load per-compile-db header resolution cache (small JSON).
        self._load_header_resolution_cache()
        cached_entry = self._header_resolution_cache.get(header_name) if isinstance(self._header_resolution_cache, dict) else None
        if isinstance(cached_entry, dict):
            cached_success = cached_entry.get("success")
            if isinstance(cached_success, str) and cached_success.strip():
                try:
                    d = Path(cached_success).expanduser()
                    if (d / header_name).exists():
                        return str(d)
                except Exception:
                    pass

        if not preferred_subpaths:
            preferred_subpaths = self._infer_preferred_subpaths_for_source(source_file)

        # Normalize hints
        norm_subpaths: List[Path] = []
        for s in (preferred_subpaths or []):
            s = (s or "").replace("\\", "/").strip().strip("/")
            if not s:
                continue
            norm_subpaths.append(Path(s))

        preferred_dirs_set: Set[Path] = set()
        try:
            if preferred_include_dirs:
                preferred_dirs_set = {Path(p) for p in preferred_include_dirs if p}
        except Exception:
            preferred_dirs_set = set()

        ohos_root_resolved: Optional[Path] = self._get_openharmony_search_root()
        extra_include_dirs = self._get_project_missing_header_include_dirs(source_file)

        def _prefix_match_len(a: List[str], b: List[str]) -> int:
            n = 0
            for x, y in zip(a, b):
                if x != y:
                    break
                n += 1
            return n

        def _score_include_dir(p: Path) -> int:
            score = 0
            p_str = str(p).replace("\\", "/")
            p_low = p_str.lower()

            # Strong preference: already-known include dirs (from previous attempts).
            if preferred_dirs_set:
                try:
                    if p in preferred_dirs_set:
                        score += 10_000
                except Exception:
                    pass

            # Prefer dirs that lie under (or close to) the project's original subtree in OHOS tree.
            if norm_subpaths and ohos_root_resolved:
                try:
                    rel = Path(p).resolve().relative_to(ohos_root_resolved)
                    rel_parts = [x.lower() for x in rel.parts]
                except Exception:
                    rel_parts = []
                for sub in norm_subpaths:
                    hint_parts = [x.lower() for x in sub.parts]
                    if not hint_parts:
                        continue
                    # Prefix match is stronger than plain substring.
                    score += _prefix_match_len(rel_parts, hint_parts) * 200
                    # Substring match is a cheap fallback when rel_to fails.
                    hint_norm = str(sub).replace("\\", "/").lower()
                    if hint_norm and hint_norm in p_low:
                        score += 500

            return score

        def _dedupe_keep_order(xs: Iterable[str]) -> List[str]:
            out: List[str] = []
            seen: Set[str] = set()
            for x in xs:
                if not isinstance(x, str):
                    continue
                s = x.strip()
                if not s or s in seen:
                    continue
                seen.add(s)
                out.append(s)
            return out

        def _cache_add_candidates(header: str, include_dirs: List[Path]) -> None:
            if not header or not include_dirs:
                return
            try:
                entry = self._header_resolution_cache.get(header)
                if not isinstance(entry, dict):
                    entry = {}
                    self._header_resolution_cache[header] = entry
                existing = entry.get("candidates")
                if not isinstance(existing, list):
                    existing = []
                merged = _dedupe_keep_order([*existing, *[str(p) for p in include_dirs if p]])
                # Keep cache bounded to avoid huge JSON for very common headers (e.g. string.h).
                try:
                    max_keep = int(os.environ.get("C2R_HEADER_RESOLUTION_CACHE_MAX_CANDIDATES", "200"))
                except Exception:
                    max_keep = 200
                if max_keep > 0 and len(merged) > max_keep:
                    # Prefer higher-scored dirs under the current hint context.
                    try:
                        merged = sorted(merged, key=lambda s: _score_include_dir(Path(s)), reverse=True)[:max_keep]
                    except Exception:
                        merged = merged[:max_keep]
                entry["candidates"] = merged
                self._save_header_resolution_cache()
            except Exception as e:
                logger.debug(f"更新 header resolution cache 失败: {e}")

        # 1. 优先在索引/缓存中查找（fast path；避免重复扫描大树）
        cand_dirs: List[str] = []
        if self._header_index and header_name in self._header_index:
            try:
                cand_dirs.extend([str(x) for x in (self._header_index.get(header_name) or [])])
            except Exception:
                pass
        if isinstance(cached_entry, dict):
            cands = cached_entry.get("candidates")
            if isinstance(cands, list):
                cand_dirs.extend([str(x) for x in cands if isinstance(x, str) and x.strip()])
        cand_dirs = _dedupe_keep_order(cand_dirs)
        if cand_dirs:
            best = None
            best_score = None
            for d in cand_dirs:
                try:
                    p = Path(d)
                except Exception:
                    continue
                sc = _score_include_dir(p)
                if best_score is None or sc > best_score:
                    best_score = sc
                    best = d
            if best:
                return best

        # 2. 懒惰扫描：遍历所有 include 目录查找该文件
        # 这比构建全量索引快，因为只针对报错的文件找
        print(f"🔎 Hunting for missing header: {header_name} ...")

        matches: List[Path] = []
        best_dir: Optional[Path] = None
        best_score: Optional[int] = None

        search_include_dirs: List[Path] = []
        seen_search_dirs: Set[str] = set()
        for include_dir in [*extra_include_dirs, *list(self._all_include_dirs_cache or [])]:
            try:
                resolved = include_dir.resolve()
            except Exception:
                resolved = include_dir
            key = str(resolved)
            if key in seen_search_dirs:
                continue
            seen_search_dirs.add(key)
            search_include_dirs.append(resolved)

        for include_dir in search_include_dirs:
            candidate = include_dir / header_name
            if not candidate.exists():
                continue

            matches.append(include_dir)
            sc = _score_include_dir(include_dir)
            if best_score is None or sc > best_score:
                best_score = sc
                best_dir = include_dir

        if best_dir is not None:
            print(f"🎉 Found {header_name} at {best_dir}")
            # 缓存所有匹配（后续同名缺失可直接打分选择）
            if self._header_index is None:
                self._header_index = defaultdict(list)
            for d in matches:
                self._header_index[header_name].append(str(d))
            _cache_add_candidates(header_name, matches)
            return str(best_dir)
        
        # 3. 如果直接路径没找到，尝试只匹配文件名（去掉子目录）
        base_name = Path(header_name).name
        if base_name != header_name:
            print(f"🔎 Trying base name: {base_name} ...")
            for include_dir in search_include_dirs:
                # 在 include 目录的所有子目录中查找
                for candidate in include_dir.rglob(base_name):
                    if candidate.is_file():
                        # 验证完整路径是否匹配
                        parent_dir = candidate.parent
                        # 计算需要添加的 include 路径
                        # 例如：找到 /a/b/c/core/hdf.h，header_name="core/hdf.h"
                        # 则需要添加 /a/b/c
                        relative_parts = header_name.split('/')
                        if len(relative_parts) > 1:
                            # 向上回溯找到正确的 include 目录
                            target_dir = parent_dir
                            for _ in range(len(relative_parts) - 1):
                                target_dir = target_dir.parent
                            # 验证
                            if (target_dir / header_name).exists():
                                print(f"🎉 Found {header_name} via rglob at {target_dir}")
                                if self._header_index is None:
                                    self._header_index = defaultdict(list)
                                self._header_index[header_name].append(str(target_dir))
                                _cache_add_candidates(header_name, [target_dir])
                                return str(target_dir)
        
        # 4. 最后尝试：在 OpenHarmony 源码树中搜索（BFS + 深度剪枝）
        search_root = self._get_openharmony_search_root()
        if search_root and search_root.exists():
            try:
                max_depth = int(os.environ.get("C2R_MISSING_HEADER_SEARCH_MAX_DEPTH", "10"))
            except Exception:
                max_depth = 10
            max_depth = max(0, int(max_depth))
            print(f"🔎 Searching OpenHarmony source tree (depth<={max_depth}) for: {header_name} ...")

            try:
                relative_parts = header_name.split("/")
            except Exception:
                relative_parts = []

            matches: List[Path] = []
            best_dir: Optional[Path] = None
            best_score: Optional[int] = None

            q = deque([(Path(search_root), 0)])
            while q:
                cur_dir, depth = q.popleft()
                try:
                    with os.scandir(cur_dir) as it:
                        for ent in it:
                            try:
                                if ent.is_dir(follow_symlinks=False):
                                    if depth < max_depth:
                                        q.append((Path(ent.path), depth + 1))
                                    continue
                                if not ent.is_file(follow_symlinks=False):
                                    continue
                                if ent.name != base_name:
                                    continue

                                parent_dir = Path(ent.path).parent
                                if header_name == base_name:
                                    target_dir = parent_dir
                                    # No further verification needed: base_name matches.
                                else:
                                    if len(relative_parts) <= 1:
                                        continue
                                    target_dir = parent_dir
                                    for _ in range(len(relative_parts) - 1):
                                        target_dir = target_dir.parent
                                    if not (target_dir / header_name).exists():
                                        continue

                                matches.append(target_dir)
                                sc = _score_include_dir(target_dir)
                                if best_score is None or sc > best_score:
                                    best_score = sc
                                    best_dir = target_dir
                            except PermissionError:
                                continue
                            except Exception:
                                continue
                except PermissionError:
                    continue
                except FileNotFoundError:
                    continue
                except Exception as e:
                    logger.debug(f"Source tree BFS scan failed at {cur_dir}: {e}")
                    continue

            if best_dir is not None:
                print(f"🎉 Found {header_name} via source tree BFS at {best_dir}")
                if self._header_index is None:
                    self._header_index = defaultdict(list)
                for d in matches:
                    self._header_index[header_name].append(str(d))
                _cache_add_candidates(header_name, matches)
                return str(best_dir)
        
        print(f"❌ Could not find header: {header_name}")
        return None
    
    def find_multiple_headers(self, header_names: List[str]) -> Dict[str, Optional[str]]:
        """
        批量查找多个头文件
        
        Args:
            header_names: 头文件名列表
        
        Returns:
            字典 { header_name: include_path_or_none }
        """
        results = {}
        for header in header_names:
            results[header] = self.find_header_path(header)
        return results
    
    def build_header_index(self, extensions: List[str] = None):
        """
        构建全局头文件索引（完整扫描）
        
        这是一个可选的优化：预先扫描所有 include 目录，建立文件名到路径的映射。
        对于大型项目（如 OpenHarmony），可能需要几分钟，但后续查找会很快。
        
        Args:
            extensions: 要索引的文件扩展名列表，默认 ['.h', '.hpp', '.hxx']
        """
        if self._header_index_built:
            return
        
        if extensions is None:
            extensions = ['.h', '.hpp', '.hxx', '.inc']
        
        if not self._all_include_dirs_cache:
            self.get_all_include_dirs()
        
        print(f"🔨 Building global header index from {len(self._all_include_dirs_cache)} include directories...")
        
        self._header_index = defaultdict(list)
        total_headers = 0
        processed_dirs = 0
        
        for include_dir in self._all_include_dirs_cache:
            processed_dirs += 1
            if processed_dirs % 500 == 0:
                print(f"   Processed {processed_dirs}/{len(self._all_include_dirs_cache)} directories, found {total_headers} headers...")
            
            try:
                # 只扫描直接子文件，不递归（避免重复）
                for item in include_dir.iterdir():
                    if item.is_file() and item.suffix.lower() in extensions:
                        self._header_index[item.name].append(str(include_dir))
                        total_headers += 1
            except PermissionError:
                continue
            except Exception as e:
                logger.debug(f"Error scanning {include_dir}: {e}")
                continue
        
        self._header_index_built = True
        print(f"✅ Header index built: {total_headers} headers in {len(self._header_index)} unique names")
        
        # 保存到缓存
        try:
            cache_file = self._get_cache_file()
            cache_data = self._load_cache(cache_file)
            if cache_data:
                cache_data['header_index'] = dict(self._header_index)
                self._save_cache(cache_file, cache_data)
        except Exception as e:
            logger.debug(f"Failed to save header index to cache: {e}")
    
    def get_resolved_includes_for_bindgen(
        self,
        missing_headers: List[str],
        current_includes: Set[Path],
        *,
        preferred_subpaths: Optional[List[str]] = None,
    ) -> Tuple[Set[Path], List[str], Dict[str, Optional[str]]]:
        """
        为 bindgen 解决缺失的头文件

        这是供 skeleton_builder 调用的便捷方法。

        Args:
            missing_headers: bindgen 报错中的缺失头文件列表
            current_includes: 当前已有的 include 路径集合

        Returns:
            (新的 include 路径集合, 仍然无法解决的头文件列表, 解析结果映射)
        """
        new_includes = set(current_includes)
        unresolved = []
        resolved_map: Dict[str, Optional[str]] = {}

        for header in missing_headers:
            found_dir = self.find_header_path(
                header,
                preferred_subpaths=preferred_subpaths,
                preferred_include_dirs=current_includes,
                source_file=None,
            )
            if found_dir:
                resolved_map[header] = found_dir
                found_path = Path(found_dir)
                if found_path not in new_includes:
                    print(f"✨ Auto-resolved: {header} -> {found_dir}")
                    new_includes.add(found_path)
            else:
                resolved_map[header] = None
                unresolved.append(header)

        return new_includes, unresolved, resolved_map

    # =========================================================================
    # 预处理上下文选择与预处理功能 (Preprocess-First Approach)
    # =========================================================================

    def get_all_entries_for_file(self, source_file: Path) -> List[Dict]:
        """
        获取源文件的所有编译数据库条目

        同一个源文件可能在不同的编译配置下有多条命令（不同 board/mode）

        Args:
            source_file: 源文件路径

        Returns:
            编译条目列表
        """
        if not self.compile_db:
            return []

        source_file = Path(source_file).resolve()
        source_file_str = str(source_file)
        source_file_name = source_file.name

        entries = []

        # 文件名匹配（使用索引）
        if source_file_name in self.file_index:
            candidates = self.file_index[source_file_name]
            for entry in candidates:
                entry_path = self._resolve_entry_file_path(entry)
                if entry_path and entry_path == source_file:
                    entries.append(entry)

            # 如果通过解析找到，返回
            if entries:
                return entries

            # 否则返回所有候选项（可能是相对路径）
            return candidates

        return entries

    def _find_proxy_entries_for_missing_file(
        self,
        source_file: Path,
        *,
        target_config: Optional[str] = None,
        max_depth: int = 6,
        max_files_per_dir: int = 40,
        max_entries: int = 30,
    ) -> List[Dict]:
        """
        When `source_file` is not present in compile_commands.json, try to find a "nearby" TU entry
        (same directory tree) and reuse its flags to preprocess `source_file`.

        This is a best-effort closure strategy: it does NOT claim semantic correctness, but it often
        restores macro/include visibility so downstream stages can operate on a stable `.i`.
        """
        try:
            source_file = Path(source_file).resolve()
        except Exception:
            source_file = Path(source_file)

        # Limit to common TU extensions.
        exts = {".c", ".cc", ".cpp", ".cxx", ".c++"}

        def _entry_key(e: Dict) -> str:
            try:
                return json.dumps(
                    {
                        "file": e.get("file"),
                        "directory": e.get("directory"),
                        "command": e.get("command"),
                        "arguments": e.get("arguments"),
                    },
                    sort_keys=True,
                )
            except Exception:
                return str(id(e))

        seen: Set[str] = set()
        proxy_entries: List[Dict] = []

        cur_dir = source_file.parent
        for _ in range(max(1, int(max_depth or 6))):
            try:
                if not cur_dir.exists() or not cur_dir.is_dir():
                    break
            except Exception:
                break

            # Scan a small number of source files in this directory as proxy candidates.
            try:
                files = [p for p in cur_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
                files.sort(key=lambda p: p.name)
            except Exception:
                files = []

            if files:
                files = files[: max(1, int(max_files_per_dir or 40))]

            for cand in files:
                if cand == source_file:
                    continue
                entries = self.get_all_entries_for_file(cand)
                if not entries:
                    continue
                # Prefer exact-path matched entries to avoid filename collision.
                for e in entries:
                    ep = self._resolve_entry_file_path(e)
                    if not ep or ep != cand.resolve():
                        continue
                    if target_config:
                        directory = e.get("directory", "") or ""
                        command = e.get("command", "") or ""
                        if target_config not in directory and target_config not in command:
                            continue
                    k = _entry_key(e)
                    if k in seen:
                        continue
                    seen.add(k)
                    proxy_entries.append(e)
                    if len(proxy_entries) >= max(1, int(max_entries or 30)):
                        break
                if len(proxy_entries) >= max(1, int(max_entries or 30)):
                    break

            if proxy_entries:
                break

            # Stop when reaching OHOS root (if configured).
            if self.ohos_root:
                try:
                    if cur_dir == self.ohos_root.resolve():
                        break
                except Exception:
                    pass
            # Walk up.
            cur_dir = cur_dir.parent

        return proxy_entries

    def preprocess_with_context(
        self,
        source_file: Path,
        entry: Dict,
        output_dir: Path = None,
        timeout_sec: int = 60,
    ) -> PreprocessingContext:
        """
        使用特定编译上下文预处理源文件

        运行 `clang -E` 生成预处理后的 .i 文件，保留 #line 信息

        Args:
            source_file: 源文件路径
            entry: compile_commands.json 中的条目
            output_dir: 输出目录（默认使用临时目录）

        Returns:
            预处理上下文对象
        """
        source_file = Path(source_file).resolve()
        context = PreprocessingContext(source_file=source_file, entry=entry)

        # 创建输出目录
        if output_dir is None:
            output_dir = Path(tempfile.gettempdir()) / "c2rust_preprocessed"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 生成输出文件名：源文件名 + entry哈希 + .i
        entry_hash = hashlib.md5(json.dumps(entry, sort_keys=True).encode()).hexdigest()[:8]
        output_file = output_dir / f"{source_file.stem}_{entry_hash}.i"

        # 获取 clang 参数
        # 关键：必须使用“本次选中的 entry”对应的参数，否则 active/best 策略会失效
        clang_flags = self.get_clang_flags_for_entry(entry, normalize_paths=True)
        base_clang_flags = list(clang_flags or [])

        def _extract_missing_headers(stderr_text: str) -> List[str]:
            if not stderr_text:
                return []
            missing = re.findall(r"'([^']+)' file not found", stderr_text)
            missing += re.findall(r"fatal error:\s*([^\s:]+)\s*file not found", stderr_text)
            if not missing:
                return []
            seen = set()
            out: List[str] = []
            for f in missing:
                f = (f or "").strip().strip('"').strip("'").strip("<>").strip()
                if f and f not in seen:
                    seen.add(f)
                    out.append(f)
            return out

        def _include_dirs_from_flags(flags: List[str]) -> Set[Path]:
            dirs: Set[Path] = set()
            i = 0
            while i < len(flags):
                f = flags[i]
                if f == "-I" and i + 1 < len(flags):
                    p = (flags[i + 1] or "").strip()
                    if p:
                        dirs.add(Path(p))
                    i += 2
                    continue
                if f == "-isystem" and i + 1 < len(flags):
                    p = (flags[i + 1] or "").strip()
                    if p:
                        dirs.add(Path(p))
                    i += 2
                    continue
                if isinstance(f, str) and f.startswith("-I") and f != "-I":
                    p = f[2:].strip()
                    if p:
                        dirs.add(Path(p))
                if isinstance(f, str) and f.startswith("-isystem") and f != "-isystem":
                    p = f[len("-isystem") :].strip()
                    if p:
                        dirs.add(Path(p))
                i += 1
            return dirs

        def _append_include_flags(flags: List[str], include_dirs: Iterable[Path]) -> List[str]:
            out = list(flags or [])
            existing = _include_dirs_from_flags(out)
            for d in include_dirs:
                try:
                    p = Path(d)
                except Exception:
                    continue
                if p in existing:
                    continue
                out.extend(["-I", str(p)])
                existing.add(p)
            return out

        enable_auto = os.environ.get("C2R_PREPROCESS_AUTO_RESOLVE_INCLUDES", "1").strip().lower() in ("1", "true", "yes", "on")
        try:
            max_rounds = int(os.environ.get("C2R_PREPROCESS_AUTO_RESOLVE_ROUNDS", "100"))
        except Exception:
            max_rounds = 100
        max_rounds = max(0, max_rounds)

        preferred_subpaths: Optional[List[str]] = self._infer_preferred_subpaths_for_source(source_file) or None

        include_dirs_base = _include_dirs_from_flags(base_clang_flags)
        # For headers with multiple candidates (e.g. los_config.h), we may need to try alternatives.
        resolved_include_by_header: Dict[str, str] = {}
        tried_include_by_header: Dict[str, Set[str]] = defaultdict(set)

        # 构建 clang -E 命令
        # 优先使用 OpenHarmony 预置 clang，避免宿主机 clang 对 OHOS target/sysroot 的兼容性问题。
        clang_bin = shutil.which("clang") or "clang"
        if self.ohos_root:
            try:
                ohos_clang = (
                    Path(self.ohos_root)
                    / "prebuilts"
                    / "clang"
                    / "ohos"
                    / "linux-x86_64"
                    / "llvm"
                    / "bin"
                    / "clang"
                )
                if ohos_clang.exists():
                    clang_bin = str(ohos_clang)
            except Exception:
                pass

        last_err = ""
        last_missing: List[str] = []
        last_resolved_map: Dict[str, Optional[str]] = {}
        last_added: List[str] = []

        # NOTE: keep the output path stable across retries (same entry hash).
        total_rounds = max(1, max_rounds)
        for round_idx in range(1, total_rounds + 1):
            # Rebuild flags from the selected compile_commands entry each round, so multi-candidate
            # header fallback can REPLACE a previously-chosen include dir (instead of appending a wrong one).
            effective_flags = list(base_clang_flags)
            if resolved_include_by_header:
                effective_flags = _append_include_flags(
                    effective_flags,
                    (Path(p) for p in resolved_include_by_header.values() if p),
                )
            cmd = [clang_bin, "-E", str(source_file)]
            cmd.extend(effective_flags)
            # Keep diagnostics in logs, but avoid false failures when upstream uses `-Werror`.
            if "-Wno-error" not in effective_flags:
                cmd.append("-Wno-error")
            cmd.extend(["-o", str(output_file)])

            try:
                logger.debug(f"运行预处理命令: {' '.join(cmd)}")
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout_sec
                )

                if result.returncode != 0:
                    stderr = result.stderr or ""
                    last_err = stderr
                    if not enable_auto or round_idx >= total_rounds:
                        break
                    missing = _extract_missing_headers(stderr)
                    last_missing = list(missing)
                    if not missing:
                        break
                    # Resolve each missing header by trying cached candidates first, then discovering new ones.
                    resolved_map: Dict[str, Optional[str]] = {}
                    changed: List[str] = []

                    current_includes: Set[Path] = set(include_dirs_base)
                    for d in (resolved_include_by_header or {}).values():
                        try:
                            current_includes.add(Path(str(d)))
                        except Exception:
                            continue

                    for header in missing:
                        cand_dirs: List[str] = []
                        try:
                            self._load_header_resolution_cache()
                            cached = self._header_resolution_cache.get(header) if isinstance(self._header_resolution_cache, dict) else None
                            if isinstance(cached, dict):
                                suc = cached.get("success")
                                if isinstance(suc, str) and suc.strip():
                                    cand_dirs.append(suc.strip())
                                cands = cached.get("candidates")
                                if isinstance(cands, list):
                                    cand_dirs.extend([str(x) for x in cands if isinstance(x, str) and x.strip()])
                        except Exception:
                            cand_dirs = []

                        # If we don't have candidates yet, trigger a discovery pass (will populate cache candidates).
                        if not cand_dirs:
                            try:
                                _ = self.find_header_path(
                                    header,
                                    preferred_subpaths=preferred_subpaths,
                                    preferred_include_dirs=current_includes,
                                    source_file=source_file,
                                )
                            except Exception:
                                pass
                            try:
                                self._load_header_resolution_cache()
                                cached = self._header_resolution_cache.get(header) if isinstance(self._header_resolution_cache, dict) else None
                                if isinstance(cached, dict):
                                    cands = cached.get("candidates")
                                    if isinstance(cands, list):
                                        cand_dirs.extend([str(x) for x in cands if isinstance(x, str) and x.strip()])
                            except Exception:
                                pass

                        # De-dup in order
                        seen: Set[str] = set()
                        deduped: List[str] = []
                        for d in cand_dirs:
                            s = (d or "").strip()
                            if not s or s in seen:
                                continue
                            seen.add(s)
                            deduped.append(s)
                        cand_dirs = deduped

                        selected_dir: Optional[str] = None
                        for d in cand_dirs:
                            if d in tried_include_by_header.get(header, set()):
                                continue
                            try:
                                inc_dir = Path(d).expanduser()
                            except Exception:
                                continue
                            if not (inc_dir / header).exists():
                                continue
                            tried_include_by_header.setdefault(header, set()).add(d)
                            selected_dir = str(inc_dir)
                            break

                        if selected_dir:
                            prev = resolved_include_by_header.get(header)
                            resolved_include_by_header[header] = selected_dir
                            resolved_map[header] = selected_dir
                            if prev != selected_dir:
                                changed.append(selected_dir)
                        else:
                            resolved_map[header] = None

                    last_resolved_map = resolved_map or {}
                    last_added = list(changed)
                    if not changed:
                        break
                    logger.info(
                        f"clang -E 缺头文件，已自动补充 include dirs: +{len(changed)} (round {round_idx}/{max_rounds})"
                    )
                    continue

                context.preprocessed_file = output_file

                # 解析 #line 指令，建立行号映射
                context.line_mapping = self._parse_line_directives(output_file)

                # 统计函数数量和宏数量
                context.function_count = self._count_functions_in_preprocessed(output_file)
                context.macro_count = len([flag for flag in base_clang_flags if str(flag).startswith("-D")])

                # If preprocessing succeeded with auto-resolved include dirs, cache the winning choice(s).
                if resolved_include_by_header:
                    try:
                        self._load_header_resolution_cache()
                        for header, inc in resolved_include_by_header.items():
                            if not header or not inc:
                                continue
                            ent = self._header_resolution_cache.get(header)
                            if not isinstance(ent, dict):
                                ent = {}
                                self._header_resolution_cache[header] = ent
                            ent["success"] = str(inc)
                            # Keep success at the front of candidates for future trials.
                            cands = ent.get("candidates")
                            if isinstance(cands, list):
                                merged = [str(inc), *[str(x) for x in cands if str(x) != str(inc)]]
                                ent["candidates"] = merged
                        self._save_header_resolution_cache()
                    except Exception:
                        pass

                logger.info(f"预处理成功: {output_file.name} (函数: {context.function_count}, 宏: {context.macro_count})")
                return context

            except subprocess.TimeoutExpired:
                context.error = f"预处理超时（{timeout_sec}秒）"
                logger.warning(context.error)
                return context
            except Exception as e:
                context.error = f"预处理异常: {str(e)}"
                logger.warning(context.error)
                return context

        # Failed after retries
        err_lines = (last_err or "").splitlines()
        err_head = "\n".join(err_lines[:80]) if err_lines else (last_err or "")
        err_tail = "\n".join(err_lines[-40:]) if len(err_lines) > 120 else ""
        err = err_head
        if err_tail and err_tail not in err_head:
            err = err_head + "\n...\n" + err_tail
        err = (err or "").strip()
        if len(err) > 8000:
            err = err[:8000] + "\n...(truncated)..."
        context.error = f"预处理失败: {err}" if err else "预处理失败: unknown error"
        if enable_auto and (last_missing or last_added or last_resolved_map):
            context.auto_resolve = {
                "missing_headers": last_missing,
                "added_include_dirs": last_added,
                "resolved_map": last_resolved_map,
            }

        return context

    def _parse_line_directives(self, preprocessed_file: Path) -> Dict[int, Tuple[str, int]]:
        """
        解析预处理文件中的 #line 指令，建立行号映射

        #line 指令格式:
        # 123 "/path/to/original/file.c" flags

        Args:
            preprocessed_file: 预处理后的 .i 文件

        Returns:
            字典: {.i 文件行号: (原文件路径, 原文件行号)}
        """
        line_mapping = {}

        try:
            with open(preprocessed_file, 'r', encoding='utf-8', errors='ignore') as f:
                current_original_file = None
                current_original_line = 0

                for i, line in enumerate(f, start=1):
                    # 检查是否是 #line 指令
                    # 格式: # 123 "/path/to/file.c" [flags]
                    match = re.match(r'^#\s+(\d+)\s+"([^"]+)"', line)
                    if match:
                        current_original_line = int(match.group(1))
                        current_original_file = match.group(2)
                        continue

                    # 记录映射
                    if current_original_file:
                        line_mapping[i] = (current_original_file, current_original_line)
                        current_original_line += 1

        except Exception as e:
            logger.debug(f"解析 #line 指令失败: {e}")

        return line_mapping

    def _count_functions_in_preprocessed(self, preprocessed_file: Path) -> int:
        """
        统计预处理文件中的函数定义数量（简单启发式）

        使用简单的模式匹配，避免引入 tree-sitter 依赖

        Args:
            preprocessed_file: 预处理后的 .i 文件

        Returns:
            函数定义数量（估计值）
        """
        try:
            with open(preprocessed_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # 简单启发式：匹配函数定义模式
            # 匹配形如: return_type function_name(...) {
            # 排除 typedef、struct、union、enum
            pattern = r'\b(?!typedef|struct|union|enum|if|for|while|switch)\w+\s+\w+\s*\([^)]*\)\s*\{'
            matches = re.findall(pattern, content)

            return len(matches)

        except Exception as e:
            logger.debug(f"统计函数数量失败: {e}")
            return 0

    def select_preprocessing_context(
        self,
        source_file: Path,
        strategy: ContextSelectionStrategy = ContextSelectionStrategy.ACTIVE,
        target_config: str = None,
        output_dir: Path = None
    ) -> Optional[PreprocessingContext]:
        """
        根据策略选择最佳预处理上下文

        Args:
            source_file: 源文件路径
            strategy: 上下文选择策略（默认 ACTIVE）
            target_config: 目标配置（用于 ACTIVE 策略，如 "rk3568"）
            output_dir: 预处理输出目录

        Returns:
            选择的预处理上下文，如果失败返回 None
        """
        def _pick_best(entries_to_try: List[Dict]) -> Tuple[Optional[PreprocessingContext], Optional[PreprocessingContext]]:
            best_context: Optional[PreprocessingContext] = None
            best_score = None
            last_failed: Optional[PreprocessingContext] = None
            for e in entries_to_try:
                ctx = self.preprocess_with_context(source_file, e, output_dir)
                if ctx.error:
                    last_failed = ctx
                    logger.debug(f"跳过失败的预处理: {ctx.error}")
                    continue
                score = (ctx.function_count * 10 + ctx.macro_count, ctx.function_count, ctx.macro_count)
                if best_score is None or score > best_score:
                    best_score = score
                    best_context = ctx
            return best_context, last_failed

        # 获取所有编译条目
        entries = self.get_all_entries_for_file(source_file)

        if not entries:
            # No compile_commands entry for this source file -> input closure is incomplete.
            # Default behavior: DO NOT try to "construct TU closure" via proxy-TU heuristics (brittle).
            # Users can explicitly re-enable proxy-TU fallback for legacy compatibility.
            logger.warning(f"未找到源文件的编译条目: {source_file}")
            proxy_env = os.environ.get("C2R_ENABLE_PROXY_TU_FALLBACK", "").strip().lower()
            if proxy_env:
                enable_proxy = proxy_env in ("1", "true", "yes", "on")
            else:
                truth_mode = os.environ.get("C2R_TRUTH_MODE", "0").strip().lower() in ("1", "true", "yes", "on")
                enable_proxy = not truth_mode
            if not enable_proxy:
                return PreprocessingContext(
                    source_file=Path(source_file),
                    entry={},
                    preprocessed_file=None,
                    function_count=0,
                    macro_count=0,
                    line_mapping={},
                    error="compile_commands_missing_entry (proxy_tu_disabled)",
                    proxy_used=False,
                    proxy_entry_file=None,
                    proxy_reason=None,
                )
            # Optional: Proxy-TU fallback (best-effort). This may recover macro/include visibility but is not guaranteed correct.
            proxy_target = target_config if strategy == ContextSelectionStrategy.ACTIVE else None
            try:
                max_depth = int(os.environ.get("C2R_PROXY_TU_MAX_DEPTH", "6"))
                max_files = int(os.environ.get("C2R_PROXY_TU_MAX_FILES_PER_DIR", "40"))
                max_entries = int(os.environ.get("C2R_PROXY_TU_MAX_ENTRIES", "30"))
            except Exception:
                max_depth, max_files, max_entries = 6, 40, 30
            proxy_entries = self._find_proxy_entries_for_missing_file(
                Path(source_file),
                target_config=proxy_target,
                max_depth=max_depth,
                max_files_per_dir=max_files,
                max_entries=max_entries,
            )
            if proxy_entries:
                logger.warning(
                    f"Proxy-TU: 使用附近 TU 条目({len(proxy_entries)}) 预处理缺失文件: {Path(source_file).name}"
                )
                best_ctx, last_failed = _pick_best(proxy_entries)
                if best_ctx:
                    try:
                        best_ctx.proxy_used = True
                        best_ctx.proxy_entry_file = str(best_ctx.entry.get("file") or "")
                        best_ctx.proxy_reason = "compile_commands_missing_entry"
                    except Exception:
                        pass
                    return best_ctx
                if last_failed:
                    try:
                        last_failed.proxy_used = True
                        last_failed.proxy_entry_file = str(last_failed.entry.get("file") or "")
                        last_failed.proxy_reason = "compile_commands_missing_entry"
                    except Exception:
                        pass
                    return last_failed
            return PreprocessingContext(
                source_file=Path(source_file),
                entry={},
                preprocessed_file=None,
                function_count=0,
                macro_count=0,
                line_mapping={},
                error="compile_commands_missing_entry (proxy_tu_failed)",
                proxy_used=False,
                proxy_entry_file=None,
                proxy_reason=None,
            )

        logger.info(f"找到 {len(entries)} 个编译配置，策略: {strategy.value}")

        # 如果存在“精确路径命中”的条目，优先使用这些，避免文件名碰撞导致的误选
        try:
            src_resolved = Path(source_file).resolve()
            exact_entries = []
            for e in entries:
                ep = self._resolve_entry_file_path(e)
                if ep and ep == src_resolved:
                    exact_entries.append(e)
            if exact_entries:
                entries = exact_entries
                logger.info(f"精确路径命中 {len(entries)} 个条目（已过滤文件名碰撞候选）")
        except Exception:
            pass

        # 防止极端情况下（文件名碰撞）尝试过多 entry 导致耗时爆炸
        try:
            max_entries = int(os.environ.get("C2R_CONTEXT_MAX_ENTRIES", "25"))
            if max_entries > 0 and len(entries) > max_entries:
                logger.warning(f"条目数量过多({len(entries)})，仅尝试前 {max_entries} 个（可用 C2R_CONTEXT_MAX_ENTRIES 调整）")
                entries = entries[:max_entries]
        except Exception:
            pass

        # ACTIVE 策略：用户指定配置（out_dir/board/mode），在匹配子集中做 best-of-matching
        if strategy == ContextSelectionStrategy.ACTIVE:
            if not target_config:
                logger.warning("ACTIVE 策略需要指定 target_config")
                # 回退到 BEST 策略
                strategy = ContextSelectionStrategy.BEST
            else:
                matched = []
                target_token = str(target_config).lower()
                for entry in entries:
                    directory = entry.get('directory', '') or ''
                    command = entry.get('command', '') or ''
                    arguments = entry.get('arguments', []) or []
                    if not isinstance(arguments, list):
                        arguments = []
                    haystack = " ".join([str(directory), str(command)] + [str(arg) for arg in arguments]).lower()
                    if target_token in haystack:
                        matched.append(entry)

                if matched:
                    logger.info(f"ACTIVE 匹配到 {len(matched)} 个条目: {target_config}，开始自动裁决")
                    best_ctx, last_failed = _pick_best(matched)
                    if best_ctx:
                        logger.info(
                            f"选择 ACTIVE 配置: {target_config} "
                            f"(函数={best_ctx.function_count}, 宏={best_ctx.macro_count})"
                        )
                        return best_ctx
                    if last_failed:
                        logger.warning(f"ACTIVE 匹配到的条目全部预处理失败，将返回失败上下文以便诊断: {target_config}")
                        return last_failed

                    logger.warning(f"ACTIVE 匹配到的条目全部预处理失败，回退到 BEST 策略: {target_config}")
                    strategy = ContextSelectionStrategy.BEST
                else:
                    if len(entries) == 1:
                        logger.info(
                            f"ACTIVE 未命中目标配置 {target_config}，但仅有 1 个可用条目，按唯一条目继续"
                        )
                    else:
                        logger.warning(f"未找到匹配 {target_config} 的配置，回退到 BEST 策略")
                    strategy = ContextSelectionStrategy.BEST

        # BEST 策略：选择产生最多函数/宏的配置
        if strategy == ContextSelectionStrategy.BEST:
            best_context, last_failed = _pick_best(entries)
            if best_context:
                logger.info(f"选择 BEST 配置: 函数={best_context.function_count}, 宏={best_context.macro_count}")
                return best_context

            if last_failed:
                logger.warning("所有预处理配置均失败，返回失败上下文以便诊断")
                return last_failed
            logger.warning("所有预处理配置均失败")
            return PreprocessingContext(
                source_file=Path(source_file),
                entry={},
                preprocessed_file=None,
                function_count=0,
                macro_count=0,
                line_mapping={},
                error="preprocess_all_failed (no context available)",
                proxy_used=False,
                proxy_entry_file=None,
                proxy_reason=None,
            )

        # UNION 策略：暂不实现（需要更复杂的合并逻辑和 #[cfg] 管理）
        if strategy == ContextSelectionStrategy.UNION:
            logger.warning("UNION 策略暂未实现，回退到 BEST 策略")
            return self.select_preprocessing_context(
                source_file,
                ContextSelectionStrategy.BEST,
                target_config,
                output_dir
            )

        return None


def get_includes_from_compile_commands(
    compile_db_path: Path,
    source_file: Path,
    ohos_root: Path = None
) -> List[Path]:
    """
    便捷函数：从 compile_commands.json 获取源文件的头文件搜索路径
    
    Args:
        compile_db_path: compile_commands.json 的路径
        source_file: 要查找的源文件路径
        ohos_root: OpenHarmony 源码根目录（可选）
    
    Returns:
        头文件搜索路径列表
    """
    parser = CompileCommandsParser(compile_db_path, ohos_root)
    return parser.get_includes_for_file(source_file)


if __name__ == '__main__':
    # 测试代码
    import sys
    
    if len(sys.argv) < 3:
        print("用法: python3 compile_commands_parser.py <compile_commands.json> <source_file>")
        sys.exit(1)
    
    compile_db_path = Path(sys.argv[1])
    source_file = Path(sys.argv[2])
    
    parser = CompileCommandsParser(compile_db_path)
    includes = parser.get_includes_for_file(source_file)
    
    print(f"文件: {source_file}")
    print(f"找到 {len(includes)} 个 include 路径:")
    for inc in includes[:20]:  # 显示前20个
        print(f"  - {inc}")
