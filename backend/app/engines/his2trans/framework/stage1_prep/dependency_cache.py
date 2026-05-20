#!/usr/bin/env python3
"""
依赖缓存模块 (Dependency Cache)

缓存已成功匹配的依赖信息，避免重复查找。

缓存内容：
- 头文件路径映射
- bindgen 成功的配置
- include 路径
- 已解析的 compile_commands.json 信息
"""

import json
import hashlib
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger(__name__)

# 缓存文件路径
# - 默认：repo 根目录下的 .cache/
# - 支持通过 C2R_CACHE_ROOT 将缓存放到某次运行目录（实验隔离更方便）
_cache_root_env = os.environ.get("C2R_CACHE_ROOT", "").strip()
if _cache_root_env:
    CACHE_DIR = Path(_cache_root_env).expanduser().resolve()
else:
    CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"
DEPENDENCY_CACHE_FILE = CACHE_DIR / "dependency_cache.json"


@dataclass
class ProjectCache:
    """单个项目的缓存数据"""
    project_name: str
    project_root: str
    source_files_hash: str  # 源文件内容的哈希，用于检测变化
    
    # 头文件映射：{源文件名: 头文件路径}
    header_mappings: Dict[str, str]
    
    # include 路径列表
    include_dirs: List[str]
    
    # bindgen 成功的配置
    bindgen_success: bool
    bindgen_args: List[str]
    
    # 函数签名缓存：{源文件名: [签名列表]}
    function_signatures: Dict[str, List[Dict]]
    
    # 缓存时间
    cached_at: str
    
    # 缓存版本（用于兼容性检查）
    cache_version: str = "1.0"


class DependencyCache:
    """
    依赖缓存管理器
    
    用法：
        cache = DependencyCache()
        
        # 检查缓存
        if cache.is_valid(project_name, source_files):
            data = cache.get(project_name)
            # 使用缓存数据
        else:
            # 执行依赖分析
            data = analyze_dependencies(...)
            # 保存缓存
            cache.set(project_name, data)
    """
    
    def __init__(self, cache_dir: Path = None):
        self.cache_dir = cache_dir or CACHE_DIR
        self.cache_file = self.cache_dir / "dependency_cache.json"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, ProjectCache] = {}
        self._load()
    
    def _load(self):
        """加载缓存文件"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for name, proj_data in data.get('projects', {}).items():
                    self._cache[name] = ProjectCache(**proj_data)
                
                logger.info(f"已加载 {len(self._cache)} 个项目的缓存")
            except Exception as e:
                logger.warning(f"加载缓存失败: {e}")
                self._cache = {}
    
    def _save(self):
        """保存缓存文件"""
        try:
            data = {
                'version': '1.0',
                'updated_at': datetime.now().isoformat(),
                'projects': {
                    name: asdict(proj) for name, proj in self._cache.items()
                }
            }
            
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"缓存已保存: {len(self._cache)} 个项目")
        except Exception as e:
            logger.warning(f"保存缓存失败: {e}")
    
    def _compute_hash(self, source_files: List[Path]) -> str:
        """计算源文件列表的哈希值"""
        hasher = hashlib.md5()
        
        for src_file in sorted(source_files):
            if src_file.exists():
                # 使用文件路径和修改时间计算哈希（比内容哈希快）
                stat = src_file.stat()
                hasher.update(f"{src_file}:{stat.st_mtime}:{stat.st_size}".encode())
        
        return hasher.hexdigest()
    
    def is_valid(self, project_name: str, source_files: List[Path]) -> bool:
        """
        检查缓存是否有效
        
        Args:
            project_name: 项目名称
            source_files: 源文件列表
        
        Returns:
            缓存是否有效
        """
        if project_name not in self._cache:
            return False
        
        cached = self._cache[project_name]
        
        # 检查版本
        if cached.cache_version != "1.0":
            return False
        
        # 检查源文件是否变化
        current_hash = self._compute_hash(source_files)
        if current_hash != cached.source_files_hash:
            logger.debug(f"项目 {project_name} 的源文件已变化，缓存无效")
            return False
        
        return True
    
    def get(self, project_name: str) -> Optional[ProjectCache]:
        """获取项目缓存"""
        return self._cache.get(project_name)
    
    def set(
        self,
        project_name: str,
        project_root: Path,
        source_files: List[Path],
        header_mappings: Dict[str, str] = None,
        include_dirs: List[str] = None,
        bindgen_success: bool = False,
        bindgen_args: List[str] = None,
        function_signatures: Dict[str, List[Dict]] = None
    ):
        """
        设置项目缓存
        
        Args:
            project_name: 项目名称
            project_root: 项目根目录
            source_files: 源文件列表
            header_mappings: 头文件映射
            include_dirs: include 路径
            bindgen_success: bindgen 是否成功
            bindgen_args: bindgen 参数
            function_signatures: 函数签名缓存
        """
        cache_entry = ProjectCache(
            project_name=project_name,
            project_root=str(project_root),
            source_files_hash=self._compute_hash(source_files),
            header_mappings=header_mappings or {},
            include_dirs=include_dirs or [],
            bindgen_success=bindgen_success,
            bindgen_args=bindgen_args or [],
            function_signatures=function_signatures or {},
            cached_at=datetime.now().isoformat()
        )
        
        self._cache[project_name] = cache_entry
        self._save()
        
        logger.info(f"已缓存项目 {project_name} 的依赖信息")
    
    def invalidate(self, project_name: str):
        """使项目缓存失效"""
        if project_name in self._cache:
            del self._cache[project_name]
            self._save()
            logger.info(f"已清除项目 {project_name} 的缓存")
    
    def clear_all(self):
        """清除所有缓存"""
        self._cache = {}
        if self.cache_file.exists():
            self.cache_file.unlink()
        logger.info("已清除所有缓存")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        return {
            'total_projects': len(self._cache),
            'cache_file': str(self.cache_file),
            'projects': list(self._cache.keys())
        }


# 全局缓存实例
_global_cache: Optional[DependencyCache] = None


def get_dependency_cache() -> DependencyCache:
    """获取全局缓存实例"""
    global _global_cache
    if _global_cache is None:
        _global_cache = DependencyCache()
    return _global_cache


# =========================================================================
# 便捷函数
# =========================================================================

def cache_header_mapping(project_name: str, src_file: str, header_file: str):
    """缓存单个头文件映射"""
    cache = get_dependency_cache()
    proj = cache.get(project_name)
    if proj:
        proj.header_mappings[src_file] = header_file
        cache._save()


def get_cached_header(project_name: str, src_file: str) -> Optional[str]:
    """获取缓存的头文件路径"""
    cache = get_dependency_cache()
    proj = cache.get(project_name)
    if proj:
        return proj.header_mappings.get(src_file)
    return None


def cache_include_dirs(project_name: str, include_dirs: List[str]):
    """缓存 include 路径"""
    cache = get_dependency_cache()
    proj = cache.get(project_name)
    if proj:
        proj.include_dirs = include_dirs
        cache._save()


def get_cached_include_dirs(project_name: str) -> List[str]:
    """获取缓存的 include 路径"""
    cache = get_dependency_cache()
    proj = cache.get(project_name)
    if proj:
        return proj.include_dirs
    return []


# =========================================================================
# 命令行接口
# =========================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="依赖缓存管理")
    parser.add_argument('--stats', action='store_true', help='显示缓存统计')
    parser.add_argument('--clear', action='store_true', help='清除所有缓存')
    parser.add_argument('--invalidate', type=str, help='使指定项目缓存失效')
    
    args = parser.parse_args()
    
    cache = get_dependency_cache()
    
    if args.stats:
        stats = cache.get_stats()
        print(f"缓存统计:")
        print(f"  项目数: {stats['total_projects']}")
        print(f"  缓存文件: {stats['cache_file']}")
        if stats['projects']:
            print(f"  已缓存项目: {', '.join(stats['projects'])}")
    elif args.clear:
        cache.clear_all()
        print("已清除所有缓存")
    elif args.invalidate:
        cache.invalidate(args.invalidate)
        print(f"已清除项目 {args.invalidate} 的缓存")
    else:
        parser.print_help()




























