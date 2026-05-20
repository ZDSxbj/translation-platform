#!/usr/bin/env python3
"""
混合 C/Rust 构建支持 (Hybrid C/Rust Build Support)

实现渐进式重写架构：
- Rust 调用现有的 C/C++ 库
- 只翻译用户关心的那个 C 文件
- 其他依赖通过 FFI 链接

核心组件：
- build.rs 自动生成器
- native/ 目录管理
- compile_commands.json 参数提取
"""

import os
import re
import json
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# =========================================================================
# 数据结构
# =========================================================================

@dataclass
class CSourceFile:
    """C 源文件信息"""
    path: Path
    include_dirs: List[str] = field(default_factory=list)
    defines: List[str] = field(default_factory=list)
    compiler_flags: List[str] = field(default_factory=list)
    is_target: bool = False  # 是否是要翻译的目标文件


@dataclass
class HybridBuildConfig:
    """混合构建配置"""
    project_name: str
    output_dir: Path
    target_files: List[Path]  # 要翻译成 Rust 的文件
    dependency_files: List[Path]  # 保留为 C 的依赖文件
    include_dirs: List[Path]
    defines: Dict[str, str]
    compiler_flags: List[str]


# =========================================================================
# Cargo.toml 生成器
# =========================================================================

def generate_cargo_toml(
    project_name: str,
    output_dir: Path,
    use_cc: bool = True,
    use_bindgen: bool = True,
    extra_deps: Dict[str, str] = None
) -> str:
    """
    生成支持混合编程的 Cargo.toml
    
    Args:
        project_name: 项目名称
        output_dir: 输出目录
        use_cc: 是否使用 cc crate 编译 C 代码
        use_bindgen: 是否使用 bindgen 生成绑定
        extra_deps: 额外的依赖
    
    Returns:
        生成的 Cargo.toml 内容
    """
    # 基本依赖
    deps = {
        "libc": '"0.2"',
    }
    
    # 构建依赖
    build_deps = {}
    if use_cc:
        build_deps["cc"] = '"1.0"'
    if use_bindgen:
        build_deps["bindgen"] = '"0.69"'
    
    # 合并额外依赖
    if extra_deps:
        deps.update(extra_deps)
    
    # 生成依赖字符串
    deps_str = "\n".join(f'{k} = {v}' for k, v in deps.items())
    build_deps_str = "\n".join(f'{k} = {v}' for k, v in build_deps.items())
    
    content = f'''[package]
name = "{project_name}"
version = "0.1.0"
edition = "2021"
build = "build.rs"

[lib]
crate-type = ["staticlib", "rlib"]

[dependencies]
{deps_str}

[build-dependencies]
{build_deps_str}

# 允许不安全代码（FFI 需要）
[lints.rust]
unsafe_code = "allow"

# 性能优化
[profile.release]
opt-level = 3
lto = true
'''
    
    cargo_path = output_dir / "Cargo.toml"
    with open(cargo_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    logger.info(f"生成 Cargo.toml: {cargo_path}")
    return content


# =========================================================================
# build.rs 生成器
# =========================================================================

def generate_build_rs(
    output_dir: Path,
    c_source_files: List[CSourceFile],
    include_dirs: List[Path],
    defines: Dict[str, str] = None,
    compiler_flags: List[str] = None,
    lib_name: str = "native_deps"
) -> str:
    """
    生成 build.rs 构建脚本
    
    Args:
        output_dir: 输出目录
        c_source_files: C 源文件列表
        include_dirs: include 路径列表
        defines: 宏定义
        compiler_flags: 编译器标志
        lib_name: 生成的库名称
    
    Returns:
        生成的 build.rs 内容
    """
    defines = defines or {}
    compiler_flags = compiler_flags or []
    
    # 生成 include 路径代码
    includes_code = []
    for inc_dir in include_dirs:
        includes_code.append(f'        .include("{inc_dir}")')
    includes_str = "\n".join(includes_code) if includes_code else ''

    # 生成宏定义代码
    defines_code = []
    for name, value in defines.items():
        if value:
            defines_code.append(f'        .define("{name}", Some("{value}"))')
        else:
            defines_code.append(f'        .define("{name}", None)')
    defines_str = "\n".join(defines_code) if defines_code else ''

    # 生成编译器标志代码
    flags_code = []
    for flag in compiler_flags:
        flags_code.append(f'        .flag("{flag}")')
    flags_str = "\n".join(flags_code) if flags_code else ''

    # Build config method chain tail
    chain_parts = []
    if includes_str:
        chain_parts.append(includes_str)
    if defines_str:
        chain_parts.append(defines_str)
    if flags_str:
        chain_parts.append(flags_str)
    chain_parts.append('        .warnings(false)')
    chain_parts.append('        .extra_warnings(false)')
    chain_parts.append('        .flag_if_supported("-w")')
    chain_parts.append('        .opt_level(2);')
    chain_str = "\n".join(chain_parts)

    content = f'''//! 自动生成的构建脚本
//!
//! 此脚本在 `cargo build` 之前运行，负责：
//! 1. 编译依赖的 C 代码为静态库
//! 2. 配置链接参数

fn main() {{
    // 告诉 Cargo 当 native/ 目录变化时重新构建
    println!("cargo:rerun-if-changed=native/");
    println!("cargo:rerun-if-changed=build.rs");

    // 检查 native/ 是否存在
    let native_dir = std::path::Path::new("native");
    if !native_dir.exists() {{
        println!("cargo:warning=没有找到 native/ 目录，跳过 C 代码编译");
        return;
    }}

    // 编译 C 代码
    let mut build = cc::Build::new();

    // Auto-discover C/C++ sources under native/
    let mut any_native = false;
    if let Ok(entries) = std::fs::read_dir(native_dir) {{
        for entry in entries.flatten() {{
            let path = entry.path();
            if !path.is_file() {{
                continue;
            }}
            let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
            // cc crate selects compiler based on extension
            if matches!(ext, "c" | "cc" | "cpp" | "cxx" | "C") {{
                any_native = true;
                build.file(path);
            }}
        }}
    }}
    if !any_native {{
        println!("cargo:warning=native/ 目录没有可编译的 C/C++ 源文件，跳过 C 代码编译");
        return;
    }}

    build
{chain_str}

    // 编译为静态库
    build.compile("{lib_name}");

    // 链接库
    println!("cargo:rustc-link-lib=static={lib_name}");

    // 可选：链接系统库（如果 C 代码依赖它们）
    // println!("cargo:rustc-link-lib=pthread");
    // println!("cargo:rustc-link-lib=m");
}}
'''

    build_rs_path = output_dir / "build.rs"
    with open(build_rs_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    logger.info(f"生成 build.rs: {build_rs_path}")
    return content


# =========================================================================
# Native 目录管理器
# =========================================================================

class NativeDirectoryManager:
    """
    管理 native/ 目录，存放原始 C 源码
    """
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.native_dir = output_dir / "native"
        self.include_dir = self.native_dir / "include"
    
    def setup(self):
        """创建目录结构"""
        self.native_dir.mkdir(parents=True, exist_ok=True)
        self.include_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"创建 native 目录: {self.native_dir}")
    
    def copy_source_file(self, source_path: Path, is_header: bool = False) -> Path:
        """
        复制源文件到 native/ 目录
        
        Args:
            source_path: 源文件路径
            is_header: 是否是头文件
        
        Returns:
            复制后的路径
        """
        if is_header:
            dest = self.include_dir / source_path.name
        else:
            dest = self.native_dir / source_path.name
        
        if source_path.exists():
            shutil.copy2(source_path, dest)
            logger.debug(f"复制文件: {source_path} -> {dest}")
        
        return dest
    
    def copy_dependency_files(
        self, 
        target_file: Path,
        dependency_files: List[Path],
        header_files: List[Path] = None
    ) -> Tuple[List[Path], List[Path]]:
        """
        复制依赖文件到 native/ 目录
        
        Args:
            target_file: 目标文件（要翻译的文件，不复制）
            dependency_files: 依赖的 C 源文件
            header_files: 头文件
        
        Returns:
            (复制的源文件路径列表, 复制的头文件路径列表)
        """
        self.setup()
        
        copied_sources = []
        copied_headers = []
        
        # 复制依赖源文件（排除目标文件）
        target_name = target_file.name if target_file else None
        for src in dependency_files:
            if src.name != target_name:
                dest = self.copy_source_file(src, is_header=False)
                copied_sources.append(dest)
        
        # 复制头文件
        if header_files:
            for h in header_files:
                dest = self.copy_source_file(h, is_header=True)
                copied_headers.append(dest)
        
        logger.info(f"复制了 {len(copied_sources)} 个源文件和 {len(copied_headers)} 个头文件到 native/")
        return copied_sources, copied_headers


# =========================================================================
# 外部符号声明生成器
# =========================================================================

def generate_extern_declarations(
    external_functions: List[Dict],
    external_variables: List[Dict],
    output_file: Path
) -> str:
    """
    生成 extern "C" 声明
    
    Args:
        external_functions: 外部函数列表 [{name, params, return_type}, ...]
        external_variables: 外部变量列表 [{name, type, is_mut}, ...]
        output_file: 输出文件路径
    
    Returns:
        生成的代码
    """
    lines = [
        "//! 外部 C 函数和变量声明",
        "//!",
        "//! 这些声明指向链接进来的 C 静态库中的符号。",
        "//! 由 hybrid_build.py 自动生成。",
        "",
        "#![allow(non_camel_case_types)]",
        "#![allow(non_snake_case)]",
        "#![allow(dead_code)]",
        "",
        "use std::os::raw::*;",
        "",
    ]
    
    # 外部函数声明
    if external_functions:
        lines.append("// ============================================================")
        lines.append("// 外部 C 函数")
        lines.append("// ============================================================")
        lines.append("")
        lines.append('extern "C" {')
        
        for func in external_functions:
            name = func.get("name", "unknown")
            params = func.get("params", [])
            return_type = func.get("return_type", "")
            
            params_str = ", ".join(f"{p['name']}: {p['type']}" for p in params)
            ret_str = f" -> {return_type}" if return_type and return_type != "void" else ""
            
            lines.append(f"    pub fn {name}({params_str}){ret_str};")
        
        lines.append("}")
        lines.append("")
    
    # 外部变量声明
    if external_variables:
        lines.append("// ============================================================")
        lines.append("// 外部 C 变量")
        lines.append("// ============================================================")
        lines.append("")
        lines.append('extern "C" {')
        
        for var in external_variables:
            name = var.get("name", "unknown")
            var_type = var.get("type", "c_int")
            is_mut = var.get("is_mut", True)
            
            mut_kw = "mut " if is_mut else ""
            lines.append(f"    pub static {mut_kw}{name}: {var_type};")
        
        lines.append("}")
        lines.append("")
    
    content = "\n".join(lines)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    logger.info(f"生成外部声明: {output_file}")
    return content


# =========================================================================
# 从 compile_commands.json 提取编译参数
# =========================================================================

def extract_compile_flags_from_commands(
    compile_commands_path: Path,
    source_files: List[Path]
) -> Tuple[List[Path], Dict[str, str], List[str]]:
    """
    从 compile_commands.json 提取编译参数
    
    Args:
        compile_commands_path: compile_commands.json 路径
        source_files: 源文件列表
    
    Returns:
        (include_dirs, defines, flags)
    """
    include_dirs = set()
    defines = {}
    flags = set()
    
    if not compile_commands_path.exists():
        logger.warning(f"compile_commands.json 不存在: {compile_commands_path}")
        return list(include_dirs), defines, list(flags)
    
    try:
        with open(compile_commands_path, 'r', encoding='utf-8') as f:
            commands = json.load(f)
    except Exception as e:
        logger.error(f"加载 compile_commands.json 失败: {e}")
        return list(include_dirs), defines, list(flags)
    
    # 创建文件名到路径的映射
    source_names = {src.name: src for src in source_files}
    
    for entry in commands:
        file_path = Path(entry.get("file", ""))
        
        # 检查是否是我们关心的文件
        if file_path.name not in source_names:
            continue
        
        # 提取编译参数
        command = entry.get("command", "") or " ".join(entry.get("arguments", []))
        
        # 提取 -I (include paths)
        for match in re.finditer(r'-I\s*([^\s]+)', command):
            inc_path = match.group(1)
            if inc_path:
                include_dirs.add(Path(inc_path))
        
        # 提取 -D (defines)
        for match in re.finditer(r'-D\s*([^=\s]+)(?:=([^\s]*))?', command):
            name = match.group(1)
            value = match.group(2) or ""
            defines[name] = value
        
        # 提取其他有用的标志
        for match in re.finditer(r'(-f[^\s]+|-W[^\s]+|-O\d)', command):
            flag = match.group(1)
            # 过滤掉一些可能有问题的标志
            if not any(x in flag for x in ['-Werror', '-pedantic']):
                flags.add(flag)
    
    logger.info(f"从 compile_commands.json 提取: {len(include_dirs)} 个 include 路径, "
                f"{len(defines)} 个宏定义, {len(flags)} 个编译标志")
    
    return list(include_dirs), defines, list(flags)


# =========================================================================
# 混合构建管理器
# =========================================================================

class HybridBuildManager:
    """
    混合 C/Rust 构建管理器
    
    整合所有组件，提供统一的接口
    """
    
    def __init__(
        self,
        project_name: str,
        output_dir: Path,
        compile_commands_path: Path = None
    ):
        self.project_name = project_name
        self.output_dir = output_dir
        self.compile_commands_path = compile_commands_path
        
        self.native_manager = NativeDirectoryManager(output_dir)
        
        self.target_files: List[Path] = []
        self.dependency_files: List[Path] = []
        self.header_files: List[Path] = []
        self.include_dirs: List[Path] = []
        self.defines: Dict[str, str] = {}
        self.compiler_flags: List[str] = []
    
    def set_target_files(self, files: List[Path]):
        """设置要翻译的目标文件"""
        self.target_files = files
    
    def set_dependency_files(self, files: List[Path]):
        """设置依赖的 C 文件（不翻译，保留为 C）"""
        self.dependency_files = files
    
    def set_header_files(self, files: List[Path]):
        """设置头文件"""
        self.header_files = files
    
    def extract_compile_flags(self):
        """从 compile_commands.json 提取编译参数"""
        if self.compile_commands_path:
            all_files = self.target_files + self.dependency_files
            self.include_dirs, self.defines, self.compiler_flags = \
                extract_compile_flags_from_commands(self.compile_commands_path, all_files)
    
    def setup_project(self) -> bool:
        """
        设置混合构建项目
        
        Returns:
            是否成功
        """
        try:
            # 1. 创建目录结构
            (self.output_dir / "src").mkdir(parents=True, exist_ok=True)
            
            # 2. 复制依赖文件到 native/
            target_file = self.target_files[0] if self.target_files else None
            self.native_manager.copy_dependency_files(
                target_file,
                self.dependency_files,
                self.header_files
            )
            
            # 3. 提取编译参数
            self.extract_compile_flags()
            
            # 4. 添加 native/include 到 include 路径
            native_include = self.output_dir / "native" / "include"
            if native_include.exists():
                self.include_dirs.insert(0, native_include)
            
            # 5. 生成 Cargo.toml
            generate_cargo_toml(
                self.project_name,
                self.output_dir,
                use_cc=len(self.dependency_files) > 0,
                use_bindgen=True
            )
            
            # 6. 生成 build.rs
            c_sources = [
                CSourceFile(
                    path=f,
                    is_target=(f in self.target_files)
                )
                for f in self.dependency_files
            ]
            
            generate_build_rs(
                self.output_dir,
                c_sources,
                self.include_dirs,
                self.defines,
                self.compiler_flags,
                lib_name=f"{self.project_name}_native"
            )
            
            logger.info(f"混合构建项目设置完成: {self.output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"设置混合构建项目失败: {e}")
            return False
    
    def get_extern_functions_for_file(self, source_file: Path) -> List[Dict]:
        """
        获取文件调用的外部函数
        
        Args:
            source_file: 源文件
        
        Returns:
            外部函数列表
        """
        # TODO: 使用 tree-sitter 分析源文件，提取调用的外部函数
        # 这里返回空列表作为占位
        return []


# =========================================================================
# 便捷函数
# =========================================================================

def setup_hybrid_project(
    project_name: str,
    output_dir: Path,
    target_files: List[Path],
    dependency_files: List[Path],
    header_files: List[Path] = None,
    compile_commands_path: Path = None
) -> bool:
    """
    一键设置混合 C/Rust 项目
    
    Args:
        project_name: 项目名称
        output_dir: 输出目录
        target_files: 要翻译的 C 文件
        dependency_files: 依赖的 C 文件（保留为 C）
        header_files: 头文件
        compile_commands_path: compile_commands.json 路径
    
    Returns:
        是否成功
    """
    manager = HybridBuildManager(
        project_name,
        output_dir,
        compile_commands_path
    )
    
    manager.set_target_files(target_files)
    manager.set_dependency_files(dependency_files)
    manager.set_header_files(header_files or [])
    
    return manager.setup_project()


# =========================================================================
# 测试
# =========================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 测试生成 Cargo.toml
    print("=== 测试 Cargo.toml 生成 ===")
    content = generate_cargo_toml("test_project", Path("/tmp/test"), use_cc=True, use_bindgen=True)
    print(content[:500])
    
    # 测试生成 build.rs
    print("\n=== 测试 build.rs 生成 ===")
    c_sources = [
        CSourceFile(path=Path("helper.c"), is_target=False),
        CSourceFile(path=Path("utils.c"), is_target=False),
    ]
    content = generate_build_rs(
        Path("/tmp/test"),
        c_sources,
        [Path("/usr/include"), Path("/usr/local/include")],
        {"DEBUG": "1", "VERSION": "1.0"},
        ["-O2", "-fPIC"],
        "my_native"
    )
    print(content[:800])





























