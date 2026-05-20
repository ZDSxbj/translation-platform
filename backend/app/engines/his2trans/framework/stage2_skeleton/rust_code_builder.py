#!/usr/bin/env python3
"""
Rust 代码生成器 (RustCodeBuilder)

使用 Builder Pattern 安全地生成 Rust 代码，避免 f-string 拼接导致的语法错误：
- 自动处理缩进
- 自动添加分号和逗号
- 自动匹配大括号
- 保证生成的代码语法正确

使用示例:
    builder = RustCodeBuilder()
    builder.add_struct("Point", repr_c=True)
    builder.add_field("x", "i32")
    builder.add_field("y", "i32")
    builder.end_block()
    print(builder.build())
"""

from typing import List, Optional
from contextlib import contextmanager


class RustCodeBuilder:
    """
    安全的 Rust 代码生成器
    
    特性:
    - 自动缩进管理
    - 自动添加必要的语法元素
    - 支持链式调用
    - 上下文管理器支持自动闭合块
    """
    
    def __init__(self):
        self.code_blocks: List[str] = []
        self.indent_level: int = 0
        self._block_stack: List[str] = []  # 跟踪未闭合的块类型
    
    def _indent(self) -> str:
        """获取当前缩进字符串"""
        return "    " * self.indent_level
    
    def add_line(self, line: str = "") -> 'RustCodeBuilder':
        """
        添加一行代码，自动处理缩进
        
        Args:
            line: 要添加的代码行（不含缩进）
        
        Returns:
            self，支持链式调用
        """
        if line:
            self.code_blocks.append(f"{self._indent()}{line}")
        else:
            self.code_blocks.append("")
        return self
    
    def add_raw(self, code: str) -> 'RustCodeBuilder':
        """
        添加原始代码（不处理缩进，用于插入已格式化的代码块）
        
        Args:
            code: 原始代码字符串
        
        Returns:
            self
        """
        self.code_blocks.append(code)
        return self
    
    def add_comment(self, text: str, doc: bool = True) -> 'RustCodeBuilder':
        """
        添加注释
        
        Args:
            text: 注释内容
            doc: True 使用 /// (doc comment)，False 使用 // (line comment)
        
        Returns:
            self
        """
        prefix = "///" if doc else "//"
        self.add_line(f"{prefix} {text}")
        return self
    
    def add_multiline_comment(self, lines: List[str], doc: bool = True) -> 'RustCodeBuilder':
        """添加多行注释"""
        for line in lines:
            self.add_comment(line, doc)
        return self
    
    def add_attribute(self, attr: str) -> 'RustCodeBuilder':
        """
        添加属性
        
        Args:
            attr: 属性内容（不含 #[]）
        
        Returns:
            self
        """
        self.add_line(f"#[{attr}]")
        return self
    
    def add_allow(self, *warnings: str) -> 'RustCodeBuilder':
        """
        添加 #![allow(...)] 或 #[allow(...)]
        
        Args:
            *warnings: 要允许的警告列表
        
        Returns:
            self
        """
        warnings_str = ", ".join(warnings)
        self.add_line(f"#![allow({warnings_str})]")
        return self
    
    # =========================================================================
    # 结构体相关
    # =========================================================================
    
    def add_struct(
        self, 
        name: str, 
        is_pub: bool = True, 
        repr_c: bool = True,
        derives: Optional[List[str]] = None,
        doc: Optional[str] = None
    ) -> 'RustCodeBuilder':
        """
        开始定义一个结构体
        
        Args:
            name: 结构体名称
            is_pub: 是否公开
            repr_c: 是否添加 #[repr(C)]
            derives: 要派生的 trait 列表，默认 ["Debug", "Copy", "Clone"]
            doc: 文档注释
        
        Returns:
            self
        """
        if doc:
            self.add_comment(doc)
        
        if repr_c:
            self.add_attribute("repr(C)")
        
        if derives is None:
            derives = ["Debug", "Copy", "Clone"]
        if derives:
            derives_str = ", ".join(derives)
            self.add_attribute(f"derive({derives_str})")
        
        vis = "pub " if is_pub else ""
        self.add_line(f"{vis}struct {name} {{")
        self.indent_level += 1
        self._block_stack.append("struct")
        return self
    
    def add_field(
        self, 
        name: str, 
        type_name: str, 
        is_pub: bool = True,
        doc: Optional[str] = None
    ) -> 'RustCodeBuilder':
        """
        添加结构体字段
        
        Args:
            name: 字段名
            type_name: 类型名
            is_pub: 是否公开
            doc: 文档注释
        
        Returns:
            self
        """
        if doc:
            self.add_comment(doc)
        vis = "pub " if is_pub else ""
        self.add_line(f"{vis}{name}: {type_name},")
        return self
    
    def add_opaque_struct(
        self, 
        name: str, 
        doc: Optional[str] = None
    ) -> 'RustCodeBuilder':
        """
        添加不透明结构体（零大小占位）
        
        Args:
            name: 结构体名称
            doc: 文档注释
        
        Returns:
            self
        """
        if doc:
            self.add_comment(doc)
        else:
            self.add_comment(f"Opaque placeholder for external type `{name}`")
        
        self.add_struct(name, is_pub=True, repr_c=True)
        self.add_field("_private", "[u8; 0]", is_pub=False)
        self.end_block()
        return self
    
    # =========================================================================
    # 枚举相关
    # =========================================================================
    
    def add_enum(
        self, 
        name: str, 
        is_pub: bool = True,
        repr_c: bool = True,
        repr_type: Optional[str] = None,
        derives: Optional[List[str]] = None,
        doc: Optional[str] = None
    ) -> 'RustCodeBuilder':
        """
        开始定义一个枚举
        
        Args:
            name: 枚举名称
            is_pub: 是否公开
            repr_c: 是否添加 #[repr(C)]
            repr_type: repr 类型，如 "u32"
            derives: 要派生的 trait 列表
            doc: 文档注释
        
        Returns:
            self
        """
        if doc:
            self.add_comment(doc)
        
        if repr_type:
            self.add_attribute(f"repr({repr_type})")
        elif repr_c:
            self.add_attribute("repr(C)")
        
        if derives is None:
            derives = ["Debug", "Copy, Clone", "PartialEq", "Eq"]
        if derives:
            derives_str = ", ".join(derives)
            self.add_attribute(f"derive({derives_str})")
        
        vis = "pub " if is_pub else ""
        self.add_line(f"{vis}enum {name} {{")
        self.indent_level += 1
        self._block_stack.append("enum")
        return self
    
    def add_variant(
        self, 
        name: str, 
        value: Optional[int] = None,
        doc: Optional[str] = None
    ) -> 'RustCodeBuilder':
        """
        添加枚举变体
        
        Args:
            name: 变体名称
            value: 可选的显式值
            doc: 文档注释
        
        Returns:
            self
        """
        if doc:
            self.add_comment(doc)
        
        if value is not None:
            self.add_line(f"{name} = {value},")
        else:
            self.add_line(f"{name},")
        return self
    
    # =========================================================================
    # 类型别名
    # =========================================================================
    
    def add_type_alias(
        self, 
        name: str, 
        target: str, 
        is_pub: bool = True,
        doc: Optional[str] = None
    ) -> 'RustCodeBuilder':
        """
        添加类型别名
        
        Args:
            name: 新类型名
            target: 目标类型
            is_pub: 是否公开
            doc: 文档注释
        
        Returns:
            self
        """
        if doc:
            self.add_comment(doc)
        vis = "pub " if is_pub else ""
        self.add_line(f"{vis}type {name} = {target};")
        self.add_line("")
        return self
    
    # =========================================================================
    # 常量
    # =========================================================================
    
    def add_const(
        self, 
        name: str, 
        type_name: str, 
        value: str,
        is_pub: bool = True,
        doc: Optional[str] = None
    ) -> 'RustCodeBuilder':
        """
        添加常量定义
        
        Args:
            name: 常量名
            type_name: 类型名
            value: 值
            is_pub: 是否公开
            doc: 文档注释
        
        Returns:
            self
        """
        if doc:
            self.add_comment(doc)
        vis = "pub " if is_pub else ""
        self.add_line(f"{vis}const {name}: {type_name} = {value};")
        return self
    
    def add_static(
        self, 
        name: str, 
        type_name: str, 
        value: str,
        is_pub: bool = True,
        is_mut: bool = False,
        doc: Optional[str] = None
    ) -> 'RustCodeBuilder':
        """
        添加静态变量定义
        
        Args:
            name: 变量名
            type_name: 类型名
            value: 初始值
            is_pub: 是否公开
            is_mut: 是否可变
            doc: 文档注释
        
        Returns:
            self
        """
        if doc:
            self.add_comment(doc)
        vis = "pub " if is_pub else ""
        mut = "mut " if is_mut else ""
        self.add_line(f"{vis}static {mut}{name}: {type_name} = {value};")
        return self
    
    # =========================================================================
    # 函数相关
    # =========================================================================
    
    def add_function(
        self, 
        name: str,
        params: List[tuple],  # [(name, type), ...]
        return_type: Optional[str] = None,
        is_pub: bool = True,
        is_unsafe: bool = False,
        extern_c: bool = True,
        doc: Optional[str] = None
    ) -> 'RustCodeBuilder':
        """
        开始定义一个函数
        
        Args:
            name: 函数名
            params: 参数列表 [(name, type), ...]
            return_type: 返回类型，None 表示无返回值
            is_pub: 是否公开
            is_unsafe: 是否 unsafe
            extern_c: 是否 extern "C"
            doc: 文档注释
        
        Returns:
            self
        """
        if doc:
            self.add_comment(doc)
        
        # 构建签名
        vis = "pub " if is_pub else ""
        unsafe_kw = "unsafe " if is_unsafe else ""
        extern_kw = 'extern "C" ' if extern_c else ""
        
        params_str = ", ".join(f"{p[0]}: {p[1]}" for p in params)
        ret_str = f" -> {return_type}" if return_type else ""
        
        self.add_line(f"{vis}{unsafe_kw}{extern_kw}fn {name}({params_str}){ret_str} {{")
        self.indent_level += 1
        self._block_stack.append("function")
        return self
    
    def add_function_stub(
        self, 
        name: str,
        params: List[tuple],
        return_type: Optional[str] = None,
        is_pub: bool = True,
        extern_c: bool = True,
        doc: Optional[str] = None
    ) -> 'RustCodeBuilder':
        """
        添加函数桩（body 为 unimplemented!()）
        
        Args:
            name: 函数名
            params: 参数列表
            return_type: 返回类型
            is_pub: 是否公开
            extern_c: 是否 extern "C"
            doc: 文档注释
        
        Returns:
            self
        """
        self.add_function(name, params, return_type, is_pub, False, extern_c, doc)
        self.add_line("unimplemented!()")
        self.end_block()
        return self
    
    def add_extern_c_block(self) -> 'RustCodeBuilder':
        """开始 extern "C" 块"""
        self.add_line('extern "C" {')
        self.indent_level += 1
        self._block_stack.append("extern")
        return self
    
    def add_extern_fn(
        self, 
        name: str,
        params: List[tuple],
        return_type: Optional[str] = None,
        doc: Optional[str] = None
    ) -> 'RustCodeBuilder':
        """
        在 extern "C" 块中添加函数声明
        
        Args:
            name: 函数名
            params: 参数列表
            return_type: 返回类型
            doc: 文档注释
        
        Returns:
            self
        """
        if doc:
            self.add_comment(doc)
        
        params_str = ", ".join(f"{p[0]}: {p[1]}" for p in params)
        ret_str = f" -> {return_type}" if return_type else ""
        
        self.add_line(f"pub fn {name}({params_str}){ret_str};")
        return self
    
    # =========================================================================
    # 块管理
    # =========================================================================
    
    def end_block(self) -> 'RustCodeBuilder':
        """
        结束当前块（自动减少缩进并添加 }）
        
        Returns:
            self
        """
        if self.indent_level > 0:
            self.indent_level -= 1
        
        if self._block_stack:
            self._block_stack.pop()
        
        self.add_line("}")
        self.add_line("")  # 空行分隔
        return self
    
    @contextmanager
    def block(self, block_type: str = "generic"):
        """
        上下文管理器，自动闭合块
        
        使用示例:
            with builder.block("struct"):
                builder.add_line("pub x: i32,")
        """
        self._block_stack.append(block_type)
        self.indent_level += 1
        try:
            yield self
        finally:
            self.end_block()
    
    # =========================================================================
    # 模块相关
    # =========================================================================
    
    def add_module_header(
        self, 
        name: str,
        description: str = "Auto-generated skeleton - function bodies are unimplemented."
    ) -> 'RustCodeBuilder':
        """
        添加模块头部注释和常用 allow 属性
        
        Args:
            name: 模块名
            description: 描述
        
        Returns:
            self
        """
        self.add_line(f"//! Module: {name}")
        self.add_line("//!")
        self.add_line(f"//! {description}")
        self.add_line("")
        self.add_allow("unused_imports")
        self.add_allow("dead_code")
        self.add_allow("unused_variables")
        self.add_allow("non_camel_case_types")
        self.add_allow("non_snake_case")
        self.add_line("")
        return self
    
    def add_use(self, path: str) -> 'RustCodeBuilder':
        """
        添加 use 语句
        
        Args:
            path: use 路径
        
        Returns:
            self
        """
        self.add_line(f"use {path};")
        return self
    
    def add_mod(self, name: str, is_pub: bool = True) -> 'RustCodeBuilder':
        """
        添加 mod 声明
        
        Args:
            name: 模块名
            is_pub: 是否公开
        
        Returns:
            self
        """
        vis = "pub " if is_pub else ""
        self.add_line(f"{vis}mod {name};")
        return self
    
    # =========================================================================
    # 输出
    # =========================================================================
    
    def build(self) -> str:
        """
        构建最终的 Rust 代码字符串
        
        Returns:
            格式化的 Rust 代码
        """
        # 检查是否有未闭合的块
        if self._block_stack:
            import warnings
            warnings.warn(f"未闭合的块: {self._block_stack}")
        
        return "\n".join(self.code_blocks)
    
    def clear(self) -> 'RustCodeBuilder':
        """清空所有内容"""
        self.code_blocks = []
        self.indent_level = 0
        self._block_stack = []
        return self
    
    def __str__(self) -> str:
        return self.build()


# =========================================================================
# 便捷工厂函数
# =========================================================================

def create_opaque_type(name: str, doc: Optional[str] = None) -> str:
    """快速创建不透明类型"""
    builder = RustCodeBuilder()
    builder.add_opaque_struct(name, doc)
    return builder.build()


def create_type_alias(name: str, target: str, doc: Optional[str] = None) -> str:
    """快速创建类型别名"""
    builder = RustCodeBuilder()
    builder.add_type_alias(name, target, doc=doc)
    return builder.build()


def create_const(name: str, type_name: str, value: str, doc: Optional[str] = None) -> str:
    """快速创建常量"""
    builder = RustCodeBuilder()
    builder.add_const(name, type_name, value, doc=doc)
    return builder.build()


def create_function_stub(
    name: str,
    params: List[tuple],
    return_type: Optional[str] = None,
    doc: Optional[str] = None
) -> str:
    """快速创建函数桩"""
    builder = RustCodeBuilder()
    builder.add_function_stub(name, params, return_type, doc=doc)
    return builder.build()


# =========================================================================
# 测试代码
# =========================================================================

if __name__ == "__main__":
    # 示例：生成完整的 types.rs
    builder = RustCodeBuilder()
    
    # 模块头
    builder.add_line("//! Types module")
    builder.add_line("")
    builder.add_allow("unused_imports", "dead_code", "non_camel_case_types")
    builder.add_line("")
    
    # 不透明类型
    builder.add_opaque_struct("HnpCfgInfo", "External configuration info")
    
    # 普通结构体
    builder.add_struct("Point", doc="A 2D point")
    builder.add_field("x", "i32")
    builder.add_field("y", "i32")
    builder.end_block()
    
    # 枚举
    builder.add_enum("Color", repr_type="u32", doc="Color enumeration")
    builder.add_variant("Red", 0)
    builder.add_variant("Green", 1)
    builder.add_variant("Blue", 2)
    builder.end_block()
    
    # 类型别名
    builder.add_type_alias("BOOL", "i32", doc="Boolean type alias")
    
    # 常量
    builder.add_const("MAX_SIZE", "usize", "1024", doc="Maximum buffer size")
    
    # 函数桩
    builder.add_function_stub(
        "calculate_distance",
        [("p1", "*const Point"), ("p2", "*const Point")],
        "f64",
        doc="Calculate distance between two points"
    )
    
    print(builder.build())































