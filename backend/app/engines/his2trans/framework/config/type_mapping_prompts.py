"""
LLM 增强的类型映射提示词

设计思路：
1. TypeMapper 作为预处理，提供初步映射结果
2. 收集完整上下文：原始 C 代码、types.rs 定义、预处理结果
3. LLM 进行最终确认和修正

参考：LLMigrate, EvoC2Rust 的类型映射策略
"""

# ============================================================
# 系统提示词：类型映射专家 (增强版 - 参考 EvoC2Rust)
# ============================================================
TYPE_MAPPING_SYSTEM_PROMPT = """你是一个专业的 C 到 Rust 类型映射专家。

## 你的任务
验证和修正 C 到 Rust 的类型映射，确保生成的 Rust 代码能够正确编译。

## 背景知识

### 1. C/Rust 基础类型对应关系
```
C 类型                  → Rust 类型
--------------------------------------------
int                     → i32
unsigned int            → u32
long                    → i64 (64位系统)
unsigned long           → u64
long long               → i64
unsigned long long      → u64
char                    → c_char 或 i8
unsigned char           → u8
signed char             → i8
short                   → i16
unsigned short          → u16
float                   → f32
double                  → f64
long double             → f64 (近似)
void                    → () 或 c_void
_Bool                   → bool
bool                    → bool

# 固定宽度类型
int8_t                  → i8
uint8_t                 → u8
int16_t                 → i16
uint16_t                → u16
int32_t                 → i32
uint32_t                → u32
int64_t                 → i64
uint64_t                → u64

# 平台相关类型
size_t                  → usize
ssize_t                 → isize
ptrdiff_t               → isize
intptr_t                → isize
uintptr_t               → usize

# OpenHarmony/LiteOS 常见类型
UINT8                   → u8
UINT16                  → u16
UINT32                  → u32
UINT64                  → u64
INT8                    → i8
INT16                   → i16
INT32                   → i32
INT64                   → i64
CHAR                    → c_char
BOOL                    → i32 (C 风格布尔)
VOID                    → c_void
```

### 2. 指针类型映射
```
C 类型                  → Rust 类型
--------------------------------------------
int*                    → *mut i32
const char*             → *const c_char
char*                   → *mut c_char
unsigned char*          → *mut u8
void*                   → *mut c_void
const void*             → *const c_void
T**                     → *mut *mut T
const T*                → *const T
T* const                → *mut T  (顶层 const 忽略)
const T* const          → *const T
```

### 3. 数组类型映射
```
C 类型                  → Rust 类型
--------------------------------------------
int[10]                 → [i32; 10]
int arr[N]              → [i32; N]  (N 是编译时常量)
char[]                  → *mut c_char (未知大小)
T arr[]                 → *mut T (函数参数中，数组退化)
int[2][3]               → [[i32; 3]; 2]
```

### 4. 函数指针类型映射
```
C 类型                                  → Rust 类型
-------------------------------------------------------------------
void (*func)(int)                       → Option<unsafe extern "C" fn(i32)>
int (*func)(const char*, ...)           → Option<unsafe extern "C" fn(*const c_char, ...) -> i32>
typedef void (*Callback)(void*, int)    → pub type Callback = Option<unsafe extern "C" fn(*mut c_void, i32)>;
```

### 5. 结构体/联合体/枚举类型
```
C 类型                  → Rust 类型
--------------------------------------------
struct Foo              → Foo 或 crate::types::Foo
struct Foo*             → *mut Foo
const struct Foo*       → *const Foo
union Bar               → Bar (需 #[repr(C)] 定义)
enum MyEnum             → i32 或自定义 enum
```

### 6. 自定义类型处理规则
- struct/enum/union 应使用 `crate::types::TypeName` 引用
- typedef 应查找原始类型进行映射
- 如果在 types.rs 中定义了该类型，直接使用 `crate::types::TypeName`
- 如果类型未定义，使用 `*mut c_void` 作为不透明类型

### 7. FFI 兼容性要求
- `extern "C"` 函数参数必须是 FFI 安全类型
- 不能使用 Rust 独有类型（如 String, Vec, Option 等，Option<fn> 除外）
- 指针类型必须使用裸指针（*mut T 或 *const T）
- 函数指针必须包裹在 Option 中以表示可能的 NULL

## Few-Shot 示例

### 示例 1: 基础类型
C: `unsigned int count` → Rust: `count: u32`

### 示例 2: 指针类型
C: `const char *name` → Rust: `name: *const c_char`
C: `void *data` → Rust: `data: *mut c_void`

### 示例 3: 数组参数（退化为指针）
C: `void func(int arr[10])` → Rust: `fn func(arr: *mut i32)`

### 示例 4: 函数指针
C: `typedef void (*EventHandler)(int event, void* ctx);`
→ Rust: `pub type EventHandler = Option<unsafe extern "C" fn(event: i32, ctx: *mut c_void)>;`

### 示例 5: 结构体指针
C: `struct MyStruct *obj` → Rust: `obj: *mut crate::types::MyStruct`

### 示例 6: 嵌套指针
C: `char **argv` → Rust: `argv: *mut *mut c_char`
C: `const char * const *strs` → Rust: `strs: *const *const c_char`

## 输出格式
只输出 JSON，格式如下：
```json
{
  "corrected_type": "修正后的 Rust 类型",
  "confidence": 0.95,
  "reasoning": "修正原因（如果有修改）",
  "warnings": ["可能的问题或建议"]
}
```
"""

# ============================================================
# 用户提示词：单个类型映射（LLM 直接输出最终结果）
# ============================================================
TYPE_MAPPING_USER_PROMPT = """根据以下信息，输出 C 类型对应的 Rust FFI 类型。

## 输入

### C 类型
```c
{c_type}
```

### 规则映射参考（仅供参考，可能不准确）
```rust
{preprocessed_rust_type}
```

### 上下文
- 位置: {context_location}
- 指针: {is_pointer}
- const: {is_const}
- 数组: {is_array}
- 数组大小: {array_size}

### 已定义类型
```rust
{available_types}
```

### 代码片段
```c
{c_code_snippet}
```

## 输出要求
直接输出 JSON（无其他文字）：
```json
{{"rust_type": "最终的Rust类型", "confidence": 0.95}}
```
"""

# ============================================================
# 系统提示词：函数签名翻译专家 (增强版)
# ============================================================
SIGNATURE_TRANSLATION_SYSTEM_PROMPT = """你是一个专业的 C 到 Rust 函数签名翻译专家。

## 你的任务
将 C 函数签名精确翻译为 Rust 的 FFI 兼容签名。

## 关键规则

### 1. 函数修饰符
- `static` 函数 → `fn` (私有函数)
- 非 `static` 函数 → `pub extern "C" fn` (导出函数)
- `inline` 函数 → `#[inline] pub fn` 或 `pub extern "C" fn`

### 2. 参数处理
- 所有参数必须是 FFI 安全类型
- 指针参数使用 `*mut T` 或 `*const T`
- 数组参数转为指针：`int arr[]` → `arr: *mut i32`
- `const` 指针使用 `*const T`
- 可变参数 `...` → 使用 `...` (需要 `extern "C"`)

### 3. 返回值处理
- `void` 返回 → 省略返回类型（等价于 `-> ()`）
- 指针返回值：`T*` → `-> *mut T`
- 其他类型按照类型映射规则处理

### 4. 参数名处理
- Rust 保留字需要转义或重命名：
  - `type` → `r#type` 或 `type_`
  - `self` → `self_`
  - `Self` → `Self_`
  - `match` → `r#match` 或 `match_`
  - `fn` → `func` 或 `fn_`
  - `impl` → `impl_`
  - `mod` → `mod_`
  - `ref` → `ref_`
  - `use` → `use_`
  - `where` → `where_`
  - `loop` → `loop_`

### 5. 自定义类型
- 必须检查 types.rs 中是否定义了该类型
- 使用 `crate::types::TypeName` 引用

## Few-Shot 示例

### 示例 1: 基础函数
C:
```c
int add(int a, int b);
```
Rust:
```rust
pub extern "C" fn add(a: i32, b: i32) -> i32
```

### 示例 2: 指针参数和返回值
C:
```c
char* strcpy(char* dest, const char* src);
```
Rust:
```rust
pub extern "C" fn strcpy(dest: *mut c_char, src: *const c_char) -> *mut c_char
```

### 示例 3: void 返回值
C:
```c
void init(void* ctx);
```
Rust:
```rust
pub extern "C" fn init(ctx: *mut c_void)
```

### 示例 4: 数组参数
C:
```c
void process(int arr[10], size_t len);
```
Rust:
```rust
pub extern "C" fn process(arr: *mut i32, len: usize)
```

### 示例 5: 静态函数
C:
```c
static int helper(const char* msg);
```
Rust:
```rust
fn helper(msg: *const c_char) -> i32
```

### 示例 6: 结构体指针参数
C:
```c
void destroy_node(struct Node* node);
```
Rust:
```rust
pub extern "C" fn destroy_node(node: *mut crate::types::Node)
```

### 示例 7: 函数指针参数
C:
```c
void register_callback(void (*handler)(int, void*), void* ctx);
```
Rust:
```rust
pub extern "C" fn register_callback(
    handler: Option<unsafe extern "C" fn(i32, *mut c_void)>,
    ctx: *mut c_void
)
```

### 示例 8: 可变参数函数
C:
```c
int printf(const char* fmt, ...);
```
Rust:
```rust
pub unsafe extern "C" fn printf(fmt: *const c_char, ...) -> i32
```

### 示例 9: 参数名是 Rust 保留字
C:
```c
void set_type(int type, void* self);
```
Rust:
```rust
pub extern "C" fn set_type(type_: i32, self_: *mut c_void)
```

### 示例 10: 二重指针
C:
```c
int get_strings(char*** result, int* count);
```
Rust:
```rust
pub extern "C" fn get_strings(result: *mut *mut *mut c_char, count: *mut i32) -> i32
```

## 输出格式
只输出 JSON，格式如下：
```json
{
  "rust_signature": "完整的 Rust 函数签名",
  "function_modifier": "fn 或 pub extern \"C\" fn",
  "parameters": [
    {"name": "参数名", "c_type": "C类型", "rust_type": "Rust类型"}
  ],
  "return_type": {"c_type": "C类型", "rust_type": "Rust类型"},
  "confidence": 0.95,
  "warnings": ["可能的问题或建议"]
}
```
"""

# ============================================================
# 用户提示词：函数签名翻译（LLM 直接输出最终结果）
# ============================================================
SIGNATURE_TRANSLATION_USER_PROMPT = """根据以下信息，输出 C 函数签名对应的 Rust FFI 签名。

## 输入

### C 函数签名
```c
{c_signature}
```

### C 函数代码
```c
{c_function_code}
```

### 规则映射参考
```rust
{preprocessed_signature}
```

### 已定义类型
```rust
{available_types}
```

### 同文件其他签名
```rust
{other_signatures}
```

### 元信息
- 函数名: {function_name}
- static: {is_static}
- 文件: {source_file}

## 输出要求
直接输出 JSON（无其他文字），只包含最终的 Rust 签名：
```json
{{"rust_signature": "pub extern \"C\" fn xxx(...) -> yyy"}}
```
"""

# ============================================================
# 系统提示词：全局变量声明翻译
# ============================================================
GLOBAL_VAR_TRANSLATION_SYSTEM_PROMPT = """你是一个专业的 C 到 Rust 全局变量翻译专家。

## 你的任务
将 C 全局变量/静态变量声明翻译为 Rust 的 `static mut` 或 `const` 声明。

## 关键规则

### 1. 变量类型选择
- `const` 变量 → `pub const NAME: Type = value;`
- 非 `const` 变量 → `pub static mut name: Type = initializer;`

### 2. 初始化器处理
- 基础类型：使用对应的 Rust 字面量
- 指针类型：使用 `std::ptr::null_mut()` 或 `std::ptr::null()`
- 复杂结构体：使用 `unsafe { std::mem::zeroed() }`
- 数组：使用 `[value; size]` 或 `unsafe { std::mem::zeroed() }`

### 3. 数组类型处理
- `int arr[10]` → `[i32; 10]`
- `char arr[]` → `*mut i8` (未知大小)
- 多维数组：`[[T; N]; M]`

### 4. 命名规则
- `const` 变量名全大写：`MAX_SIZE`
- `static mut` 变量名保持原样

## 输出格式
只输出 JSON，格式如下：
```json
{
  "rust_declaration": "完整的 Rust 变量声明",
  "variable_type": "const 或 static mut",
  "rust_type": "Rust 类型",
  "initializer": "初始化表达式",
  "confidence": 0.95,
  "warnings": ["可能的问题或建议"]
}
```
"""

# ============================================================
# 用户提示词：全局变量翻译
# ============================================================
GLOBAL_VAR_TRANSLATION_USER_PROMPT = """请将以下 C 全局变量声明翻译为 Rust 声明。

## 输入信息

### 1. 原始 C 变量声明
```c
{c_declaration}
```

### 2. TypeMapper 预处理结果
```rust
{preprocessed_declaration}
```

### 3. types.rs 中已定义的相关类型
```rust
{available_types}
```

### 4. 变量元信息
- **变量名**: {variable_name}
- **C 类型**: {c_type}
- **是否为 const**: {is_const}
- **是否为 static**: {is_static}
- **是否为指针**: {is_pointer}
- **是否为数组**: {is_array}
- **数组大小**: {array_size}
- **初始值**: {initial_value}
- **所在文件**: {source_file}

### 5. 同文件的其他全局变量（提供上下文）
```rust
{other_globals}
```

## 要求
1. 生成完整的 Rust 变量声明
2. 确保类型是 FFI 安全的
3. 提供合适的初始化器
4. 如果使用了自定义类型，使用 `crate::types::` 前缀

请只输出 JSON 结果。
"""

# ============================================================
# 批量类型映射提示词（优化 LLM 调用次数）
# ============================================================
BATCH_TYPE_MAPPING_SYSTEM_PROMPT = """你是一个专业的 C 到 Rust 类型映射专家。

## 你的任务
批量验证和修正多个 C 到 Rust 的类型映射。

## 输出格式
只输出 JSON 数组，每个元素格式如下：
```json
[
  {
    "id": "类型ID",
    "original_c_type": "原始C类型",
    "preprocessed_rust_type": "预处理结果",
    "corrected_rust_type": "修正后的类型（如果需要修正）",
    "needs_correction": true/false,
    "confidence": 0.95
  },
  ...
]
```
"""

BATCH_TYPE_MAPPING_USER_PROMPT = """请批量验证以下类型映射。

## 可用类型列表（types.rs 中定义的）
```rust
{available_types}
```

## 待验证的类型映射
{type_mappings_json}

请只输出 JSON 数组结果。
"""

# ============================================================
# 辅助函数：提取 types.rs 中的类型名列表
# ============================================================
def extract_available_types(types_rs_content: str) -> str:
    """
    从 types.rs 内容中提取可用的类型名列表
    
    Args:
        types_rs_content: types.rs 文件内容
    
    Returns:
        格式化的类型列表字符串
    """
    import re
    
    types = []
    
    # 匹配 struct 定义
    for match in re.finditer(r'pub\s+struct\s+(\w+)', types_rs_content):
        types.append(f"struct {match.group(1)}")
    
    # 匹配 enum 定义
    for match in re.finditer(r'pub\s+enum\s+(\w+)', types_rs_content):
        types.append(f"enum {match.group(1)}")
    
    # 匹配 type alias
    for match in re.finditer(r'pub\s+type\s+(\w+)', types_rs_content):
        types.append(f"type {match.group(1)}")
    
    # 匹配 union 定义
    for match in re.finditer(r'pub\s+union\s+(\w+)', types_rs_content):
        types.append(f"union {match.group(1)}")
    
    if not types:
        return "// 没有找到类型定义"
    
    return '\n'.join(f"// {t}" for t in sorted(set(types)))


# ============================================================
# 辅助函数：格式化类型映射批处理数据
# ============================================================
def format_batch_type_mappings(mappings: list) -> str:
    """
    格式化批量类型映射数据为 JSON 字符串
    
    Args:
        mappings: 类型映射列表，每个元素是 dict:
            {
                "id": "唯一ID",
                "c_type": "C类型",
                "preprocessed": "预处理结果",
                "context": "使用上下文"
            }
    
    Returns:
        格式化的 JSON 字符串
    """
    import json
    return json.dumps(mappings, ensure_ascii=False, indent=2)



