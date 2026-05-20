"""
配置模块

包含：
- predefines: 可配置的类型/常量预定义（支持自适应学习）
"""

from .predefines import (
    PredefineManager,
    get_predefine_manager,
    learn_type_from_error,
    learn_constant_from_error,
    POSIX_TYPES,
    POSIX_CONSTANTS,
    OHOS_TYPES,
    OHOS_CONSTANTS,
    CONSTANT_PREFIX_RULES,
)

__all__ = [
    'PredefineManager',
    'get_predefine_manager',
    'learn_type_from_error',
    'learn_constant_from_error',
    'POSIX_TYPES',
    'POSIX_CONSTANTS',
    'OHOS_TYPES',
    'OHOS_CONSTANTS',
    'CONSTANT_PREFIX_RULES',
]






















