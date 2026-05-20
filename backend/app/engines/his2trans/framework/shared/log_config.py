#!/usr/bin/env python3
"""
统一日志配置模块

解决问题：
1. HTTP 请求日志过多（httpx/openai 每次请求都打印）
2. 各模块日志级别不一致
3. 日志格式不统一

使用方法：
    在主入口文件的开头添加:
    from log_config import setup_logging
    setup_logging()
"""

import logging
import os
import sys

# 需要静默的第三方库（产生大量无用日志）
QUIET_LOGGERS = [
    "httpx",           # HTTP 客户端，每次请求都打印 200 OK
    "httpcore",        # httpx 的底层库
    "openai",          # OpenAI 客户端
    "urllib3",         # HTTP 库
    "requests",        # HTTP 库
    "charset_normalizer",  # 字符编码检测
    "asyncio",         # 异步 IO
    "concurrent.futures",  # 线程池
]

# 日志格式
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FORMAT_SIMPLE = "%(levelname)s - %(message)s"


def setup_logging(
    level: int = logging.INFO,
    log_file: str = None,
    quiet_mode: bool = True,
    simple_format: bool = False
):
    """
    配置全局日志
    
    Args:
        level: 日志级别（默认 INFO）
        log_file: 日志文件路径（可选，不设置则只输出到控制台）
        quiet_mode: 是否静默第三方库日志（默认 True）
        simple_format: 是否使用简化格式（默认 False）
    """
    # 选择格式
    log_format = LOG_FORMAT_SIMPLE if simple_format else LOG_FORMAT
    
    # 配置根 logger
    handlers = []
    
    # 控制台输出
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(log_format))
    handlers.append(console_handler)
    
    # 文件输出（如果指定）
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        handlers.append(file_handler)
    
    # 配置根 logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # 清除现有的 handlers
    root_logger.handlers.clear()
    
    for handler in handlers:
        root_logger.addHandler(handler)
    
    # 静默第三方库
    if quiet_mode:
        for logger_name in QUIET_LOGGERS:
            logging.getLogger(logger_name).setLevel(logging.WARNING)
        
        # 特别处理 httpx：它会在每次 HTTP 请求时打印日志
        # 设置为 ERROR 级别，只有出错时才打印
        logging.getLogger("httpx").setLevel(logging.ERROR)
        logging.getLogger("httpcore").setLevel(logging.ERROR)
    
    logging.info("日志配置完成")


def get_logger(name: str) -> logging.Logger:
    """
    获取模块 logger
    
    Args:
        name: 模块名称（通常使用 __name__）
        
    Returns:
        配置好的 logger 实例
    """
    return logging.getLogger(name)


# 便捷函数：快速设置静默模式
def silence_http_logs():
    """静默所有 HTTP 相关日志（httpx, openai 等）"""
    for logger_name in ["httpx", "httpcore", "openai", "urllib3"]:
        logging.getLogger(logger_name).setLevel(logging.ERROR)


# =========================================================================
# 日志与 print 桥接器（支持渐进式迁移）
# =========================================================================

class LogPrinter:
    """
    日志与 print 桥接器

    在迁移期间，同时输出到日志和控制台，确保向后兼容。
    迁移完成后，可以禁用 print 输出。

    使用方法：
        from log_config import LogPrinter
        log = LogPrinter(__name__)
        log.info("这条消息同时输出到日志和控制台")
    """

    def __init__(self, name: str, enable_print: bool = True):
        """
        Args:
            name: 模块名称（通常使用 __name__）
            enable_print: 是否同时输出到控制台（默认 True，迁移完成后可设为 False）
        """
        self.logger = logging.getLogger(name)
        self.enable_print = enable_print

    def _log_and_print(self, level: int, msg: str, *args):
        """内部方法：同时记录日志和打印"""
        formatted_msg = msg % args if args else msg
        self.logger.log(level, formatted_msg)
        if self.enable_print:
            # 添加级别前缀使输出更清晰
            prefix = {
                logging.DEBUG: "[DEBUG]",
                logging.INFO: "[INFO]",
                logging.WARNING: "[WARN]",
                logging.ERROR: "[ERROR]",
                logging.CRITICAL: "[CRIT]"
            }.get(level, "")
            print(f"{prefix} {formatted_msg}")

    def debug(self, msg: str, *args):
        self._log_and_print(logging.DEBUG, msg, *args)

    def info(self, msg: str, *args):
        self._log_and_print(logging.INFO, msg, *args)

    def warning(self, msg: str, *args):
        self._log_and_print(logging.WARNING, msg, *args)

    def error(self, msg: str, *args):
        self._log_and_print(logging.ERROR, msg, *args)

    def critical(self, msg: str, *args):
        self._log_and_print(logging.CRITICAL, msg, *args)

    def progress(self, current: int, total: int, msg: str = ""):
        """进度输出（不记录到日志，只打印到控制台）"""
        if self.enable_print:
            percent = 100.0 * current / total if total > 0 else 0
            print(f"\r[进度] {current}/{total} ({percent:.1f}%) {msg}", end="", flush=True)
            if current >= total:
                print()  # 完成时换行


def create_file_logger(
    name: str,
    log_file: str,
    level: int = logging.INFO,
    console: bool = True
) -> logging.Logger:
    """
    创建独立的文件日志器

    用于将特定模块的日志写入单独文件，便于分析。

    Args:
        name: 日志器名称
        log_file: 日志文件路径
        level: 日志级别
        console: 是否同时输出到控制台

    Returns:
        配置好的 logger 实例
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    # 文件输出
    file_handler = logging.FileHandler(log_file, encoding='utf-8', mode='a')
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(file_handler)

    # 控制台输出
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(logging.Formatter(LOG_FORMAT_SIMPLE))
        logger.addHandler(console_handler)

    return logger


# 全局日志初始化标志
_logging_initialized = False

def ensure_logging_setup():
    """确保日志已初始化（幂等）"""
    global _logging_initialized
    if not _logging_initialized:
        setup_logging()
        _logging_initialized = True


# 如果直接运行此文件，进行测试
if __name__ == "__main__":
    setup_logging(level=logging.DEBUG)

    logger = get_logger(__name__)

    logger.debug("这是 DEBUG 消息")
    logger.info("这是 INFO 消息")
    logger.warning("这是 WARNING 消息")
    logger.error("这是 ERROR 消息")

    print("\n测试第三方库日志静默...")

    # 模拟 httpx 日志（应该被静默）
    httpx_logger = logging.getLogger("httpx")
    httpx_logger.info("这条 HTTP 日志应该被静默")
    httpx_logger.error("这条 HTTP 错误日志应该显示")

    print("\n测试 LogPrinter 桥接器...")
    log = LogPrinter(__name__)
    log.info("桥接器信息消息")
    log.warning("桥接器警告消息")
    log.error("桥接器错误消息")

    print("\n测试进度输出...")
    for i in range(1, 6):
        log.progress(i, 5, f"处理项目 {i}")
        import time
        time.sleep(0.2)


