"""日志记录系统模块 - 提供统一的日志记录功能。

该模块使用Python标准库logging实现，提供以下功能：
- 配置不同日志级别（DEBUG、INFO、WARNING、ERROR、CRITICAL）
- 支持控制台和文件输出
- 统一的日志格式（包含时间戳、级别、模块名、消息）
- 支持日志轮转
"""

import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Optional, Union
from pathlib import Path


# 日志级别映射
LOG_LEVELS = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL
}

# 默认日志格式
DEFAULT_FORMAT = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
DEFAULT_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# 详细日志格式（包含文件名和行号）
DETAILED_FORMAT = '%(asctime)s - %(levelname)s - %(name)s - [%(filename)s:%(lineno)d] - %(message)s'

# 全局日志器实例缓存
_loggers = {}


class LoggerConfig:
    """日志配置类。
    
    Attributes:
        level: 日志级别
        log_dir: 日志文件目录
        log_file: 日志文件名
        console_output: 是否输出到控制台
        file_output: 是否输出到文件
        max_file_size: 单个日志文件最大大小（字节）
        backup_count: 保留的备份文件数量
        format_string: 日志格式字符串
        date_format: 日期格式字符串
    """
    
    def __init__(
        self,
        level: str = 'INFO',
        log_dir: str = 'logs',
        log_file: str = 'poker_ai.log',
        console_output: bool = True,
        file_output: bool = True,
        max_file_size: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        format_string: str = DEFAULT_FORMAT,
        date_format: str = DEFAULT_DATE_FORMAT
    ):
        """初始化日志配置。
        
        Args:
            level: 日志级别（DEBUG、INFO、WARNING、ERROR、CRITICAL）
            log_dir: 日志文件目录
            log_file: 日志文件名
            console_output: 是否输出到控制台
            file_output: 是否输出到文件
            max_file_size: 单个日志文件最大大小（字节）
            backup_count: 保留的备份文件数量
            format_string: 日志格式字符串
            date_format: 日期格式字符串
        """
        self.level = level.upper()
        self.log_dir = log_dir
        self.log_file = log_file
        self.console_output = console_output
        self.file_output = file_output
        self.max_file_size = max_file_size
        self.backup_count = backup_count
        self.format_string = format_string
        self.date_format = date_format
        
        # 验证日志级别
        if self.level not in LOG_LEVELS:
            raise ValueError(f"无效的日志级别: {self.level}。有效级别: {list(LOG_LEVELS.keys())}")


def setup_logger(
    name: str,
    config: Optional[LoggerConfig] = None
) -> logging.Logger:
    """设置并返回一个日志器。
    
    如果已存在同名日志器，则返回现有实例。
    
    Args:
        name: 日志器名称
        config: 日志配置，如果为None则使用默认配置
        
    Returns:
        配置好的日志器实例
    """
    # 如果已存在，返回缓存的日志器
    if name in _loggers:
        return _loggers[name]
    
    # 使用默认配置
    if config is None:
        config = LoggerConfig()
    
    # 创建日志器
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVELS[config.level])
    
    # 清除现有处理器
    logger.handlers.clear()
    
    # 创建格式化器
    formatter = logging.Formatter(config.format_string, config.date_format)
    
    # 添加控制台处理器
    if config.console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(LOG_LEVELS[config.level])
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # 添加文件处理器
    if config.file_output:
        # 确保日志目录存在
        log_dir = Path(config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_path = log_dir / config.log_file
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=config.max_file_size,
            backupCount=config.backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(LOG_LEVELS[config.level])
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # 防止日志传播到根日志器
    logger.propagate = False
    
    # 缓存日志器
    _loggers[name] = logger
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """获取日志器。
    
    如果日志器不存在，则使用默认配置创建。
    
    Args:
        name: 日志器名称
        
    Returns:
        日志器实例
    """
    if name not in _loggers:
        return setup_logger(name)
    return _loggers[name]


def set_log_level(name: str, level: str) -> None:
    """设置日志器的日志级别。
    
    Args:
        name: 日志器名称
        level: 日志级别
        
    Raises:
        ValueError: 如果日志级别无效
    """
    level = level.upper()
    if level not in LOG_LEVELS:
        raise ValueError(f"无效的日志级别: {level}。有效级别: {list(LOG_LEVELS.keys())}")
    
    logger = get_logger(name)
    logger.setLevel(LOG_LEVELS[level])
    
    # 同时更新所有处理器的级别
    for handler in logger.handlers:
        handler.setLevel(LOG_LEVELS[level])


def clear_loggers() -> None:
    """清除所有缓存的日志器。
    
    主要用于测试目的。
    """
    global _loggers
    for logger in _loggers.values():
        logger.handlers.clear()
    _loggers.clear()


class LoggerMixin:
    """日志器混入类。
    
    为类提供便捷的日志记录功能。
    继承此类的类将自动获得一个以类名命名的日志器。
    
    Example:
        class MyClass(LoggerMixin):
            def do_something(self):
                self.logger.info("正在执行操作...")
    """
    
    @property
    def logger(self) -> logging.Logger:
        """获取类的日志器。"""
        if not hasattr(self, '_logger'):
            self._logger = get_logger(self.__class__.__name__)
        return self._logger


# 预定义的模块日志器
def get_training_logger() -> logging.Logger:
    """获取训练模块的日志器。"""
    return get_logger('training')


def get_environment_logger() -> logging.Logger:
    """获取环境模块的日志器。"""
    return get_logger('environment')


def get_model_logger() -> logging.Logger:
    """获取模型模块的日志器。"""
    return get_logger('model')


def get_monitoring_logger() -> logging.Logger:
    """获取监控模块的日志器。"""
    return get_logger('monitoring')


def get_analysis_logger() -> logging.Logger:
    """获取分析模块的日志器。"""
    return get_logger('analysis')


def get_utils_logger() -> logging.Logger:
    """获取工具模块的日志器。"""
    return get_logger('utils')


# 全局日志配置函数
_global_config: Optional[LoggerConfig] = None


def configure_logging(
    level: str = 'INFO',
    log_dir: str = 'logs',
    log_file: str = 'poker_ai.log',
    console_output: bool = True,
    file_output: bool = True,
    detailed: bool = False
) -> None:
    """配置全局日志设置。
    
    此函数应在应用程序启动时调用一次。
    
    Args:
        level: 日志级别
        log_dir: 日志文件目录
        log_file: 日志文件名
        console_output: 是否输出到控制台
        file_output: 是否输出到文件
        detailed: 是否使用详细格式（包含文件名和行号）
    """
    global _global_config
    
    format_string = DETAILED_FORMAT if detailed else DEFAULT_FORMAT
    
    _global_config = LoggerConfig(
        level=level,
        log_dir=log_dir,
        log_file=log_file,
        console_output=console_output,
        file_output=file_output,
        format_string=format_string
    )
    
    # 清除现有日志器并使用新配置重新创建
    clear_loggers()


def get_global_config() -> LoggerConfig:
    """获取全局日志配置。
    
    Returns:
        全局日志配置，如果未配置则返回默认配置
    """
    global _global_config
    if _global_config is None:
        _global_config = LoggerConfig()
    return _global_config
