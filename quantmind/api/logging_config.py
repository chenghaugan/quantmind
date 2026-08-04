"""API 层日志配置"""
import logging
import sys
from typing import Optional

def setup_api_logger(level: str = "INFO") -> logging.Logger:
    """配置 API 层日志
    
    Args:
        level: 日志级别 (DEBUG/INFO/WARNING/ERROR)
    
    Returns:
        配置好的 logger 实例
    """
    logger = logging.getLogger("quantmind.api")
    logger.setLevel(getattr(logging, level.upper()))
    
    # 避免重复添加 handler
    if logger.handlers:
        return logger
    
    # 控制台 handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    
    # 日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger
