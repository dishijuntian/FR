"""
通用工具函数模块
"""

import os
import psutil
import time
from pathlib import Path
from functools import wraps
import logging


# 常用常量
DEFAULT_MISSING_VALUE = -1
DEFAULT_CHUNK_SIZE = 200000
DEFAULT_N_JOBS = min(os.cpu_count(), 8)

# 快捷函数
def get_project_root() -> Path:
    """获取项目根目录"""
    current = Path(__file__).parent
    while current != current.parent:
        if (current / 'main.py').exists() or (current / 'config').exists():
            return current
        current = current.parent
    return Path.cwd()

def format_size(size_bytes: int) -> str:
    """格式化文件大小"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f}TB"

def format_duration(seconds: float) -> str:
    """格式化时间段"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"
    
# 通用函数
def setup_logger(name: str, level: str = "INFO", format_str: str = None) -> logging.Logger:
    """快速设置logger"""
    logger = logging.getLogger(name)
    if not logger.handlers:  # 避免重复添加handler
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            format_str or '%(asctime)s | %(levelname)8s | %(name)s | %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(getattr(logging, level.upper()))
    return logger

def timer(func):
    """性能计时装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        
        # 尝试获取logger
        logger = None
        if hasattr(args[0], 'logger'):
            logger = args[0].logger
        else:
            logger = logging.getLogger(func.__module__)
        
        logger.info(f"{func.__name__} 耗时: {duration:.2f}s")
        return result
    return wrapper

def memory_monitor(func):
    """内存监控装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024**2  # MB
        
        result = func(*args, **kwargs)
        
        mem_after = process.memory_info().rss / 1024**2  # MB
        mem_diff = mem_after - mem_before
        
        # 尝试获取logger
        logger = None
        if hasattr(args[0], 'logger'):
            logger = args[0].logger
        else:
            logger = logging.getLogger(func.__module__)
        
        logger.info(f"{func.__name__} 内存变化: {mem_diff:+.1f}MB (当前: {mem_after:.1f}MB)")
        return result
    return wrapper
