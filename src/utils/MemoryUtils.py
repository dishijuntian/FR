"""
内存管理工具类
"""

import gc
import psutil
import pandas as pd
import numpy as np
from typing import Dict


class MemoryUtils:    
    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """获取当前内存使用情况"""
        process = psutil.Process()
        return {
            'rss_mb': process.memory_info().rss / 1024**2,
            'vms_mb': process.memory_info().vms / 1024**2,
            'percent': process.memory_percent()
        }
    
    @staticmethod
    def force_gc():
        """强制垃圾回收"""
        before = MemoryUtils.get_memory_usage()
        gc.collect()
        after = MemoryUtils.get_memory_usage()
        freed = before['rss_mb'] - after['rss_mb']
        return freed
    
    @staticmethod
    def optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
        """优化DataFrame内存使用"""
        start_memory = df.memory_usage(deep=True).sum() / 1024**2
        
        # 优化整数类型
        for col in df.select_dtypes(include=['int']).columns:
            col_min, col_max = df[col].min(), df[col].max()
            if col_min > np.iinfo(np.int8).min and col_max < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif col_min > np.iinfo(np.int16).min and col_max < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif col_min > np.iinfo(np.int32).min and col_max < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
        
        # 优化浮点类型
        for col in df.select_dtypes(include=['float']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
        
        end_memory = df.memory_usage(deep=True).sum() / 1024**2
        reduction = (start_memory - end_memory) / start_memory * 100
        
        return df
