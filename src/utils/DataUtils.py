"""
数据处理工具类
"""

import pandas as pd
from typing import Any, Dict, List, Optional, Union


class DataUtils:    
    @staticmethod
    def safe_divide(a: Union[int, float], b: Union[int, float], default: float = 0.0) -> float:
        """安全除法"""
        try:
            return a / b if b != 0 else default
        except (TypeError, ZeroDivisionError):
            return default
    
    @staticmethod
    def percentile_clip(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
        """基于百分位数的异常值裁剪"""
        lower_bound = series.quantile(lower)
        upper_bound = series.quantile(upper)
        return series.clip(lower_bound, upper_bound)
    
    @staticmethod
    def encode_categorical(series: pd.Series, method: str = 'hash') -> pd.Series:
        """分类变量编码"""
        if method == 'hash':
            return series.apply(lambda x: hash(str(x)) & 0x7FFFFFFF if pd.notna(x) else -1)
        elif method == 'label':
            return pd.Categorical(series).codes
        else:
            raise ValueError(f"不支持的编码方法: {method}")
    
    @staticmethod
    def create_time_features(dt_series: pd.Series) -> pd.DataFrame:
        """从时间序列创建时间特征"""
        dt_series = pd.to_datetime(dt_series)
        
        features = pd.DataFrame({
            'hour': dt_series.dt.hour,
            'day_of_week': dt_series.dt.dayofweek,
            'day_of_month': dt_series.dt.day,
            'month': dt_series.dt.month,
            'quarter': dt_series.dt.quarter,
            'is_weekend': (dt_series.dt.dayofweek >= 5).astype(int)
        })
        
        return features
