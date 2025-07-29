"""
数据验证工具类
"""

import os
import pandas as pd
import numpy as np
from typing import Any, Dict, List


class ValidationUtils:    
    @staticmethod
    def check_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
        """检查数据质量"""
        quality_report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicated_rows': df.duplicated().sum(),
            'data_types': df.dtypes.to_dict(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
        }
        
        # 数值列统计
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            quality_report['numeric_stats'] = df[numeric_cols].describe().to_dict()
        
        return quality_report
    
    @staticmethod
    def validate_segment_integrity(segment_files: List[str], original_file: str) -> Dict[str, bool]:
        """验证数据段完整性"""
        # 读取原始文件
        original_df = pd.read_parquet(original_file)
        original_rows = len(original_df)
        original_rankers = set(original_df['ranker_id'].unique())
        
        # 读取分段文件
        segment_dfs = []
        for file_path in segment_files:
            if os.path.exists(file_path):
                segment_dfs.append(pd.read_parquet(file_path))
        
        if not segment_dfs:
            return {'row_integrity': False, 'ranker_integrity': False}
        
        # 合并分段数据
        combined_df = pd.concat(segment_dfs, ignore_index=True)
        combined_rows = len(combined_df)
        combined_rankers = set(combined_df['ranker_id'].unique())
        
        return {
            'row_integrity': original_rows == combined_rows,
            'ranker_integrity': original_rankers == combined_rankers,
            'original_rows': original_rows,
            'combined_rows': combined_rows,
            'original_rankers': len(original_rankers),
            'combined_rankers': len(combined_rankers)
        }