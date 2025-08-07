"""
数据处理模块 - 重构版

专注于：
- 数据加载和预处理
- 特征工程
- 数据分割
- 排名分配

作者: Flight Ranking Team
版本: 4.0 (重构版)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')


class DataProcessor:
    """数据处理器 - 专注于数据处理逻辑"""
    
    def __init__(self, exclude_features: List[str] = None):
        """
        初始化数据处理器
        
        Args:
            exclude_features: 需要排除的特征列表
        """
        self.exclude_features = exclude_features or [
            'Id', 'selected', 'ranker_id', 'profileId', 'companyID'
        ]
        self.feature_columns = None
        
    def load_data(self, file_path: Path, use_sampling: bool = False, 
                  num_groups: int = 2000, min_group_size: int = 20) -> pd.DataFrame:
        """
        加载数据
        
        Args:
            file_path: 数据文件路径
            use_sampling: 是否使用抽样
            num_groups: 抽样组数
            min_group_size: 最小组大小
            
        Returns:
            pd.DataFrame: 加载的数据
        """
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        # 读取数据
        df = pd.read_parquet(file_path)
        print(f"原始数据形状: {df.shape}")
        
        if use_sampling:
            df = self._sample_data(df, num_groups, min_group_size)
            print(f"抽样后数据形状: {df.shape}")
        
        # 优化内存
        df = self._optimize_memory(df)
        
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        准备特征数据
        
        Args:
            df: 原始数据
            
        Returns:
            Tuple: (特征矩阵, 标签向量, 特征名列表)
        """
        # 选择数值型特征
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col not in self.exclude_features]
        
        # 处理缺失值
        df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())
        
        # 准备特征和标签
        X = df[feature_cols].values.astype(np.float32)
        y = df['selected'].values.astype(np.float32)
        
        self.feature_columns = feature_cols
        
        return X, y, feature_cols
    
    def split_ranking_data(self, df: pd.DataFrame, test_size: float = 0.2, 
                          random_state: int = 42) -> Tuple[np.ndarray, ...]:
        """
        分割排序数据
        
        Args:
            df: 原始数据
            test_size: 测试集比例
            random_state: 随机种子
            
        Returns:
            Tuple: 训练测试数据元组
        """
        # 准备特征
        X, y, feature_cols = self.prepare_features(df)
        groups = df['ranker_id'].values
        
        # 按组分割
        unique_groups = df['ranker_id'].unique()
        train_groups, test_groups = train_test_split(
            unique_groups, test_size=test_size, random_state=random_state
        )
        
        train_mask = df['ranker_id'].isin(train_groups)
        test_mask = df['ranker_id'].isin(test_groups)
        
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        
        # 计算组大小
        train_group_sizes = self._calculate_group_sizes(groups[train_mask])
        test_group_sizes = self._calculate_group_sizes(groups[test_mask])
        
        test_info = df[test_mask][['ranker_id', 'selected']].copy()
        
        return (X_train, X_test, y_train, y_test, 
                train_group_sizes, test_group_sizes, feature_cols, test_info)
    
    def prepare_test_features(self, test_df: pd.DataFrame, feature_names: List[str] = None) -> Tuple[np.ndarray, List[int]]:
        """
        准备测试特征
        
        Args:
            test_df: 测试数据
            feature_names: 特征名称列表（可选，如果未提供则自动推断）
            
        Returns:
            Tuple: (特征矩阵, 组大小列表)
        """
        # 如果提供了特征名称，使用提供的；否则尝试自动推断
        if feature_names is not None:
            feature_cols = feature_names
        elif self.feature_columns is not None:
            feature_cols = self.feature_columns
        else:
            # 自动推断特征列：选择数值型特征，排除指定的列
            numeric_cols = test_df.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [col for col in numeric_cols if col not in self.exclude_features]
            self.feature_columns = feature_cols
            print(f"自动推断特征列，共 {len(feature_cols)} 个特征")
        
        # 确保包含所需特征
        for feature in feature_cols:
            if feature not in test_df.columns:
                test_df[feature] = 0.0
        
        # 处理缺失值
        test_df[feature_cols] = test_df[feature_cols].fillna(
            test_df[feature_cols].median()
        )
        
        X_test = test_df[feature_cols].values.astype(np.float32)
        group_sizes = self._calculate_group_sizes(test_df['ranker_id'].values)
        
        return X_test, group_sizes
    
    def assign_rankings(self, test_df: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
        """
        分配排名
        
        Args:
            test_df: 测试数据
            scores: 预测分数
            
        Returns:
            pd.DataFrame: 包含排名的结果
        """
        result_df = test_df[['Id', 'ranker_id']].copy()
        result_df['scores'] = scores
        
        # 按分数降序排列，使用Id作为tie-breaker
        result_df = result_df.sort_values(['ranker_id', 'scores', 'Id'], 
                                        ascending=[True, False, True])
        result_df['selected'] = result_df.groupby('ranker_id').cumcount() + 1
        
        return result_df[['Id', 'ranker_id', 'selected']]
    
    def _sample_data(self, df: pd.DataFrame, num_groups: int, min_group_size: int) -> pd.DataFrame:
        """基于ranker_id的分组抽样"""
        group_counts = df['ranker_id'].value_counts()
        valid_groups = group_counts[group_counts >= min_group_size].index
        
        if len(valid_groups) < num_groups:
            num_groups = len(valid_groups)
        
        np.random.seed(42)
        selected_groups = np.random.choice(valid_groups, size=num_groups, replace=False)
        
        return df[df['ranker_id'].isin(selected_groups)].copy()
    
    def _optimize_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """优化内存使用"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].astype(np.float32)
        return df
    
    def _calculate_group_sizes(self, groups: np.ndarray) -> List[int]:
        """计算每组大小"""
        group_sizes = []
        current_group = groups[0]
        current_size = 1
        
        for i in range(1, len(groups)):
            if groups[i] == current_group:
                current_size += 1
            else:
                group_sizes.append(current_size)
                current_group = groups[i]
                current_size = 1
        group_sizes.append(current_size)
        
        return group_sizes


class PredictionMerger:
    """预测结果合并器"""
    
    def merge_predictions(self, prediction_files: List[Path], 
                         submission_file: Path, output_file: Path,
                         ensemble_method: str = 'average') -> Path:
        """
        合并预测结果
        
        Args:
            prediction_files: 预测文件列表
            submission_file: 提交模板文件
            output_file: 输出文件
            ensemble_method: 集成方法
            
        Returns:
            Path: 输出文件路径
        """
        # 读取所有预测文件
        prediction_dfs = []
        for file_path in prediction_files:
            if file_path.exists():
                pred_df = pd.read_parquet(file_path)
                prediction_dfs.append(pred_df)
        
        if not prediction_dfs:
            raise ValueError("没有有效的预测文件")
        
        # 合并预测
        merged_df = pd.concat(prediction_dfs, ignore_index=True)
        
        # 简化集成
        if ensemble_method == 'average':
            score_cols = [col for col in merged_df.columns if 'score' in col]
            if score_cols:
                merged_df['final_score'] = merged_df[score_cols].mean(axis=1)
                # 重新分配排名
                merged_df = merged_df.sort_values(['ranker_id', 'final_score', 'Id'], 
                                                ascending=[True, False, True])
                merged_df['selected'] = merged_df.groupby('ranker_id').cumcount() + 1
        
        # 保存结果
        final_df = merged_df[['Id', 'ranker_id', 'selected']]
        final_df.to_csv(output_file, index=False)
        
        print(f"合并完成，保存到: {output_file}")
        return output_file