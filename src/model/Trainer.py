"""
航班排名模型训练器 - 修复版本
主要修复：
1. 修复全量数据模式下的模型保存路径问题
2. 统一训练和预测的路径逻辑
3. 添加更好的错误处理和回退机制
"""

import os
import time
import logging
import gc
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import ndcg_score

# 直接导入优化的模型创建函数
from .Models import create_model_fast


class FlightRankingTrainer:
    """航班排名训练器 - 修复版本"""
    
    def __init__(self, config: Dict, logger=None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # 训练配置
        training_config = config.get('training', {})
        self.segments = training_config.get('segments', [0, 1, 2])
        self.model_names = training_config.get('model_names', ['XGBRanker', 'LGBMRanker'])
        self.use_gpu = training_config.get('use_gpu', True)
        self.random_state = training_config.get('random_state', 42)
        self.model_configs = training_config.get('model_configs', {})
        self.use_full_data = training_config.get('use_full_data', False)
        
        # 路径配置
        self.data_path = Path(config['paths']['model_input_dir'])
        self.model_save_path = Path(config['paths']['model_save_dir'])
        self.model_save_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"[INIT] 快速训练器初始化完成")
        self.logger.info(f"[INIT] 处理segments: {self.segments}")
        self.logger.info(f"[INIT] 全量数据模式: {self.use_full_data}")
    
    def prepare_data_fast(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """快速数据预处理 - 最小化转换"""
        # 1. 快速特征选择
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = {'Id', 'selected', 'ranker_id', 'profileId', 'companyID'}
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # 2. 简单缺失值处理
        df_work = df[feature_cols + ['selected', 'ranker_id']].copy()
        df_work[feature_cols] = df_work[feature_cols].fillna(0)
        
        # 3. 直接转换为数组
        X = df_work[feature_cols].values.astype(np.float32)
        y = df_work['selected'].values.astype(np.float32)
        groups = df_work['ranker_id'].values
        
        return X, y, groups, feature_cols
    
    def train_segment_fast(self, segment_id: int) -> Dict:
        """快速训练单个数据段"""
        self.logger.info(f"[SEGMENT] 开始训练 segment_{segment_id}")
        start_time = time.time()
        
        # 1. 加载数据
        train_file = self.data_path / "train" / f"train_segment_{segment_id}.parquet"
        if not train_file.exists():
            raise FileNotFoundError(f"训练文件不存在: {train_file}")
        
        df = pd.read_parquet(train_file)
        self.logger.info(f"[DATA] 加载完成: shape={df.shape}")
        
        # 2. 快速数据预处理
        X, y, groups, feature_cols = self.prepare_data_fast(df)
        del df  # 立即释放内存
        gc.collect()
        
        # 3. 根据use_full_data决定训练策略
        validation_scores = {}
        trained_models = {}
        
        if self.use_full_data:
            # 使用全部数据训练
            X_train, y_train, groups_train = X, y, groups
            X_val, y_val = None, None
            self.logger.info(f"[TRAIN] 全量训练模式: {X_train.shape[0]:,} 样本")
        else:
            # 快速划分train/val
            unique_groups = np.unique(groups)
            train_groups, val_groups = train_test_split(
                unique_groups, test_size=0.2, random_state=self.random_state
            )
            
            train_mask = np.isin(groups, train_groups)
            val_mask = np.isin(groups, val_groups)
            
            X_train, y_train, groups_train = X[train_mask], y[train_mask], groups[train_mask]
            X_val, y_val = X[val_mask], y[val_mask]
            
            self.logger.info(f"[TRAIN] 划分训练: {X_train.shape[0]:,} 样本，验证: {X_val.shape[0]:,} 样本")
        
        # 4. 直接训练模型 - 避免管理器开销
        for model_name in self.model_names:
            try:
                self.logger.info(f"[TRAIN] 训练模型: {model_name}")
                train_start = time.time()
                
                # 获取模型参数
                model_params = self.model_configs.get(model_name, {})
                
                # 直接创建模型
                if model_name in ['RankNet', 'NeuralRanker', 'TransformerRanker']:
                    model = create_model_fast(
                        model_name, 
                        use_gpu=self.use_gpu,
                        input_dim=X.shape[1],
                        **model_params
                    )
                else:
                    model = create_model_fast(
                        model_name, 
                        use_gpu=self.use_gpu,
                        **model_params
                    )
                
                # 训练模型
                model.fit(X_train, y_train, groups_train)
                trained_models[model_name] = model
                
                # 如果有验证集，计算验证分数
                if X_val is not None and y_val is not None:
                    val_pred = model.predict(X_val)
                    val_score = ndcg_score([y_val], [val_pred], k=10)
                    validation_scores[model_name] = val_score
                else:
                    validation_scores[model_name] = 0.0
                
                train_time = time.time() - train_start
                score_info = f"NDCG@10={validation_scores[model_name]:.4f}" if validation_scores[model_name] > 0 else "全量模式"
                self.logger.info(f"[TRAIN] {model_name} 完成: {score_info}, 耗时={train_time:.2f}s")
                
            except Exception as e:
                self.logger.error(f"[TRAIN] {model_name} 失败: {e}")
                continue
        
        # 5. 清理内存
        del X, y, groups, X_train, y_train, groups_train
        if X_val is not None:
            del X_val, y_val
        gc.collect()
        
        # 6. 保存结果
        results = {
            'segment_id': segment_id,
            'models': trained_models,
            'validation_scores': validation_scores,
            'training_time': time.time() - start_time,
            'feature_names': feature_cols,
            'n_rankers': len(trained_models),
            'use_full_data': self.use_full_data
        }
        
        self.save_segment_results_fast(segment_id, results)
        
        total_time = time.time() - start_time
        self.logger.info(f"[SEGMENT] segment_{segment_id} 完成: 总耗时={total_time:.1f}s")
        
        return results
    
    def save_segment_results_fast(self, segment_id: int, results: Dict):
        """快速保存分段训练结果 - 修复路径问题"""
        # 修复：根据use_full_data决定保存路径
        if self.use_full_data:
            # 全量数据模式：保存到统一目录
            segment_dir = self.model_save_path / "full_data"
            self.logger.info(f"[SAVE] 全量数据模式，保存到: {segment_dir}")
        else:
            # 分段模式：保存到各自目录
            segment_dir = self.model_save_path / f"segment_{segment_id}"
            self.logger.info(f"[SAVE] 分段模式，保存到: {segment_dir}")
        
        segment_dir.mkdir(exist_ok=True)
        
        # 保存模型 - 直接保存，避免管理器开销
        for model_name, model in results['models'].items():
            model_path = segment_dir / f"{model_name}.pkl"
            try:
                model.save_model(str(model_path))
                self.logger.info(f"[SAVE] {model_name} 模型已保存")
            except Exception as e:
                self.logger.warning(f"[SAVE] 保存{model_name}失败: {e}")
        
        # 保存特征名称
        import joblib
        features_path = segment_dir / "features.pkl"
        joblib.dump(results['feature_names'], features_path)
        self.logger.info(f"[SAVE] 特征名称已保存: {len(results['feature_names'])} 个特征")
        
        # 保存训练报告
        report = {
            'segment_id': segment_id,
            'validation_scores': results['validation_scores'],
            'training_time': results['training_time'],
            'n_rankers': results['n_rankers'],
            'model_count': len(results['models']),
            'use_full_data': results['use_full_data'],
            'feature_count': len(results['feature_names'])
        }
        
        report_path = segment_dir / "training_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"[SAVE] segment_{segment_id} 结果已保存")
    
    def train_all_segments(self) -> Dict:
        """训练所有数据段 - 支持全量数据模式"""
        if self.use_full_data:
            return self._train_full_data_unified()
        else:
            return self._train_segments_separately()
    
    def _train_full_data_unified(self) -> Dict:
        """统一全量数据训练 - 新增方法"""
        self.logger.info(f"[FULL] 开始全量数据统一训练")
        total_start = time.time()
        
        # 1. 加载所有段数据
        all_dfs = []
        total_samples = 0
        
        for segment_id in self.segments:
            train_file = self.data_path / "train" / f"train_segment_{segment_id}.parquet"
            if train_file.exists():
                df = pd.read_parquet(train_file)
                all_dfs.append(df)
                total_samples += len(df)
                self.logger.info(f"[FULL] 加载segment_{segment_id}: {len(df):,} 样本")
            else:
                self.logger.warning(f"[FULL] 跳过不存在的文件: {train_file}")
        
        if not all_dfs:
            raise ValueError("没有找到任何训练数据文件")
        
        # 2. 合并数据
        self.logger.info(f"[FULL] 合并 {len(all_dfs)} 个数据段，总样本: {total_samples:,}")
        combined_df = pd.concat(all_dfs, ignore_index=True)
        del all_dfs  # 释放内存
        gc.collect()
        
        # 3. 数据预处理
        X, y, groups, feature_cols = self.prepare_data_fast(combined_df)
        del combined_df
        gc.collect()
        
        # 4. 训练模型
        trained_models = {}
        validation_scores = {}
        
        for model_name in self.model_names:
            try:
                self.logger.info(f"[FULL] 训练模型: {model_name}")
                train_start = time.time()
                
                # 获取模型参数
                model_params = self.model_configs.get(model_name, {})
                
                # 创建模型
                if model_name in ['RankNet', 'NeuralRanker', 'TransformerRanker']:
                    model = create_model_fast(
                        model_name, 
                        use_gpu=self.use_gpu,
                        input_dim=X.shape[1],
                        **model_params
                    )
                else:
                    model = create_model_fast(
                        model_name, 
                        use_gpu=self.use_gpu,
                        **model_params
                    )
                
                # 训练模型
                model.fit(X, y, groups)
                trained_models[model_name] = model
                validation_scores[model_name] = 0.0  # 全量模式无验证分数
                
                train_time = time.time() - train_start
                self.logger.info(f"[FULL] {model_name} 完成: 全量模式, 耗时={train_time:.2f}s")
                
            except Exception as e:
                self.logger.error(f"[FULL] {model_name} 失败: {e}")
                continue
        
        # 5. 保存结果
        results = {
            'full_data': {
                'models': trained_models,
                'validation_scores': validation_scores,
                'training_time': time.time() - total_start,
                'feature_names': feature_cols,
                'n_rankers': len(trained_models),
                'use_full_data': True,
                'total_samples': len(X)
            }
        }
        
        self._save_full_data_results(results['full_data'])
        
        total_time = time.time() - total_start
        self.logger.info(f"[FULL] 全量数据训练完成: 总耗时={total_time:.1f}s")
        
        return results
    
    def _train_segments_separately(self) -> Dict:
        """分别训练各段 - 原有逻辑"""
        self.logger.info(f"[BATCH] 开始批量训练 {len(self.segments)} 个段")
        
        all_results = {}
        total_start = time.time()
        
        for i, segment_id in enumerate(self.segments, 1):
            self.logger.info(f"[BATCH] 处理段 {i}/{len(self.segments)}: segment_{segment_id}")
            
            try:
                results = self.train_segment_fast(segment_id)
                all_results[f'segment_{segment_id}'] = results
                
                # 段间内存清理
                gc.collect()
                
            except Exception as e:
                self.logger.error(f"[BATCH] segment_{segment_id} 失败: {e}")
                continue
        
        total_time = time.time() - total_start
        success_count = len(all_results)
        self.logger.info(f"[BATCH] 批量训练完成: 成功={success_count}/{len(self.segments)}, 总耗时={total_time:.1f}s")
        
        return all_results
    
    def _save_full_data_results(self, results: Dict):
        """保存全量数据结果"""
        full_data_dir = self.model_save_path / "full_data"
        full_data_dir.mkdir(exist_ok=True)
        
        # 保存模型
        for model_name, model in results['models'].items():
            model_path = full_data_dir / f"{model_name}.pkl"
            try:
                model.save_model(str(model_path))
                self.logger.info(f"[SAVE] 全量{model_name} 模型已保存")
            except Exception as e:
                self.logger.warning(f"[SAVE] 保存全量{model_name}失败: {e}")
        
        # 保存特征名称
        import joblib
        features_path = full_data_dir / "features.pkl"
        joblib.dump(results['feature_names'], features_path)
        
        # 保存训练报告
        report = {
            'mode': 'full_data',
            'validation_scores': results['validation_scores'],
            'training_time': results['training_time'],
            'n_rankers': results['n_rankers'],
            'model_count': len(results['models']),
            'use_full_data': True,
            'total_samples': results['total_samples'],
            'feature_count': len(results['feature_names'])
        }
        
        report_path = full_data_dir / "training_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"[SAVE] 全量数据结果已保存到: {full_data_dir}")


# 保持原有的快速数据处理函数
def prepare_ranking_data_fast(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """快速排序数据准备 - 模仿第二个文件夹的高效做法"""
    
    # 1. 清理数据 - 简化版本
    if 'selected' in df.columns:
        selected_per_group = df.groupby('ranker_id')['selected'].sum()
        invalid_groups = selected_per_group[selected_per_group != 1].index
        if len(invalid_groups) > 0:
            df = df[~df['ranker_id'].isin(invalid_groups)]
    
    # 2. 特征选择 - 直接处理
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = {'Id', 'selected', 'ranker_id', 'profileId', 'companyID'}
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    # 3. 处理缺失值 - 最简单的方式
    for col in feature_cols:
        if df[col].dtype in ['float32', 'float64']:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(0)
    
    # 4. 准备数据
    X = df[feature_cols].values.astype(np.float32)
    y = df['selected'].values.astype(np.float32)
    groups = df['ranker_id'].values
    
    # 5. 按组分割 - 快速版本
    unique_groups = np.unique(groups)
    train_groups, test_groups = train_test_split(
        unique_groups, test_size=test_size, random_state=random_state
    )
    
    train_mask = np.isin(groups, train_groups)
    test_mask = np.isin(groups, test_groups)
    
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    groups_train, groups_test = groups[train_mask], groups[test_mask]
    
    # 6. 计算组大小 - 快速版本
    def calculate_group_sizes_fast(group_array):
        unique, counts = np.unique(group_array, return_counts=True)
        return counts.tolist()
    
    train_group_sizes = calculate_group_sizes_fast(groups_train)
    test_group_sizes = calculate_group_sizes_fast(groups_test)
    
    test_info = df[test_mask][['ranker_id', 'selected']].copy()
    
    return (X_train, X_test, y_train, y_test, 
            train_group_sizes, test_group_sizes, feature_cols, test_info)