"""
航班排名模型训练器 - 正确版本
use_full_data 控制每个segment内是否使用全部数据训练
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

from .Manager import FlightRankingModelsManager


def timer_clean(func):
    """计时装饰器"""
    def wrapper(*args, **kwargs):
        import time
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        if hasattr(args[0], 'logger'):
            args[0].logger.info(f"[TIMER] {func.__name__}: {duration:.2f}s")
        return result
    return wrapper


class FlightRankingTrainer:
    """航班排名训练器 - 正确理解需求版本"""
    
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
        
        # 关键配置：是否在segment内使用全部数据训练
        self.use_full_data = training_config.get('use_full_data', False)
        
        # 路径配置
        self.data_path = Path(config['paths']['model_input_dir'])
        self.model_save_path = Path(config['paths']['model_save_dir'])
        self.model_save_path.mkdir(parents=True, exist_ok=True)
        
        # 全量数据配置
        self.full_data_config = training_config.get('full_data_config', {})
        
        # 解释配置
        if self.use_full_data:
            mode_desc = "每个segment使用全部数据训练（无train/val划分）"
        else:
            mode_desc = "每个segment划分train/val，仅用train部分训练"
        
        self.logger.info(f"[INIT] 训练器初始化完成")
        self.logger.info(f"[INIT] 训练模式: {mode_desc}")
        self.logger.info(f"[INIT] 处理segments: {self.segments}")
    
    def _optimize_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """内存优化"""
        initial_memory = df.memory_usage(deep=True).sum() / 1024**2
        
        # 优化整数类型
        for col in df.select_dtypes(include=['int']).columns:
            col_min, col_max = df[col].min(), df[col].max()
            if col_min >= 0:
                if col_max < 255:
                    df[col] = df[col].astype(np.uint8)
                elif col_max < 65535:
                    df[col] = df[col].astype(np.uint16)
            else:
                if col_min > np.iinfo(np.int16).min and col_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif col_min > np.iinfo(np.int32).min and col_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
        
        # 优化浮点类型
        for col in df.select_dtypes(include=['float']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
        
        final_memory = df.memory_usage(deep=True).sum() / 1024**2
        reduction = (initial_memory - final_memory) / initial_memory * 100
        self.logger.info(f"[MEMORY] 优化: {initial_memory:.1f}MB -> {final_memory:.1f}MB (-{reduction:.1f}%)")
        
        return df
    
    def _create_train_val_split_fast(self, groups: np.ndarray, test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
        """快速数据划分"""
        self.logger.info(f"[SPLIT] 开始数据划分: {len(groups):,} 样本")
        start_time = time.time()
        
        unique_groups, inverse_indices = np.unique(groups, return_inverse=True)
        n_groups = len(unique_groups)
        
        n_val = max(1, int(n_groups * test_size))
        np.random.seed(self.random_state)
        
        all_indices = np.arange(n_groups)
        np.random.shuffle(all_indices)
        val_group_indices = all_indices[:n_val]
        
        is_val_group = np.zeros(n_groups, dtype=bool)
        is_val_group[val_group_indices] = True
        
        val_mask = is_val_group[inverse_indices]
        train_mask = ~val_mask
        
        train_indices = np.where(train_mask)[0]
        val_indices = np.where(val_mask)[0]
        
        split_time = time.time() - start_time
        self.logger.info(f"[SPLIT] 完成: train={len(train_indices):,}, val={len(val_indices):,}, 耗时={split_time:.2f}s")
        
        return train_indices, val_indices
    
    @timer_clean
    def train_segment(self, segment_id: int) -> Dict:
        """训练单个数据段 - 根据use_full_data配置决定使用多少数据"""
        self.logger.info(f"[SEGMENT] ========== 开始训练 segment_{segment_id} ==========")
        start_time = time.time()
        
        # 1. 内存预清理
        gc.collect()
        
        # 2. 加载数据
        self.logger.info("[DATA] 加载训练数据")
        train_file = self.data_path / "train" / f"train_segment_{segment_id}.parquet"
        if not train_file.exists():
            raise FileNotFoundError(f"训练文件不存在: {train_file}")
        
        load_start = time.time()
        df = pd.read_parquet(train_file)
        load_time = time.time() - load_start
        self.logger.info(f"[DATA] 加载完成: shape={df.shape}, 耗时={load_time:.2f}s")
        
        # 3. 内存优化
        df = self._optimize_memory(df)
        
        # 4. 数据预处理
        self.logger.info("[PREP] 开始数据预处理")
        prep_start = time.time()
        
        models_manager = FlightRankingModelsManager(use_gpu=self.use_gpu, logger=self.logger)
        X, y, groups, feature_cols, _ = models_manager.prepare_data(df)
        
        del df
        gc.collect()
        
        prep_time = time.time() - prep_start
        self.logger.info(f"[PREP] 预处理完成: 特征数={len(feature_cols)}, 耗时={prep_time:.2f}s")
        
        # 5. 根据use_full_data决定训练策略
        validation_scores = {}
        trained_models = {}
        
        if self.use_full_data:
            # 使用全部数据训练，无验证
            self.logger.info("[STRATEGY] 使用全部数据训练模式（无train/val划分）")
            
            # 直接用全部数据训练
            X_train, y_train, groups_train = X, y, groups
            X_val, y_val = None, None  # 无验证集
            
            self.logger.info(f"[TRAIN] 开始全量训练: {X_train.shape[0]:,} 样本")
            
        else:
            # 划分train/val，只用train部分训练
            self.logger.info("[STRATEGY] 使用train/val划分模式")
            
            train_idx, val_idx = self._create_train_val_split_fast(groups)
            X_train, y_train, groups_train = X[train_idx], y[train_idx], groups[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            
            self.logger.info(f"[TRAIN] 开始训练: {X_train.shape[0]:,} 样本（验证集: {X_val.shape[0]:,} 样本）")
        
        # 6. 创建并训练模型
        model_start = time.time()
        input_dim = X.shape[1]
        models_manager.create_models(input_dim, self.model_configs, self.model_names)
        
        for i, model_name in enumerate(self.model_names, 1):
            if model_name not in models_manager.models:
                continue
            
            self.logger.info(f"[TRAIN] 训练模型 {i}/{len(self.model_names)}: {model_name}")
            
            try:
                train_model_start = time.time()
                
                # 获取训练参数
                training_kwargs = self._get_training_kwargs(model_name)
                
                # 训练模型
                result_models = models_manager.train_models(
                    X_train, y_train, groups_train, [model_name], **training_kwargs
                )
                
                if model_name in result_models:
                    trained_model = result_models[model_name]
                    trained_models[model_name] = trained_model
                    
                    # 如果有验证集，计算验证分数
                    if X_val is not None and y_val is not None:
                        val_pred = trained_model.predict(X_val)
                        val_score = ndcg_score([y_val], [val_pred], k=10)
                        validation_scores[model_name] = val_score
                        
                        train_model_time = time.time() - train_model_start
                        self.logger.info(f"[TRAIN] {model_name} 完成: NDCG@10={val_score:.4f}, 耗时={train_model_time:.2f}s")
                    else:
                        # 全量数据模式，无验证分数
                        validation_scores[model_name] = 0.0  # 占位符
                        
                        train_model_time = time.time() - train_model_start
                        self.logger.info(f"[TRAIN] {model_name} 完成（全量数据，无验证）, 耗时={train_model_time:.2f}s")
                
            except Exception as e:
                train_model_time = time.time() - train_model_start if 'train_model_start' in locals() else 0
                self.logger.error(f"[TRAIN] {model_name} 失败: {e}, 耗时={train_model_time:.2f}s")
                continue
        
        model_time = time.time() - model_start
        self.logger.info(f"[TRAIN] 全部模型完成: 成功={len(trained_models)}/{len(self.model_names)}, 耗时={model_time:.2f}s")
        
        # 7. 清理内存
        del X, y, groups, X_train, y_train, groups_train
        if X_val is not None:
            del X_val, y_val
        gc.collect()
        
        # 8. 组织结果
        results = {
            'segment_id': segment_id,
            'models': trained_models,
            'validation_scores': validation_scores,
            'training_time': time.time() - start_time,
            'feature_names': feature_cols,
            'n_rankers': len(trained_models),
            'use_full_data': self.use_full_data  # 记录使用的训练模式
        }
        
        # 9. 保存结果
        self._save_segment_results(segment_id, results)
        
        total_time = time.time() - start_time
        self.logger.info(f"[SEGMENT] ========== segment_{segment_id} 完成: 总耗时={total_time:.1f}s ==========")
        
        return results
    
    def train_all_segments(self) -> Dict:
        """训练所有数据段"""
        self.logger.info(f"[BATCH] ========== 开始批量训练 {len(self.segments)} 个段 ==========")
        
        if self.use_full_data:
            self.logger.info("[BATCH] 模式: 每个segment使用全部数据训练")
        else:
            self.logger.info("[BATCH] 模式: 每个segment划分train/val后训练")
        
        all_results = {}
        total_start = time.time()
        
        for i, segment_id in enumerate(self.segments, 1):
            self.logger.info(f"[BATCH] 处理段 {i}/{len(self.segments)}: segment_{segment_id}")
            
            try:
                results = self.train_segment(segment_id)
                all_results[f'segment_{segment_id}'] = results
                
                # 段间内存清理
                gc.collect()
                
            except Exception as e:
                self.logger.error(f"[BATCH] segment_{segment_id} 失败: {e}")
                continue
        
        total_time = time.time() - total_start
        success_count = len(all_results)
        self.logger.info(f"[BATCH] ========== 批量训练完成: 成功={success_count}/{len(self.segments)}, 总耗时={total_time:.1f}s ==========")
        
        return all_results
    
    def _get_training_kwargs(self, model_name: str) -> Dict:
        """获取训练参数"""
        pytorch_models = {'RankNet', 'TransformerRanker', 'NeuralRanker'}
        
        if model_name in pytorch_models:
            # 根据是否使用全量数据调整训练参数
            if self.use_full_data:
                return {
                    'epochs': self.full_data_config.get('final_epochs', 50),
                    'batch_size': self.full_data_config.get('final_batch_size', 2048)
                }
            else:
                return {
                    'epochs': self.full_data_config.get('pytorch_epochs', 30),
                    'batch_size': self.full_data_config.get('pytorch_batch_size', 1024)
                }
        return {}
    
    def _save_segment_results(self, segment_id: int, results: Dict):
        """保存分段训练结果"""
        segment_dir = self.model_save_path / f"segment_{segment_id}"
        segment_dir.mkdir(exist_ok=True)
        
        # 保存模型
        models_manager = FlightRankingModelsManager(logger=self.logger)
        models_manager.models = results['models']
        models_manager.feature_names = results['feature_names']
        models_manager.save_models(str(segment_dir))
        
        # 保存训练报告
        report = {
            'segment_id': segment_id,
            'validation_scores': results['validation_scores'],
            'training_time': results['training_time'],
            'n_rankers': results['n_rankers'],
            'model_count': len(results['models']),
            'use_full_data': results['use_full_data'],
            'training_strategy': 'full_segment_data' if results['use_full_data'] else 'train_val_split'
        }
        
        with open(segment_dir / "training_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"[SAVE] segment_{segment_id} 结果已保存")
        
        # 输出关键信息
        if results['use_full_data']:
            self.logger.info(f"[INFO] segment_{segment_id} 使用全量数据训练，共{len(results['models'])}个模型")
        else:
            best_model = max(results['validation_scores'].items(), key=lambda x: x[1]) if results['validation_scores'] else None
            if best_model:
                self.logger.info(f"[INFO] segment_{segment_id} 最佳模型: {best_model[0]} (NDCG@10={best_model[1]:.4f})")