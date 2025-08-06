"""
优化后的航班排名模型训练器 - 重构版
专注于训练逻辑，移除重复的数据加载和配置逻辑
"""

import os
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import ndcg_score

from .Manager import FlightRankingModelsManager


def timer(func):
    """简单的计时装饰器"""
    def wrapper(*args, **kwargs):
        import time
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        if hasattr(args[0], 'logger'):
            args[0].logger.info(f"{func.__name__} 耗时: {duration:.2f}s")
        return result
    return wrapper


class FlightRankingTrainer:
    """航班排名训练器 - 重构版，专注于训练协调"""
    
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
        
        # 全量数据配置
        self.full_data_config = training_config.get('full_data_config', {})
        
        mode = "全量数据" if self.use_full_data else "分段数据"
        self.logger.info(f"训练器初始化完成 - {mode}模式")
    
    @timer
    def train_segment(self, segment_id: int) -> Dict:
        """训练单个数据段"""
        self.logger.info(f"开始训练 segment_{segment_id}")
        start_time = time.time()
        
        # 加载数据
        train_file = self.data_path / "train" / f"train_segment_{segment_id}.parquet"
        if not train_file.exists():
            raise FileNotFoundError(f"训练文件不存在: {train_file}")
        
        df = pd.read_parquet(train_file)
        
        # 数据预处理和模型训练
        models_manager = FlightRankingModelsManager(use_gpu=self.use_gpu, logger=self.logger)
        X, y, groups, feature_cols, _ = models_manager.prepare_data(df)
        
        # 创建训练验证集划分
        train_idx, val_idx = self._create_train_val_split(groups)
        X_train, y_train, groups_train = X[train_idx], y[train_idx], groups[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        # 创建并训练模型
        input_dim = X.shape[1]
        models_manager.create_models(input_dim, self.model_configs, self.model_names)
        
        # 训练和验证
        validation_scores = {}
        final_models = {}
        
        for model_name in self.model_names:
            if model_name not in models_manager.models:
                continue
            
            try:
                # 训练模型
                trained_models = models_manager.train_models(
                    X_train, y_train, groups_train, [model_name]
                )
                
                if model_name in trained_models:
                    # 验证模型
                    val_pred = trained_models[model_name].predict(X_val)
                    val_score = ndcg_score([y_val], [val_pred], k=10)
                    validation_scores[model_name] = val_score
                    
                    # 重新训练最终模型（使用全部数据）
                    final_model = self._train_final_model(
                        models_manager, model_name, X, y, groups
                    )
                    final_models[model_name] = final_model
                    
                    self.logger.info(f"✓ {model_name} - NDCG@10: {val_score:.4f}")
                
            except Exception as e:
                self.logger.error(f"✗ {model_name} 训练失败: {e}")
                continue
        
        # 组织结果
        results = {
            'segment_id': segment_id,
            'models': final_models,
            'validation_scores': validation_scores,
            'training_time': time.time() - start_time,
            'feature_names': feature_cols,
            'n_rankers': len(np.unique(groups))
        }
        
        # 保存结果
        self._save_segment_results(segment_id, results)
        
        self.logger.info(f"✓ segment_{segment_id} 训练完成 (时间: {results['training_time']:.1f}s)")
        return results
    
    @timer
    def train_full_data_mode(self) -> Dict:
        """全量数据训练模式"""
        self.logger.info("开始全量数据训练模式")
        start_time = time.time()
        
        # 加载全量数据
        df = self._load_full_data()
        
        # 数据预处理
        models_manager = FlightRankingModelsManager(use_gpu=self.use_gpu, logger=self.logger)
        X, y, groups, feature_cols, _ = models_manager.prepare_data(df, use_full_data=True)
        
        # 创建训练验证集划分
        train_idx, val_idx = self._create_train_val_split(groups)
        X_train, y_train, groups_train = X[train_idx], y[train_idx], groups[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        # 创建并训练模型
        input_dim = X.shape[1]
        models_manager.create_models(input_dim, self.model_configs, self.model_names)
        
        # 训练和验证
        validation_scores = {}
        final_models = {}
        
        for model_name in self.model_names:
            if model_name not in models_manager.models:
                continue
            
            try:
                # 训练模型（全量数据配置）
                training_kwargs = self._get_full_data_training_kwargs(model_name)
                trained_models = models_manager.train_models(
                    X_train, y_train, groups_train, [model_name], **training_kwargs
                )
                
                if model_name in trained_models:
                    # 验证模型
                    val_pred = trained_models[model_name].predict(X_val)
                    val_score = ndcg_score([y_val], [val_pred], k=10)
                    validation_scores[model_name] = val_score
                    
                    # 重新训练最终模型
                    final_model = self._train_final_model(
                        models_manager, model_name, X, y, groups, use_full_data=True
                    )
                    final_models[model_name] = final_model
                    
                    self.logger.info(f"✓ {model_name} - NDCG@10: {val_score:.4f}")
                
            except Exception as e:
                self.logger.error(f"✗ {model_name} 训练失败: {e}")
                continue
        
        # 组织结果
        results = {
            'training_mode': 'full_data',
            'models': final_models,
            'validation_scores': validation_scores,
            'training_time': time.time() - start_time,
            'feature_names': feature_cols,
            'n_rankers': len(np.unique(groups))
        }
        
        # 保存结果
        self._save_full_data_results(results)
        
        self.logger.info(f"✓ 全量数据训练完成 (时间: {results['training_time']:.1f}s)")
        return results
    
    def train_all_segments(self) -> Dict:
        """训练所有数据段或全量数据"""
        if self.use_full_data:
            results = self.train_full_data_mode()
            return {'full_data': results}
        else:
            all_results = {}
            for segment_id in self.segments:
                try:
                    results = self.train_segment(segment_id)
                    all_results[f'segment_{segment_id}'] = results
                except Exception as e:
                    self.logger.error(f"✗ segment_{segment_id} 训练失败: {e}")
                    continue
            return all_results
    
    def _load_full_data(self) -> pd.DataFrame:
        """加载全量数据"""
        # 查找训练文件
        train_dir = self.data_path / "train"
        if train_dir.exists():
            train_files = sorted(train_dir.glob("*.parquet"))
            if train_files:
                dfs = [pd.read_parquet(f) for f in train_files]
                return pd.concat(dfs, ignore_index=True)
        
        # 尝试单个文件
        possible_files = [
            self.data_path / "train.parquet",
            self.data_path / "training_data.parquet"
        ]
        
        for file_path in possible_files:
            if file_path.exists():
                return pd.read_parquet(file_path)
        
        raise FileNotFoundError("未找到训练数据文件")
    
    def _create_train_val_split(self, groups: np.ndarray, test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
        """创建训练验证集划分"""
        unique_groups = np.unique(groups)
        train_groups, val_groups = train_test_split(
            unique_groups, test_size=test_size, random_state=self.random_state
        )
        
        train_mask = np.isin(groups, train_groups)
        val_mask = np.isin(groups, val_groups)
        
        return np.where(train_mask)[0], np.where(val_mask)[0]
    
    def _get_full_data_training_kwargs(self, model_name: str) -> Dict:
        """获取全量数据训练参数"""
        pytorch_models = {'RankNet', 'TransformerRanker', 'NeuralRanker'}
        
        if model_name in pytorch_models:
            return {
                'epochs': self.full_data_config.get('pytorch_epochs', 20),
                'batch_size': self.full_data_config.get('pytorch_batch_size', 2048)
            }
        return {}
    
    def _train_final_model(self, models_manager: FlightRankingModelsManager, 
                          model_name: str, X: np.ndarray, y: np.ndarray, 
                          groups: np.ndarray, use_full_data: bool = False) -> object:
        """训练最终模型"""
        # 创建新的模型实例
        input_dim = X.shape[1]
        final_models = models_manager.create_models(input_dim, self.model_configs, [model_name])
        final_model = final_models[model_name]
        
        # 训练参数
        if use_full_data:
            training_kwargs = self._get_full_data_training_kwargs(model_name)
        else:
            training_kwargs = {}
        
        # 训练
        models_manager.models = {model_name: final_model}
        trained = models_manager.train_models(X, y, groups, [model_name], **training_kwargs)
        
        return trained[model_name]
    
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
            'model_count': len(results['models'])
        }
        
        with open(segment_dir / "training_report.json", 'w') as f:
            json.dump(report, f, indent=2)
    
    def _save_full_data_results(self, results: Dict):
        """保存全量数据训练结果"""
        full_data_dir = self.model_save_path / "full_data"
        full_data_dir.mkdir(exist_ok=True)
        
        # 保存模型
        models_manager = FlightRankingModelsManager(logger=self.logger)
        models_manager.models = results['models']
        models_manager.feature_names = results['feature_names']
        models_manager.save_models(str(full_data_dir))
        
        # 保存训练报告
        report = {
            'training_mode': 'full_data',
            'validation_scores': results['validation_scores'],
            'training_time': results['training_time'],
            'n_rankers': results['n_rankers'],
            'model_count': len(results['models'])
        }
        
        with open(full_data_dir / "training_report.json", 'w') as f:
            json.dump(report, f, indent=2)