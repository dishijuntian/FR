"""
自动调参模块 - 重构版

专注于：
- 超参数优化
- 贝叶斯优化
- 评估指标计算

作者: Flight Ranking Team
版本: 4.0 (重构版)
"""

import optuna
import numpy as np
import torch
from typing import Dict, List, Any, Optional, Callable
from sklearn.metrics import ndcg_score
import warnings
import gc

warnings.filterwarnings('ignore')


class MetricsCalculator:
    """评估指标计算器"""
    
    @staticmethod
    def calculate_hit_rate(y_true: np.ndarray, y_pred: np.ndarray, 
                          group_sizes: List[int], k: int = 3) -> float:
        """计算Hit Rate@K"""
        hits = 0
        total_groups = 0
        start_idx = 0
        
        for group_size in group_sizes:
            end_idx = start_idx + group_size
            
            group_pred = y_pred[start_idx:end_idx]
            group_true = y_true[start_idx:end_idx]
            
            positive_indices = np.where(group_true == 1)[0]
            
            if len(positive_indices) > 0:
                top_k_indices = np.argsort(group_pred)[-k:]
                if any(idx in positive_indices for idx in top_k_indices):
                    hits += 1
            
            total_groups += 1
            start_idx = end_idx
        
        return hits / total_groups if total_groups > 0 else 0.0
    
    @staticmethod
    def calculate_ndcg(y_true: np.ndarray, y_pred: np.ndarray, 
                      group_sizes: List[int], k: int = 5) -> float:
        """计算NDCG@K"""
        ndcg_scores = []
        start_idx = 0
        
        for group_size in group_sizes:
            if group_size >= k:
                end_idx = start_idx + group_size
                group_true = y_true[start_idx:end_idx].reshape(1, -1)
                group_pred = y_pred[start_idx:end_idx].reshape(1, -1)
                
                try:
                    ndcg = ndcg_score(group_true, group_pred, k=k)
                    ndcg_scores.append(ndcg)
                except:
                    pass
            
            start_idx += group_size
        
        return np.mean(ndcg_scores) if ndcg_scores else 0.0


class AutoTuner:
    """自动调参器 - 简化版"""
    
    def __init__(self, model_name: str, search_space: Dict[str, Any],
                 n_trials: int = 50, objective_metric: str = 'hit_rate'):
        """
        初始化调参器
        
        Args:
            model_name: 模型名称
            search_space: 搜索空间
            n_trials: 试验次数
            objective_metric: 目标指标
        """
        self.model_name = model_name
        self.search_space = search_space
        self.n_trials = n_trials
        self.objective_metric = objective_metric
        self.metrics_calc = MetricsCalculator()
        
        # 存储训练数据
        self.X_train = None
        self.y_train = None
        self.train_groups = None
        self.X_val = None
        self.y_val = None
        self.val_groups = None
        
        self.study = None
        self.best_params = None
    
    def optimize(self, X_train: np.ndarray, y_train: np.ndarray,
                 train_groups: List[int], X_val: np.ndarray,
                 y_val: np.ndarray, val_groups: List[int]) -> Dict[str, Any]:
        """
        执行超参数优化
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            train_groups: 训练组大小
            X_val: 验证特征
            y_val: 验证标签
            val_groups: 验证组大小
            
        Returns:
            Dict: 最优参数
        """
        # 存储数据
        self.X_train = X_train
        self.y_train = y_train
        self.train_groups = train_groups
        self.X_val = X_val
        self.y_val = y_val
        self.val_groups = val_groups
        
        # 创建优化研究
        self.study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        # 执行优化
        self.study.optimize(
            self._objective,
            n_trials=self.n_trials,
            show_progress_bar=True
        )
        
        self.best_params = self.study.best_params
        return self.best_params
    
    def _objective(self, trial: optuna.Trial) -> float:
        """目标函数"""
        try:
            # 建议参数
            params = self._suggest_params(trial)
            
            # 创建并训练模型
            from models import ModelFactory
            
            if self.model_name in ['NeuralRanker', 'RankNet', 'TransformerRanker']:
                model = ModelFactory.create_model(
                    self.model_name,
                    input_dim=self.X_train.shape[1],
                    **params
                )
                # PyTorch模型使用较少epochs进行快速调优
                model.fit(self.X_train, self.y_train, self.train_groups, epochs=5)
            else:
                model = ModelFactory.create_model(self.model_name, **params)
                model.fit(self.X_train, self.y_train, self.train_groups)
            
            # 预测和评估
            y_pred = model.predict(self.X_val)
            
            if self.objective_metric == 'hit_rate':
                score = self.metrics_calc.calculate_hit_rate(
                    self.y_val, y_pred, self.val_groups
                )
            else:
                score = self.metrics_calc.calculate_ndcg(
                    self.y_val, y_pred, self.val_groups
                )
            
            # 清理内存
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return score
            
        except Exception as e:
            print(f"试验失败: {e}")
            return 0.0
    
    def _suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """根据搜索空间建议参数"""
        params = {}
        
        for param_name, param_config in self.search_space.items():
            if isinstance(param_config, list):
                # 分类参数
                params[param_name] = trial.suggest_categorical(param_name, param_config)
            elif isinstance(param_config, dict):
                param_type = param_config.get('type', 'float')
                if param_type == 'int':
                    params[param_name] = trial.suggest_int(
                        param_name,
                        param_config['low'],
                        param_config['high']
                    )
                elif param_type == 'float':
                    if param_config.get('log', False):
                        params[param_name] = trial.suggest_float(
                            param_name,
                            param_config['low'],
                            param_config['high'],
                            log=True
                        )
                    else:
                        params[param_name] = trial.suggest_float(
                            param_name,
                            param_config['low'],
                            param_config['high']
                        )
        
        return params


def create_auto_tuner(model_name: str, search_space: Dict[str, Any] = None,
                     n_trials: int = 50) -> AutoTuner:
    """
    创建自动调参器
    
    Args:
        model_name: 模型名称
        search_space: 搜索空间
        n_trials: 试验次数
        
    Returns:
        AutoTuner: 调参器实例
    """
    # PyTorch模型使用较少试验次数
    if model_name in ['NeuralRanker', 'RankNet', 'TransformerRanker']:
        n_trials = min(n_trials, 30)
    
    return AutoTuner(
        model_name=model_name,
        search_space=search_space or {},
        n_trials=n_trials
    )