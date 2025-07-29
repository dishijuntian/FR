<<<<<<< HEAD
"""
高效GPU加速航班排名模型训练器
支持多进程、GPU加速、自动调参和集成训练
"""

import os
import time
import logging
import warnings
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pickle
import json

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from sklearn.metrics import ndcg_score
import optuna
from joblib import Parallel, delayed

# 导入模型
from .Models import FlightRankingModels, BaseRankingModel

warnings.filterwarnings('ignore')


class GPUTrainingAccelerator:
    """GPU训练加速器"""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gpu_available = torch.cuda.is_available()
        
        if self.gpu_available:
            self.logger.info(f"GPU加速可用: {torch.cuda.get_device_name(0)}")
            self.logger.info(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        else:
            self.logger.info("GPU不可用，使用CPU模式")
    
    def optimize_gpu_settings(self):
        """优化GPU设置"""
        if not self.gpu_available:
            return
        
        try:
            # 启用混合精度训练
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            # 设置内存策略
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            
            self.logger.info("GPU设置已优化")
        except Exception as e:
            self.logger.warning(f"GPU优化失败: {e}")
    
    def get_optimal_batch_size(self, model_size: str = 'medium') -> int:
        """根据GPU内存自动设置批次大小"""
        if not self.gpu_available:
            return 512
        
        try:
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            # 根据GPU内存和模型大小估算最优批次大小
            size_multipliers = {'small': 1.0, 'medium': 0.5, 'large': 0.25}
            multiplier = size_multipliers.get(model_size, 0.5)
            
            if gpu_memory_gb >= 12:
                base_batch_size = 2048
            elif gpu_memory_gb >= 8:
                base_batch_size = 1024
            elif gpu_memory_gb >= 6:
                base_batch_size = 512
            else:
                base_batch_size = 256
            
            optimal_size = int(base_batch_size * multiplier)
            self.logger.info(f"推荐批次大小: {optimal_size}")
            return optimal_size
            
        except Exception as e:
            self.logger.warning(f"批次大小优化失败: {e}")
            return 512


class ParallelTrainingManager:
    """智能并行训练管理器"""
    
    def __init__(self, use_gpu: bool, n_jobs: int = -1, logger=None):
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.logger = logger or logging.getLogger(__name__)
        
        if self.use_gpu:
            # GPU模式：使用单进程避免GPU竞争
            self.n_jobs = 1
            self.backend = 'sequential'
            self.logger.info("GPU模式: 使用单进程训练避免GPU竞争")
        else:
            # CPU模式：使用多进程加速
            self.n_jobs = n_jobs if n_jobs != -1 else mp.cpu_count()
            self.backend = 'loky'
            self.logger.info(f"CPU模式: 使用 {self.n_jobs} 进程并行训练")
    
    def train_folds(self, train_func, fold_data: List[Tuple], **kwargs) -> List:
        """智能训练fold"""
        if self.use_gpu:
            # GPU模式：串行训练避免资源竞争
            return self._sequential_train_folds(train_func, fold_data, **kwargs)
        else:
            # CPU模式：并行训练
            return self._parallel_train_folds(train_func, fold_data, **kwargs)
    
    def _sequential_train_folds(self, train_func, fold_data: List[Tuple], **kwargs) -> List:
        """串行训练fold（GPU模式）"""
        self.logger.info(f"GPU串行训练 {len(fold_data)} 个fold")
        
        results = []
        for i, (train_idx, val_idx) in enumerate(fold_data):
            try:
                result = train_func(i, train_idx, val_idx, **kwargs)
                if result:
                    results.append(result)
                    self.logger.info(f"✓ Fold {i+1}/{len(fold_data)} 完成")
            except Exception as e:
                self.logger.error(f"✗ Fold {i+1} 训练失败: {e}")
                continue
        
        self.logger.info(f"成功训练: {len(results)}/{len(fold_data)} fold")
        return results
    
    def _parallel_train_folds(self, train_func, fold_data: List[Tuple], **kwargs) -> List:
        """并行训练fold（CPU模式）"""
        self.logger.info(f"CPU并行训练 {len(fold_data)} 个fold")
        
        def train_single_fold(fold_idx, train_idx, val_idx):
            try:
                return train_func(fold_idx, train_idx, val_idx, **kwargs)
            except Exception as e:
                self.logger.error(f"Fold {fold_idx} 训练失败: {e}")
                return None
        
        # 并行执行
        results = Parallel(n_jobs=self.n_jobs, backend=self.backend)(
            delayed(train_single_fold)(i, train_idx, val_idx) 
            for i, (train_idx, val_idx) in enumerate(fold_data)
        )
        
        successful_results = [r for r in results if r is not None]
        self.logger.info(f"成功训练: {len(successful_results)}/{len(fold_data)} fold")
        return successful_results


class HyperparameterOptimizer:
    """超参数优化器"""
    
    def __init__(self, n_trials: int = 20, direction: str = 'maximize', logger=None):
        self.n_trials = n_trials
        self.direction = direction
        self.logger = logger or logging.getLogger(__name__)
        self.study = None
    
    def create_study(self, study_name: str = "flight_ranking_optimization"):
        """创建优化研究"""
        self.study = optuna.create_study(
            direction=self.direction,
            study_name=study_name,
            sampler=optuna.samplers.TPESampler()
        )
        self.logger.info(f"创建优化研究: {study_name}")
    
    def suggest_xgb_params(self, trial) -> Dict:
        """建议XGBoost参数"""
        return {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        }
    
    def suggest_lgb_params(self, trial) -> Dict:
        """建议LightGBM参数"""
        return {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'num_leaves': trial.suggest_int('num_leaves', 10, 300),
        }
    
    def suggest_neural_params(self, trial) -> Dict:
        """建议神经网络参数"""
        n_layers = trial.suggest_int('n_layers', 2, 4)
        hidden_dims = []
        for i in range(n_layers):
            dim = trial.suggest_int(f'hidden_dim_{i}', 32, 256)
            hidden_dims.append(dim)
        
        return {
            'hidden_dims': hidden_dims,
            'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.01),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
        }
    
    def optimize(self, objective_func, **kwargs):
        """执行优化"""
        if self.study is None:
            self.create_study()
        
        self.logger.info(f"开始超参数优化，trials: {self.n_trials}")
        
        self.study.optimize(
            lambda trial: objective_func(trial, **kwargs),
            n_trials=self.n_trials,
            show_progress_bar=True
        )
        
        self.logger.info(f"优化完成，最佳得分: {self.study.best_value:.4f}")
        self.logger.info(f"最佳参数: {self.study.best_params}")
        
        return self.study.best_params, self.study.best_value


class FlightRankingTrainer:
    """航班排名高效训练器"""
    
    def __init__(self, 
                 data_path: str = "data/aeroclub-recsys-2025",
                 model_save_path: str = "models",
                 use_gpu: bool = True,
                 enable_parallel: bool = True,
                 enable_optimization: bool = True,
                 n_folds: int = 5,
                 random_state: int = 42,
                 logger=None):
        
        self.data_path = Path(data_path)
        self.model_save_path = Path(model_save_path)
        self.use_gpu = use_gpu
        self.enable_parallel = enable_parallel
        self.enable_optimization = enable_optimization
        self.n_folds = n_folds
        self.random_state = random_state
        
        # 设置logger
        self.logger = logger or self._setup_logger()
        
        # 初始化组件
        self.gpu_accelerator = GPUTrainingAccelerator(self.logger)
        self.parallel_manager = ParallelTrainingManager(self.use_gpu, logger=self.logger) if enable_parallel else None
        self.optimizer = HyperparameterOptimizer(n_trials=20, logger=self.logger) if enable_optimization else None  # 减少trials数量
        
        # 确保目录存在
        self.model_save_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("训练器初始化完成")
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s | %(levelname)8s | %(name)s | %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def load_segment_data(self, segment_id: int) -> pd.DataFrame:
        """加载数据段"""
        train_file = self.data_path / "train" / f"train_segment_{segment_id}.parquet"
        if not train_file.exists():
            raise FileNotFoundError(f"训练文件不存在: {train_file}")
        
        df = pd.read_parquet(train_file)
        self.logger.info(f"加载 segment_{segment_id}: {df.shape}")
        return df
    
    def create_cv_folds(self, groups: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """创建交叉验证fold"""
        unique_groups = np.unique(groups)
        n_groups = len(unique_groups)
        
        if n_groups < self.n_folds:
            self.logger.warning(f"组数量({n_groups})少于fold数({self.n_folds})，调整为{n_groups}fold")
            n_folds = n_groups
        else:
            n_folds = self.n_folds
        
        gkf = GroupKFold(n_splits=n_folds)
        
        # 创建虚拟数据用于分割
        dummy_data = np.zeros(len(groups))
        folds = list(gkf.split(dummy_data, groups=groups))
        
        self.logger.info(f"创建 {len(folds)} 个CV fold")
        return folds
    
    def optimize_model_hyperparameters(self, model_name: str, X: np.ndarray, 
                                     y: np.ndarray, groups: np.ndarray) -> Dict:
        """优化模型超参数"""
        if not self.enable_optimization:
            return {}
        
        # GPU模式下减少优化强度，避免资源竞争
        if self.use_gpu:
            self.logger.info(f"GPU模式：简化 {model_name} 超参数优化")
            n_folds_for_opt = 2  # 只用2个fold快速评估
        else:
            self.logger.info(f"CPU模式：完整 {model_name} 超参数优化")
            n_folds_for_opt = 3
        
        def objective(trial, X, y, groups):
            """优化目标函数"""
            # 根据模型类型建议参数
            if model_name == 'XGBRanker':
                params = self.optimizer.suggest_xgb_params(trial)
            elif model_name == 'LGBMRanker':
                params = self.optimizer.suggest_lgb_params(trial)
            elif model_name in ['RankNet', 'TransformerRanker']:
                params = self.optimizer.suggest_neural_params(trial)
            else:
                return 0.0
            
            # 快速CV评估
            folds = self.create_cv_folds(groups)
            scores = []
            
            for train_idx, val_idx in folds[:n_folds_for_opt]:  # 减少fold数量
                try:
                    # 创建模型
                    models_manager = FlightRankingModels(use_gpu=self.use_gpu, logger=self.logger)
                    input_dim = X.shape[1] if hasattr(X, 'shape') else len(X[0])
                    
                    if model_name in ['RankNet', 'TransformerRanker']:
                        params['input_dim'] = input_dim
                    
                    model_configs = {model_name: params}
                    created_models = models_manager.create_models(input_dim, model_configs)
                    
                    if model_name not in created_models:
                        return 0.0
                    
                    # 训练和评估
                    X_train, y_train = X[train_idx], y[train_idx]
                    X_val, y_val = X[val_idx], y[val_idx]
                    groups_train = groups[train_idx]
                    
                    model = created_models[model_name]
                    
                    # 减少训练轮数加快优化
                    if hasattr(model, 'fit'):
                        if model_name in ['RankNet', 'TransformerRanker']:
                            model.fit(X_train, y_train, groups_train, epochs=20)  # 减少epochs
                        else:
                            model.fit(X_train, y_train, groups_train)
                    
                    pred = model.predict(X_val)
                    score = ndcg_score([y_val], [pred], k=10)
                    scores.append(score)
                    
                except Exception as e:
                    self.logger.warning(f"参数评估失败: {e}")
                    continue
            
            return np.mean(scores) if scores else 0.0
        
        # 执行优化
        best_params, best_score = self.optimizer.optimize(
            objective, X=X, y=y, groups=groups
        )
        
        self.logger.info(f"{model_name} 最佳参数: {best_params}")
        return best_params
    
    def train_single_model(self, model_name: str, X: np.ndarray, y: np.ndarray, 
                          groups: np.ndarray, model_configs: Dict = None) -> BaseRankingModel:
        """训练单个模型"""
        self.logger.info(f"训练 {model_name}")
        
        # 优化GPU设置
        if self.use_gpu:
            self.gpu_accelerator.optimize_gpu_settings()
        
        try:
            # 创建模型管理器
            models_manager = FlightRankingModels(use_gpu=self.use_gpu, logger=self.logger)
            input_dim = X.shape[1]
            
            # 使用优化的参数（如果有）
            if model_configs is None:
                model_configs = {}
            
            # 超参数优化
            if self.enable_optimization and model_name in ['XGBRanker', 'LGBMRanker', 'RankNet']:
                optimized_params = self.optimize_model_hyperparameters(model_name, X, y, groups)
                if optimized_params:
                    model_configs[model_name] = optimized_params
            
            # 创建模型
            created_models = models_manager.create_models(input_dim, model_configs)
            
            if model_name not in created_models:
                raise ValueError(f"模型 {model_name} 创建失败")
            
            # 训练模型
            model = created_models[model_name]
            
            if isinstance(model, (models_manager.models.get('RankNet', type(None)), 
                                models_manager.models.get('TransformerRanker', type(None)))):
                # 深度学习模型使用更多epochs
                model.fit(X, y, groups, epochs=100)
            else:
                model.fit(X, y, groups)
            
            self.logger.info(f"✓ {model_name} 训练完成")
            return model
            
        except Exception as e:
            self.logger.error(f"✗ {model_name} 训练失败: {e}")
            return None
    
    def train_segment_with_cv(self, segment_id: int, 
                             model_names: List[str] = None,
                             model_configs: Dict = None) -> Dict:
        """使用交叉验证训练数据段"""
        self.logger.info(f"开始CV训练 segment_{segment_id}")
        
        start_time = time.time()
        
        # 加载数据
        df = self.load_segment_data(segment_id)
        
        # 数据预处理
        models_manager = FlightRankingModels(use_gpu=self.use_gpu, logger=self.logger)
        X, y, groups, feature_cols, _ = models_manager.prepare_data(df)
        
        # 默认模型列表
        if model_names is None:
            model_names = ['XGBRanker', 'LGBMRanker', 'RankNet']
        
        # 创建CV fold
        folds = self.create_cv_folds(groups)
        
        # 训练结果
        results = {
            'segment_id': segment_id,
            'models': {},
            'cv_scores': {},
            'training_time': 0,
            'feature_names': feature_cols
        }
        
        # 训练每个模型
        for model_name in model_names:
            self.logger.info(f"训练 {model_name} with CV")
            
            try:
                model_start_time = time.time()
                
                # CV训练
                fold_scores = []
                fold_models = []
                
                def train_fold(fold_idx, train_idx, val_idx):
                    """训练单个fold"""
                    X_train, y_train = X[train_idx], y[train_idx]
                    X_val, y_val = X[val_idx], y[val_idx]
                    groups_train = groups[train_idx]
                    
                    # 训练模型
                    model = self.train_single_model(
                        model_name, X_train, y_train, groups_train, model_configs
                    )
                    
                    if model is None:
                        return None
                    
                    # 验证
                    pred = model.predict(X_val)
                    score = ndcg_score([y_val], [pred], k=10)
                    
                    return {
                        'fold': fold_idx,
                        'model': model,
                        'score': score
                    }
                
                # 智能训练fold
                if self.enable_parallel and len(folds) > 1:
                    fold_results = self.parallel_manager.train_folds(
                        train_fold, folds
                    )
                else:
                    fold_results = []
                    for i, (train_idx, val_idx) in enumerate(folds):
                        result = train_fold(i, train_idx, val_idx)
                        if result:
                            fold_results.append(result)
                
                # 收集结果
                if fold_results:
                    fold_scores = [r['score'] for r in fold_results]
                    fold_models = [r['model'] for r in fold_results]
                    
                    # 选择最佳模型（最高分数的fold）
                    best_idx = np.argmax(fold_scores)
                    best_model = fold_models[best_idx]
                    
                    # 记录结果
                    results['models'][model_name] = best_model
                    results['cv_scores'][model_name] = {
                        'mean': np.mean(fold_scores),
                        'std': np.std(fold_scores),
                        'scores': fold_scores,
                        'best_score': fold_scores[best_idx]
                    }
                    
                    model_time = time.time() - model_start_time
                    self.logger.info(f"✓ {model_name} CV完成: {np.mean(fold_scores):.4f}±{np.std(fold_scores):.4f} "
                                   f"(时间: {model_time:.1f}s)")
                else:
                    self.logger.warning(f"✗ {model_name} 所有fold都失败")
                
            except Exception as e:
                self.logger.error(f"✗ {model_name} CV训练失败: {e}")
                continue
        
        # 训练最终集成模型（使用全部数据）
        if results['models']:
            self.logger.info("训练最终集成模型...")
            final_models = {}
            
            for model_name in results['models'].keys():
                try:
                    final_model = self.train_single_model(
                        model_name, X, y, groups, model_configs
                    )
                    if final_model:
                        final_models[model_name] = final_model
                except Exception as e:
                    self.logger.warning(f"最终模型训练失败 {model_name}: {e}")
            
            results['final_models'] = final_models
        
        # 保存结果
        total_time = time.time() - start_time
        results['training_time'] = total_time
        
        # 保存模型和结果
        self._save_segment_results(segment_id, results)
        
        self.logger.info(f"✓ segment_{segment_id} 训练完成 (总时间: {total_time:.1f}s)")
        return results
    
    def _save_segment_results(self, segment_id: int, results: Dict):
        """保存训练结果"""
        segment_dir = self.model_save_path / f"segment_{segment_id}"
        segment_dir.mkdir(exist_ok=True)
        
        # 保存最终模型
        if 'final_models' in results:
            for model_name, model in results['final_models'].items():
                model_path = segment_dir / f"{model_name}.pkl"
                model.save_model(str(model_path))
        
        # 保存特征名称
        feature_path = segment_dir / "features.pkl"
        with open(feature_path, 'wb') as f:
            pickle.dump(results['feature_names'], f)
        
        # 保存训练报告
        report = {
            'segment_id': results['segment_id'],
            'cv_scores': results['cv_scores'],
            'training_time': results['training_time'],
            'model_count': len(results.get('final_models', {}))
        }
        
        report_path = segment_dir / "training_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"结果已保存到: {segment_dir}")
    
    def train_all_segments(self, segments: List[int] = None,
                          model_names: List[str] = None,
                          model_configs: Dict = None) -> Dict:
        """训练所有数据段"""
        if segments is None:
            segments = [0, 1, 2]
        
        if model_names is None:
            model_names = ['XGBRanker', 'LGBMRanker', 'RankNet']
        
        self.logger.info(f"开始训练所有段: {segments}")
        self.logger.info(f"使用模型: {model_names}")
        
        all_results = {}
        total_start_time = time.time()
        
        for segment_id in segments:
            try:
                self.logger.info(f"\n{'='*50}")
                self.logger.info(f"训练 segment_{segment_id}")
                self.logger.info(f"{'='*50}")
                
                results = self.train_segment_with_cv(
                    segment_id, model_names, model_configs
                )
                all_results[f'segment_{segment_id}'] = results
                
            except Exception as e:
                self.logger.error(f"✗ segment_{segment_id} 训练失败: {e}")
                continue
        
        # 生成总体报告
        total_time = time.time() - total_start_time
        self._generate_final_report(all_results, total_time)
        
        return all_results
    
    def _generate_final_report(self, all_results: Dict, total_time: float):
        """生成最终训练报告"""
        report = {
            'total_segments': len(all_results),
            'total_training_time': total_time,
            'segment_summary': {},
            'model_performance': {}
        }
        
        # 汇总每个segment的结果
        all_model_scores = {}
        
        for segment_name, results in all_results.items():
            cv_scores = results.get('cv_scores', {})
            training_time = results.get('training_time', 0)
            
            report['segment_summary'][segment_name] = {
                'training_time': training_time,
                'models_trained': len(cv_scores),
                'best_model': max(cv_scores.keys(), key=lambda x: cv_scores[x]['mean']) if cv_scores else None,
                'best_score': max(cv_scores[x]['mean'] for x in cv_scores.keys()) if cv_scores else 0
            }
            
            # 收集模型分数
            for model_name, score_info in cv_scores.items():
                if model_name not in all_model_scores:
                    all_model_scores[model_name] = []
                all_model_scores[model_name].append(score_info['mean'])
        
        # 计算模型平均性能
        for model_name, scores in all_model_scores.items():
            report['model_performance'][model_name] = {
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'min_score': np.min(scores),
                'max_score': np.max(scores)
            }
        
        # 保存报告
        report_path = self.model_save_path / "final_training_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # 打印总结
        self.logger.info(f"\n{'='*60}")
        self.logger.info("训练总结")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"总训练时间: {total_time:.1f}s")
        self.logger.info(f"成功训练段数: {len(all_results)}")
        
        if report['model_performance']:
            self.logger.info("\n模型性能排名:")
            sorted_models = sorted(
                report['model_performance'].items(),
                key=lambda x: x[1]['mean_score'],
                reverse=True
            )
            for i, (model_name, perf) in enumerate(sorted_models, 1):
                self.logger.info(f"{i}. {model_name}: {perf['mean_score']:.4f}±{perf['std_score']:.4f}")
        
        self.logger.info(f"\n详细报告已保存到: {report_path}")
    
    # 保持原有接口兼容性
    def train_segment(self, segment_id: int, **kwargs):
        """兼容原有接口"""
        return self.train_segment_with_cv(segment_id, **kwargs)
    
    def train_all(self, segments: List[int] = None, **kwargs):
        """兼容原有接口"""
        results = self.train_all_segments(segments, **kwargs)
        return len(results) > 0  # 返回是否有成功的训练
=======
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.model.FlightRankingAnalyzer import FlightRankingAnalyzer
import joblib
from pathlib import Path

class FlightRankingTrainer:
    def __init__(self, data_path="data/aeroclub-recsys-2025/segmented", 
                 model_save_path="models", use_gpu=False, random_state=42):
        self.data_path = Path(data_path)
        self.model_save_path = self.data_path / model_save_path
        self.use_gpu = use_gpu
        self.random_state = random_state
        
        # 确保模型保存目录存在
        self.model_save_path.mkdir(parents=True, exist_ok=True)
    
    def train_segment(self, segment_id):
        """训练单个数据段"""
        print(f"开始训练 segment_{segment_id}")
        
        # 加载数据
        train_file = self.data_path / "train" / f"train_segment_{segment_id}.parquet"
        df = pd.read_parquet(train_file)
        print(f"数据形状: {df.shape}")
        
        # 初始化分析器
        analyzer = FlightRankingAnalyzer(use_gpu=self.use_gpu, random_state=self.random_state)
        
        # 准备数据
        X, y, groups, feature_cols, df_processed = analyzer.prepare_data(df)
        print(f"特征数量: {len(feature_cols)}")
        
        # 按ranker_id进行训练集验证集划分
        unique_rankers = np.unique(groups)
        train_rankers, test_rankers = train_test_split(
            unique_rankers, test_size=0.2, random_state=self.random_state
        )
        
        train_mask = np.isin(groups, train_rankers)
        test_mask = np.isin(groups, test_rankers)
        
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        groups_train, groups_test = groups[train_mask], groups[test_mask]
        
        print(f"训练集: {X_train.shape}, 验证集: {X_test.shape}")
        
        # 训练模型
        results = analyzer.train_models(X_train, X_test, y_train, y_test, groups_train, groups_test)
        
        # 保存模型
        for model_name in analyzer.trained_models:
            model_path = self.model_save_path / f"{model_name}_segment_{segment_id}.pkl"
            analyzer.save_model(str(model_path), model_name)
            print(f"已保存模型: {model_path}")
        
        # 保存特征名称
        feature_path = self.model_save_path / f"features_segment_{segment_id}.pkl"
        joblib.dump(feature_cols, feature_path)
        
        # 输出结果
        for result in results:
            print(f"段{segment_id} - {result['Model']}: HitRate@3 = {result['HitRate@3']}")
        
        return results
    
    def train_all(self, segments=[0, 1, 2]):
        """训练所有指定数据段"""
        all_results = {}
        
        for segment_id in segments:
            try:
                print(f"\n{'='*50}")
                results = self.train_segment(segment_id)
                all_results[f"segment_{segment_id}"] = results
                print(f"完成训练 segment_{segment_id}\n")
            except Exception as e:
                print(f"训练 segment_{segment_id} 失败: {e}")
                continue
        
        # 汇总结果
        print("\n" + "="*50)
        print("训练结果汇总:")
        for segment, results in all_results.items():
            print(f"\n{segment}:")
            for result in results:
                print(f"  {result['Model']}: HitRate@3 = {result['HitRate@3']}")
        
        return all_results
>>>>>>> e9addff7041905ba228e279124f71a4a54b1d4e3
