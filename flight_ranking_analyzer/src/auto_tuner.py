"""
自动超参数调优模块

该模块提供自动超参数调优功能：
- 基于Optuna的贝叶斯优化
- 支持多种模型的参数搜索
- 自定义目标函数
- 并行优化支持

作者: Flight Ranking Team
版本: 2.1
"""

import optuna
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from sklearn.model_selection import cross_val_score
import warnings

# 导入本地模块
try:
    from .models import ModelFactory, BaseRanker
    from .config import Config
except ImportError:
    from models import ModelFactory, BaseRanker
    from config import Config

warnings.filterwarnings('ignore')


class AutoTuner:
    """自动超参数调优器"""
    
    def __init__(self, 
                 model_name: str,
                 search_space: Dict[str, Any],
                 objective_metric: str = 'hit_rate',
                 n_trials: int = 50,
                 timeout: Optional[int] = None,
                 n_jobs: int = 1,
                 random_state: int = 42,
                 logger=None):
        """
        初始化自动调优器
        
        Args:
            model_name: 模型名称
            search_space: 搜索空间定义
            objective_metric: 目标指标 ('hit_rate', 'ndcg')
            n_trials: 试验次数
            timeout: 超时时间（秒）
            n_jobs: 并行作业数
            random_state: 随机种子
            logger: 日志记录器
        """
        self.model_name = model_name
        self.search_space = search_space
        self.objective_metric = objective_metric
        self.n_trials = n_trials
        self.timeout = timeout
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.logger = logger
        
        # 存储优化结果
        self.study = None
        self.best_params = None
        self.best_score = None
        
        # 数据存储
        self.X_train = None
        self.y_train = None
        self.train_groups = None
        self.X_val = None
        self.y_val = None
        self.val_groups = None
        
        self._log(f"AutoTuner 初始化完成: {model_name}")
    
    def _log(self, message: str):
        """记录日志"""
        if self.logger:
            self.logger.info(message)
        else:
            print(message)
    
    def optimize(self, 
                 X_train: np.ndarray, 
                 y_train: np.ndarray,
                 train_groups: List[int],
                 X_val: np.ndarray,
                 y_val: np.ndarray,
                 val_groups: List[int]) -> Dict[str, Any]:
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
        self._log(f"开始优化 {self.model_name} 的超参数...")
        
        # 存储数据
        self.X_train = X_train
        self.y_train = y_train
        self.train_groups = train_groups
        self.X_val = X_val
        self.y_val = y_val
        self.val_groups = val_groups
        
        # 创建研究对象
        study_name = f"{self.model_name}_optimization"
        self.study = optuna.create_study(
            direction='maximize',
            study_name=study_name,
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        
        # 执行优化
        try:
            self.study.optimize(
                self._objective,
                n_trials=self.n_trials,
                timeout=self.timeout,
                n_jobs=self.n_jobs,
                show_progress_bar=True
            )
            
            # 获取最优结果
            self.best_params = self.study.best_params
            self.best_score = self.study.best_value
            
            self._log(f"优化完成! 最优 {self.objective_metric}: {self.best_score:.4f}")
            self._log(f"最优参数: {self.best_params}")
            
            return self.best_params
            
        except Exception as e:
            self._log(f"优化过程中出错: {str(e)}")
            # 返回默认参数
            return Config.DEFAULT_MODEL_PARAMS.get(self.model_name, {})
    
    def _objective(self, trial: optuna.Trial) -> float:
        """目标函数"""
        try:
            # 根据搜索空间建议参数
            params = self._suggest_params(trial)
            
            # 创建模型
            if self.model_name in ['NeuralRanker', 'RankNet', 'TransformerRanker']:
                model = ModelFactory.create_model(
                    self.model_name,
                    input_dim=self.X_train.shape[1],
                    **params
                )
            else:
                model = ModelFactory.create_model(self.model_name, **params)
            
            # 训练模型
            model.fit(self.X_train, self.y_train, self.train_groups)
            
            # 预测
            y_pred = model.predict(self.X_val)
            
            # 计算目标指标
            if self.objective_metric == 'hit_rate':
                score = self._calculate_hit_rate(y_pred)
            elif self.objective_metric == 'ndcg':
                score = self._calculate_ndcg(y_pred)
            else:
                score = self._calculate_hit_rate(y_pred)  # 默认使用hit_rate
            
            return score
            
        except Exception as e:
            self._log(f"试验失败: {str(e)}")
            return 0.0  # 返回最低分数
    
    def _suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """根据搜索空间建议参数"""
        params = {}
        
        for param_name, param_config in self.search_space.items():
            if isinstance(param_config, list):
                # 分类参数
                params[param_name] = trial.suggest_categorical(param_name, param_config)
            elif isinstance(param_config, dict):
                if 'type' in param_config:
                    param_type = param_config['type']
                    if param_type == 'int':
                        params[param_name] = trial.suggest_int(
                            param_name,
                            param_config['low'],
                            param_config['high'],
                            step=param_config.get('step', 1)
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
                                param_config['high'],
                                step=param_config.get('step')
                            )
                else:
                    # 假设是范围参数 [min, max]
                    if len(param_config) == 2:
                        low, high = param_config
                        if isinstance(low, int) and isinstance(high, int):
                            params[param_name] = trial.suggest_int(param_name, low, high)
                        else:
                            params[param_name] = trial.suggest_float(param_name, low, high)
        
        return params
    
    def _calculate_hit_rate(self, y_pred: np.ndarray, k: int = 3) -> float:
        """计算Hit Rate@K"""
        try:
            hits = 0
            total_groups = 0
            start_idx = 0
            
            for group_size in self.val_groups:
                end_idx = start_idx + group_size
                
                # 获取该组的预测分数和真实标签
                group_pred = y_pred[start_idx:end_idx]
                group_true = self.y_val[start_idx:end_idx]
                
                # 找到真实的正样本
                positive_indices = np.where(group_true == 1)[0]
                
                if len(positive_indices) > 0:
                    # 按预测分数排序，获取前k个
                    top_k_indices = np.argsort(group_pred)[-k:]
                    
                    # 检查前k个中是否包含正样本
                    if any(idx in positive_indices for idx in top_k_indices):
                        hits += 1
                
                total_groups += 1
                start_idx = end_idx
            
            return hits / total_groups if total_groups > 0 else 0.0
            
        except Exception as e:
            self._log(f"计算Hit Rate时出错: {str(e)}")
            return 0.0
    
    def _calculate_ndcg(self, y_pred: np.ndarray, k: int = 5) -> float:
        """计算NDCG@K"""
        try:
            from sklearn.metrics import ndcg_score
            
            ndcg_scores = []
            start_idx = 0
            
            for group_size in self.val_groups:
                end_idx = start_idx + group_size
                
                if group_size >= k:
                    group_true = self.y_val[start_idx:end_idx].reshape(1, -1)
                    group_pred = y_pred[start_idx:end_idx].reshape(1, -1)
                    
                    try:
                        ndcg = ndcg_score(group_true, group_pred, k=k)
                        ndcg_scores.append(ndcg)
                    except:
                        continue
                
                start_idx = end_idx
            
            return np.mean(ndcg_scores) if ndcg_scores else 0.0
            
        except Exception as e:
            self._log(f"计算NDCG时出错: {str(e)}")
            return 0.0
    
    def get_optimization_history(self) -> Dict[str, Any]:
        """获取优化历史"""
        if self.study is None:
            return {}
        
        trials_df = self.study.trials_dataframe()
        
        return {
            'n_trials': len(self.study.trials),
            'best_value': self.study.best_value,
            'best_params': self.study.best_params,
            'trials_dataframe': trials_df,
            'optimization_history': [trial.value for trial in self.study.trials if trial.value is not None]
        }
    
    def plot_optimization_history(self):
        """绘制优化历史"""
        if self.study is None:
            self._log("尚未执行优化，无法绘制历史")
            return
        
        try:
            import matplotlib.pyplot as plt
            
            # 优化历史
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            optuna.visualization.matplotlib.plot_optimization_history(self.study)
            plt.title(f'{self.model_name} 优化历史')
            
            plt.subplot(1, 2, 2)
            optuna.visualization.matplotlib.plot_param_importances(self.study)
            plt.title(f'{self.model_name} 参数重要性')
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            self._log("无法导入matplotlib，跳过可视化")
        except Exception as e:
            self._log(f"绘制优化历史时出错: {str(e)}")


def create_auto_tuner(model_name: str, 
                     search_space: Dict[str, Any] = None,
                     objective_metric: str = 'hit_rate',
                     n_trials: int = 50,
                     timeout: Optional[int] = None,
                     logger=None) -> AutoTuner:
    """
    创建自动调优器实例
    
    Args:
        model_name: 模型名称
        search_space: 搜索空间，如果为None则使用默认配置
        objective_metric: 目标指标
        n_trials: 试验次数
        timeout: 超时时间
        logger: 日志记录器
        
    Returns:
        AutoTuner: 调优器实例
    """
    if search_space is None:
        search_space = Config.TUNING_SEARCH_SPACES.get(model_name, {})
    
    return AutoTuner(
        model_name=model_name,
        search_space=search_space,
        objective_metric=objective_metric,
        n_trials=n_trials,
        timeout=timeout,
        logger=logger
    )


def optimize_all_models(models: List[str],
                       X_train: np.ndarray,
                       y_train: np.ndarray,
                       train_groups: List[int],
                       X_val: np.ndarray,
                       y_val: np.ndarray,
                       val_groups: List[int],
                       n_trials: int = 50,
                       logger=None) -> Dict[str, Dict[str, Any]]:
    """
    优化所有指定模型的超参数
    
    Args:
        models: 模型名称列表
        X_train: 训练特征
        y_train: 训练标签
        train_groups: 训练组大小
        X_val: 验证特征
        y_val: 验证标签
        val_groups: 验证组大小
        n_trials: 每个模型的试验次数
        logger: 日志记录器
        
    Returns:
        Dict: 每个模型的最优参数
    """
    all_best_params = {}
    
    for model_name in models:
        if model_name in Config.TUNING_SEARCH_SPACES:
            try:
                tuner = create_auto_tuner(
                    model_name=model_name,
                    n_trials=n_trials,
                    logger=logger
                )
                
                best_params = tuner.optimize(
                    X_train, y_train, train_groups,
                    X_val, y_val, val_groups
                )
                
                all_best_params[model_name] = best_params
                
            except Exception as e:
                if logger:
                    logger.error(f"优化 {model_name} 时出错: {str(e)}")
                else:
                    print(f"优化 {model_name} 时出错: {str(e)}")
                
                # 使用默认参数
                all_best_params[model_name] = Config.DEFAULT_MODEL_PARAMS.get(model_name, {})
        else:
            # 使用默认参数
            all_best_params[model_name] = Config.DEFAULT_MODEL_PARAMS.get(model_name, {})
    
    return all_best_params