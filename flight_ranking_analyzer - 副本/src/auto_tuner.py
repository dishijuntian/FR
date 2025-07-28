"""
自动调参模块 - 修复排名重复问题版本

该模块实现了基于Optuna的超参数自动调优功能
- 修复了验证过程中的排名重复问题
- 强化了排名唯一性保证
- 改进了HitRate计算的准确性

作者: Flight Ranking Team  
版本: 2.2 (修复排名重复问题)
"""

import optuna
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Callable
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

try:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except:
    pass


class AutoTuner:
    """自动超参数调优器 - 修复排名重复问题版本"""
    
    def __init__(self, 
                 model_factory: Callable,
                 hitrate_calculator: Callable,
                 n_trials: int = 50,
                 timeout: int = 3600,
                 n_jobs: int = 1,
                 random_state: int = 42):
        """
        初始化自动调参器
        
        Args:
            model_factory: 模型工厂函数
            hitrate_calculator: HitRate计算函数
            n_trials: 试验次数
            timeout: 超时时间(秒)
            n_jobs: 并行作业数
            random_state: 随机种子
        """
        self.model_factory = model_factory
        self.hitrate_calculator = hitrate_calculator
        self.n_trials = n_trials
        self.timeout = timeout
        self.n_jobs = n_jobs
        self.random_state = random_state
        
        # 设置日志级别
        logging.getLogger('optuna').setLevel(logging.WARNING)
    
    def _create_objective(self, model_name: str, X_train, y_train, train_groups, 
                         X_val, y_val, val_info, use_gpu: bool, input_dim: Optional[int] = None):
        """创建优化目标函数"""
        
        def objective(trial):
            try:
                # 根据模型类型定义搜索空间
                params = self._get_search_space(trial, model_name)
                
                # 创建模型
                model = self.model_factory(
                    model_name=model_name,
                    use_gpu=use_gpu,
                    input_dim=input_dim,
                    **params
                )
                
                # 训练模型
                if model_name == 'NeuralRanker':
                    model.fit(X_train, y_train, group=train_groups, 
                             epochs=params.get('epochs', 10),
                             batch_size=params.get('batch_size', 32))
                else:
                    model.fit(X_train, y_train, group=train_groups)
                
                # 预测
                y_pred_scores = model.predict(X_val)
                
                # 计算组内排名（关键修复：确保排名唯一）
                y_pred_ranks = self._calculate_group_ranks_robust(
                    y_pred_scores, 
                    self._get_val_group_sizes(val_info),
                    trial_number=trial.number
                )
                
                # 验证排名唯一性
                is_valid = self._validate_ranking_uniqueness(
                    y_pred_ranks, 
                    self._get_val_group_sizes(val_info),
                    f"Trial_{trial.number}_{model_name}"
                )
                
                if not is_valid:
                    logging.warning(f"Trial {trial.number} 排名验证失败，强制修复")
                    y_pred_ranks = self._force_fix_rankings(
                        y_pred_ranks,
                        self._get_val_group_sizes(val_info),
                        trial.number
                    )
                
                # 计算HitRate@3（这是我们要最大化的指标）
                hitrate = self.hitrate_calculator(val_info, y_pred_ranks, k=3)
                
                return hitrate
                
            except Exception as e:
                # 如果出现错误，返回很低的分数
                logging.warning(f"Trial {trial.number} failed with error: {str(e)}")
                return 0.0
        
        return objective
    
    def _get_search_space(self, trial, model_name: str) -> Dict[str, Any]:
        """获取模型的超参数搜索空间"""
        
        if model_name in ['XGBRanker', 'LambdaMART']:
            return {
                'n_estimators': trial.suggest_categorical('n_estimators', [50, 100, 200, 300]),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'random_state': self.random_state
            }
        
        elif model_name in ['LGBMRanker', 'ListNet']:
            return {
                'n_estimators': trial.suggest_categorical('n_estimators', [50, 100, 200, 300]),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'num_leaves': trial.suggest_int('num_leaves', 10, 300),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'random_state': self.random_state
            }
        
        elif model_name == 'NeuralRanker':
            # 神经网络架构搜索
            n_layers = trial.suggest_int('n_layers', 2, 4)
            hidden_units = []
            
            for i in range(n_layers):
                units = trial.suggest_categorical(f'units_layer_{i}', [64, 128, 256, 512])
                hidden_units.append(units)
            
            return {
                'hidden_units': hidden_units,
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
                'epochs': trial.suggest_categorical('epochs', [5, 10, 15, 20]),
                'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128])
            }
        
        elif model_name == 'BM25Ranker':
            return {}  # BM25没有需要调优的超参数
        
        else:
            return {}
    
    def _calculate_group_ranks_robust(self, scores: np.ndarray, group_sizes: List[int], 
                                     trial_number: int = 0) -> np.ndarray:
        """
        稳健的排名计算函数 - 确保排名唯一且连续
        
        Args:
            scores: 预测分数
            group_sizes: 组大小列表
            trial_number: 试验编号（用于生成唯一随机种子）
            
        Returns:
            np.ndarray: 唯一且连续的排名
        """
        ranks = np.zeros_like(scores, dtype=int)
        start_idx = 0
        
        for group_idx, group_size in enumerate(group_sizes):
            end_idx = start_idx + group_size
            group_scores = scores[start_idx:end_idx]
            
            if group_size == 1:
                # 单个元素的组，排名直接为1
                ranks[start_idx:end_idx] = 1
            else:
                # 多个元素的组，确保排名唯一
                # 创建基于试验编号和组索引的唯一随机种子
                unique_seed = ((trial_number * 1009 + group_idx * 2017) % 2147483647)
                np.random.seed(unique_seed)
                
                # 添加适中强度的随机噪声
                noise_scale = 1e-7
                noise = np.random.random(len(group_scores)) * noise_scale
                
                # 为每个位置添加不同的噪声偏移
                position_offset = np.arange(len(group_scores)) * 1e-10
                noisy_scores = group_scores + noise + position_offset
                
                # 计算排名：分数越高，排名越靠前（rank=1最好）
                sorted_indices = np.argsort(-noisy_scores)  # 降序排列的索引
                group_ranks = np.zeros(group_size, dtype=int)
                
                # 分配唯一且连续的排名
                for rank, idx in enumerate(sorted_indices):
                    group_ranks[idx] = rank + 1
                
                ranks[start_idx:end_idx] = group_ranks
                
                # 验证当前组的排名
                unique_ranks = set(group_ranks)
                expected_ranks = set(range(1, group_size + 1))
                if unique_ranks != expected_ranks:
                    # 如果仍有问题，强制修复
                    logging.warning(f"Trial {trial_number} 组{group_idx}排名计算失败，强制修复")
                    ranks[start_idx:end_idx] = self._generate_forced_unique_ranks(
                        group_size, group_idx, trial_number
                    )
            
            start_idx = end_idx
        
        return ranks
    
    def _generate_forced_unique_ranks(self, group_size: int, group_idx: int, 
                                     trial_number: int) -> np.ndarray:
        """
        强制生成唯一排名
        
        Args:
            group_size: 组大小
            group_idx: 组索引
            trial_number: 试验编号
            
        Returns:
            np.ndarray: 唯一排名数组
        """
        # 使用确定性但独特的方法生成排名
        forced_seed = ((trial_number * 13 + group_idx * 23) % 1000000)
        np.random.seed(forced_seed)
        
        # 直接生成1到group_size的随机排列
        unique_ranks = np.random.permutation(range(1, group_size + 1))
        return unique_ranks
    
    def _validate_ranking_uniqueness(self, ranks: np.ndarray, group_sizes: List[int], 
                                   context: str = "") -> bool:
        """
        验证排名的唯一性
        
        Args:
            ranks: 排名数组
            group_sizes: 组大小列表
            context: 上下文信息
            
        Returns:
            bool: 排名是否唯一有效
        """
        start_idx = 0
        all_valid = True
        
        for group_idx, group_size in enumerate(group_sizes):
            end_idx = start_idx + group_size
            group_ranks = ranks[start_idx:end_idx]
            
            # 检查排名是否唯一且连续
            unique_ranks = set(group_ranks)
            expected_ranks = set(range(1, group_size + 1))
            
            if unique_ranks != expected_ranks:
                logging.warning(f"排名验证失败 - {context} 组{group_idx}: "
                               f"期望{sorted(expected_ranks)}, 实际{sorted(unique_ranks)}")
                all_valid = False
            
            start_idx = end_idx
        
        return all_valid
    
    def _force_fix_rankings(self, ranks: np.ndarray, group_sizes: List[int], 
                           trial_number: int = 0) -> np.ndarray:
        """
        强制修复排名唯一性问题
        
        Args:
            ranks: 原始排名
            group_sizes: 组大小列表
            trial_number: 试验编号
            
        Returns:
            np.ndarray: 修复后的排名
        """
        fixed_ranks = ranks.copy()
        start_idx = 0
        
        for group_idx, group_size in enumerate(group_sizes):
            end_idx = start_idx + group_size
            group_ranks = fixed_ranks[start_idx:end_idx]
            
            # 检查是否需要修复
            unique_ranks = set(group_ranks)
            expected_ranks = set(range(1, group_size + 1))
            
            if unique_ranks != expected_ranks:
                # 强制分配连续排名
                # 使用多重因子生成唯一种子
                fix_seed = ((trial_number * 37 + group_idx * 67) % 1000000)
                np.random.seed(fix_seed)
                new_ranks = np.random.permutation(range(1, group_size + 1))
                fixed_ranks[start_idx:end_idx] = new_ranks
                
                logging.warning(f"强制修复Trial {trial_number} 组{group_idx}的排名")
            
            start_idx = end_idx
        
        return fixed_ranks
    
    def _get_val_group_sizes(self, val_info):
        """从验证集信息中获取组大小"""
        groups = val_info['ranker_id'].values
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
    
    def optimize(self, model_name: str, X_train, y_train, train_groups,
                 X_val, y_val, val_info, use_gpu: bool = True, 
                 input_dim: Optional[int] = None) -> Dict[str, Any]:
        """
        执行超参数优化（修复排名重复问题版本）
        
        Args:
            model_name: 模型名称
            X_train, y_train: 训练数据
            train_groups: 训练组信息
            X_val, y_val: 验证数据
            val_info: 验证集信息（包含ranker_id和selected）
            use_gpu: 是否使用GPU
            input_dim: 输入维度（仅NeuralRanker需要）
            
        Returns:
            Dict: 包含最佳参数和最佳分数的字典
        """
        print(f"开始为 {model_name} 进行自动调参...")
        print(f"试验次数: {self.n_trials}, 超时时间: {self.timeout}秒")
        
        # 创建研究对象
        study = optuna.create_study(
            direction='maximize',  # 最大化HitRate@3
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )
        
        # 创建目标函数
        objective = self._create_objective(
            model_name, X_train, y_train, train_groups,
            X_val, y_val, val_info, use_gpu, input_dim
        )
        
        # 执行优化
        try:
            study.optimize(
                objective,
                n_trials=self.n_trials,
                timeout=self.timeout,
                n_jobs=self.n_jobs
            )
        except KeyboardInterrupt:
            print("优化过程被用户中断")
        
        # 获取最佳结果
        best_params = study.best_params
        best_score = study.best_value
        
        print(f"{model_name} 调参完成!")
        print(f"最佳HitRate@3: {best_score:.4f}")
        print(f"最佳参数: {best_params}")
        
        # 额外验证：使用最佳参数重新训练和验证
        print(f"验证最佳参数...")
        validation_score = self._validate_best_params(
            model_name, best_params, X_train, y_train, train_groups,
            X_val, y_val, val_info, use_gpu, input_dim
        )
        
        if abs(validation_score - best_score) > 0.01:  # 如果差异较大
            print(f"⚠️ 验证分数 {validation_score:.4f} 与最佳分数 {best_score:.4f} 存在差异")
        else:
            print(f"✅ 最佳参数验证通过: {validation_score:.4f}")
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'validation_score': validation_score,
            'study': study
        }
    
    def _validate_best_params(self, model_name: str, best_params: Dict[str, Any],
                             X_train, y_train, train_groups, X_val, y_val, val_info,
                             use_gpu: bool, input_dim: Optional[int] = None) -> float:
        """
        验证最佳参数
        
        Args:
            model_name: 模型名称
            best_params: 最佳参数
            其他参数同optimize方法
            
        Returns:
            float: 验证分数
        """
        try:
            # 使用最佳参数创建模型
            model = self.model_factory(
                model_name=model_name,
                use_gpu=use_gpu,
                input_dim=input_dim,
                **best_params
            )
            
            # 训练模型
            if model_name == 'NeuralRanker':
                model.fit(X_train, y_train, group=train_groups, 
                         epochs=best_params.get('epochs', 10),
                         batch_size=best_params.get('batch_size', 32))
            else:
                model.fit(X_train, y_train, group=train_groups)
            
            # 预测
            y_pred_scores = model.predict(X_val)
            
            # 计算排名（使用验证专用的种子）
            y_pred_ranks = self._calculate_group_ranks_robust(
                y_pred_scores, 
                self._get_val_group_sizes(val_info),
                trial_number=99999  # 特殊的验证试验编号
            )
            
            # 验证排名唯一性
            is_valid = self._validate_ranking_uniqueness(
                y_pred_ranks, 
                self._get_val_group_sizes(val_info),
                f"Validation_{model_name}"
            )
            
            if not is_valid:
                print(f"⚠️ 验证排名不唯一，强制修复...")
                y_pred_ranks = self._force_fix_rankings(
                    y_pred_ranks,
                    self._get_val_group_sizes(val_info),
                    99999
                )
            
            # 计算HitRate@3
            hitrate = self.hitrate_calculator(val_info, y_pred_ranks, k=3)
            
            return hitrate
            
        except Exception as e:
            print(f"验证最佳参数时出错: {str(e)}")
            return 0.0
    
    def optimize_all_models(self, model_names: List[str], 
                          X_train, y_train, train_groups,
                          X_val, y_val, val_info, 
                          use_gpu: bool = True,
                          input_dim: Optional[int] = None) -> Dict[str, Dict[str, Any]]:
        """
        为所有指定模型进行超参数优化
        
        Args:
            model_names: 模型名称列表
            其他参数同optimize方法
            
        Returns:
            Dict: 每个模型的优化结果
        """
        results = {}
        
        for model_name in model_names:
            if model_name == 'BM25Ranker':
                # BM25不需要调参
                results[model_name] = {
                    'best_params': {},
                    'best_score': None,
                    'validation_score': None,
                    'study': None
                }
                continue
            
            try:
                result = self.optimize(
                    model_name, X_train, y_train, train_groups,
                    X_val, y_val, val_info, use_gpu, input_dim
                )
                results[model_name] = result
            except Exception as e:
                print(f"优化 {model_name} 时出错: {str(e)}")
                results[model_name] = {
                    'best_params': {},
                    'best_score': 0.0,
                    'validation_score': 0.0,
                    'study': None,
                    'error': str(e)
                }
        
        return results


def create_auto_tuner(model_factory, hitrate_calculator, 
                     n_trials: int = 50, timeout: int = 3600,
                     n_jobs: int = 1, random_state: int = 42) -> AutoTuner:
    """
    创建自动调参器的工厂函数
    
    Args:
        model_factory: 模型工厂函数
        hitrate_calculator: HitRate计算函数
        n_trials: 试验次数
        timeout: 超时时间(秒)
        n_jobs: 并行作业数
        random_state: 随机种子
        
    Returns:
        AutoTuner: 自动调参器实例
    """
    return AutoTuner(
        model_factory=model_factory,
        hitrate_calculator=hitrate_calculator,
        n_trials=n_trials,
        timeout=timeout,
        n_jobs=n_jobs,
        random_state=random_state
    )