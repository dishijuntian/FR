"""
训练和预测模块 - 重构版
统一管理模型训练、调参和预测功能

作者: Flight Ranking Team
版本: 5.0 (重构版)
"""

import pandas as pd
import numpy as np
import joblib
import torch
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import warnings
import gc
from abc import ABC, abstractmethod

warnings.filterwarnings('ignore')

# 导入其他模块
from evaluation_metrics import ModelEvaluator, calculate_hit_rate
from models_module import ModelFactory, BaseRanker

# 尝试导入optuna用于超参数优化
try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class AutoTuner:
    """超参数自动调优器"""
    
    def __init__(self, model_name: str, search_space: Dict[str, Any],
                 n_trials: int = 50, objective_metric: str = 'hit_rate_3'):
        """
        初始化调优器
        
        Args:
            model_name: 模型名称
            search_space: 搜索空间
            n_trials: 试验次数
            objective_metric: 目标指标
        """
        if not HAS_OPTUNA:
            raise ImportError("Optuna未安装，无法使用自动调参功能")
        
        self.model_name = model_name
        self.search_space = search_space
        self.n_trials = n_trials
        self.objective_metric = objective_metric
        
        # 存储训练数据
        self.X_train = None
        self.y_train = None
        self.train_groups = None
        self.X_val = None
        self.y_val = None
        self.val_groups = None
        self.input_dim = None
        
        self.study = None
        self.best_params = None
    
    def optimize(self, X_train: np.ndarray, y_train: np.ndarray,
                 train_groups: List[int], X_val: np.ndarray,
                 y_val: np.ndarray, val_groups: List[int]) -> Dict[str, Any]:
        """
        执行超参数优化
        
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
        self.input_dim = X_train.shape[1]
        
        # 创建优化研究
        self.study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        # 执行优化
        print(f"开始超参数优化: {self.model_name}")
        self.study.optimize(
            self._objective,
            n_trials=self.n_trials,
            show_progress_bar=True
        )
        
        self.best_params = self.study.best_params
        print(f"最优参数: {self.best_params}")
        print(f"最优分数: {self.study.best_value:.4f}")
        
        return self.best_params
    
    def _objective(self, trial: optuna.Trial) -> float:
        """目标函数"""
        try:
            # 建议参数
            params = self._suggest_params(trial)
            
            # 创建并训练模型
            if ModelFactory.is_pytorch_model(self.model_name):
                model = ModelFactory.create_model(
                    self.model_name,
                    input_dim=self.input_dim,
                    **params
                )
                # PyTorch模型使用较少epochs进行快速调优
                model.fit(self.X_train, self.y_train, self.train_groups, epochs=10)
            else:
                model = ModelFactory.create_model(self.model_name, **params)
                model.fit(self.X_train, self.y_train, self.train_groups)
            
            # 预测和评估
            y_pred = model.predict(self.X_val)
            
            # 计算目标指标
            if self.objective_metric == 'hit_rate_3':
                score = calculate_hit_rate(self.y_val, y_pred, self.val_groups, k=3)
            else:
                # 可以添加其他指标
                score = calculate_hit_rate(self.y_val, y_pred, self.val_groups, k=3)
            
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


class ModelManager:
    """模型管理器 - 负责模型的保存和加载"""
    
    def __init__(self, model_save_path: Path):
        self.model_save_path = Path(model_save_path)
        self.model_save_path.mkdir(parents=True, exist_ok=True)
        self.loaded_models = {}
        self.loaded_features = {}
    
    def save_model(self, model: BaseRanker, model_name: str, segment_id: int, 
                   feature_names: List[str], performance: float = None):
        """保存模型和相关信息"""
        try:
            # 保存模型
            if isinstance(model, torch.nn.Module):
                self._save_pytorch_model(model, model_name, segment_id)
            else:
                self._save_traditional_model(model, model_name, segment_id)
            
            # 保存特征和信息
            self._save_model_info(model_name, segment_id, feature_names, performance)
            
            print(f"已保存模型: {model_name}_segment_{segment_id}")
            
        except Exception as e:
            print(f"保存模型失败: {e}")
    
    def load_model(self, model_name: str, segment_id: int) -> Tuple[BaseRanker, List[str]]:
        """加载模型和特征信息"""
        cache_key = f"{model_name}_segment_{segment_id}"
        
        # 检查缓存
        if cache_key in self.loaded_models:
            return self.loaded_models[cache_key], self.loaded_features[cache_key]
        
        # 加载特征
        feature_path = self.model_save_path / f"features_segment_{segment_id}.pkl"
        if not feature_path.exists():
            raise FileNotFoundError(f"特征文件不存在: {feature_path}")
        
        feature_names = joblib.load(feature_path)
        
        # 加载模型
        if ModelFactory.is_pytorch_model(model_name):
            model = self._load_pytorch_model(model_name, segment_id)
        else:
            model = self._load_traditional_model(model_name, segment_id)
        
        # 缓存
        self.loaded_models[cache_key] = model
        self.loaded_features[cache_key] = feature_names
        
        return model, feature_names
    
    def get_available_models(self) -> Dict[str, List[int]]:
        """获取可用的模型和对应段"""
        available_models = {}
        
        for model_file in self.model_save_path.glob("*"):
            if model_file.suffix in ['.pkl', '.pth']:
                name_parts = model_file.stem.split("_")
                if len(name_parts) >= 3 and name_parts[-2] == "segment":
                    model_name = "_".join(name_parts[:-2])
                    try:
                        segment_id = int(name_parts[-1])
                        if model_name not in available_models:
                            available_models[model_name] = []
                        if segment_id not in available_models[model_name]:
                            available_models[model_name].append(segment_id)
                    except ValueError:
                        continue
        
        # 排序段ID
        for model_name in available_models:
            available_models[model_name].sort()
        
        return available_models
    
    def _save_pytorch_model(self, model: torch.nn.Module, model_name: str, segment_id: int):
        """保存PyTorch模型"""
        model_path = self.model_save_path / f"{model_name}_segment_{segment_id}.pth"
        
        model_info = {
            'model_name': model_name,
            'state_dict': model.state_dict(),
            'model_params': getattr(model, 'params', {}),
            'input_dim': getattr(model, 'input_dim', None)
        }
        
        torch.save(model_info, model_path)
    
    def _save_traditional_model(self, model: BaseRanker, model_name: str, segment_id: int):
        """保存传统模型"""
        model_path = self.model_save_path / f"{model_name}_segment_{segment_id}.pkl"
        joblib.dump(model, model_path)
    
    def _save_model_info(self, model_name: str, segment_id: int, 
                        feature_names: List[str], performance: float):
        """保存模型信息"""
        # 保存特征
        feature_path = self.model_save_path / f"features_segment_{segment_id}.pkl"
        joblib.dump(feature_names, feature_path)
        
        # 保存信息
        info_path = self.model_save_path / f"info_{model_name}_segment_{segment_id}.json"
        import json
        info = {
            'model_name': model_name,
            'segment_id': segment_id,
            'feature_count': len(feature_names),
            'performance': performance,
            'is_pytorch_model': ModelFactory.is_pytorch_model(model_name)
        }
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
    
    def _load_pytorch_model(self, model_name: str, segment_id: int) -> BaseRanker:
        """加载PyTorch模型"""
        model_path = self.model_save_path / f"{model_name}_segment_{segment_id}.pth"
        
        if not model_path.exists():
            raise FileNotFoundError(f"PyTorch模型文件不存在: {model_path}")
        
        model_info = torch.load(model_path, map_location=DEVICE)
        
        # 重新创建模型
        model = ModelFactory.create_model(
            model_name,
            input_dim=model_info['input_dim'],
            **model_info['model_params']
        )
        
        model.load_state_dict(model_info['state_dict'])
        model.eval()
        
        return model
    
    def _load_traditional_model(self, model_name: str, segment_id: int) -> BaseRanker:
        """加载传统模型"""
        model_path = self.model_save_path / f"{model_name}_segment_{segment_id}.pkl"
        
        if not model_path.exists():
            raise FileNotFoundError(f"传统模型文件不存在: {model_path}")
        
        return joblib.load(model_path)


class ModelTrainer:
    """模型训练器"""
    
    def __init__(self, model_manager: ModelManager, 
                 enable_auto_tuning: bool = False,
                 auto_tuning_trials: int = 50):
        """
        初始化训练器
        
        Args:
            model_manager: 模型管理器
            enable_auto_tuning: 是否启用自动调参
            auto_tuning_trials: 调参试验次数
        """
        self.model_manager = model_manager
        self.enable_auto_tuning = enable_auto_tuning
        self.auto_tuning_trials = auto_tuning_trials
        self.evaluator = ModelEvaluator()
    
    def train_model(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray,
                   train_groups: List[int], X_val: np.ndarray, y_val: np.ndarray,
                   val_groups: List[int], model_params: Dict[str, Any] = None,
                   tuning_space: Dict[str, Any] = None) -> Tuple[BaseRanker, float]:
        """
        训练单个模型
        
        Returns:
            Tuple: (训练好的模型, 验证性能)
        """
        print(f"训练模型: {model_name}")
        
        # 获取默认参数
        if model_params is None:
            model_params = {}
        
        # 自动调参
        if self.enable_auto_tuning and tuning_space and HAS_OPTUNA:
            print(f"执行自动调参: {model_name}")
            tuner = AutoTuner(
                model_name=model_name,
                search_space=tuning_space,
                n_trials=self.auto_tuning_trials
            )
            
            best_params = tuner.optimize(
                X_train, y_train, train_groups,
                X_val, y_val, val_groups
            )
            model_params.update(best_params)
        
        # 创建和训练模型
        if ModelFactory.is_pytorch_model(model_name):
            model = ModelFactory.create_model(
                model_name, 
                input_dim=X_train.shape[1],
                **model_params
            )
        else:
            model = ModelFactory.create_model(model_name, **model_params)
        
        model.fit(X_train, y_train, train_groups)
        
        # 评估模型
        results = self.evaluator.evaluate_model(model, X_val, y_val, val_groups)
        performance = results.get('hit_rate_3', 0.0)
        
        print(f"模型 {model_name} 验证性能 HitRate@3: {performance:.4f}")
        
        return model, performance
    
    def train_models(self, model_names: List[str], 
                    X_train: np.ndarray, y_train: np.ndarray,
                    train_groups: List[int], X_val: np.ndarray, 
                    y_val: np.ndarray, val_groups: List[int],
                    segment_id: int, feature_names: List[str],
                    model_configs: Dict[str, Dict[str, Any]] = None,
                    tuning_spaces: Dict[str, Dict[str, Any]] = None,
                    save_models: bool = True) -> pd.DataFrame:
        """
        训练多个模型
        
        Returns:
            pd.DataFrame: 模型结果汇总
        """
        results = []
        trained_models = {}
        
        for model_name in model_names:
            try:
                # 获取模型配置
                model_params = model_configs.get(model_name, {}) if model_configs else {}
                tuning_space = tuning_spaces.get(model_name, {}) if tuning_spaces else {}
                
                # 训练模型
                model, performance = self.train_model(
                    model_name, X_train, y_train, train_groups,
                    X_val, y_val, val_groups, model_params, tuning_space
                )
                
                # 保存模型
                if save_models:
                    self.model_manager.save_model(
                        model, model_name, segment_id, feature_names, performance
                    )
                
                # 记录结果
                trained_models[model_name] = model
                results.append({
                    'Model': model_name,
                    'HitRate@3': performance,
                    'Segment': segment_id
                })
                
                # 清理GPU内存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"训练模型 {model_name} 失败: {e}")
                continue
        
        return pd.DataFrame(results)


class ModelPredictor:
    """模型预测器"""
    
    def __init__(self, model_manager: ModelManager, data_processor):
        """
        初始化预测器
        
        Args:
            model_manager: 模型管理器
            data_processor: 数据处理器
        """
        self.model_manager = model_manager
        self.data_processor = data_processor
    
    def predict_segment(self, test_file: Path, segment_id: int, 
                       model_names: List[str],
                       ensemble_method: str = 'average') -> Optional[pd.DataFrame]:
        """
        预测单个数据段
        
        Args:
            test_file: 测试文件路径
            segment_id: 段ID
            model_names: 模型名称列表
            ensemble_method: 集成方法
            
        Returns:
            Optional[pd.DataFrame]: 预测结果
        """
        try:
            print(f"预测段 {segment_id}: {test_file.name}")
            
            # 加载测试数据
            test_df = pd.read_parquet(test_file)
            
            # 收集所有模型的预测
            all_predictions = {}
            
            for model_name in model_names:
                try:
                    # 加载模型和特征
                    model, feature_names = self.model_manager.load_model(model_name, segment_id)
                    
                    # 准备特征
                    X_test, _ = self.data_processor.prepare_test_features(test_df, feature_names)
                    
                    # 预测
                    scores = model.predict(X_test)
                    all_predictions[model_name] = scores
                    
                except Exception as e:
                    print(f"模型 {model_name} 预测失败: {e}")
                    continue
            
            if not all_predictions:
                print(f"段 {segment_id} 没有可用的预测结果")
                return None
            
            # 集成预测结果
            final_scores = self._ensemble_predictions(all_predictions, ensemble_method)
            
            # 分配排名
            result_df = self.data_processor.assign_rankings(test_df, final_scores)
            
            return result_df
            
        except Exception as e:
            print(f"预测段 {segment_id} 失败: {e}")
            return None
        finally:
            # 清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def predict_all_segments(self, segments: List[int], model_names: List[str],
                           test_data_path: Path, ensemble_method: str = 'average',
                           output_path: Optional[Path] = None) -> Optional[pd.DataFrame]:
        """
        预测所有数据段
        
        Returns:
            Optional[pd.DataFrame]: 最终预测结果
        """
        print(f"开始预测数据段: {segments}")
        print(f"使用模型: {model_names}")
        
        all_predictions = []
        
        for segment_id in segments:
            # 查找测试文件
            test_file = test_data_path / f"test_segment_{segment_id}.parquet"
            if not test_file.exists():
                print(f"找不到测试文件: {test_file}")
                continue
            
            # 预测该段
            segment_result = self.predict_segment(
                test_file, segment_id, model_names, ensemble_method
            )
            
            if segment_result is not None:
                all_predictions.append(segment_result)
        
        if not all_predictions:
            print("没有成功的预测结果")
            return None
        
        # 合并所有预测
        final_result = pd.concat(all_predictions, ignore_index=True)
        final_result = final_result.sort_values('Id').reset_index(drop=True)
        
        # 保存结果
        if output_path:
            output_file = output_path / f"final_predictions_{'+'.join(model_names)}.csv"
            final_result.to_csv(output_file, index=False)
            print(f"预测结果已保存: {output_file}")
        
        print(f"预测完成，总记录数: {len(final_result)}")
        return final_result
    
    def _ensemble_predictions(self, predictions: Dict[str, np.ndarray], 
                            method: str = 'average') -> np.ndarray:
        """集成多个预测结果"""
        if len(predictions) == 1:
            return list(predictions.values())[0]
        
        if method == 'average':
            # 简单平均
            all_scores = list(predictions.values())
            return np.mean(all_scores, axis=0)
        
        elif method == 'weighted_average':
            # 加权平均（可以根据模型性能设置权重）
            all_scores = list(predictions.values())
            weights = np.ones(len(all_scores)) / len(all_scores)  # 均等权重
            return np.average(all_scores, axis=0, weights=weights)
        
        else:
            # 默认使用平均
            all_scores = list(predictions.values())
            return np.mean(all_scores, axis=0)


def create_tuning_space(model_name: str) -> Dict[str, Any]:
    """创建默认调参空间"""
    spaces = {
        'XGBRanker': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 4, 5, 6, 7],
            'subsample': [0.6, 0.7, 0.8, 0.9],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9]
        },
        'LGBMRanker': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 4, 5, 6, 7],
            'subsample': [0.6, 0.7, 0.8, 0.9],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9]
        },
        'NeuralRanker': {
            'hidden_units': [
                [128, 64], [256, 128], [256, 128, 64], [128, 64, 32]
            ],
            'learning_rate': [0.0001, 0.0005, 0.001, 0.005],
            'epochs': [10, 15, 20],
            'batch_size': [32, 64, 128],
            'dropout_rate': [0.1, 0.2, 0.3],
            'weight_decay': [1e-6, 1e-5, 1e-4]
        }
    }
    return spaces.get(model_name, {})