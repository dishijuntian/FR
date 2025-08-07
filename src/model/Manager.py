"""
航班排名模型集合管理器 - 清洁输出版本
优化日志输出，移除emoji，提供清晰的进度信息
"""

import os
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
import warnings
import joblib
import time
import gc

from .Models import (
    LightGBMRanker, XGBoostRanker, RankNet, 
    LambdaMART, ListNet, TransformerRanker, BM25Ranker, NeuralRanker
)

warnings.filterwarnings('ignore')


class FlightRankingModelsManager:
    """航班排名模型管理器 - 清洁输出版本"""
    
    def __init__(self, use_gpu: bool = True, logger=None):
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.logger = logger or logging.getLogger(__name__)
        self.models: Dict[str, object] = {}
        self.feature_names: List[str] = []
        self._model_configs: Dict = {}
        
        # 模型类型映射
        self.model_classes = {
            'XGBRanker': XGBoostRanker,
            'LGBMRanker': LightGBMRanker,
            'RankNet': RankNet,
            'LambdaMART': LambdaMART,
            'ListNet': ListNet,
            'TransformerRanker': TransformerRanker,
            'BM25Ranker': BM25Ranker,
            'NeuralRanker': NeuralRanker
        }
        
        # PyTorch模型列表
        self.pytorch_models = {'RankNet', 'TransformerRanker', 'NeuralRanker'}
        
        if self.use_gpu:
            gpu_name = torch.cuda.get_device_name(0)
            self.logger.info(f"[GPU] 已启用GPU加速: {gpu_name}")
        else:
            self.logger.info("[CPU] 使用CPU模式")
    
    def create_models(self, input_dim: int, model_configs: Dict = None, 
                     model_names: List[str] = None) -> Dict:
        """创建指定的模型"""
        self.logger.info(f"[MODEL] 开始创建模型: 请求{len(model_names) if model_names else 0}个")
        start_time = time.time()
        
        if model_configs is None:
            model_configs = {}
        
        if model_names is None:
            model_names = list(self.model_classes.keys())
        
        self._model_configs = model_configs
        created_models = {}
        
        # 默认配置
        default_configs = self._get_default_configs(input_dim)
        
        for i, model_name in enumerate(model_names, 1):
            if model_name not in self.model_classes:
                self.logger.warning(f"[MODEL] 跳过不支持的模型: {model_name}")
                continue
            
            self.logger.info(f"[MODEL] 创建 {i}/{len(model_names)}: {model_name}")
            model_start = time.time()
            
            try:
                # 合并配置
                config = default_configs.get(model_name, {}).copy()
                config.update(model_configs.get(model_name, {}))
                
                # 创建模型
                model_class = self.model_classes[model_name]
                
                if model_name == 'BM25Ranker':
                    created_models[model_name] = model_class(logger=self.logger, **config)
                else:
                    created_models[model_name] = model_class(
                        use_gpu=self.use_gpu, logger=self.logger, **config
                    )
                
                model_time = time.time() - model_start
                self.logger.info(f"[MODEL] {model_name} 创建成功: {model_time:.2f}s")
                
            except Exception as e:
                model_time = time.time() - model_start
                self.logger.error(f"[MODEL] {model_name} 创建失败: {e} ({model_time:.2f}s)")
                continue
        
        self.models.update(created_models)
        total_time = time.time() - start_time
        self.logger.info(f"[MODEL] 模型创建完成: {len(created_models)}/{len(model_names)} 成功, 总耗时{total_time:.2f}s")
        
        return created_models
    
    def _get_default_configs(self, input_dim: int) -> Dict:
        """获取默认配置"""
        return {
            'XGBRanker': {'n_estimators': 200, 'max_depth': 8, 'learning_rate': 0.05},
            'LGBMRanker': {'n_estimators': 200, 'max_depth': 8, 'learning_rate': 0.05},
            'RankNet': {'input_dim': input_dim, 'hidden_dims': [128, 64, 32], 'learning_rate': 0.001},
            'LambdaMART': {'n_estimators': 200, 'max_depth': 8, 'learning_rate': 0.05},
            'ListNet': {'n_estimators': 200, 'max_depth': 7, 'learning_rate': 0.05},
            'TransformerRanker': {
                'input_dim': input_dim, 'num_heads': 4, 'num_layers': 2, 
                'd_model': 64, 'learning_rate': 0.001
            },
            'BM25Ranker': {'k1': 1.2, 'b': 0.75},
            'NeuralRanker': {
                'input_dim': input_dim, 'hidden_units': [256, 128, 64], 'learning_rate': 0.001
            }
        }
    
    def prepare_data(self, df: pd.DataFrame, target_col: str = 'selected', **kwargs) -> Tuple:
        """数据预处理 - 清洁输出版本"""
        self.logger.info(f"[PREP] 开始数据预处理: shape={df.shape}")
        prep_start = time.time()
        
        # 1. 数据清理
        if target_col in df.columns:
            clean_start = time.time()
            selected_per_group = df.groupby('ranker_id')[target_col].sum()
            invalid_groups = selected_per_group[selected_per_group != 1].index
            if len(invalid_groups) > 0:
                df = df[~df['ranker_id'].isin(invalid_groups)]
                clean_time = time.time() - clean_start
                self.logger.info(f"[PREP] 数据清理: 移除{len(invalid_groups)}个无效组, 耗时{clean_time:.2f}s")
        
        # 2. 特征选择
        feature_start = time.time()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = {'Id', target_col, 'ranker_id', 'profileId', 'companyID'}
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # 3. 批量零方差检查
        self.logger.info("[PREP] 检查零方差特征")
        variance_start = time.time()
        
        feature_matrix = df[feature_cols].values
        with np.errstate(invalid='ignore', divide='ignore'):
            variances = np.var(feature_matrix, axis=0)
        
        valid_feature_mask = variances > 1e-8
        zero_var_count = np.sum(~valid_feature_mask)
        feature_cols = [col for col, is_valid in zip(feature_cols, valid_feature_mask) if is_valid]
        
        variance_time = time.time() - variance_start
        if zero_var_count > 0:
            self.logger.info(f"[PREP] 特征筛选: 移除{zero_var_count}个零方差特征, 耗时{variance_time:.2f}s")
        
        feature_time = time.time() - feature_start
        self.logger.info(f"[PREP] 特征选择完成: 保留{len(feature_cols)}个特征, 耗时{feature_time:.2f}s")
        
        # 4. 数据提取和类型转换
        extract_start = time.time()
        
        df_features = df[feature_cols].fillna(0)
        X = df_features.values.astype(np.float32)
        y = df[target_col].values.astype(np.float32) if target_col in df.columns else np.zeros(len(df), dtype=np.float32)
        groups = df['ranker_id'].values
        
        extract_time = time.time() - extract_start
        self.logger.info(f"[PREP] 数据提取完成: 耗时{extract_time:.2f}s")
        
        self.feature_names = feature_cols
        total_time = time.time() - prep_start
        self.logger.info(f"[PREP] 数据预处理完成: 输出shape={X.shape}, 总耗时{total_time:.2f}s")
        
        return X, y, groups, feature_cols, df
    
    def train_models(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                    model_names: List[str] = None, **training_kwargs) -> Dict:
        """训练模型"""
        if model_names is None:
            model_names = list(self.models.keys())
        
        trained_models = {}
        total_start = time.time()
        
        self.logger.info(f"[TRAIN] 开始模型训练: {len(model_names)}个模型")
        
        for i, name in enumerate(model_names, 1):
            if name not in self.models:
                self.logger.warning(f"[TRAIN] 跳过不存在的模型: {name}")
                continue
            
            try:
                self.logger.info(f"[TRAIN] 训练 {i}/{len(model_names)}: {name}")
                train_start = time.time()
                
                model = self.models[name]
                
                # 根据模型类型选择训练参数
                if name in self.pytorch_models:
                    epochs = training_kwargs.get('epochs', 50)
                    batch_size = training_kwargs.get('batch_size', 1024)
                    model.fit(X, y, groups, epochs=epochs, batch_size=batch_size)
                else:
                    model.fit(X, y, groups)
                
                trained_models[name] = model
                train_time = time.time() - train_start
                self.logger.info(f"[TRAIN] {name} 训练完成: {train_time:.2f}s")
                
            except Exception as e:
                train_time = time.time() - train_start if 'train_start' in locals() else 0
                self.logger.error(f"[TRAIN] {name} 训练失败: {e} ({train_time:.2f}s)")
                continue
        
        total_time = time.time() - total_start
        self.logger.info(f"[TRAIN] 模型训练完成: {len(trained_models)}/{len(model_names)} 成功, 总耗时{total_time:.2f}s")
        
        return trained_models
    
    def predict_model(self, X: np.ndarray, validation_scores: Dict[str, float] = None, 
                    model_names: List[str] = None) -> np.ndarray:
        """基于验证得分的加权预测"""
        predict_start = time.time()
        
        if model_names is None:
            model_names = list(self.models.keys())
        
        # 过滤可用模型
        available_models = []
        available_scores = {}
        
        for name in model_names:
            if name in self.models and hasattr(self.models[name], 'is_fitted') and self.models[name].is_fitted:
                available_models.append(name)
                available_scores[name] = validation_scores.get(name, 1.0) if validation_scores else 1.0
        
        if not available_models:
            raise ValueError("没有可用的已训练模型")
        
        self.logger.info(f"[PREDICT] 开始集成预测: {len(available_models)}个模型")
        
        # 计算权重
        if validation_scores:
            weights = self._calculate_performance_weights(available_scores)
            self.logger.info(f"[PREDICT] 使用验证得分权重: {dict(zip(available_models, [f'{w:.3f}' for w in weights]))}")
        else:
            weights = [1.0 / len(available_models)] * len(available_models)
            self.logger.info("[PREDICT] 使用等权重")
        
        # 收集预测结果
        predictions = []
        valid_weights = []
        
        for i, model_name in enumerate(available_models):
            try:
                model_pred_start = time.time()
                model = self.models[model_name]
                pred = model.predict(X)
                predictions.append(pred)
                valid_weights.append(weights[i])
                
                model_pred_time = time.time() - model_pred_start
                self.logger.info(f"[PREDICT] {model_name} 预测完成: 权重{weights[i]:.3f}, 耗时{model_pred_time:.2f}s")
            except Exception as e:
                self.logger.warning(f"[PREDICT] {model_name} 预测失败: {e}")
                continue
        
        if not predictions:
            raise ValueError("所有模型预测都失败")
        
        # 加权平均
        predictions = np.array(predictions)
        valid_weights = np.array(valid_weights)
        valid_weights = valid_weights / np.sum(valid_weights)
        
        final_predictions = np.average(predictions, axis=0, weights=valid_weights)
        
        total_time = time.time() - predict_start
        self.logger.info(f"[PREDICT] 集成预测完成: {len(predictions)}个模型, 耗时{total_time:.2f}s")
        
        return final_predictions

    def _calculate_performance_weights(self, scores: Dict[str, float], 
                                    weight_strategy: str = 'softmax') -> List[float]:
        """基于验证得分计算模型权重"""
        if not scores:
            return []
        
        score_array = np.array(list(scores.values()))
        
        if weight_strategy == 'softmax':
            scaled_scores = (score_array - np.min(score_array)) * 10
            exp_scores = np.exp(scaled_scores - np.max(scaled_scores))
            weights = exp_scores / np.sum(exp_scores)
            
        elif weight_strategy == 'linear':
            min_score = np.min(score_array)
            max_score = np.max(score_array)
            if max_score == min_score:
                weights = np.ones(len(score_array)) / len(score_array)
            else:
                normalized_scores = (score_array - min_score) / (max_score - min_score)
                weights = 0.1 + 0.9 * normalized_scores
                weights = weights / np.sum(weights)
                
        else:
            weights = np.ones(len(score_array)) / len(score_array)
        
        return weights.tolist()

    def get_best_model_name(self, validation_scores: Dict[str, float] = None) -> Optional[str]:
        """获取最佳模型名称"""
        if not validation_scores:
            return None
        
        best_model = max(validation_scores.items(), key=lambda x: x[1])
        self.logger.info(f"[BEST] 最佳模型: {best_model[0]} (NDCG@10={best_model[1]:.4f})")
        
        return best_model[0]

    def predict_with_best_model(self, X: np.ndarray, validation_scores: Dict[str, float] = None) -> np.ndarray:
        """使用最佳模型预测"""
        best_model_name = self.get_best_model_name(validation_scores)
        
        if best_model_name is None or best_model_name not in self.models:
            available_models = [name for name in self.models.keys() 
                            if hasattr(self.models[name], 'is_fitted') and self.models[name].is_fitted]
            if not available_models:
                raise ValueError("没有可用的已训练模型")
            best_model_name = available_models[0]
            self.logger.info(f"[BEST] 使用第一个可用模型: {best_model_name}")
        else:
            self.logger.info(f"[BEST] 使用最佳模型: {best_model_name}")
        
        return self.models[best_model_name].predict(X)
    
    def save_models(self, save_dir: str):
        """保存模型"""
        self.logger.info(f"[SAVE] 开始保存模型到: {save_dir}")
        os.makedirs(save_dir, exist_ok=True)
        saved_count = 0
        save_start = time.time()
        
        for name, model in self.models.items():
            if hasattr(model, 'is_fitted') and model.is_fitted:
                filepath = os.path.join(save_dir, f"{name}.pkl")
                try:
                    model_save_start = time.time()
                    model.save_model(filepath)
                    model_save_time = time.time() - model_save_start
                    saved_count += 1
                    self.logger.info(f"[SAVE] {name} 保存完成: {model_save_time:.2f}s")
                except Exception as e:
                    self.logger.error(f"[SAVE] {name} 保存失败: {e}")
        
        # 保存元数据
        if self.feature_names:
            joblib.dump(self.feature_names, os.path.join(save_dir, "features.pkl"))
        joblib.dump(self._model_configs, os.path.join(save_dir, "model_configs.pkl"))
        
        save_time = time.time() - save_start
        self.logger.info(f"[SAVE] 模型保存完成: {saved_count}个模型, 耗时{save_time:.2f}s")
    
    def load_models(self, save_dir: str, model_names: List[str] = None):
        """加载模型"""
        if not os.path.exists(save_dir):
            raise FileNotFoundError(f"模型目录不存在: {save_dir}")
        
        self.logger.info(f"[LOAD] 开始加载模型: {save_dir}")
        load_start = time.time()
        
        if model_names is None:
            model_files = [f for f in os.listdir(save_dir) 
                          if f.endswith('.pkl') and f not in ['features.pkl', 'model_configs.pkl']]
            model_names = [f.replace('.pkl', '') for f in model_files]
        
        loaded_count = 0
        for name in model_names:
            filepath = os.path.join(save_dir, f"{name}.pkl")
            if os.path.exists(filepath) and name in self.model_classes:
                try:
                    model_load_start = time.time()
                    self.models[name] = self.model_classes[name].load_model(filepath)
                    model_load_time = time.time() - model_load_start
                    loaded_count += 1
                    self.logger.info(f"[LOAD] {name} 加载完成: {model_load_time:.2f}s")
                except Exception as e:
                    self.logger.warning(f"[LOAD] {name} 加载失败: {e}")
        
        # 加载元数据
        features_path = os.path.join(save_dir, "features.pkl")
        if os.path.exists(features_path):
            self.feature_names = joblib.load(features_path)
        
        config_path = os.path.join(save_dir, "model_configs.pkl")
        if os.path.exists(config_path):
            self._model_configs = joblib.load(config_path)
        
        load_time = time.time() - load_start
        self.logger.info(f"[LOAD] 模型加载完成: {loaded_count}个模型, 耗时{load_time:.2f}s")
        return loaded_count > 0
    
    def get_model_summary(self) -> Dict:
        """获取模型概要信息"""
        fitted_models = []
        unfitted_models = []
        
        for name, model in self.models.items():
            if hasattr(model, 'is_fitted') and model.is_fitted:
                fitted_models.append(name)
            else:
                unfitted_models.append(name)
        
        return {
            'total_models': len(self.models),
            'fitted_models': fitted_models,
            'unfitted_models': unfitted_models,
            'feature_count': len(self.feature_names),
            'gpu_enabled': self.use_gpu
        }