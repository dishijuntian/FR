"""
航班排名模型集合管理器 - 重构版
专注于模型创建、训练协调和预测集成，移除重复的数据预处理逻辑
"""

import os
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
import warnings
import joblib

from .Models import (
    LightGBMRanker, XGBoostRanker, RankNet, 
    LambdaMART, ListNet, TransformerRanker, BM25Ranker, NeuralRanker
)

warnings.filterwarnings('ignore')


class FlightRankingModelsManager:
    """航班排名模型管理器 - 重构版，专注于模型管理"""
    
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
            self.logger.info(f"GPU可用: {torch.cuda.get_device_name(0)}")
    
    def create_models(self, input_dim: int, model_configs: Dict = None, 
                     model_names: List[str] = None) -> Dict:
        """创建指定的模型"""
        if model_configs is None:
            model_configs = {}
        
        if model_names is None:
            model_names = list(self.model_classes.keys())
        
        self._model_configs = model_configs
        created_models = {}
        
        # 默认配置
        default_configs = self._get_default_configs(input_dim)
        
        for model_name in model_names:
            if model_name not in self.model_classes:
                self.logger.warning(f"不支持的模型: {model_name}")
                continue
            
            try:
                # 合并配置
                config = default_configs.get(model_name, {}).copy()
                config.update(model_configs.get(model_name, {}))
                
                # 创建模型
                model_class = self.model_classes[model_name]
                
                if model_name == 'BM25Ranker':
                    # BM25不需要use_gpu参数
                    created_models[model_name] = model_class(logger=self.logger, **config)
                else:
                    created_models[model_name] = model_class(
                        use_gpu=self.use_gpu, logger=self.logger, **config
                    )
                
                self.logger.info(f"✓ {model_name}模型创建成功")
                
            except Exception as e:
                self.logger.warning(f"✗ {model_name}创建失败: {e}")
                continue
        
        self.models.update(created_models)
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
        """简化的数据预处理（主要数据处理由DataProcessor负责）"""
        # 基本数据清理
        if target_col in df.columns:
            selected_per_group = df.groupby('ranker_id')[target_col].sum()
            invalid_groups = selected_per_group[selected_per_group != 1].index
            if len(invalid_groups) > 0:
                df = df[~df['ranker_id'].isin(invalid_groups)]
                self.logger.info(f"移除 {len(invalid_groups)} 个无效组")
        
        # 特征选择
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['Id', target_col, 'ranker_id', 'profileId', 'companyID']
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # 移除零方差特征
        feature_data = df[feature_cols]
        zero_var_cols = feature_data.columns[feature_data.var() == 0].tolist()
        if zero_var_cols:
            feature_cols = [col for col in feature_cols if col not in zero_var_cols]
            self.logger.info(f"移除 {len(zero_var_cols)} 个零方差特征")
        
        # 处理缺失值和转换数据类型
        df_features = df[feature_cols].fillna(df[feature_cols].median())
        X = df_features.values.astype(np.float32)
        y = df[target_col].values if target_col in df.columns else np.zeros(len(df))
        groups = df['ranker_id'].values
        
        self.feature_names = feature_cols
        self.logger.info(f"数据预处理完成: {X.shape}, 特征数: {len(feature_cols)}")
        
        return X, y, groups, feature_cols, df
    
    def train_models(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                    model_names: List[str] = None, **training_kwargs) -> Dict:
        """训练模型"""
        if model_names is None:
            model_names = list(self.models.keys())
        
        trained_models = {}
        
        for name in model_names:
            if name not in self.models:
                self.logger.warning(f"模型 {name} 不存在，跳过")
                continue
            
            try:
                self.logger.info(f"开始训练 {name}...")
                model = self.models[name]
                
                # 根据模型类型选择训练参数
                if name in self.pytorch_models:
                    epochs = training_kwargs.get('epochs', 50)
                    batch_size = training_kwargs.get('batch_size', 1024)
                    model.fit(X, y, groups, epochs=epochs, batch_size=batch_size)
                else:
                    model.fit(X, y, groups)
                
                trained_models[name] = model
                self.logger.info(f"✓ {name} 训练完成")
                
            except Exception as e:
                self.logger.error(f"✗ {name} 训练失败: {e}")
                continue
        
        return trained_models
    
    def predict_model(self, X: np.ndarray, validation_scores: Dict[str, float] = None, 
                    model_names: List[str] = None) -> np.ndarray:
        """
        基于验证得分的加权预测函数
        
        Args:
            X: 特征矩阵
            validation_scores: 各模型的验证得分字典 {model_name: score}
            model_names: 要使用的模型名称列表，如果为None则使用所有已加载的模型
        
        Returns:
            预测分数数组
        """
        if model_names is None:
            model_names = list(self.models.keys())
        
        # 过滤出可用的模型
        available_models = []
        available_scores = {}
        
        for name in model_names:
            if name in self.models and hasattr(self.models[name], 'is_fitted') and self.models[name].is_fitted:
                available_models.append(name)
                # 如果有验证分数，使用验证分数，否则设为相等权重
                if validation_scores and name in validation_scores:
                    available_scores[name] = validation_scores[name]
                else:
                    available_scores[name] = 1.0  # 默认权重
        
        if not available_models:
            raise ValueError("没有可用的已训练模型")
        
        self.logger.info(f"使用 {len(available_models)} 个模型进行预测: {available_models}")
        
        # 计算加权平均的权重
        if validation_scores:
            # 使用验证得分计算权重（得分越高权重越大）
            weights = self._calculate_performance_weights(available_scores)
            self.logger.info(f"基于验证得分的权重: {dict(zip(available_models, weights))}")
        else:
            # 等权重
            weights = [1.0 / len(available_models)] * len(available_models)
            self.logger.info("使用等权重预测")
        
        # 收集所有模型的预测结果
        predictions = []
        valid_weights = []
        
        for i, model_name in enumerate(available_models):
            try:
                model = self.models[model_name]
                pred = model.predict(X)
                predictions.append(pred)
                valid_weights.append(weights[i])
                self.logger.info(f"✓ {model_name} 预测完成，权重: {weights[i]:.4f}")
            except Exception as e:
                self.logger.warning(f"✗ {model_name} 预测失败: {e}")
                continue
        
        if not predictions:
            raise ValueError("所有模型预测都失败")
        
        # 加权平均预测
        predictions = np.array(predictions)
        valid_weights = np.array(valid_weights)
        
        # 重新归一化权重
        valid_weights = valid_weights / np.sum(valid_weights)
        
        # 计算加权平均
        final_predictions = np.average(predictions, axis=0, weights=valid_weights)
        
        self.logger.info(f"✓ 加权集成预测完成，使用 {len(predictions)} 个模型")
        
        return final_predictions

    def _calculate_performance_weights(self, scores: Dict[str, float], 
                                    weight_strategy: str = 'softmax') -> List[float]:
        """
        基于验证得分计算模型权重
        
        Args:
            scores: 各模型的验证得分
            weight_strategy: 权重计算策略 ('softmax', 'linear', 'rank')
        
        Returns:
            权重列表
        """
        if not scores:
            return []
        
        score_values = list(scores.values())
        score_array = np.array(score_values)
        
        if weight_strategy == 'softmax':
            # 使用softmax函数，突出最好的模型
            # 先放大差异，然后应用softmax
            scaled_scores = (score_array - np.min(score_array)) * 10
            exp_scores = np.exp(scaled_scores)
            weights = exp_scores / np.sum(exp_scores)
            
        elif weight_strategy == 'linear':
            # 线性归一化权重
            min_score = np.min(score_array)
            max_score = np.max(score_array)
            if max_score == min_score:
                weights = np.ones(len(score_array)) / len(score_array)
            else:
                normalized_scores = (score_array - min_score) / (max_score - min_score)
                # 避免权重为0，给最小权重0.1
                weights = 0.1 + 0.9 * normalized_scores
                weights = weights / np.sum(weights)
                
        elif weight_strategy == 'rank':
            # 基于排名的权重
            ranks = len(score_array) - np.argsort(np.argsort(score_array))
            weights = ranks / np.sum(ranks)
            
        else:
            # 默认等权重
            weights = np.ones(len(score_array)) / len(score_array)
        
        return weights.tolist()

    def get_best_model_name(self, validation_scores: Dict[str, float] = None) -> Optional[str]:
        """
        获取验证得分最好的模型名称
        
        Args:
            validation_scores: 验证得分字典
        
        Returns:
            最好模型的名称，如果没有得分则返回None
        """
        if not validation_scores:
            return None
        
        # 找到得分最高的模型
        best_model = max(validation_scores.items(), key=lambda x: x[1])
        self.logger.info(f"最佳模型: {best_model[0]} (NDCG@10: {best_model[1]:.4f})")
        
        return best_model[0]

    def predict_with_best_model(self, X: np.ndarray, validation_scores: Dict[str, float] = None) -> np.ndarray:
        """
        使用验证得分最好的单个模型进行预测
        
        Args:
            X: 特征矩阵
            validation_scores: 验证得分字典
        
        Returns:
            预测分数数组
        """
        best_model_name = self.get_best_model_name(validation_scores)
        
        if best_model_name is None or best_model_name not in self.models:
            # 如果没有验证得分或最佳模型不可用，使用第一个可用模型
            available_models = [name for name in self.models.keys() 
                            if hasattr(self.models[name], 'is_fitted') and self.models[name].is_fitted]
            if not available_models:
                raise ValueError("没有可用的已训练模型")
            best_model_name = available_models[0]
            self.logger.info(f"使用第一个可用模型: {best_model_name}")
        else:
            self.logger.info(f"使用最佳模型进行预测: {best_model_name}")
        
        return self.models[best_model_name].predict(X)
    
    def save_models(self, save_dir: str):
        """保存模型"""
        os.makedirs(save_dir, exist_ok=True)
        saved_count = 0
        
        for name, model in self.models.items():
            if hasattr(model, 'is_fitted') and model.is_fitted:
                filepath = os.path.join(save_dir, f"{name}.pkl")
                try:
                    model.save_model(filepath)
                    saved_count += 1
                    self.logger.info(f"✓ 保存模型: {name}")
                except Exception as e:
                    self.logger.error(f"✗ 保存模型失败 {name}: {e}")
        
        # 保存特征名称和配置
        if self.feature_names:
            joblib.dump(self.feature_names, os.path.join(save_dir, "features.pkl"))
        joblib.dump(self._model_configs, os.path.join(save_dir, "model_configs.pkl"))
        
        self.logger.info(f"已保存 {saved_count} 个模型到: {save_dir}")
    
    def load_models(self, save_dir: str, model_names: List[str] = None):
        """加载模型"""
        if not os.path.exists(save_dir):
            raise FileNotFoundError(f"模型目录不存在: {save_dir}")
        
        if model_names is None:
            model_files = [f for f in os.listdir(save_dir) 
                          if f.endswith('.pkl') and f not in ['features.pkl', 'model_configs.pkl']]
            model_names = [f.replace('.pkl', '') for f in model_files]
        
        loaded_count = 0
        for name in model_names:
            filepath = os.path.join(save_dir, f"{name}.pkl")
            if os.path.exists(filepath) and name in self.model_classes:
                try:
                    self.models[name] = self.model_classes[name].load_model(filepath)
                    loaded_count += 1
                    self.logger.info(f"✓ 加载模型: {name}")
                except Exception as e:
                    self.logger.warning(f"✗ 加载模型失败 {name}: {e}")
        
        # 加载特征名称和配置
        features_path = os.path.join(save_dir, "features.pkl")
        if os.path.exists(features_path):
            self.feature_names = joblib.load(features_path)
        
        config_path = os.path.join(save_dir, "model_configs.pkl")
        if os.path.exists(config_path):
            self._model_configs = joblib.load(config_path)
        
        self.logger.info(f"成功加载 {loaded_count} 个模型")
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