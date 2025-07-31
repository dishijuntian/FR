"""
简化版航班排名模型集合
支持XGBoost、LightGBM和RankNet
"""

import os
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging
import warnings
import joblib

from .Models import LightGBMRanker, XGBoostRanker, RankNet


warnings.filterwarnings('ignore')


class FlightRankingModelsManager:
    """航班排名模型管理器"""
    
    def __init__(self, use_gpu: bool = True, logger=None):
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.logger = logger or logging.getLogger(__name__)
        self.models: Dict[str, object] = {}
        self.feature_names: List[str] = []
        
        if self.use_gpu:
            self.logger.info(f"GPU可用: {torch.cuda.get_device_name(0)}")
        else:
            self.logger.info("使用CPU模式")
    
    def create_models(self, input_dim: int, model_configs: Dict = None) -> Dict:
        """创建模型"""
        if model_configs is None:
            model_configs = {}
        
        default_configs = {
            'XGBRanker': {'n_estimators': 200, 'max_depth': 8, 'learning_rate': 0.05},
            'LGBMRanker': {'n_estimators': 200, 'max_depth': 8, 'learning_rate': 0.05},
            'RankNet': {'input_dim': input_dim, 'hidden_dims': [128, 64, 32]}
        }
        
        # 更新配置
        for model_name in default_configs:
            if model_name in model_configs:
                default_configs[model_name].update(model_configs[model_name])
        
        created_models = {}
        
        # XGBoost
        try:
            created_models['XGBRanker'] = XGBoostRanker(
                use_gpu=self.use_gpu, 
                logger=self.logger,
                **default_configs['XGBRanker']
            )
            self.logger.info("✓ XGBoost模型创建成功")
        except Exception as e:
            self.logger.warning(f"✗ XGBoost创建失败: {e}")
        
        # LightGBM
        try:
            created_models['LGBMRanker'] = LightGBMRanker(
                use_gpu=self.use_gpu,
                logger=self.logger,
                **default_configs['LGBMRanker']
            )
            self.logger.info("✓ LightGBM模型创建成功")
        except Exception as e:
            self.logger.warning(f"✗ LightGBM创建失败: {e}")
        
        # RankNet
        try:
            created_models['RankNet'] = RankNet(
                use_gpu=self.use_gpu,
                logger=self.logger,
                **default_configs['RankNet']
            )
            self.logger.info("✓ RankNet模型创建成功")
        except Exception as e:
            self.logger.warning(f"✗ RankNet创建失败: {e}")
        
        self.models.update(created_models)
        return created_models
    
    def prepare_data(self, df: pd.DataFrame, target_col: str = 'selected') -> Tuple:
        """数据预处理"""
        # 数据清理
        if target_col in df.columns:
            selected_per_group = df.groupby('ranker_id')[target_col].sum()
            invalid_groups = selected_per_group[selected_per_group != 1].index
            if len(invalid_groups) > 0:
                df = df[~df['ranker_id'].isin(invalid_groups)]
        
        # 特征选择
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['Id', target_col, 'ranker_id', 'profileId', 'companyID']
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # 处理缺失值
        df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())
        
        X = df[feature_cols].values.astype(np.float32)
        y = df[target_col].values if target_col in df.columns else np.zeros(len(df))
        groups = df['ranker_id'].values
        
        self.feature_names = feature_cols
        return X, y, groups, feature_cols, df
    
    def train_models(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                    model_names: List[str] = None):
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
                
                if name == 'RankNet':
                    model.fit(X, y, groups, epochs=50)
                else:
                    model.fit(X, y, groups)
                
                trained_models[name] = model
                self.logger.info(f"✓ {name} 训练完成")
                
            except Exception as e:
                self.logger.error(f"✗ {name} 训练失败: {e}")
                continue
        
        return trained_models
    
    def predict_ensemble(self, X: np.ndarray, model_names: List[str] = None,
                        weights: List[float] = None) -> np.ndarray:
        """集成预测"""
        if model_names is None:
            model_names = [name for name, model in self.models.items() if model.is_fitted]
        
        if not model_names:
            raise ValueError("没有已训练的模型可用于预测")
        
        if weights is None:
            weights = [1.0] * len(model_names)
        
        if len(weights) != len(model_names):
            raise ValueError("权重数量必须与模型数量相同")
        
        weights = np.array(weights) / np.sum(weights)
        
        predictions = []
        valid_weights = []
        
        for i, name in enumerate(model_names):
            if name in self.models and self.models[name].is_fitted:
                try:
                    pred = self.models[name].predict(X)
                    predictions.append(pred)
                    valid_weights.append(weights[i])
                except Exception as e:
                    self.logger.warning(f"✗ {name} 预测失败: {e}")
        
        if not predictions:
            raise ValueError("所有模型预测都失败")
        
        valid_weights = np.array(valid_weights) / np.sum(valid_weights)
        ensemble_pred = np.average(predictions, axis=0, weights=valid_weights)
        
        return ensemble_pred
    
    def save_models(self, save_dir: str):
        """保存模型"""
        os.makedirs(save_dir, exist_ok=True)
        
        for name, model in self.models.items():
            if model.is_fitted:
                filepath = os.path.join(save_dir, f"{name}.pkl")
                model.save_model(filepath)
        
        # 保存特征名称
        feature_path = os.path.join(save_dir, "features.pkl")
        joblib.dump(self.feature_names, feature_path)
        
        self.logger.info(f"所有模型已保存到: {save_dir}")
    
    def load_models(self, save_dir: str, model_names: List[str] = None):
        """加载模型"""
        if model_names is None:
            model_files = [f for f in os.listdir(save_dir) if f.endswith('.pkl') and f != 'features.pkl']
            model_names = [f.replace('.pkl', '') for f in model_files]
        
        for name in model_names:
            filepath = os.path.join(save_dir, f"{name}.pkl")
            if os.path.exists(filepath):
                try:
                    if name == 'XGBRanker':
                        self.models[name] = XGBoostRanker.load_model(filepath)
                    elif name == 'LGBMRanker':
                        self.models[name] = LightGBMRanker.load_model(filepath)
                    elif name == 'RankNet':
                        self.models[name] = RankNet.load_model(filepath)
                    self.logger.info(f"✓ 加载模型: {name}")
                except Exception as e:
                    self.logger.warning(f"✗ 加载模型失败 {name}: {e}")
        
        # 加载特征名称
        feature_path = os.path.join(save_dir, "features.pkl")
        if os.path.exists(feature_path):
            self.feature_names = joblib.load(feature_path)