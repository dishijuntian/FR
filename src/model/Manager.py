"""
航班排名模型管理器
"""

import os
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
import warnings
import joblib

# from .Models import LightGBMRanker, XGBoostRanker, RankNet, GraphRanker, CNNRanker, TransformerRanker
from .Models import LightGBMRanker, XGBoostRanker, RankNet

warnings.filterwarnings('ignore')


class FlightRankingModelsManager:
    """简化版航班排名模型管理器"""
    
    def __init__(self, use_gpu: bool = True, logger=None):
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.logger = logger or logging.getLogger(__name__)
        self.models: Dict[str, object] = {}
        self.feature_names: List[str] = []
        self._model_configs: Dict = {}
        
        if self.use_gpu:
            self.logger.info(f"GPU可用: {torch.cuda.get_device_name(0)}")
        else:
            self.logger.info("使用CPU模式")
    
    def create_models(self, model_names: List[str], input_dim: int, model_configs: Dict = None) -> Dict[str, object]:
        """批量创建多个模型实例"""
        if model_configs is None:
            model_configs = {}
        
        created_models = {}
        for model_name in model_names:
            model = self.create_model(model_name, input_dim, model_configs.get(model_name, {}))
            if model is not None:
                created_models[model_name] = model
        
        return created_models
    
    def create_model(self, model_name: str, input_dim: int, model_config: Dict = None) -> object:
        """创建单个模型实例"""
        if model_config is None:
            model_config = {}
        
        # 保存配置
        self._model_configs[model_name] = model_config
        
        # 默认配置
        default_configs = {
            'XGBRanker': {
                'n_estimators': 200, 
                'max_depth': 8, 
                'learning_rate': 0.05
            },
            'LGBMRanker': {
                'n_estimators': 200, 
                'max_depth': 8, 
                'learning_rate': 0.05
            },
            'RankNet': {
                'input_dim': input_dim, 
                'hidden_dims': [128, 64, 32],
                'learning_rate': 0.001,
                'dropout_rate': 0.2
            },
            'GraphRanker': {
                'input_dim': input_dim,
                'hidden_dims': [64, 32],
                'num_gnn_layers': 3,
                'learning_rate': 0.001,
                'dropout_rate': 0.2
            },
            'CNNRanker': {
                'input_dim': input_dim,
                'sequence_length': 10,
                'conv_channels': [32, 64, 128],
                'kernel_sizes': [3, 5, 7],
                'hidden_dims': [128, 64],
                'learning_rate': 0.001,
                'dropout_rate': 0.2
            },
            'TransformerRanker': {
                'input_dim': input_dim,
                'd_model': 128,
                'nhead': 8,
                'num_layers': 3,
                'dim_feedforward': 512,
                'learning_rate': 0.001,
                'dropout_rate': 0.1
            }
        }
        
        # 合并用户配置
        config = default_configs.get(model_name, {})
        config.update(model_config)
        
        try:
            if model_name == 'XGBRanker':
                model = XGBoostRanker(
                    use_gpu=self.use_gpu, 
                    logger=self.logger,
                    **config
                )
            elif model_name == 'LGBMRanker':
                model = LightGBMRanker(
                    use_gpu=self.use_gpu,
                    logger=self.logger,
                    **config
                )
            elif model_name == 'RankNet':
                model = RankNet(
                    use_gpu=self.use_gpu,
                    logger=self.logger,
                    **config
                )
            elif model_name == 'GraphRanker':
                model = GraphRanker(
                    use_gpu=self.use_gpu,
                    logger=self.logger,
                    **config
                )
            elif model_name == 'CNNRanker':
                model = CNNRanker(
                    use_gpu=self.use_gpu,
                    logger=self.logger,
                    **config
                )
            elif model_name == 'TransformerRanker':
                model = TransformerRanker(
                    use_gpu=self.use_gpu,
                    logger=self.logger,
                    **config
                )
            else:
                raise ValueError(f"不支持的模型类型: {model_name}")
            
            self.models[model_name] = model
            self.logger.info(f"✓ {model_name}模型创建成功")
            return model
            
        except Exception as e:
            self.logger.error(f"✗ {model_name}创建失败: {e}")
            return None
    
    def prepare_data(self, df: pd.DataFrame, target_col: str = 'selected') -> Tuple:
        """数据预处理"""
        original_shape = df.shape
        
        # 数据清理 - 移除无效组
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
        
        # 过滤掉方差为0的特征
        feature_data = df[feature_cols]
        zero_var_cols = feature_data.columns[feature_data.var() == 0].tolist()
        if zero_var_cols:
            feature_cols = [col for col in feature_cols if col not in zero_var_cols]
            self.logger.info(f"移除 {len(zero_var_cols)} 个零方差特征")
        
        # 处理缺失值
        df_features = df[feature_cols].copy()
        numeric_medians = df_features.median()
        df_features = df_features.fillna(numeric_medians)
        
        # 转换为numpy数组
        X = df_features.values.astype(np.float32)
        y = df[target_col].values if target_col in df.columns else np.zeros(len(df))
        groups = df['ranker_id'].values
        
        self.feature_names = feature_cols
        
        self.logger.info(f"数据预处理完成: {original_shape} → {X.shape}, "
                        f"特征数: {len(feature_cols)}")
        
        return X, y, groups, feature_cols, df
    
    def train_model(self, model_name: str, X: np.ndarray, y: np.ndarray, 
                   groups: np.ndarray, **training_kwargs) -> bool:
        """训练单个模型"""
        if model_name not in self.models:
            self.logger.warning(f"模型 {model_name} 不存在")
            return False
        
        try:
            self.logger.info(f"开始训练 {model_name}...")
            model = self.models[model_name]
            
            # 根据模型类型选择训练参数
            if model_name in ['RankNet', 'GraphRanker', 'CNNRanker', 'TransformerRanker']:
                epochs = training_kwargs.get('epochs', 100)
                if model_name in ['GraphRanker', 'TransformerRanker']:
                    # 这些模型需要groups参数进行预测
                    model.fit(X, y, groups, epochs=epochs)
                else:
                    model.fit(X, y, groups, epochs=epochs)
            else:
                # XGBoost和LightGBM
                model.fit(X, y, groups)
            
            self.logger.info(f"✓ {model_name} 训练完成")
            return True
            
        except Exception as e:
            self.logger.error(f"✗ {model_name} 训练失败: {e}")
            return False
    
    def predict_model(self, model_name: str, X: np.ndarray, 
                     groups: np.ndarray = None) -> np.ndarray:
        """使用单个模型进行预测"""
        if model_name not in self.models:
            raise ValueError(f"模型 {model_name} 不存在")
        
        model = self.models[model_name]
        if not hasattr(model, 'is_fitted') or not model.is_fitted:
            raise ValueError(f"模型 {model_name} 未训练")
        
        try:
            # 根据模型类型选择预测方法
            if model_name in ['GraphRanker', 'TransformerRanker']:
                if groups is None:
                    raise ValueError(f"{model_name} 预测需要groups参数")
                predictions = model.predict(X, groups)
            else:
                predictions = model.predict(X)
            
            self.logger.debug(f"{model_name} 预测完成")
            return predictions
            
        except Exception as e:
            self.logger.error(f"✗ {model_name} 预测失败: {e}")
            raise
    
    def get_model_summary(self) -> Dict:
        """获取模型概要信息"""
        summary = {
            'total_models': len(self.models),
            'fitted_models': [],
            'unfitted_models': [],
            'feature_count': len(self.feature_names),
            'gpu_enabled': self.use_gpu,
            'available_models': [
                'XGBRanker', 'LGBMRanker', 'RankNet', 
                'GraphRanker', 'CNNRanker', 'TransformerRanker'
            ]
        }
        
        for name, model in self.models.items():
            if hasattr(model, 'is_fitted') and model.is_fitted:
                summary['fitted_models'].append(name)
            else:
                summary['unfitted_models'].append(name)
        
        return summary
    
    def save_model(self, model_name: str, save_dir: str):
        """保存单个模型"""
        if model_name not in self.models:
            raise ValueError(f"模型 {model_name} 不存在")
        
        model = self.models[model_name]
        if not hasattr(model, 'is_fitted') or not model.is_fitted:
            raise ValueError(f"模型 {model_name} 未训练，无法保存")
        
        os.makedirs(save_dir, exist_ok=True)
        
        try:
            filepath = os.path.join(save_dir, f"{model_name}.pkl")
            model.save_model(filepath)
            self.logger.info(f"✓ 保存模型: {model_name}")
            
            # 保存特征名称
            if self.feature_names:
                feature_path = os.path.join(save_dir, "features.pkl")
                joblib.dump(self.feature_names, feature_path)
            
            # 保存模型配置
            config_path = os.path.join(save_dir, "model_configs.pkl")
            joblib.dump(self._model_configs, config_path)
            
        except Exception as e:
            self.logger.error(f"✗ 保存模型失败 {model_name}: {e}")
            raise
    
    def load_model(self, model_name: str, save_dir: str):
        """加载单个模型"""
        filepath = os.path.join(save_dir, f"{model_name}.pkl")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"模型文件不存在: {filepath}")
        
        try:
            if model_name == 'XGBRanker':
                self.models[model_name] = XGBoostRanker.load_model(filepath)
            elif model_name == 'LGBMRanker':
                self.models[model_name] = LightGBMRanker.load_model(filepath)
            elif model_name == 'RankNet':
                self.models[model_name] = RankNet.load_model(filepath)
            elif model_name == 'GraphRanker':
                self.models[model_name] = GraphRanker.load_model(filepath)
            elif model_name == 'CNNRanker':
                self.models[model_name] = CNNRanker.load_model(filepath)
            elif model_name == 'TransformerRanker':
                self.models[model_name] = TransformerRanker.load_model(filepath)
            else:
                raise ValueError(f"未知模型类型: {model_name}")
            
            self.logger.info(f"✓ 加载模型: {model_name}")
            
            # 加载特征名称
            feature_path = os.path.join(save_dir, "features.pkl")
            if os.path.exists(feature_path):
                self.feature_names = joblib.load(feature_path)
                self.logger.info(f"✓ 加载特征名称: {len(self.feature_names)} 个")
            
            # 加载模型配置
            config_path = os.path.join(save_dir, "model_configs.pkl")
            if os.path.exists(config_path):
                self._model_configs = joblib.load(config_path)
                self.logger.info("✓ 加载模型配置")
            
            return True
            
        except Exception as e:
            self.logger.error(f"✗ 加载模型失败 {model_name}: {e}")
            return False
    
    def clear_models(self):
        """清空模型缓存"""
        self.models.clear()
        self.feature_names.clear()
        self._model_configs.clear()
        self.logger.info("模型缓存已清空")
    
    def validate_model(self, model_name: str) -> bool:
        """验证单个模型状态"""
        if model_name not in self.models:
            return False
        
        try:
            model = self.models[model_name]
            
            # 检查模型是否已训练
            is_fitted = hasattr(model, 'is_fitted') and model.is_fitted
            
            # 检查模型是否可以预测（使用小数据测试）
            if is_fitted and self.feature_names:
                test_X = np.random.random((10, len(self.feature_names))).astype(np.float32)
                
                if model_name in ['GraphRanker', 'TransformerRanker']:
                    test_groups = np.array([0] * 5 + [1] * 5)
                    _ = model.predict(test_X, test_groups)
                else:
                    _ = model.predict(test_X)
                
                return True
            else:
                return False
                
        except Exception as e:
            self.logger.warning(f"模型 {model_name} 验证失败: {e}")
            return False