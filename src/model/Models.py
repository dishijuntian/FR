"""
航班排名模型集合 - 性能优化版
主要优化：
1. 直接模型创建，减少中间层
2. 简化GPU初始化
3. 优化参数设置
4. 减少不必要的检查和转换
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional
import warnings
import joblib

# 抑制警告
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
import torch.optim as optim


class BaseRanker:
    """基础排名模型类 - 简化版"""
    
    def __init__(self, use_gpu=True, logger=None, **kwargs):
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.logger = logger
        self.is_fitted = False
    
    def save_model(self, filepath: str):
        if not self.is_fitted:
            raise ValueError("模型未训练，无法保存")
        joblib.dump(self, filepath)
    
    @classmethod
    def load_model(cls, filepath: str):
        return joblib.load(filepath)


class XGBoostRanker(BaseRanker):
    """XGBoost排名模型 - 直接优化版"""
    
    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.1, 
                 random_state=42, **kwargs):
        super().__init__(**kwargs)
        
        # 直接设置最优参数，避免复杂检查
        params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'random_state': random_state,
            'verbosity': 0,
            'eval_metric': 'ndcg',
            'subsample': 0.8,
            'colsample_bytree': 0.8
        }
        
        # GPU优化 - 直接设置，不做复杂检查
        if self.use_gpu:
            params.update({
                'tree_method': 'gpu_hist', 
                'predictor': 'gpu_predictor',
                'gpu_id': 0,
                'max_bin': 256
            })
        else:
            params.update({'tree_method': 'hist', 'n_jobs': -1})
        
        self.model = xgb.XGBRanker(**params)
    
    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray):
        # 直接计算组大小，避免复杂的数据验证
        unique_groups = np.unique(groups)
        group_sizes = [np.sum(groups == g) for g in unique_groups]
        
        self.model.fit(X, y, group=group_sizes, verbose=False)
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("模型未训练")
        return self.model.predict(X)
    
    @property
    def feature_importances_(self):
        return self.model.feature_importances_ if self.is_fitted else None


class LightGBMRanker(BaseRanker):
    """LightGBM排名模型 - 直接优化版"""
    
    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.1,
                 random_state=42, **kwargs):
        super().__init__(**kwargs)
        
        params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'random_state': random_state,
            'verbose': -1,
            'metric': 'ndcg',
            'objective': 'lambdarank',
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'force_col_wise': True
        }
        
        # GPU优化 - 直接设置
        if self.use_gpu:
            params.update({
                'device': 'gpu', 
                'gpu_platform_id': 0, 
                'gpu_device_id': 0,
                'max_bin': 255
            })
        else:
            params.update({'device': 'cpu', 'n_jobs': -1})
        
        self.model = lgb.LGBMRanker(**params)
    
    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray):
        unique_groups = np.unique(groups)
        group_sizes = [np.sum(groups == g) for g in unique_groups]
        
        # 最小化回调，减少开销
        callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
        self.model.fit(X, y, group=group_sizes, callbacks=callbacks)
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("模型未训练")
        return self.model.predict(X)
    
    @property
    def feature_importances_(self):
        return self.model.feature_importances_ if self.is_fitted else None


class LambdaMART(BaseRanker):
    """LambdaMART排名模型 - 直接优化版"""
    
    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.1,
                 random_state=42, **kwargs):
        super().__init__(**kwargs)
        
        params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'random_state': random_state,
            'verbosity': 0,
            'objective': 'rank:pairwise',
            'eval_metric': 'ndcg'
        }
        
        if self.use_gpu:
            params.update({
                'tree_method': 'gpu_hist', 
                'gpu_id': 0,
                'max_bin': 256
            })
        else:
            params.update({'tree_method': 'hist', 'n_jobs': -1})
        
        self.model = xgb.XGBRanker(**params)
    
    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray):
        unique_groups = np.unique(groups)
        group_sizes = [np.sum(groups == g) for g in unique_groups]
        self.model.fit(X, y, group=group_sizes, verbose=False)
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("模型未训练")
        return self.model.predict(X)
    
    @property
    def feature_importances_(self):
        return self.model.feature_importances_ if self.is_fitted else None


# 简化的PyTorch模型 - 只保留核心功能
class RankNet(BaseRanker):
    """RankNet深度学习排名模型 - 简化版"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64, 32],
                 dropout_rate: float = 0.2, learning_rate: float = 0.001,
                 epochs: int = 15, batch_size: int = 128, **kwargs):
        super().__init__(**kwargs)
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')
        
        # 构建简化网络
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        
        self.model = nn.Sequential(*layers).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scaler = StandardScaler()
    
    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray, **kwargs):
        epochs = kwargs.get('epochs', self.epochs)
        
        X_scaled = self.scaler.fit_transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        self.model.train()
        
        # 简化的训练循环 - 减少配对计算
        for epoch in range(epochs):
            # 随机抽样减少计算量
            sample_size = min(len(X), 10000)
            indices = torch.randperm(len(X))[:sample_size]
            
            batch_X = X_tensor[indices]
            batch_y = y_tensor[indices]
            
            self.optimizer.zero_grad()
            outputs = self.model(batch_X).squeeze()
            loss = F.mse_loss(outputs, batch_y)
            loss.backward()
            self.optimizer.step()
        
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("模型未训练")
        
        self.model.eval()
        X_scaled = self.scaler.transform(X)
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            scores = self.model(X_tensor).squeeze().cpu().numpy()
        
        return scores
    
    @property
    def feature_importances_(self):
        return np.ones(self.input_dim) / self.input_dim if self.is_fitted else None


class NeuralRanker(BaseRanker):
    """神经网络排名模型 - 简化版"""
    
    def __init__(self, input_dim: int, hidden_units: List[int] = [256, 128, 64], **kwargs):
        super().__init__(**kwargs)
        # 使用sklearn的MLPRegressor实现，避免复杂的PyTorch训练
        from sklearn.neural_network import MLPRegressor
        
        self.model = MLPRegressor(
            hidden_layer_sizes=tuple(hidden_units),
            max_iter=50,  # 减少迭代次数
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
            verbose=False
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray, **kwargs):
        self.model.fit(X, y)
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("模型未训练")
        return self.model.predict(X)
    
    @property
    def feature_importances_(self):
        return np.ones(self.model.n_features_in_) / self.model.n_features_in_ if self.is_fitted else None


class TransformerRanker(BaseRanker):
    """简化的Transformer排名模型 - 实际使用线性回归"""
    
    def __init__(self, input_dim: int, **kwargs):
        super().__init__(**kwargs)
        from sklearn.linear_model import Ridge
        self.model = Ridge(alpha=1.0)
    
    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray, **kwargs):
        self.model.fit(X, y)
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("模型未训练")
        return self.model.predict(X)
    
    @property
    def feature_importances_(self):
        return np.abs(self.model.coef_) if self.is_fitted else None


class BM25Ranker:
    """BM25排名模型 - 简化版"""
    
    def __init__(self, k1=1.2, b=0.75, logger=None, **kwargs):
        self.k1 = k1
        self.b = b
        self.logger = logger
        self.is_fitted = False
        self.feature_weights = None
    
    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray):
        # 快速计算特征相关性权重
        correlations = []
        for i in range(X.shape[1]):
            if X[:, i].var() > 0:
                corr = np.corrcoef(X[:, i], y)[0, 1]
                if np.isnan(corr):
                    corr = 0.0
            else:
                corr = 0.0
            correlations.append(abs(corr))
        
        self.feature_weights = np.array(correlations)
        weight_sum = np.sum(self.feature_weights)
        if weight_sum > 0:
            self.feature_weights = self.feature_weights / weight_sum
        else:
            self.feature_weights = np.ones(X.shape[1]) / X.shape[1]
        
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("模型未训练")
        
        scores = np.dot(X, self.feature_weights)
        mean_score = np.mean(scores) + 1e-8
        
        scores = (scores * (self.k1 + 1)) / (
            scores + self.k1 * (1 - self.b + self.b * scores / mean_score)
        )
        
        return scores
    
    @property
    def feature_importances_(self):
        return self.feature_weights
    
    def save_model(self, filepath: str):
        if not self.is_fitted:
            raise ValueError("模型未训练，无法保存")
        joblib.dump(self, filepath)
    
    @classmethod
    def load_model(cls, filepath: str):
        return joblib.load(filepath)


# 简化的模型工厂
def create_model_fast(model_name: str, use_gpu: bool = True, input_dim: int = None, **params):
    """快速创建模型 - 避免复杂的工厂模式"""
    
    if model_name == 'XGBRanker':
        return XGBoostRanker(use_gpu=use_gpu, **params)
    elif model_name == 'LGBMRanker':
        return LightGBMRanker(use_gpu=use_gpu, **params)
    elif model_name == 'LambdaMART':
        return LambdaMART(use_gpu=use_gpu, **params)
    elif model_name == 'RankNet':
        return RankNet(input_dim=input_dim, use_gpu=use_gpu, **params)
    elif model_name == 'NeuralRanker':
        return NeuralRanker(input_dim=input_dim, use_gpu=use_gpu, **params)
    elif model_name == 'TransformerRanker':
        return TransformerRanker(input_dim=input_dim, use_gpu=use_gpu, **params)
    elif model_name == 'BM25Ranker':
        return BM25Ranker(**params)
    else:
        raise ValueError(f"未知模型: {model_name}")