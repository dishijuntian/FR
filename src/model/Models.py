"""
航班排名模型集合 - 重构版
简化模型实现，移除重复的配置和通用逻辑
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import warnings
import joblib
import math

import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
import torch.optim as optim

warnings.filterwarnings('ignore')


class BaseRanker:
    """基础排名模型类 - 避免重复代码"""
    
    def __init__(self, use_gpu=True, logger=None, **kwargs):
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.logger = logger or logging.getLogger(__name__)
        self.is_fitted = False
    
    def save_model(self, filepath: str):
        """保存模型"""
        if not self.is_fitted:
            raise ValueError("模型未训练，无法保存")
        joblib.dump(self, filepath)
    
    @classmethod
    def load_model(cls, filepath: str):
        """加载模型"""
        return joblib.load(filepath)


class XGBoostRanker(BaseRanker):
    """XGBoost排名模型"""
    
    def __init__(self, n_estimators=200, max_depth=8, learning_rate=0.05, 
                 random_state=42, **kwargs):
        super().__init__(**kwargs)
        
        params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'random_state': random_state,
            'verbosity': 0,
            'eval_metric': 'ndcg'
        }
        
        if self.use_gpu:
            try:
                params.update({'tree_method': 'gpu_hist', 'gpu_id': 0})
                # 测试GPU可用性
                test_model = xgb.XGBRanker(**params)
                test_X, test_y = np.random.random((10, 5)), np.random.randint(0, 3, 10)
                test_model.fit(test_X, test_y, group=[5, 5])
                self.logger.info("XGBoost GPU加速可用")
            except Exception as e:
                self.logger.warning(f"XGBoost GPU失败，使用CPU: {e}")
                params.update({'tree_method': 'hist', 'n_jobs': -1})
                self.use_gpu = False
        else:
            params.update({'tree_method': 'hist', 'n_jobs': -1})
        
        self.model = xgb.XGBRanker(**params)
    
    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray):
        unique_groups = np.unique(groups)
        group_sizes = [np.sum(groups == g) for g in unique_groups]
        self.model.fit(X, y, group=group_sizes)
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("模型未训练")
        return self.model.predict(X)
    
    @property
    def feature_importances_(self):
        return self.model.feature_importances_ if self.is_fitted else None


class LightGBMRanker(BaseRanker):
    """LightGBM排名模型"""
    
    def __init__(self, n_estimators=200, max_depth=8, learning_rate=0.05,
                 random_state=42, **kwargs):
        super().__init__(**kwargs)
        
        params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'random_state': random_state,
            'verbose': -1,
            'metric': 'ndcg'
        }
        
        if self.use_gpu:
            try:
                params.update({'device': 'gpu', 'gpu_platform_id': 0, 'gpu_device_id': 0})
                test_model = lgb.LGBMRanker(**params)
                test_X, test_y = np.random.random((10, 5)), np.random.randint(0, 3, 10)
                test_model.fit(test_X, test_y, group=[5, 5])
                self.logger.info("LightGBM GPU加速可用")
            except Exception as e:
                self.logger.warning(f"LightGBM GPU失败，使用CPU: {e}")
                params.update({'device': 'cpu', 'n_jobs': -1})
                self.use_gpu = False
        else:
            params.update({'device': 'cpu', 'n_jobs': -1})
        
        self.model = lgb.LGBMRanker(**params)
    
    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray):
        unique_groups = np.unique(groups)
        group_sizes = [np.sum(groups == g) for g in unique_groups]
        self.model.fit(X, y, group=group_sizes)
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("模型未训练")
        return self.model.predict(X)
    
    @property
    def feature_importances_(self):
        return self.model.feature_importances_ if self.is_fitted else None


class LambdaMART(BaseRanker):
    """LambdaMART排名模型"""
    
    def __init__(self, n_estimators=200, max_depth=8, learning_rate=0.05,
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
            params.update({'tree_method': 'gpu_hist', 'gpu_id': 0})
        else:
            params.update({'tree_method': 'hist', 'n_jobs': -1})
        
        self.model = xgb.XGBRanker(**params)
    
    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray):
        unique_groups = np.unique(groups)
        group_sizes = [np.sum(groups == g) for g in unique_groups]
        self.model.fit(X, y, group=group_sizes)
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("模型未训练")
        return self.model.predict(X)
    
    @property
    def feature_importances_(self):
        return self.model.feature_importances_ if self.is_fitted else None


class ListNet(BaseRanker):
    """ListNet排名模型"""
    
    def __init__(self, n_estimators=200, max_depth=8, learning_rate=0.05,
                 random_state=42, **kwargs):
        super().__init__(**kwargs)
        
        params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'random_state': random_state,
            'verbose': -1,
            'objective': 'lambdarank',
            'metric': 'ndcg'
        }
        
        if self.use_gpu:
            params.update({'device': 'gpu', 'gpu_platform_id': 0, 'gpu_device_id': 0})
        else:
            params.update({'device': 'cpu', 'n_jobs': -1})
        
        self.model = lgb.LGBMRanker(**params)
    
    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray):
        unique_groups = np.unique(groups)
        group_sizes = [np.sum(groups == g) for g in unique_groups]
        self.model.fit(X, y, group=group_sizes)
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("模型未训练")
        return self.model.predict(X)
    
    @property
    def feature_importances_(self):
        return self.model.feature_importances_ if self.is_fitted else None


class BM25Ranker:
    """BM25排名模型"""
    
    def __init__(self, k1=1.2, b=0.75, logger=None, **kwargs):
        self.k1 = k1
        self.b = b
        self.logger = logger or logging.getLogger(__name__)
        self.is_fitted = False
        self.feature_weights = None
    
    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray):
        # 计算特征相关性权重
        feature_weights = []
        for i in range(X.shape[1]):
            correlation = np.corrcoef(X[:, i], y)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
            feature_weights.append(abs(correlation))
        
        self.feature_weights = np.array(feature_weights)
        if np.sum(self.feature_weights) > 0:
            self.feature_weights = self.feature_weights / np.sum(self.feature_weights)
        else:
            self.feature_weights = np.ones(X.shape[1]) / X.shape[1]
        
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("模型未训练")
        
        scores = np.dot(X, self.feature_weights)
        mean_score = np.mean(scores) + 1e-8
        scores = (scores * (self.k1 + 1)) / (scores + self.k1 * (1 - self.b + self.b * scores / mean_score))
        
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


class RankNet(BaseRanker):
    """RankNet深度学习排名模型"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64, 32],
                 dropout_rate: float = 0.2, learning_rate: float = 0.001,
                 random_state: int = 42, **kwargs):
        super().__init__(**kwargs)
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')
        
        # 构建网络
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
        
        self.model = nn.Sequential(*layers)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scaler = StandardScaler()
        
        if self.use_gpu:
            self.model = self.model.to(self.device)
    
    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray, 
            epochs: int = 100, batch_size: int = 1024):
        X_scaled = self.scaler.fit_transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        self.model.train()
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            unique_groups = np.unique(groups)
            for group_id in unique_groups:
                group_mask = groups == group_id
                group_X = X_tensor[group_mask]
                group_y = y_tensor[group_mask]
                
                if len(group_X) < 2:
                    continue
                
                indices = torch.arange(len(group_X), device=self.device)
                pairs = torch.combinations(indices, 2)
                
                for i in range(0, len(pairs), batch_size):
                    batch_pairs = pairs[i:i+batch_size]
                    idx1, idx2 = batch_pairs[:, 0], batch_pairs[:, 1]
                    
                    scores1 = self.model(group_X[idx1]).squeeze()
                    scores2 = self.model(group_X[idx2]).squeeze()
                    
                    label_diff = group_y[idx1] - group_y[idx2]
                    score_diff = scores1 - scores2
                    
                    mask = (label_diff != 0)
                    if mask.sum() == 0:
                        continue
                    
                    prob = torch.sigmoid(score_diff[mask])
                    target = (label_diff[mask] > 0).float()
                    loss = F.binary_cross_entropy(prob, target)
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                    epoch_loss += loss.item()
                    num_batches += 1
        
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("模型未训练")
        
        self.model.eval()
        with torch.no_grad():
            X_scaled = self.scaler.transform(X)
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            scores = self.model(X_tensor).squeeze().cpu().numpy()
        
        return scores
    
    @property
    def feature_importances_(self):
        return np.ones(self.input_dim) / self.input_dim if self.is_fitted else None


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)
        
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.W_o(attn_output)


class TransformerRanker(BaseRanker):
    """Transformer排名模型"""
    
    def __init__(self, input_dim: int, d_model: int = 64, num_heads: int = 4, 
                 num_layers: int = 2, max_seq_length: int = 16,
                 dropout_rate: float = 0.1, learning_rate: float = 0.001, **kwargs):
        super().__init__(**kwargs)
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        self.learning_rate = learning_rate
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')
        
        # 构建网络
        target_dim = max_seq_length * d_model
        self.input_projection = nn.Linear(input_dim, target_dim)
        
        self.attention_layers = nn.ModuleList([
            MultiHeadAttention(d_model, num_heads, dropout_rate)
            for _ in range(num_layers)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)
        ])
        
        self.output_layer = nn.Sequential(
            nn.Linear(d_model * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1)
        )
        
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.scaler = StandardScaler()
        
        if self.use_gpu:
            self.to(self.device)
    
    def parameters(self):
        """获取所有参数"""
        params = []
        params.extend(list(self.input_projection.parameters()))
        for layer in self.attention_layers:
            params.extend(list(layer.parameters()))
        for layer in self.layer_norms:
            params.extend(list(layer.parameters()))
        params.extend(list(self.output_layer.parameters()))
        return params
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # 输入投影和重塑
        x = self.input_projection(x)
        x = x.view(batch_size, self.max_seq_length, self.d_model)
        
        # Transformer层
        for attention, layer_norm in zip(self.attention_layers, self.layer_norms):
            attn_output = attention(x)
            x = layer_norm(x + attn_output)
        
        # 全局池化
        avg_pool = torch.mean(x, dim=1)
        max_pool = torch.max(x, dim=1)[0]
        pooled = torch.cat([avg_pool, max_pool], dim=1)
        
        return self.output_layer(pooled)
    
    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray, 
            epochs: int = 50, batch_size: int = 64):
        X_scaled = self.scaler.fit_transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.FloatTensor(y).reshape(-1, 1).to(self.device)
        
        self.train()
        criterion = nn.MSELoss()
        
        for epoch in range(epochs):
            total_loss = 0.0
            num_batches = 0
            
            for i in range(0, len(X_tensor), batch_size):
                batch_X = X_tensor[i:i+batch_size]
                batch_y = y_tensor[i:i+batch_size]
                
                self.optimizer.zero_grad()
                outputs = self.forward(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
        
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("模型未训练")
        
        self.eval()
        with torch.no_grad():
            X_scaled = self.scaler.transform(X)
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            scores = self.forward(X_tensor).squeeze().cpu().numpy()
        
        return scores
    
    @property
    def feature_importances_(self):
        return np.ones(self.input_dim) / self.input_dim if self.is_fitted else None


class NeuralRanker(BaseRanker):
    """神经网络排名模型"""
    
    def __init__(self, input_dim: int, hidden_units: List[int] = [256, 128, 64],
                 dropout_rate: float = 0.2, learning_rate: float = 0.001, **kwargs):
        super().__init__(**kwargs)
        
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')
        
        # 构建网络
        layers = []
        prev_dim = input_dim
        for units in hidden_units:
            layers.extend([
                nn.Linear(prev_dim, units),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.BatchNorm1d(units)
            ])
            prev_dim = units
        layers.append(nn.Linear(prev_dim, 1))
        
        self.model = nn.Sequential(*layers)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scaler = StandardScaler()
        
        if self.use_gpu:
            self.model = self.model.to(self.device)
    
    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray, 
            epochs: int = 50, batch_size: int = 64):
        X_scaled = self.scaler.fit_transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.FloatTensor(y).reshape(-1, 1).to(self.device)
        
        self.model.train()
        criterion = nn.MSELoss()
        
        for epoch in range(epochs):
            total_loss = 0.0
            num_batches = 0
            
            for i in range(0, len(X_tensor), batch_size):
                batch_X = X_tensor[i:i+batch_size]
                batch_y = y_tensor[i:i+batch_size]
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
        
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("模型未训练")
        
        self.model.eval()
        with torch.no_grad():
            X_scaled = self.scaler.transform(X)
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            scores = self.model(X_tensor).squeeze().cpu().numpy()
        
        return scores
    
    @property
    def feature_importances_(self):
        return np.ones(self.input_dim) / self.input_dim if self.is_fitted else None