import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import warnings
import joblib

import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
import torch.optim as optim

warnings.filterwarnings('ignore')


class XGBoostRanker:
    """XGBoost排名模型"""
    
    def __init__(self, n_estimators=200, max_depth=8, learning_rate=0.05, 
                 use_gpu=True, random_state=42, logger=None):
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.logger = logger or logging.getLogger(__name__)
        self.is_fitted = False
        
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
                params.update({
                    'tree_method': 'gpu_hist',
                    'gpu_id': 0
                })
                # 测试GPU可用性
                test_model = xgb.XGBRanker(**params)
                test_X = np.random.random((10, 5))
                test_y = np.random.randint(0, 3, 10)
                test_model.fit(test_X, test_y, group=[5, 5])
                self.logger.info("XGBoost GPU加速可用")
            except Exception as e:
                self.logger.warning(f"XGBoost GPU失败，使用CPU: {e}")
                params.update({
                    'tree_method': 'hist',
                    'n_jobs': -1
                })
                self.use_gpu = False
        else:
            params.update({
                'tree_method': 'hist',
                'n_jobs': -1
            })
        
        self.model = xgb.XGBRanker(**params)
    
    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray):
        """训练模型"""
        unique_groups = np.unique(groups)
        group_sizes = [np.sum(groups == g) for g in unique_groups]
        
        self.model.fit(X, y, group=group_sizes)
        self.is_fitted = True
        self.logger.info("XGBoost训练完成")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测分数"""
        if not self.is_fitted:
            raise ValueError("模型未训练")
        return self.model.predict(X)
    
    def save_model(self, filepath: str):
        """保存模型"""
        if not self.is_fitted:
            raise ValueError("模型未训练，无法保存")
        joblib.dump(self, filepath)
    
    @classmethod
    def load_model(cls, filepath: str):
        """加载模型"""
        return joblib.load(filepath)


class LightGBMRanker:
    """LightGBM排名模型"""
    
    def __init__(self, n_estimators=200, max_depth=8, learning_rate=0.05,
                 use_gpu=True, random_state=42, logger=None):
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.logger = logger or logging.getLogger(__name__)
        self.is_fitted = False
        
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
                params.update({
                    'device': 'gpu',
                    'gpu_platform_id': 0,
                    'gpu_device_id': 0
                })
                # 测试GPU可用性
                test_model = lgb.LGBMRanker(**params)
                test_X = np.random.random((10, 5))
                test_y = np.random.randint(0, 3, 10)
                test_model.fit(test_X, test_y, group=[5, 5])
                self.logger.info("LightGBM GPU加速可用")
            except Exception as e:
                self.logger.warning(f"LightGBM GPU失败，使用CPU: {e}")
                params.update({
                    'device': 'cpu',
                    'n_jobs': -1
                })
                self.use_gpu = False
        else:
            params.update({
                'device': 'cpu',
                'n_jobs': -1
            })
        
        self.model = lgb.LGBMRanker(**params)
    
    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray):
        """训练模型"""
        unique_groups = np.unique(groups)
        group_sizes = [np.sum(groups == g) for g in unique_groups]
        
        self.model.fit(X, y, group=group_sizes)
        self.is_fitted = True
        self.logger.info("LightGBM训练完成")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测分数"""
        if not self.is_fitted:
            raise ValueError("模型未训练")
        return self.model.predict(X)
    
    def save_model(self, filepath: str):
        """保存模型"""
        if not self.is_fitted:
            raise ValueError("模型未训练，无法保存")
        joblib.dump(self, filepath)
    
    @classmethod
    def load_model(cls, filepath: str):
        """加载模型"""
        return joblib.load(filepath)


class RankNet:
    """RankNet深度学习排名模型"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64, 32],
                 dropout_rate: float = 0.2, learning_rate: float = 0.001,
                 use_gpu: bool = True, random_state: int = 42, logger=None):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.logger = logger or logging.getLogger(__name__)
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')
        self.is_fitted = False
        
        # 构建网络
        self._build_network()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scaler = StandardScaler()
        
        if self.use_gpu:
            self.model = self.model.to(self.device)
            self.logger.info("RankNet GPU加速可用")
    
    def _build_network(self):
        """构建网络结构"""
        layers = []
        prev_dim = self.input_dim
        
        for hidden_dim in self.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.model = nn.Sequential(*layers)
    
    def _pairwise_loss(self, scores1: torch.Tensor, scores2: torch.Tensor, 
                      labels1: torch.Tensor, labels2: torch.Tensor) -> torch.Tensor:
        """计算成对损失"""
        label_diff = labels1 - labels2
        score_diff = scores1 - scores2
        
        prob = torch.sigmoid(score_diff)
        target = (label_diff > 0).float()
        
        mask = (label_diff != 0)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=self.device)
        
        loss = F.binary_cross_entropy(prob[mask], target[mask])
        return loss
    
    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray, epochs: int = 100, batch_size: int = 1024):
        """训练模型"""
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
                
                if len(pairs) == 0:
                    continue
                
                for i in range(0, len(pairs), batch_size):
                    batch_pairs = pairs[i:i+batch_size]
                    
                    idx1, idx2 = batch_pairs[:, 0], batch_pairs[:, 1]
                    X1, X2 = group_X[idx1], group_X[idx2]
                    y1, y2 = group_y[idx1], group_y[idx2]
                    
                    scores1 = self.model(X1).squeeze()
                    scores2 = self.model(X2).squeeze()
                    
                    loss = self._pairwise_loss(scores1, scores2, y1, y2)
                    
                    if loss > 0:
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                        
                        epoch_loss += loss.item()
                        num_batches += 1
            
            if (epoch + 1) % 20 == 0 and num_batches > 0:
                avg_loss = epoch_loss / num_batches
                self.logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        self.is_fitted = True
        self.logger.info("RankNet训练完成")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测分数"""
        if not self.is_fitted:
            raise ValueError("模型未训练")
        
        self.model.eval()
        with torch.no_grad():
            X_scaled = self.scaler.transform(X)
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            scores = self.model(X_tensor).squeeze().cpu().numpy()
        
        return scores
    
    def save_model(self, filepath: str):
        """保存模型"""
        if not self.is_fitted:
            raise ValueError("模型未训练，无法保存")
        joblib.dump(self, filepath)
    
    @classmethod
    def load_model(cls, filepath: str):
        """加载模型"""
        return joblib.load(filepath)

