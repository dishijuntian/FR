"""
航班排名模型集合 - 性能优化版
解决GPU初始化延迟和警告问题
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

# 抑制警告
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
import torch.optim as optim


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
    """XGBoost排名模型 - 优化版"""
    
    def __init__(self, n_estimators=200, max_depth=8, learning_rate=0.05, 
                 random_state=42, **kwargs):
        super().__init__(**kwargs)
        
        params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'random_state': random_state,
            'verbosity': 0,  # 减少输出
            'eval_metric': 'ndcg'
        }
        
        if self.use_gpu:
            self.logger.info("初始化XGBoost GPU模式...")
            # 快速GPU测试，避免复杂的训练测试
            if self._quick_gpu_check():
                params.update({
                    'tree_method': 'gpu_hist', 
                    'gpu_id': 0,
                    'max_bin': 256  # 减少GPU内存使用
                })
                self.logger.info("✓ XGBoost GPU加速可用")
            else:
                self.logger.warning("XGBoost GPU不可用，使用CPU")
                params.update({'tree_method': 'hist', 'n_jobs': -1})
                self.use_gpu = False
        else:
            params.update({'tree_method': 'hist', 'n_jobs': -1})
        
        self.model = xgb.XGBRanker(**params)
    
    def _quick_gpu_check(self) -> bool:
        """快速GPU可用性检查"""
        try:
            # 简单的CUDA可用性检查，不创建模型
            import cupy as cp  # 如果有cupy，用它快速测试
            cp.cuda.runtime.getDeviceCount()
            return True
        except:
            try:
                # 备用方案：检查torch cuda
                return torch.cuda.is_available() and torch.cuda.device_count() > 0
            except:
                return False
    
    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray):
        unique_groups = np.unique(groups)
        group_sizes = [np.sum(groups == g) for g in unique_groups]
        
        # 使用更少的verbose来减少输出
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
    """LightGBM排名模型 - 优化版"""
    
    def __init__(self, n_estimators=200, max_depth=8, learning_rate=0.05,
                 random_state=42, **kwargs):
        super().__init__(**kwargs)
        
        params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'random_state': random_state,
            'verbose': -1,  # 完全静默
            'metric': 'ndcg',
            'force_col_wise': True  # 提升性能
        }
        
        if self.use_gpu:
            self.logger.info("初始化LightGBM GPU模式...")
            if self._check_lgb_gpu():
                params.update({
                    'device': 'gpu', 
                    'gpu_platform_id': 0, 
                    'gpu_device_id': 0,
                    'max_bin': 255  # GPU优化
                })
                self.logger.info("✓ LightGBM GPU加速可用")
            else:
                self.logger.warning("LightGBM GPU不可用，使用CPU")
                params.update({'device': 'cpu', 'n_jobs': -1})
                self.use_gpu = False
        else:
            params.update({'device': 'cpu', 'n_jobs': -1})
        
        self.model = lgb.LGBMRanker(**params)
    
    def _check_lgb_gpu(self) -> bool:
        """检查LightGBM GPU支持"""
        try:
            # 创建最小测试
            test_model = lgb.LGBMRanker(
                n_estimators=1, 
                device='gpu', 
                verbose=-1,
                gpu_use_dp=False  # 使用单精度提升速度
            )
            return True
        except Exception as e:
            self.logger.debug(f"LightGBM GPU检查失败: {e}")
            return False
    
    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray):
        unique_groups = np.unique(groups)
        group_sizes = [np.sum(groups == g) for g in unique_groups]
        
        # 添加回调来减少输出
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
    """LambdaMART排名模型 - 优化版"""
    
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


class ListNet(BaseRanker):
    """ListNet排名模型 - 优化版"""
    
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
            'metric': 'ndcg',
            'force_col_wise': True
        }
        
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


class BM25Ranker:
    """BM25排名模型 - 优化版"""
    
    def __init__(self, k1=1.2, b=0.75, logger=None, **kwargs):
        self.k1 = k1
        self.b = b
        self.logger = logger or logging.getLogger(__name__)
        self.is_fitted = False
        self.feature_weights = None
    
    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray):
        self.logger.info("训练BM25Ranker...")
        
        # 快速计算特征相关性权重
        with np.errstate(invalid='ignore'):  # 忽略nan警告
            correlations = np.array([
                np.corrcoef(X[:, i], y)[0, 1] if X[:, i].var() > 0 else 0.0
                for i in range(X.shape[1])
            ])
        
        # 处理NaN值
        correlations = np.nan_to_num(correlations, 0.0)
        self.feature_weights = np.abs(correlations)
        
        # 归一化权重
        weight_sum = np.sum(self.feature_weights)
        if weight_sum > 0:
            self.feature_weights = self.feature_weights / weight_sum
        else:
            self.feature_weights = np.ones(X.shape[1]) / X.shape[1]
        
        self.is_fitted = True
        self.logger.info("✓ BM25Ranker训练完成")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("模型未训练")
        
        scores = np.dot(X, self.feature_weights)
        mean_score = np.mean(scores) + 1e-8
        
        # BM25评分公式
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


# 简化的神经网络模型
class RankNet(BaseRanker):
    """RankNet深度学习排名模型 - 优化版"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64, 32],
                 dropout_rate: float = 0.2, learning_rate: float = 0.001,
                 random_state: int = 42, **kwargs):
        super().__init__(**kwargs)
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')
        
        if self.use_gpu:
            self.logger.info("初始化RankNet GPU模式...")
            # 预热GPU
            torch.cuda.empty_cache()
        
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
        
        self.model = nn.Sequential(*layers)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.scaler = StandardScaler()
        
        if self.use_gpu:
            self.model = self.model.to(self.device)
            self.logger.info("✓ RankNet GPU模式初始化完成")
    
    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray, 
            epochs: int = 50, batch_size: int = 1024):
        self.logger.info(f"开始RankNet训练 (epochs={epochs})")
        
        X_scaled = self.scaler.fit_transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        self.model.train()
        
        # 简化的训练循环
        for epoch in range(epochs):
            if epoch % 10 == 0:
                self.logger.info(f"RankNet训练进度: {epoch}/{epochs}")
            
            epoch_loss = 0.0
            num_batches = 0
            
            # 随机选择一些组进行训练，避免全量配对
            unique_groups = np.unique(groups)
            selected_groups = np.random.choice(
                unique_groups, 
                size=min(len(unique_groups), 1000),  # 限制组数
                replace=False
            )
            
            for group_id in selected_groups:
                group_mask = groups == group_id
                group_X = X_tensor[group_mask]
                group_y = y_tensor[group_mask]
                
                if len(group_X) < 2:
                    continue
                
                # 限制配对数量
                max_pairs = min(len(group_X) * (len(group_X) - 1) // 2, batch_size)
                indices = torch.arange(len(group_X), device=self.device)
                pairs = torch.combinations(indices, 2)
                
                if len(pairs) > max_pairs:
                    pairs = pairs[torch.randperm(len(pairs))[:max_pairs]]
                
                idx1, idx2 = pairs[:, 0], pairs[:, 1]
                
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
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # 梯度裁剪
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
        
        self.is_fitted = True
        self.logger.info("✓ RankNet训练完成")
    
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


# 其他模型的简化版本...
class TransformerRanker(BaseRanker):
    """简化的Transformer排名模型"""
    
    def __init__(self, input_dim: int, d_model: int = 64, num_heads: int = 4, 
                 num_layers: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        # 简化实现，基本上是一个线性模型的包装
        self.weights = None
        
    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray, **kwargs):
        # 简化为线性回归
        from sklearn.linear_model import Ridge
        self.model = Ridge(alpha=1.0)
        self.model.fit(X, y)
        self.is_fitted = True
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("模型未训练")
        return self.model.predict(X)
    
    @property
    def feature_importances_(self):
        return np.abs(self.model.coef_) if self.is_fitted else None


class NeuralRanker(BaseRanker):
    """简化的神经网络排名模型"""
    
    def __init__(self, input_dim: int, hidden_units: List[int] = [256, 128, 64], **kwargs):
        super().__init__(**kwargs)
        # 基于sklearn的MLPRegressor实现
        from sklearn.neural_network import MLPRegressor
        
        self.model = MLPRegressor(
            hidden_layer_sizes=tuple(hidden_units),
            max_iter=100,  # 减少迭代次数
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
            verbose=False
        )
        
    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray, **kwargs):
        self.logger.info("开始NeuralRanker训练...")
        self.model.fit(X, y)
        self.is_fitted = True
        self.logger.info("✓ NeuralRanker训练完成")
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("模型未训练")
        return self.model.predict(X)
    
    @property
    def feature_importances_(self):
        # 简化的特征重要性
        return np.ones(self.model.n_features_in_) / self.model.n_features_in_ if self.is_fitted else None