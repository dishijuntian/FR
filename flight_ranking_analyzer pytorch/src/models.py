"""
模型定义模块 - 重构版

专注于：
- 模型基类定义
- 各种排序模型实现
- 模型工厂
- PyTorch模型组件

作者: Flight Ranking Team
版本: 4.0 (重构版)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRanker
from lightgbm import LGBMRanker
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import warnings
import math

warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BaseRanker(ABC):
    """排序模型基类"""
    
    @abstractmethod
    def fit(self, X, y, group, **kwargs):
        """训练模型"""
        pass
    
    @abstractmethod
    def predict(self, X):
        """预测分数"""
        pass
    
    @property
    def feature_importances_(self):
        """特征重要性"""
        return getattr(self, '_feature_importance', None)


class TraditionalRanker(BaseRanker):
    """传统排序模型基类"""
    
    def __init__(self, model_class, use_gpu: bool = False, **params):
        self.model_class = model_class
        self.use_gpu = use_gpu
        self.params = params
        self._setup_params()
        self.model = self.model_class(**self.params)
    
    def _setup_params(self):
        """设置模型参数"""
        if self.use_gpu and hasattr(self, '_gpu_params'):
            self.params.update(self._gpu_params())
        else:
            self.params['n_jobs'] = -1
    
    def fit(self, X, y, group, **kwargs):
        self.model.fit(X, y, group=group, **kwargs)
    
    def predict(self, X):
        return self.model.predict(X)
    
    @property
    def feature_importances_(self):
        return getattr(self.model, 'feature_importances_', None)


class XGBRankerModel(TraditionalRanker):
    """XGBoost排序模型"""
    
    def __init__(self, use_gpu: bool = False, **params):
        default_params = {
            'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 6,
            'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42
        }
        default_params.update(params)
        super().__init__(XGBRanker, use_gpu, **default_params)
    
    def _gpu_params(self):
        return {'tree_method': 'gpu_hist', 'predictor': 'gpu_predictor'}


class LGBMRankerModel(TraditionalRanker):
    """LightGBM排序模型"""
    
    def __init__(self, use_gpu: bool = False, **params):
        default_params = {
            'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 6,
            'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42,
            'objective': 'lambdarank', 'metric': 'ndcg'
        }
        default_params.update(params)
        super().__init__(LGBMRanker, use_gpu, **default_params)
    
    def _gpu_params(self):
        return {'device': 'gpu'}


class PyTorchRanker(nn.Module, BaseRanker):
    """PyTorch排序模型基类"""
    
    def __init__(self, input_dim: int, **params):
        super().__init__()
        self.input_dim = input_dim
        self.params = params
        self._feature_importance = None
        self.to(DEVICE)
    
    def _create_network(self, layers_config: List[int]) -> nn.Sequential:
        """创建网络层"""
        layers = []
        prev_dim = self.input_dim
        
        for units in layers_config:
            layers.extend([
                nn.Linear(prev_dim, units),
                nn.ReLU(),
                nn.Dropout(self.params.get('dropout_rate', 0.2)),
                nn.BatchNorm1d(units)
            ])
            prev_dim = units
        
        layers.append(nn.Linear(prev_dim, 1))
        return nn.Sequential(*layers)
    
    def _setup_optimizer(self):
        """设置优化器"""
        self.optimizer = optim.Adam(
            self.parameters(), 
            lr=self.params.get('learning_rate', 0.001)
        )
        self.criterion = nn.MSELoss()
    
    def fit(self, X, y, group, **kwargs):
        """训练模型"""
        self._setup_optimizer()
        epochs = kwargs.get('epochs', self.params.get('epochs', 10))
        batch_size = kwargs.get('batch_size', self.params.get('batch_size', 32))
        
        # 准备数据
        X_tensor = torch.FloatTensor(X).to(DEVICE)
        y_tensor = torch.FloatTensor(y).reshape(-1, 1).to(DEVICE)
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # 训练循环
        self.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                outputs = self(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            if epoch % 5 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}')
        
        # 计算特征重要性
        self._compute_feature_importance(X[:min(1000, len(X))])
    
    def predict(self, X):
        """预测"""
        self.eval()
        predictions = []
        batch_size = 1000
        
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch = X[i:i+batch_size]
                batch_tensor = torch.FloatTensor(batch).to(DEVICE)
                batch_pred = self(batch_tensor)
                predictions.append(batch_pred.cpu().numpy().flatten())
        
        return np.concatenate(predictions)
    
    def _compute_feature_importance(self, X_sample):
        """计算特征重要性"""
        try:
            self.eval()
            X_tensor = torch.FloatTensor(X_sample).to(DEVICE)
            X_tensor.requires_grad_(True)
            
            outputs = self(X_tensor)
            loss = outputs.mean()
            loss.backward()
            
            grads = X_tensor.grad
            if grads is not None:
                importance = torch.mean(torch.abs(grads), dim=0).cpu().numpy()
                self._feature_importance = importance / (np.sum(importance) + 1e-8)
            else:
                self._feature_importance = np.ones(self.input_dim) / self.input_dim
        except:
            self._feature_importance = np.ones(self.input_dim) / self.input_dim


class NeuralRanker(PyTorchRanker):
    """神经网络排序模型"""
    
    def __init__(self, input_dim: int, **params):
        super().__init__(input_dim, **params)
        hidden_units = params.get('hidden_units', [256, 128, 64])
        self.network = self._create_network(hidden_units)
    
    def forward(self, x):
        return self.network(x)


class RankNet(PyTorchRanker):
    """RankNet模型"""
    
    def __init__(self, input_dim: int, **params):
        super().__init__(input_dim, **params)
        hidden_units = params.get('hidden_units', [128, 64, 32])
        self.network = self._create_network(hidden_units)
    
    def forward(self, x):
        return self.network(x)


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # 多头注意力计算
        Q = self.W_q(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        return self.W_o(attn_output)


class TransformerRanker(PyTorchRanker):
    """Transformer排序模型"""
    
    def __init__(self, input_dim: int, **params):
        super().__init__(input_dim, **params)
        
        self.d_model = params.get('d_model', 64)
        self.num_heads = params.get('num_heads', 4)
        self.num_layers = params.get('num_layers', 2)
        self.max_seq_length = params.get('max_seq_length', 16)
        
        # 输入投影
        target_dim = self.max_seq_length * self.d_model
        self.input_projection = nn.Linear(input_dim, target_dim)
        
        # Transformer层
        self.attention_layers = nn.ModuleList([
            MultiHeadAttention(self.d_model, self.num_heads, params.get('dropout_rate', 0.1))
            for _ in range(self.num_layers)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(self.d_model) for _ in range(self.num_layers)
        ])
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(self.d_model * 2, 64),
            nn.ReLU(),
            nn.Dropout(params.get('dropout_rate', 0.1)),
            nn.Linear(64, 1)
        )
    
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


class ModelFactory:
    """模型工厂类 - 简化版"""
    
    MODEL_REGISTRY = {
        'XGBRanker': XGBRankerModel,
        'LGBMRanker': LGBMRankerModel,
        'LambdaMART': lambda **kwargs: XGBRankerModel(objective="rank:pairwise", **kwargs),
        'ListNet': lambda **kwargs: LGBMRankerModel(objective="lambdarank", **kwargs),
        'NeuralRanker': NeuralRanker,
        'RankNet': RankNet,
        'TransformerRanker': TransformerRanker,
    }
    
    @classmethod
    def create_model(cls, model_name: str, use_gpu: bool = True, 
                    input_dim: Optional[int] = None, **params) -> BaseRanker:
        """
        创建模型实例
        
        Args:
            model_name: 模型名称
            use_gpu: 是否使用GPU
            input_dim: 输入维度（PyTorch模型需要）
            **params: 模型参数
            
        Returns:
            BaseRanker: 模型实例
        """
        if model_name not in cls.MODEL_REGISTRY:
            raise ValueError(f"未知模型: {model_name}")
        
        model_class = cls.MODEL_REGISTRY[model_name]
        
        # PyTorch模型需要input_dim
        if model_name in ['NeuralRanker', 'RankNet', 'TransformerRanker']:
            if input_dim is None:
                raise ValueError(f"{model_name}需要指定input_dim参数")
            return model_class(input_dim=input_dim, **params)
        else:
            return model_class(use_gpu=use_gpu, **params)
    
    @classmethod
    def get_available_models(cls) -> List[str]:
        """获取所有可用模型名称"""
        return list(cls.MODEL_REGISTRY.keys())