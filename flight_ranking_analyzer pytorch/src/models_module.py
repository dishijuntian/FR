"""
模型模块 - 重构版
统一管理所有模型定义，简化模型创建流程

作者: Flight Ranking Team
版本: 5.0 (重构版)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Union
import warnings
import math

warnings.filterwarnings('ignore')

# 设备配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 尝试导入传统模型库
try:
    from xgboost import XGBRanker
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMRanker
    HAS_LGB = True
except ImportError:
    HAS_LGB = False


class BaseRanker(ABC):
    """排序模型基类"""
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, group: List[int], **kwargs):
        """训练模型"""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测分数"""
        pass
    
    @property
    def feature_importances_(self) -> Optional[np.ndarray]:
        """特征重要性"""
        return getattr(self, '_feature_importance', None)


class TraditionalRankerWrapper(BaseRanker):
    """传统排序模型包装器"""
    
    def __init__(self, model_class, **params):
        """
        初始化传统模型包装器
        
        Args:
            model_class: 模型类
            **params: 模型参数
        """
        self.model_class = model_class
        self.params = params
        self.model = None
        self._setup_params()
    
    def _setup_params(self):
        """设置模型参数"""
        # GPU参数设置
        if torch.cuda.is_available() and hasattr(self, '_get_gpu_params'):
            gpu_params = self._get_gpu_params()
            self.params.update(gpu_params)
        else:
            self.params['n_jobs'] = -1
    
    def fit(self, X: np.ndarray, y: np.ndarray, group: List[int], **kwargs):
        """训练模型"""
        self.model = self.model_class(**self.params)
        self.model.fit(X, y, group=group, **kwargs)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        if self.model is None:
            raise ValueError("模型未训练，请先调用fit方法")
        return self.model.predict(X)
    
    @property
    def feature_importances_(self) -> Optional[np.ndarray]:
        """获取特征重要性"""
        if self.model is None:
            return None
        return getattr(self.model, 'feature_importances_', None)


class XGBRankerModel(TraditionalRankerWrapper):
    """XGBoost排序模型"""
    
    def __init__(self, **params):
        if not HAS_XGB:
            raise ImportError("XGBoost未安装")
        
        default_params = {
            'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 6,
            'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42,
            'eval_metric': 'ndcg'
        }
        default_params.update(params)
        super().__init__(XGBRanker, **default_params)
    
    def _get_gpu_params(self) -> Dict[str, str]:
        """获取GPU参数"""
        return {'tree_method': 'gpu_hist', 'predictor': 'gpu_predictor'}


class LGBMRankerModel(TraditionalRankerWrapper):
    """LightGBM排序模型"""
    
    def __init__(self, **params):
        if not HAS_LGB:
            raise ImportError("LightGBM未安装")
        
        default_params = {
            'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 6,
            'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42,
            'objective': 'lambdarank', 'metric': 'ndcg'
        }
        default_params.update(params)
        super().__init__(LGBMRanker, **default_params)
    
    def _get_gpu_params(self) -> Dict[str, str]:
        """获取GPU参数"""
        return {'device': 'gpu'}


class PyTorchRankerBase(nn.Module, BaseRanker):
    """PyTorch排序模型基类"""
    
    def __init__(self, input_dim: int, **params):
        super().__init__()
        self.input_dim = input_dim
        self.params = params
        self._feature_importance = None
        self.optimizer = None
        self.criterion = None
        
        # 移动到设备
        self.to(DEVICE)
    
    def _create_linear_layers(self, layers_config: List[int]) -> nn.Sequential:
        """创建线性网络层"""
        layers = []
        prev_dim = self.input_dim
        dropout_rate = self.params.get('dropout_rate', 0.2)
        
        for units in layers_config:
            layers.extend([
                nn.Linear(prev_dim, units),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.BatchNorm1d(units)
            ])
            prev_dim = units
        
        # 输出层
        layers.append(nn.Linear(prev_dim, 1))
        return nn.Sequential(*layers)
    
    def _setup_training(self):
        """设置训练组件"""
        self.optimizer = optim.Adam(
            self.parameters(), 
            lr=self.params.get('learning_rate', 0.001),
            weight_decay=self.params.get('weight_decay', 1e-5)
        )
        self.criterion = nn.MSELoss()
    
    def fit(self, X: np.ndarray, y: np.ndarray, group: List[int], **kwargs):
        """训练模型"""
        self._setup_training()
        
        epochs = kwargs.get('epochs', self.params.get('epochs', 15))
        batch_size = kwargs.get('batch_size', self.params.get('batch_size', 64))
        
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
            
            # 每5个epoch打印一次损失
            if epoch % 5 == 0:
                avg_loss = total_loss / len(dataloader)
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
        
        # 计算特征重要性
        self._compute_feature_importance(X[:min(1000, len(X))])
    
    def predict(self, X: np.ndarray) -> np.ndarray:
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
    
    def _compute_feature_importance(self, X_sample: np.ndarray):
        """计算特征重要性（梯度方法）"""
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
        except Exception:
            self._feature_importance = np.ones(self.input_dim) / self.input_dim


class NeuralRanker(PyTorchRankerBase):
    """神经网络排序模型"""
    
    def __init__(self, input_dim: int, **params):
        super().__init__(input_dim, **params)
        hidden_units = params.get('hidden_units', [256, 128, 64])
        self.network = self._create_linear_layers(hidden_units)
    
    def forward(self, x):
        return self.network(x)


class RankNet(PyTorchRankerBase):
    """RankNet模型"""
    
    def __init__(self, input_dim: int, **params):
        super().__init__(input_dim, **params)
        hidden_units = params.get('hidden_units', [128, 64, 32])
        self.network = self._create_linear_layers(hidden_units)
    
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


class TransformerRanker(PyTorchRankerBase):
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
            MultiHeadAttention(
                self.d_model, 
                self.num_heads, 
                params.get('dropout_rate', 0.1)
            )
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
    """模型工厂"""
    
    # 模型注册表
    _models = {
        'XGBRanker': (XGBRankerModel, False),
        'LGBMRanker': (LGBMRankerModel, False),
        'NeuralRanker': (NeuralRanker, True),
        'RankNet': (RankNet, True),
        'TransformerRanker': (TransformerRanker, True),
    }
    
    @classmethod
    def create_model(cls, model_name: str, input_dim: Optional[int] = None, 
                    **params) -> BaseRanker:
        """
        创建模型实例
        
        Args:
            model_name: 模型名称
            input_dim: 输入维度（PyTorch模型需要）
            **params: 模型参数
            
        Returns:
            BaseRanker: 模型实例
        """
        if model_name not in cls._models:
            raise ValueError(f"未知模型: {model_name}，可用模型: {cls.get_available_models()}")
        
        model_class, is_pytorch = cls._models[model_name]
        
        # PyTorch模型需要input_dim
        if is_pytorch:
            if input_dim is None:
                raise ValueError(f"{model_name}需要指定input_dim参数")
            return model_class(input_dim=input_dim, **params)
        else:
            return model_class(**params)
    
    @classmethod
    def get_available_models(cls) -> List[str]:
        """获取可用模型列表"""
        available = []
        for name, (model_class, is_pytorch) in cls._models.items():
            # 检查依赖
            if name == 'XGBRanker' and not HAS_XGB:
                continue
            if name == 'LGBMRanker' and not HAS_LGB:
                continue
            available.append(name)
        return available
    
    @classmethod
    def is_pytorch_model(cls, model_name: str) -> bool:
        """判断是否为PyTorch模型"""
        if model_name not in cls._models:
            return False
        return cls._models[model_name][1]
    
    @classmethod
    def register_model(cls, name: str, model_class, is_pytorch: bool = False):
        """注册新模型"""
        cls._models[name] = (model_class, is_pytorch)


# 别名定义（向后兼容）
def LambdaMART(**params):
    """LambdaMART模型（XGBoost实现）"""
    params.setdefault('objective', 'rank:pairwise')
    return XGBRankerModel(**params)


def ListNet(**params):
    """ListNet模型（LightGBM实现）"""
    params.setdefault('objective', 'lambdarank')
    return LGBMRankerModel(**params)


# 注册别名模型
ModelFactory.register_model('LambdaMART', LambdaMART, False)
ModelFactory.register_model('ListNet', ListNet, False)


def check_dependencies() -> Dict[str, bool]:
    """检查模型依赖"""
    return {
        'xgboost': HAS_XGB,
        'lightgbm': HAS_LGB,
        'torch': True,  # PyTorch总是可用的
    }


def get_model_info(model_name: str) -> Dict[str, Any]:
    """获取模型信息"""
    if model_name not in ModelFactory._models:
        return {}
    
    model_class, is_pytorch = ModelFactory._models[model_name]
    
    info = {
        'name': model_name,
        'type': 'PyTorch' if is_pytorch else 'Traditional',
        'class': model_class.__name__,
        'available': True
    }
    
    # 检查依赖
    if model_name == 'XGBRanker':
        info['available'] = HAS_XGB
        info['dependency'] = 'xgboost'
    elif model_name == 'LGBMRanker':
        info['available'] = HAS_LGB
        info['dependency'] = 'lightgbm'
    
    return info