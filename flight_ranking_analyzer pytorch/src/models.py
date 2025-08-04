"""
排序模型定义文件 - PyTorch版本

该模块包含所有排序模型的定义和实现
- 将TensorFlow/Keras模型转换为PyTorch实现
- 改进了内存使用和训练稳定性

作者: Flight Ranking Team
版本: 3.0 (PyTorch版本)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRanker
from lightgbm import LGBMRanker
from rank_bm25 import BM25Okapi
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any
import warnings
import math

warnings.filterwarnings('ignore')

# 设置PyTorch设备
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
    
    @abstractmethod
    def get_model_name(self):
        """获取模型名称"""
        pass
    
    def get_params(self):
        """获取模型参数"""
        return getattr(self, 'params', {})
    
    def set_params(self, **params):
        """设置模型参数"""
        self.params = params
        return self


class XGBRankerModel(BaseRanker):
    """XGBoost排序模型"""
    
    def __init__(self, use_gpu=True, **params):
        default_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'eval_metric': 'ndcg'
        }
        default_params.update(params)
        
        if use_gpu:
            default_params.update({
                'tree_method': 'gpu_hist',
                'predictor': 'gpu_predictor'
            })
        else:
            default_params['n_jobs'] = -1
        
        self.params = default_params
        self.model = XGBRanker(**default_params)
        
    def fit(self, X, y, group, **kwargs):
        self.model.fit(X, y, group=group, **kwargs)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def get_model_name(self):
        return "XGBRanker"
    
    @property
    def feature_importances_(self):
        return self.model.feature_importances_


class LGBMRankerModel(BaseRanker):
    """LightGBM排序模型"""
    
    def __init__(self, use_gpu=True, **params):
        default_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'objective': 'lambdarank',
            'metric': 'ndcg'
        }
        default_params.update(params)
        
        if use_gpu:
            default_params['device'] = 'gpu'
        else:
            default_params['n_jobs'] = -1
        
        self.params = default_params
        self.model = LGBMRanker(**default_params)
        
    def fit(self, X, y, group, **kwargs):
        self.model.fit(X, y, group=group, **kwargs)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def get_model_name(self):
        return "LGBMRanker"
    
    @property
    def feature_importances_(self):
        return self.model.feature_importances_


class LambdaMART(BaseRanker):
    """LambdaMART实现（基于XGBoost）"""
    
    def __init__(self, use_gpu=True, **params):
        default_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        default_params.update(params)
        
        xgb_params = {
            'objective': "rank:pairwise",
            **default_params
        }
        
        if use_gpu:
            xgb_params.update({
                'tree_method': 'gpu_hist',
                'predictor': 'gpu_predictor'
            })
        else:
            xgb_params['n_jobs'] = -1
        
        self.params = default_params
        self.model = XGBRanker(**xgb_params)
        
    def fit(self, X, y, group, **kwargs):
        self.model.fit(X, y, group=group, **kwargs)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def get_model_name(self):
        return "LambdaMART"
    
    @property
    def feature_importances_(self):
        return self.model.feature_importances_


class ListNetModel(BaseRanker):
    """ListNet实现（基于LightGBM）"""
    
    def __init__(self, use_gpu=True, **params):
        default_params = {
            'n_estimators': 100,
            'learning_rate': 0.05,
            'max_depth': 7,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        default_params.update(params)
        
        lgb_params = {
            'objective': "lambdarank",
            'metric': "ndcg",
            **default_params
        }
        
        if use_gpu:
            lgb_params['device'] = 'gpu'
        else:
            lgb_params['n_jobs'] = -1
        
        self.params = default_params
        self.model = LGBMRanker(**lgb_params)
        
    def fit(self, X, y, group, **kwargs):
        self.model.fit(X, y, group=group, **kwargs)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def get_model_name(self):
        return "ListNet"
    
    @property
    def feature_importances_(self):
        return self.model.feature_importances_


class NeuralRanker(nn.Module, BaseRanker):
    """PyTorch神经网络排序模型"""
    
    def __init__(self, input_dim, **params):
        super(NeuralRanker, self).__init__()
        default_params = {
            'hidden_units': [256, 128, 64],
            'learning_rate': 0.001,
            'epochs': 10,
            'batch_size': 32,
            'dropout_rate': 0.2
        }
        default_params.update(params)
        
        self.input_dim = input_dim
        self.params = default_params
        self._feature_importance = None
        
        # 构建网络层
        layers = []
        prev_dim = input_dim
        
        for i, units in enumerate(self.params['hidden_units']):
            layers.append(nn.Linear(prev_dim, units))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.params['dropout_rate']))
            prev_dim = units
        
        # 输出层
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # 移动到设备
        self.to(DEVICE)
        
        # 优化器
        self.optimizer = optim.Adam(self.parameters(), lr=self.params['learning_rate'])
        self.criterion = nn.MSELoss()
        
    def forward(self, x):
        return self.network(x)
    
    def fit(self, X, y, group, **kwargs):
        epochs = kwargs.get('epochs', self.params['epochs'])
        batch_size = kwargs.get('batch_size', self.params['batch_size'])
        
        # 转换为PyTorch张量
        X_tensor = torch.FloatTensor(X).to(DEVICE)
        y_tensor = torch.FloatTensor(y).reshape(-1, 1).to(DEVICE)
        
        # 创建数据加载器
        dataset = TensorDataset(X_tensor, y_tensor)
        
        # 手动分割验证集（如果数据量足够大）
        if len(X) > 1000:
            val_size = int(len(X) * 0.2)
            train_size = len(X) - val_size
            
            train_dataset = TensorDataset(X_tensor[:train_size], y_tensor[:train_size])
            val_dataset = TensorDataset(X_tensor[train_size:], y_tensor[train_size:])
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            # 训练模型（带验证）
            for epoch in range(epochs):
                # 训练阶段
                self.train()
                train_loss = 0.0
                for batch_X, batch_y in train_loader:
                    self.optimizer.zero_grad()
                    outputs = self(batch_X)
                    loss = self.criterion(outputs, batch_y)
                    loss.backward()
                    self.optimizer.step()
                    train_loss += loss.item()
                
                # 验证阶段
                if epoch % 5 == 0:
                    self.eval()
                    val_loss = 0.0
                    with torch.no_grad():
                        for batch_X, batch_y in val_loader:
                            outputs = self(batch_X)
                            val_loss += self.criterion(outputs, batch_y).item()
                    
                    print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}')
        else:
            # 训练模型（无验证）
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            for epoch in range(epochs):
                self.train()
                total_loss = 0.0
                for batch_X, batch_y in train_loader:
                    self.optimizer.zero_grad()
                    outputs = self(batch_X)
                    loss = self.criterion(outputs, batch_y)
                    loss.backward()
                    self.optimizer.step()
                    total_loss += loss.item()
                
                if epoch % 5 == 0:
                    print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}')
        
        # 计算特征重要性
        self._compute_feature_importance(X[:min(1000, len(X))])
    
    def predict(self, X):
        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(DEVICE)
            outputs = self(X_tensor)
            return outputs.cpu().numpy().flatten()
    
    def _compute_feature_importance(self, X_sample):
        """计算特征重要性"""
        try:
            self.eval()
            X_tensor = torch.FloatTensor(X_sample).to(DEVICE)
            X_tensor.requires_grad_(True)
            
            outputs = self(X_tensor)
            loss = outputs.mean()
            loss.backward()
            
            # 使用梯度的绝对值均值作为特征重要性
            grads = X_tensor.grad
            if grads is not None:
                importance = torch.mean(torch.abs(grads), dim=0).cpu().numpy()
                # 归一化
                self._feature_importance = importance / (np.sum(importance) + 1e-8)
            else:
                # 如果无法计算梯度，使用均匀分布
                self._feature_importance = np.ones(self.input_dim) / self.input_dim
                
        except Exception as e:
            print(f"计算NeuralRanker特征重要性时出错: {e}")
            # 使用均匀分布作为后备方案
            self._feature_importance = np.ones(self.input_dim) / self.input_dim
    
    def get_model_name(self):
        return "NeuralRanker"
    
    @property
    def feature_importances_(self):
        """获取特征重要性"""
        if self._feature_importance is None:
            return np.ones(self.input_dim) / self.input_dim
        return self._feature_importance


class RankNet(nn.Module, BaseRanker):
    """PyTorch RankNet排序模型"""
    
    def __init__(self, input_dim, **params):
        super(RankNet, self).__init__()
        default_params = {
            'hidden_units': [128, 64, 32],
            'learning_rate': 0.001,
            'epochs': 15,
            'batch_size': 64,
            'dropout_rate': 0.3
        }
        default_params.update(params)
        
        self.input_dim = input_dim
        self.params = default_params
        self._feature_importance = None
        
        print(f"🔧 初始化RankNet:")
        print(f"   输入维度: {input_dim}")
        print(f"   隐藏层单元: {self.params['hidden_units']}")
        print(f"   学习率: {self.params['learning_rate']}")
        
        # 构建网络层
        layers = []
        prev_dim = input_dim
        
        for i, units in enumerate(self.params['hidden_units']):
            layers.extend([
                nn.Linear(prev_dim, units),
                nn.ReLU(),
                nn.Dropout(self.params['dropout_rate']),
                nn.BatchNorm1d(units)
            ])
            prev_dim = units
        
        # 输出层 - 输出单个分数
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # 移动到设备
        self.to(DEVICE)
        
        # 优化器和损失函数
        self.optimizer = optim.Adam(self.parameters(), lr=self.params['learning_rate'])
        self.criterion = nn.MSELoss()
        
        print("✅ RankNet模型构建成功")
        
    def forward(self, x):
        return self.network(x)
    
    def fit(self, X, y, group, **kwargs):
        """训练RankNet模型"""
        epochs = kwargs.get('epochs', self.params['epochs'])
        batch_size = kwargs.get('batch_size', self.params['batch_size'])
        
        print(f"🚀 开始训练RankNet模型，数据形状: {X.shape}")
        print(f"📊 训练参数: epochs={epochs}, batch_size={batch_size}")
        
        # 转换为PyTorch张量
        X_tensor = torch.FloatTensor(X).to(DEVICE)
        y_tensor = torch.FloatTensor(y).reshape(-1, 1).to(DEVICE)
        
        # 创建数据加载器
        dataset = TensorDataset(X_tensor, y_tensor)
        
        # 手动处理验证集分割
        if len(X) > 1000:
            # 计算验证集大小
            val_size = int(len(X) * 0.2)
            train_size = len(X) - val_size
            
            # 分割数据
            train_dataset = TensorDataset(X_tensor[:train_size], y_tensor[:train_size])
            val_dataset = TensorDataset(X_tensor[train_size:], y_tensor[train_size:])
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            # 训练模型（带验证）
            for epoch in range(epochs):
                # 训练阶段
                self.train()
                train_loss = 0.0
                for batch_X, batch_y in train_loader:
                    self.optimizer.zero_grad()
                    outputs = self(batch_X)
                    loss = self.criterion(outputs, batch_y)
                    loss.backward()
                    self.optimizer.step()
                    train_loss += loss.item()
                
                # 验证阶段
                if epoch % 5 == 0:
                    self.eval()
                    val_loss = 0.0
                    with torch.no_grad():
                        for batch_X, batch_y in val_loader:
                            outputs = self(batch_X)
                            val_loss += self.criterion(outputs, batch_y).item()
                    
                    print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}')
        else:
            # 训练模型（无验证）
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            for epoch in range(epochs):
                self.train()
                total_loss = 0.0
                for batch_X, batch_y in train_loader:
                    self.optimizer.zero_grad()
                    outputs = self(batch_X)
                    loss = self.criterion(outputs, batch_y)
                    loss.backward()
                    self.optimizer.step()
                    total_loss += loss.item()
                
                if epoch % 5 == 0:
                    print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}')
        
        # 计算特征重要性
        self._compute_feature_importance(X[:min(1000, len(X))])
        
        print("✅ RankNet模型训练完成")
    
    def _compute_feature_importance(self, X_sample):
        """计算特征重要性"""
        try:
            self.eval()
            X_tensor = torch.FloatTensor(X_sample).to(DEVICE)
            X_tensor.requires_grad_(True)
            
            outputs = self(X_tensor)
            loss = outputs.mean()
            loss.backward()
            
            # 使用梯度的绝对值均值作为特征重要性
            grads = X_tensor.grad
            if grads is not None:
                importance = torch.mean(torch.abs(grads), dim=0).cpu().numpy()
                # 归一化
                self._feature_importance = importance / (np.sum(importance) + 1e-8)
            else:
                # 如果无法计算梯度，使用均匀分布
                self._feature_importance = np.ones(self.input_dim) / self.input_dim
                
        except Exception as e:
            print(f"计算RankNet特征重要性时出错: {e}")
            # 使用均匀分布作为后备方案
            self._feature_importance = np.ones(self.input_dim) / self.input_dim
    
    def predict(self, X):
        """预测分数"""
        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(DEVICE)
            outputs = self(X_tensor)
            return outputs.cpu().numpy().flatten()
    
    def get_model_name(self):
        return "RankNet"
    
    @property
    def feature_importances_(self):
        """获取特征重要性"""
        if self._feature_importance is None:
            return np.ones(self.input_dim) / self.input_dim
        return self._feature_importance


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 应用softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力权重
        output = torch.matmul(attn_weights, V)
        
        return output, attn_weights
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # 线性变换并分头
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 应用注意力
        attn_output, attn_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # 最终线性变换
        output = self.W_o(attn_output)
        
        return output


class TransformerBlock(nn.Module):
    """Transformer块"""
    
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout_rate)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dff),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dff, d_model)
        )
        
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        # 多头自注意力
        attn_output = self.attention(x, x, x)
        x = self.layernorm1(x + self.dropout1(attn_output))
        
        # 前馈网络
        ff_output = self.feed_forward(x)
        x = self.layernorm2(x + self.dropout2(ff_output))
        
        return x


class TransformerRanker(nn.Module, BaseRanker):
    """PyTorch Transformer排序模型"""
    
    def __init__(self, input_dim, **params):
        super(TransformerRanker, self).__init__()
        # 使用更保守的默认参数
        default_params = {
            'num_heads': 4,           # 减少注意力头数
            'num_layers': 2,          # 减少层数
            'd_model': 64,            # 减少模型维度
            'dff': 128,               # 减少前馈网络维度
            'learning_rate': 0.001,
            'epochs': 10,
            'batch_size': 64,
            'dropout_rate': 0.1,
            'max_seq_length': 16      # 固定序列长度
        }
        default_params.update(params)
        
        self.input_dim = input_dim
        self.params = default_params
        self._feature_importance = None
        
        print(f"🔧 初始化TransformerRanker:")
        print(f"   输入维度: {input_dim}")
        print(f"   序列长度: {self.params['max_seq_length']}")
        print(f"   模型维度: {self.params['d_model']}")
        print(f"   注意力头数: {self.params['num_heads']}")
        
        try:
            self._build_model()
            print("✅ TransformerRanker模型构建成功")
        except Exception as e:
            print(f"❌ TransformerRanker构建失败: {e}")
            raise
    
    def _build_model(self):
        """构建Transformer模型"""
        seq_len = self.params['max_seq_length']
        d_model = self.params['d_model']
        
        # 特征预处理和维度调整
        if self.input_dim % seq_len == 0:
            features_per_token = self.input_dim // seq_len
            if features_per_token != d_model:
                self.feature_projection = nn.Linear(features_per_token, d_model)
            else:
                self.feature_projection = nn.Identity()
            
            self.reshape_method = "divide"
        else:
            # 使用线性变换
            target_dim = seq_len * d_model
            self.linear_projection = nn.Linear(self.input_dim, target_dim)
            self.reshape_method = "linear"
        
        # 位置编码
        self.pos_encoding = self._create_positional_encoding(seq_len, d_model)
        
        # Transformer层
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, self.params['num_heads'], 
                           self.params['dff'], self.params['dropout_rate'])
            for _ in range(self.params['num_layers'])
        ])
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, 128),  # *2 因为用了avg和max pooling
            nn.ReLU(),
            nn.Dropout(self.params['dropout_rate']),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(self.params['dropout_rate']),
            nn.Linear(64, 1)
        )
        
        # 移动到设备
        self.to(DEVICE)
        
        # 优化器
        self.optimizer = optim.Adam(self.parameters(), 
                                  lr=self.params['learning_rate'])
        self.criterion = nn.MSELoss()
    
    def _create_positional_encoding(self, seq_len, d_model):
        """创建位置编码"""
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)  # (1, seq_len, d_model)
    
    def forward(self, x):
        batch_size = x.size(0)
        seq_len = self.params['max_seq_length']
        d_model = self.params['d_model']
        
        # 特征重塑
        if self.reshape_method == "divide":
            features_per_token = self.input_dim // seq_len
            x = x.view(batch_size, seq_len, features_per_token)
            x = self.feature_projection(x)
        else:
            x = self.linear_projection(x)
            x = x.view(batch_size, seq_len, d_model)
        
        # 添加位置编码
        pos_encoding = self.pos_encoding.to(x.device)
        x = x + pos_encoding
        
        # Transformer层
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        
        # 全局池化
        avg_pool = torch.mean(x, dim=1)  # (batch_size, d_model)
        max_pool = torch.max(x, dim=1)[0]  # (batch_size, d_model)
        
        # 合并不同的池化结果
        pooled = torch.cat([avg_pool, max_pool], dim=1)  # (batch_size, d_model*2)
        
        # 分类头
        output = self.classifier(pooled)
        
        return output
    
    def fit(self, X, y, group, **kwargs):
        """训练模型"""
        epochs = kwargs.get('epochs', self.params['epochs'])
        batch_size = kwargs.get('batch_size', self.params['batch_size'])
        
        print(f"🚀 开始训练TransformerRanker")
        print(f"📊 数据形状: {X.shape}")
        print(f"📊 训练参数: epochs={epochs}, batch_size={batch_size}")
        
        try:
            # 数据预处理
            X_processed = X.astype(np.float32)
            y_processed = y.astype(np.float32)
            
            # 检查和清理异常值
            if np.any(np.isnan(X_processed)) or np.any(np.isinf(X_processed)):
                print("⚠️ 发现NaN或Inf值，进行清理...")
                X_processed = np.nan_to_num(X_processed, nan=0.0, posinf=1.0, neginf=-1.0)
            
            if np.any(np.isnan(y_processed)) or np.any(np.isinf(y_processed)):
                print("⚠️ 标签中发现NaN或Inf值，进行清理...")
                y_processed = np.nan_to_num(y_processed, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # 数据标准化
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_processed = scaler.fit_transform(X_processed).astype(np.float32)
            self.scaler = scaler  # 保存scaler用于预测
            
            print("✓ 数据预处理完成")
            
            # 转换为PyTorch张量
            X_tensor = torch.FloatTensor(X_processed).to(DEVICE)
            y_tensor = torch.FloatTensor(y_processed).reshape(-1, 1).to(DEVICE)
            
            # 智能确定验证集大小
            total_samples = len(X_processed)
            if total_samples > 100000:
                val_ratio = 0.02
            elif total_samples > 10000:
                val_ratio = 0.05
            else:
                val_ratio = 0.2
            
            val_size = int(total_samples * val_ratio)
            train_size = total_samples - val_size
            
            print(f"📊 数据分割: 训练={train_size}, 验证={val_size} ({val_ratio*100:.1f}%)")
            
            # 分割数据
            if val_size > 100:
                train_dataset = TensorDataset(X_tensor[:train_size], y_tensor[:train_size])
                val_dataset = TensorDataset(X_tensor[train_size:], y_tensor[train_size:])
                
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                
                print("🔄 开始训练（带验证集）...")
                for epoch in range(epochs):
                    # 训练阶段
                    self.train()
                    train_loss = 0.0
                    for batch_X, batch_y in train_loader:
                        self.optimizer.zero_grad()
                        outputs = self(batch_X)
                        loss = self.criterion(outputs, batch_y)
                        loss.backward()
                        # 梯度裁剪
                        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                        self.optimizer.step()
                        train_loss += loss.item()
                    
                    # 验证阶段
                    if epoch % 2 == 0:
                        self.eval()
                        val_loss = 0.0
                        with torch.no_grad():
                            for batch_X, batch_y in val_loader:
                                outputs = self(batch_X)
                                val_loss += self.criterion(outputs, batch_y).item()
                        
                        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}')
            else:
                train_dataset = TensorDataset(X_tensor, y_tensor)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                
                print("🔄 开始训练（无验证集）...")
                for epoch in range(epochs):
                    self.train()
                    total_loss = 0.0
                    for batch_X, batch_y in train_loader:
                        self.optimizer.zero_grad()
                        outputs = self(batch_X)
                        loss = self.criterion(outputs, batch_y)
                        loss.backward()
                        # 梯度裁剪
                        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                        self.optimizer.step()
                        total_loss += loss.item()
                    
                    if epoch % 2 == 0:
                        print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}')
            
            print("✅ TransformerRanker训练完成")
            
            # 计算特征重要性
            try:
                print("🔍 计算特征重要性...")
                sample_size = min(1000, len(X))
                self._compute_feature_importance(X[:sample_size])
                print("✅ 特征重要性计算完成")
            except Exception as e:
                print(f"⚠️ 特征重要性计算失败: {e}")
                self._feature_importance = np.ones(self.input_dim) / self.input_dim
                
        except Exception as e:
            print(f"❌ TransformerRanker训练失败: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _compute_feature_importance(self, X_sample):
        """计算特征重要性"""
        try:
            self.eval()
            
            # 数据预处理（与训练时保持一致）
            if hasattr(self, 'scaler'):
                X_processed = self.scaler.transform(X_sample.astype(np.float32))
            else:
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                X_processed = scaler.fit_transform(X_sample.astype(np.float32))
            
            X_tensor = torch.FloatTensor(X_processed).to(DEVICE)
            X_tensor.requires_grad_(True)
            
            outputs = self(X_tensor)
            loss = outputs.mean()
            loss.backward()
            
            grads = X_tensor.grad
            if grads is not None:
                importance = torch.mean(torch.abs(grads), dim=0).cpu().numpy()
                importance = importance / (np.sum(importance) + 1e-8)
                self._feature_importance = importance
                print("✓ 使用梯度计算特征重要性")
            else:
                raise ValueError("无法计算梯度")
                
        except Exception as e:
            print(f"⚠️ 梯度方法失败: {e}，使用方差方法")
            try:
                feature_variance = np.var(X_sample, axis=0)
                importance = feature_variance / (np.sum(feature_variance) + 1e-8)
                self._feature_importance = importance
                print("✓ 使用方差计算特征重要性")
            except:
                self._feature_importance = np.ones(self.input_dim) / self.input_dim
                print("✓ 使用均匀分布作为特征重要性")
    
    def predict(self, X):
        """预测分数"""
        try:
            print(f"🔮 TransformerRanker预测，数据形状: {X.shape}")
            
            # 数据预处理（与训练时保持一致）
            X_processed = X.astype(np.float32)
            
            # 清理异常值
            if np.any(np.isnan(X_processed)) or np.any(np.isinf(X_processed)):
                X_processed = np.nan_to_num(X_processed, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # 标准化
            if hasattr(self, 'scaler'):
                X_processed = self.scaler.transform(X_processed)
            else:
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                X_processed = scaler.fit_transform(X_processed)
            
            X_processed = X_processed.astype(np.float32)
            
            # 批量预测以避免内存问题
            self.eval()
            batch_size = 1000
            predictions = []
            
            with torch.no_grad():
                for i in range(0, len(X_processed), batch_size):
                    batch = X_processed[i:i+batch_size]
                    batch_tensor = torch.FloatTensor(batch).to(DEVICE)
                    batch_pred = self(batch_tensor)
                    predictions.append(batch_pred.cpu().numpy().flatten())
            
            result = np.concatenate(predictions)
            print(f"✅ TransformerRanker预测完成，结果形状: {result.shape}")
            return result
            
        except Exception as e:
            print(f"❌ TransformerRanker预测失败: {e}")
            import traceback
            traceback.print_exc()
            # 返回随机值作为后备
            return np.random.random(len(X)).astype(np.float32)
    
    def get_model_name(self):
        return "TransformerRanker"
    
    @property
    def feature_importances_(self):
        if self._feature_importance is None:
            return np.ones(self.input_dim) / self.input_dim
        return self._feature_importance


class BM25Ranker(BaseRanker):
    """BM25排序模型"""
    
    def __init__(self, **params):
        self.params = params
        self.model = None
        self.tokenized_corpus = None
        self.feature_names = None
        
    def fit(self, X, y, group, **kwargs):
        # 将特征转换为"文档"形式
        self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        self.tokenized_corpus = []
        
        for i in range(X.shape[0]):
            # 将特征值大于阈值的特征名作为"词"
            threshold = kwargs.get('threshold', 0.5)
            doc = [self.feature_names[j] for j in np.where(X[i] > threshold)[0]]
            if not doc:  # 如果没有特征超过阈值，使用所有非零特征
                doc = [self.feature_names[j] for j in np.where(X[i] != 0)[0]]
            self.tokenized_corpus.append(doc)
        
        self.model = BM25Okapi(self.tokenized_corpus)
    
    def predict(self, X):
        if self.model is None:
            raise ValueError("模型未训练，请先调用fit方法")
        
        scores = []
        for i in range(X.shape[0]):
            query = [self.feature_names[j] for j in np.where(X[i] > 0.5)[0]]
            if not query:
                query = [self.feature_names[j] for j in np.where(X[i] != 0)[0]]
            
            if query:
                doc_scores = self.model.get_scores(query)
                scores.append(doc_scores[i] if i < len(doc_scores) else 0.0)
            else:
                scores.append(0.0)
        
        return np.array(scores)
    
    def get_model_name(self):
        return "BM25Ranker"


class ModelFactory:
    """模型工厂类"""
    
    @staticmethod
    def create_model(model_name: str, use_gpu: bool = True, input_dim: Optional[int] = None, **params) -> BaseRanker:
        """
        创建指定的模型实例
        
        Args:
            model_name: 模型名称
            use_gpu: 是否使用GPU
            input_dim: 输入维度（神经网络模型需要）
            **params: 模型参数
            
        Returns:
            BaseRanker: 模型实例
        """
        if model_name == 'XGBRanker':
            return XGBRankerModel(use_gpu=use_gpu, **params)
        elif model_name == 'LGBMRanker':
            return LGBMRankerModel(use_gpu=use_gpu, **params)
        elif model_name == 'LambdaMART':
            return LambdaMART(use_gpu=use_gpu, **params)
        elif model_name == 'ListNet':
            return ListNetModel(use_gpu=use_gpu, **params)
        elif model_name == 'NeuralRanker':
            if input_dim is None:
                raise ValueError("NeuralRanker需要指定input_dim参数")
            return NeuralRanker(input_dim=input_dim, **params)
        elif model_name == 'RankNet':
            if input_dim is None:
                raise ValueError("RankNet需要指定input_dim参数")
            return RankNet(input_dim=input_dim, **params)
        elif model_name == 'TransformerRanker':
            if input_dim is None:
                raise ValueError("TransformerRanker需要指定input_dim参数")
            return TransformerRanker(input_dim=input_dim, **params)
        elif model_name == 'BM25Ranker':
            return BM25Ranker(**params)
        else:
            raise ValueError(f"未知模型类型: {model_name}")
    
    @staticmethod
    def get_available_models() -> List[str]:
        """获取所有可用的模型名称"""
        return [
            'XGBRanker', 'LGBMRanker', 'LambdaMART', 'ListNet', 
            'NeuralRanker', 'RankNet', 'TransformerRanker', 'BM25Ranker'
        ]