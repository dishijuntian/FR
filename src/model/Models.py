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
# from torch_geometric.nn import GCNConv, global_mean_pool
# from torch_geometric.data import Data, Batch

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
    
    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray, epochs: int = 100):
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
                
                idx1, idx2 = pairs[:, 0], pairs[:, 1]
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


# class GraphRanker:
#     """图神经网络排名模型 - 建模航班间的竞争关系"""
    
#     def __init__(self, input_dim: int, hidden_dims: List[int] = [64, 32], 
#                  num_gnn_layers: int = 3, dropout_rate: float = 0.2,
#                  learning_rate: float = 0.001, use_gpu: bool = True, 
#                  random_state: int = 42, logger=None):
#         self.input_dim = input_dim
#         self.hidden_dims = hidden_dims
#         self.num_gnn_layers = num_gnn_layers
#         self.dropout_rate = dropout_rate
#         self.learning_rate = learning_rate
#         self.use_gpu = use_gpu and torch.cuda.is_available()
#         self.logger = logger or logging.getLogger(__name__)
#         self.device = torch.device('cuda' if self.use_gpu else 'cpu')
#         self.is_fitted = False
        
#         # 构建网络
#         self._build_network()
#         self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
#         self.scaler = StandardScaler()
        
#         if self.use_gpu:
#             self.model = self.model.to(self.device)
#             self.logger.info("GraphRanker GPU加速可用")
    
#     def _build_network(self):
#         """构建图神经网络"""
#         class GNNRankingModel(nn.Module):
#             def __init__(self, input_dim, hidden_dims, num_gnn_layers, dropout_rate):
#                 super().__init__()
#                 self.num_gnn_layers = num_gnn_layers
                
#                 # GNN层
#                 self.gnn_layers = nn.ModuleList()
#                 prev_dim = input_dim
                
#                 for i in range(num_gnn_layers):
#                     if i < len(hidden_dims):
#                         curr_dim = hidden_dims[i]
#                     else:
#                         curr_dim = hidden_dims[-1]
                    
#                     self.gnn_layers.append(GCNConv(prev_dim, curr_dim))
#                     prev_dim = curr_dim
                
#                 # 最终预测层
#                 self.final_layers = nn.Sequential(
#                     nn.Linear(prev_dim, hidden_dims[-1] if hidden_dims else 32),
#                     nn.ReLU(),
#                     nn.Dropout(dropout_rate),
#                     nn.Linear(hidden_dims[-1] if hidden_dims else 32, 1)
#                 )
            
#             def forward(self, x, edge_index, batch):
#                 # GNN层
#                 for i, gnn_layer in enumerate(self.gnn_layers):
#                     x = gnn_layer(x, edge_index)
#                     if i < len(self.gnn_layers) - 1:
#                         x = F.relu(x)
#                         x = F.dropout(x, training=self.training)
                
#                 # 最终预测
#                 x = self.final_layers(x)
#                 return x.squeeze()
        
#         self.model = GNNRankingModel(
#             self.input_dim, self.hidden_dims, 
#             self.num_gnn_layers, self.dropout_rate
#         )
    
#     def _create_flight_graph(self, X: np.ndarray, groups: np.ndarray) -> List[Data]:
#         """为每个搜索会话创建航班竞争图"""
#         graph_list = []
#         unique_groups = np.unique(groups)
        
#         for group_id in unique_groups:
#             group_mask = groups == group_id
#             group_X = X[group_mask]
            
#             if len(group_X) < 2:
#                 continue
            
#             # 节点特征
#             node_features = torch.FloatTensor(group_X)
            
#             # 创建边：同一搜索会话中的航班互相连接（完全图）
#             num_nodes = len(group_X)
#             edge_list = []
            
#             for i in range(num_nodes):
#                 for j in range(num_nodes):
#                     if i != j:
#                         edge_list.append([i, j])
            
#             if edge_list:
#                 edge_index = torch.LongTensor(edge_list).t().contiguous()
#             else:
#                 edge_index = torch.empty((2, 0), dtype=torch.long)
            
#             # 创建图数据
#             graph_data = Data(x=node_features, edge_index=edge_index)
#             graph_list.append((graph_data, group_mask))
        
#         return graph_list
    
#     def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray, epochs: int = 100):
#         """训练模型"""
#         X_scaled = self.scaler.fit_transform(X)
        
#         self.model.train()
        
#         for epoch in range(epochs):
#             epoch_loss = 0.0
#             num_batches = 0
            
#             # 为每个组创建图
#             graph_list = self._create_flight_graph(X_scaled, groups)
            
#             for graph_data, group_mask in graph_list:
#                 group_y = torch.FloatTensor(y[group_mask]).to(self.device)
                
#                 if len(group_y) < 2:
#                     continue
                
#                 graph_data = graph_data.to(self.device)
                
#                 # 前向传播
#                 scores = self.model(graph_data.x, graph_data.edge_index, None)
                
#                 # 计算ranking loss (listwise loss)
#                 # 使用softmax + cross entropy for ranking
#                 if torch.sum(group_y) == 1:  # 只有一个正样本
#                     target_idx = torch.argmax(group_y)
#                     loss = F.cross_entropy(scores.unsqueeze(0), target_idx.unsqueeze(0))
#                 else:
#                     # 使用pairwise ranking loss
#                     pos_mask = group_y > 0
#                     neg_mask = group_y == 0
                    
#                     if pos_mask.sum() > 0 and neg_mask.sum() > 0:
#                         pos_scores = scores[pos_mask].mean()
#                         neg_scores = scores[neg_mask].mean()
#                         loss = F.relu(1.0 - (pos_scores - neg_scores))
#                     else:
#                         continue
                
#                 if loss > 0:
#                     self.optimizer.zero_grad()
#                     loss.backward()
#                     self.optimizer.step()
                    
#                     epoch_loss += loss.item()
#                     num_batches += 1
            
#             if (epoch + 1) % 20 == 0 and num_batches > 0:
#                 avg_loss = epoch_loss / num_batches
#                 self.logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
#         self.is_fitted = True
#         self.logger.info("GraphRanker训练完成")
    
#     def predict(self, X: np.ndarray, groups: np.ndarray) -> np.ndarray:
#         """预测分数"""
#         if not self.is_fitted:
#             raise ValueError("模型未训练")
        
#         self.model.eval()
#         X_scaled = self.scaler.transform(X)
#         all_scores = np.zeros(len(X))
        
#         with torch.no_grad():
#             graph_list = self._create_flight_graph(X_scaled, groups)
            
#             for graph_data, group_mask in graph_list:
#                 graph_data = graph_data.to(self.device)
#                 scores = self.model(graph_data.x, graph_data.edge_index, None)
#                 all_scores[group_mask] = scores.cpu().numpy()
        
#         return all_scores
    
#     def save_model(self, filepath: str):
#         """保存模型"""
#         if not self.is_fitted:
#             raise ValueError("模型未训练，无法保存")
#         joblib.dump(self, filepath)
    
#     @classmethod
#     def load_model(cls, filepath: str):
#         """加载模型"""
#         return joblib.load(filepath)


# class CNNRanker:
#     """CNN排名模型 - 处理特征序列模式"""
    
#     def __init__(self, input_dim: int, sequence_length: int = 10,
#                  conv_channels: List[int] = [32, 64, 128], 
#                  kernel_sizes: List[int] = [3, 5, 7],
#                  hidden_dims: List[int] = [128, 64],
#                  dropout_rate: float = 0.2, learning_rate: float = 0.001,
#                  use_gpu: bool = True, random_state: int = 42, logger=None):
#         self.input_dim = input_dim
#         self.sequence_length = sequence_length
#         self.conv_channels = conv_channels
#         self.kernel_sizes = kernel_sizes
#         self.hidden_dims = hidden_dims
#         self.dropout_rate = dropout_rate
#         self.learning_rate = learning_rate
#         self.use_gpu = use_gpu and torch.cuda.is_available()
#         self.logger = logger or logging.getLogger(__name__)
#         self.device = torch.device('cuda' if self.use_gpu else 'cpu')
#         self.is_fitted = False
        
#         # 构建网络
#         self._build_network()
#         self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
#         self.scaler = StandardScaler()
        
#         if self.use_gpu:
#             self.model = self.model.to(self.device)
#             self.logger.info("CNNRanker GPU加速可用")
    
#     def _build_network(self):
#         """构建CNN网络"""
#         class CNNRankingModel(nn.Module):
#             def __init__(self, input_dim, sequence_length, conv_channels, 
#                         kernel_sizes, hidden_dims, dropout_rate):
#                 super().__init__()
#                 self.sequence_length = sequence_length
                
#                 # 多尺度卷积层
#                 self.conv_layers = nn.ModuleList()
#                 for i, (channels, kernel_size) in enumerate(zip(conv_channels, kernel_sizes)):
#                     if i == 0:
#                         in_channels = 1  # 输入是1D信号
#                     else:
#                         in_channels = conv_channels[i-1]
                    
#                     self.conv_layers.append(nn.Sequential(
#                         nn.Conv1d(in_channels, channels, kernel_size, padding=kernel_size//2),
#                         nn.ReLU(),
#                         nn.BatchNorm1d(channels),
#                         nn.Dropout(dropout_rate)
#                     ))
                
#                 # 全局池化
#                 self.global_pool = nn.AdaptiveAvgPool1d(1)
                
#                 # 全连接层
#                 fc_input_dim = sum(conv_channels) + input_dim  # 卷积特征 + 原始特征
                
#                 fc_layers = []
#                 prev_dim = fc_input_dim
#                 for hidden_dim in hidden_dims:
#                     fc_layers.extend([
#                         nn.Linear(prev_dim, hidden_dim),
#                         nn.ReLU(),
#                         nn.Dropout(dropout_rate)
#                     ])
#                     prev_dim = hidden_dim
                
#                 fc_layers.append(nn.Linear(prev_dim, 1))
#                 self.fc = nn.Sequential(*fc_layers)
            
#             def forward(self, x):
#                 batch_size = x.size(0)
                
#                 # 将特征重塑为序列格式
#                 # 简单的方法：将特征分割成多个片段
#                 feature_dim = x.size(1)
#                 if feature_dim >= self.sequence_length:
#                     # 将特征分割成sequence_length段
#                     segment_size = feature_dim // self.sequence_length
#                     x_seq = x[:, :segment_size * self.sequence_length]
#                     x_seq = x_seq.view(batch_size, 1, segment_size * self.sequence_length)
#                 else:
#                     # 重复特征以达到所需长度
#                     repeat_times = self.sequence_length // feature_dim + 1
#                     x_seq = x.repeat(1, repeat_times)[:, :self.sequence_length]
#                     x_seq = x_seq.view(batch_size, 1, self.sequence_length)
                
#                 # 多尺度卷积特征提取
#                 conv_features = []
#                 current_x = x_seq
                
#                 for conv_layer in self.conv_layers:
#                     current_x = conv_layer(current_x)
#                     # 全局平均池化
#                     pooled = self.global_pool(current_x).squeeze(-1)
#                     conv_features.append(pooled)
                
#                 # 合并卷积特征和原始特征
#                 conv_concat = torch.cat(conv_features, dim=1)
#                 combined_features = torch.cat([conv_concat, x], dim=1)
                
#                 # 最终预测
#                 output = self.fc(combined_features)
#                 return output.squeeze()
        
#         self.model = CNNRankingModel(
#             self.input_dim, self.sequence_length, self.conv_channels,
#             self.kernel_sizes, self.hidden_dims, self.dropout_rate
#         )
    
#     def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray, epochs: int = 100):
#         """训练模型"""
#         X_scaled = self.scaler.fit_transform(X)
#         X_tensor = torch.FloatTensor(X_scaled).to(self.device)
#         y_tensor = torch.FloatTensor(y).to(self.device)
        
#         self.model.train()
        
#         for epoch in range(epochs):
#             epoch_loss = 0.0
#             num_batches = 0
            
#             unique_groups = np.unique(groups)
#             for group_id in unique_groups:
#                 group_mask = groups == group_id
#                 group_X = X_tensor[group_mask]
#                 group_y = y_tensor[group_mask]
                
#                 if len(group_X) < 2:
#                     continue
                
#                 # 前向传播
#                 scores = self.model(group_X)
                
#                 # 计算ranking loss
#                 if torch.sum(group_y) == 1:  # 只有一个正样本
#                     target_idx = torch.argmax(group_y)
#                     loss = F.cross_entropy(scores.unsqueeze(0), target_idx.unsqueeze(0))
#                 else:
#                     # 使用pairwise ranking loss
#                     pos_mask = group_y > 0
#                     neg_mask = group_y == 0
                    
#                     if pos_mask.sum() > 0 and neg_mask.sum() > 0:
#                         pos_scores = scores[pos_mask].mean()
#                         neg_scores = scores[neg_mask].mean()
#                         loss = F.relu(1.0 - (pos_scores - neg_scores))
#                     else:
#                         continue
                
#                 if loss > 0:
#                     self.optimizer.zero_grad()
#                     loss.backward()
#                     self.optimizer.step()
                    
#                     epoch_loss += loss.item()
#                     num_batches += 1
            
#             if (epoch + 1) % 20 == 0 and num_batches > 0:
#                 avg_loss = epoch_loss / num_batches
#                 self.logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
#         self.is_fitted = True
#         self.logger.info("CNNRanker训练完成")
    
#     def predict(self, X: np.ndarray) -> np.ndarray:
#         """预测分数"""
#         if not self.is_fitted:
#             raise ValueError("模型未训练")
        
#         self.model.eval()
#         with torch.no_grad():
#             X_scaled = self.scaler.transform(X)
#             X_tensor = torch.FloatTensor(X_scaled).to(self.device)
#             scores = self.model(X_tensor).cpu().numpy()
        
#         return scores
    
#     def save_model(self, filepath: str):
#         """保存模型"""
#         if not self.is_fitted:
#             raise ValueError("模型未训练，无法保存")
#         joblib.dump(self, filepath)
    
#     @classmethod
#     def load_model(cls, filepath: str):
#         """加载模型"""
#         return joblib.load(filepath)


# class TransformerRanker:
#     """Transformer排名模型 - 处理序列依赖关系"""
    
#     def __init__(self, input_dim: int, d_model: int = 128, nhead: int = 8,
#                  num_layers: int = 3, dim_feedforward: int = 512,
#                  dropout_rate: float = 0.1, learning_rate: float = 0.001,
#                  use_gpu: bool = True, random_state: int = 42, logger=None):
#         self.input_dim = input_dim
#         self.d_model = d_model
#         self.nhead = nhead
#         self.num_layers = num_layers
#         self.dim_feedforward = dim_feedforward
#         self.dropout_rate = dropout_rate
#         self.learning_rate = learning_rate
#         self.use_gpu = use_gpu and torch.cuda.is_available()
#         self.logger = logger or logging.getLogger(__name__)
#         self.device = torch.device('cuda' if self.use_gpu else 'cpu')
#         self.is_fitted = False
        
#         # 构建网络
#         self._build_network()
#         self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
#         self.scaler = StandardScaler()
        
#         if self.use_gpu:
#             self.model = self.model.to(self.device)
#             self.logger.info("TransformerRanker GPU加速可用")
    
#     def _build_network(self):
#         """构建Transformer网络"""
#         class TransformerRankingModel(nn.Module):
#             def __init__(self, input_dim, d_model, nhead, num_layers, 
#                         dim_feedforward, dropout_rate):
#                 super().__init__()
                
#                 # 输入投影
#                 self.input_projection = nn.Linear(input_dim, d_model)
                
#                 # Transformer编码器
#                 encoder_layer = nn.TransformerEncoderLayer(
#                     d_model=d_model,
#                     nhead=nhead,
#                     dim_feedforward=dim_feedforward,
#                     dropout=dropout_rate,
#                     batch_first=True
#                 )
#                 self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
                
#                 # 输出层
#                 self.output_layer = nn.Sequential(
#                     nn.Linear(d_model, d_model // 2),
#                     nn.ReLU(),
#                     nn.Dropout(dropout_rate),
#                     nn.Linear(d_model // 2, 1)
#                 )
            
#             def forward(self, x, mask=None):
#                 # 输入投影
#                 x = self.input_projection(x)
                
#                 # Transformer编码
#                 x = self.transformer(x, src_key_padding_mask=mask)
                
#                 # 输出预测
#                 x = self.output_layer(x)
#                 return x.squeeze(-1)
        
#         self.model = TransformerRankingModel(
#             self.input_dim, self.d_model, self.nhead, 
#             self.num_layers, self.dim_feedforward, self.dropout_rate
#         )
    
#     def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray, epochs: int = 100):
#         """训练模型"""
#         X_scaled = self.scaler.fit_transform(X)
        
#         self.model.train()
        
#         for epoch in range(epochs):
#             epoch_loss = 0.0
#             num_batches = 0
            
#             unique_groups = np.unique(groups)
#             for group_id in unique_groups:
#                 group_mask = groups == group_id
#                 group_X = X_scaled[group_mask]
#                 group_y = y[group_mask]
                
#                 if len(group_X) < 2:
#                     continue
                
#                 # 转换为tensor并添加序列维度
#                 group_X_tensor = torch.FloatTensor(group_X).unsqueeze(0).to(self.device)
#                 group_y_tensor = torch.FloatTensor(group_y).to(self.device)
                
#                 # 前向传播
#                 scores = self.model(group_X_tensor).squeeze(0)
                
#                 # 计算ranking loss
#                 if torch.sum(group_y_tensor) == 1:  # 只有一个正样本
#                     target_idx = torch.argmax(group_y_tensor)
#                     loss = F.cross_entropy(scores.unsqueeze(0), target_idx.unsqueeze(0))
#                 else:
#                     # 使用pairwise ranking loss
#                     pos_mask = group_y_tensor > 0
#                     neg_mask = group_y_tensor == 0
                    
#                     if pos_mask.sum() > 0 and neg_mask.sum() > 0:
#                         pos_scores = scores[pos_mask].mean()
#                         neg_scores = scores[neg_mask].mean()
#                         loss = F.relu(1.0 - (pos_scores - neg_scores))
#                     else:
#                         continue
                
#                 if loss > 0:
#                     self.optimizer.zero_grad()
#                     loss.backward()
#                     self.optimizer.step()
                    
#                     epoch_loss += loss.item()
#                     num_batches += 1
            
#             if (epoch + 1) % 20 == 0 and num_batches > 0:
#                 avg_loss = epoch_loss / num_batches
#                 self.logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
#         self.is_fitted = True
#         self.logger.info("TransformerRanker训练完成")
    
#     def predict(self, X: np.ndarray, groups: np.ndarray) -> np.ndarray:
#         """预测分数"""
#         if not self.is_fitted:
#             raise ValueError("模型未训练")
        
#         self.model.eval()
#         X_scaled = self.scaler.transform(X)
#         all_scores = np.zeros(len(X))
        
#         with torch.no_grad():
#             unique_groups = np.unique(groups)
#             for group_id in unique_groups:
#                 group_mask = groups == group_id
#                 group_X = X_scaled[group_mask]
                
#                 if len(group_X) == 0:
#                     continue
                
#                 # 转换为tensor并添加序列维度
#                 group_X_tensor = torch.FloatTensor(group_X).unsqueeze(0).to(self.device)
#                 scores = self.model(group_X_tensor).squeeze(0).cpu().numpy()
#                 all_scores[group_mask] = scores
        
#         return all_scores
    
#     def save_model(self, filepath: str):
#         """保存模型"""
#         if not self.is_fitted:
#             raise ValueError("模型未训练，无法保存")
#         joblib.dump(self, filepath)
    
#     @classmethod
#     def load_model(cls, filepath: str):
#         """加载模型"""
#         return joblib.load(filepath)