"""
GPU加速航班排名模型集合
支持传统机器学习、深度学习和Transformer排名模型
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
import logging
import warnings
import joblib
from abc import ABC, abstractmethod

# 传统ML库
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler

# 深度学习库
import torch.optim as optim


warnings.filterwarnings('ignore')


class BaseRankingModel(ABC):
    """排名模型基类"""
    
    def __init__(self, use_gpu: bool = True, random_state: int = 42, logger=None):
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.random_state = random_state
        self.logger = logger or logging.getLogger(__name__)
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')
        self.feature_names = []
        self.is_fitted = False
        
        if self.use_gpu:
            self.logger.info(f"使用GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.logger.info("使用CPU")
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray):
        """训练模型"""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测分数"""
        pass
    
    def save_model(self, filepath: str):
        """保存模型"""
        if not self.is_fitted:
            raise ValueError("模型未训练，无法保存")
        joblib.dump(self, filepath)
        self.logger.info(f"模型已保存到: {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str):
        """加载模型"""
        return joblib.load(filepath)


class XGBoostRanker(BaseRankingModel):
    """GPU加速的XGBoost排名模型"""
    
    def __init__(self, n_estimators=200, max_depth=8, learning_rate=0.05, 
                 use_gpu=True, **kwargs):
        super().__init__(use_gpu, kwargs.get('random_state', 42))
        
        # GPU配置
        params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'random_state': self.random_state,
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
                self.logger.warning(f"XGBoost GPU失败，回退到CPU: {e}")
                params.update({
                    'tree_method': 'hist',
                    'n_jobs': -1
                })
        else:
            params.update({
                'tree_method': 'hist',
                'n_jobs': -1
            })
        
        self.model = xgb.XGBRanker(**params)
    
    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray):
        """训练XGBoost排名模型"""
        try:
            # 计算组大小
            unique_groups = np.unique(groups)
            group_sizes = [np.sum(groups == g) for g in unique_groups]
            
            self.model.fit(X, y, group=group_sizes)
            self.is_fitted = True
            self.logger.info("XGBoost训练完成")
        except Exception as e:
            self.logger.error(f"XGBoost训练失败: {e}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测排名分数"""
        if not self.is_fitted:
            raise ValueError("模型未训练")
        return self.model.predict(X)


class LightGBMRanker(BaseRankingModel):
    """GPU加速的LightGBM排名模型"""
    
    def __init__(self, n_estimators=200, max_depth=8, learning_rate=0.05,
                 use_gpu=True, **kwargs):
        super().__init__(use_gpu, kwargs.get('random_state', 42))
        
        params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'random_state': self.random_state,
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
                self.logger.warning(f"LightGBM GPU失败，回退到CPU: {e}")
                params.update({
                    'device': 'cpu',
                    'n_jobs': -1
                })
        else:
            params.update({
                'device': 'cpu',
                'n_jobs': -1
            })
        
        self.model = lgb.LGBMRanker(**params)
    
    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray):
        """训练LightGBM排名模型"""
        try:
            unique_groups = np.unique(groups)
            group_sizes = [np.sum(groups == g) for g in unique_groups]
            
            self.model.fit(X, y, group=group_sizes)
            self.is_fitted = True
            self.logger.info("LightGBM训练完成")
        except Exception as e:
            self.logger.error(f"LightGBM训练失败: {e}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测排名分数"""
        if not self.is_fitted:
            raise ValueError("模型未训练")
        return self.model.predict(X)


class RankNet(BaseRankingModel):
    """深度学习RankNet模型"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64, 32],
                 dropout_rate: float = 0.2, learning_rate: float = 0.001,
                 use_gpu: bool = True, **kwargs):
        super().__init__(use_gpu, kwargs.get('random_state', 42))
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        
        # 构建网络
        self._build_network()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scaler = StandardScaler()
        
        # 移动到GPU
        if self.use_gpu:
            self.model = self.model.to(self.device)
    
    def _build_network(self):
        """构建RankNet网络结构"""
        layers = []
        prev_dim = self.input_dim
        
        for hidden_dim in self.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, 1))
        
        self.model = nn.Sequential(*layers)
    
    def _pairwise_loss(self, scores1: torch.Tensor, scores2: torch.Tensor, 
                      labels1: torch.Tensor, labels2: torch.Tensor) -> torch.Tensor:
        """计算RankNet成对损失"""
        # 标签差异 (1: doc1更相关, -1: doc2更相关, 0: 相同相关性)
        label_diff = labels1 - labels2
        score_diff = scores1 - scores2
        
        # RankNet损失：交叉熵
        prob = torch.sigmoid(score_diff)
        target = (label_diff > 0).float()
        
        # 只考虑标签不同的对
        mask = (label_diff != 0)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=self.device)
        
        loss = F.binary_cross_entropy(prob[mask], target[mask])
        return loss
    
    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray, 
            epochs: int = 100, batch_size: int = 1024):
        """训练RankNet模型"""
        try:
            # 数据预处理
            X_scaled = self.scaler.fit_transform(X)
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            y_tensor = torch.FloatTensor(y).to(self.device)
            
            self.model.train()
            
            for epoch in range(epochs):
                epoch_loss = 0.0
                num_batches = 0
                
                # 生成成对样本
                unique_groups = np.unique(groups)
                for group_id in unique_groups:
                    group_mask = groups == group_id
                    group_X = X_tensor[group_mask]
                    group_y = y_tensor[group_mask]
                    
                    if len(group_X) < 2:
                        continue
                    
                    # 生成所有可能的对
                    indices = torch.arange(len(group_X), device=self.device)
                    pairs = torch.combinations(indices, 2)
                    
                    if len(pairs) == 0:
                        continue
                    
                    # 分批处理
                    for i in range(0, len(pairs), batch_size):
                        batch_pairs = pairs[i:i+batch_size]
                        
                        idx1, idx2 = batch_pairs[:, 0], batch_pairs[:, 1]
                        X1, X2 = group_X[idx1], group_X[idx2]
                        y1, y2 = group_y[idx1], group_y[idx2]
                        
                        # 前向传播
                        scores1 = self.model(X1).squeeze()
                        scores2 = self.model(X2).squeeze()
                        
                        # 计算损失
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
            
        except Exception as e:
            self.logger.error(f"RankNet训练失败: {e}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测排名分数"""
        if not self.is_fitted:
            raise ValueError("模型未训练")
        
        self.model.eval()
        with torch.no_grad():
            X_scaled = self.scaler.transform(X)
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            scores = self.model(X_tensor).squeeze().cpu().numpy()
        
        return scores


class TransformerRanker(BaseRankingModel):
    """基于Transformer的排名模型"""
    
    def __init__(self, input_dim: int, d_model: int = 128, nhead: int = 8,
                 num_layers: int = 3, max_seq_len: int = 100,
                 learning_rate: float = 0.001, use_gpu: bool = True, **kwargs):
        super().__init__(use_gpu, kwargs.get('random_state', 42))
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.learning_rate = learning_rate
        
        # 构建模型
        self._build_model(d_model, nhead, num_layers)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scaler = StandardScaler()
        
        if self.use_gpu:
            self.model = self.model.to(self.device)
    
    def _build_model(self, d_model: int, nhead: int, num_layers: int):
        """构建Transformer排名模型"""
        self.model = nn.ModuleDict({
            'input_projection': nn.Linear(self.input_dim, d_model),
            'pos_encoding': nn.Embedding(self.max_seq_len, d_model),
            'transformer': nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=d_model, 
                    nhead=nhead, 
                    dim_feedforward=d_model*4,
                    dropout=0.1,
                    batch_first=True
                ),
                num_layers=num_layers
            ),
            'output_projection': nn.Linear(d_model, 1)
        })
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向传播"""
        batch_size, seq_len, _ = x.shape
        
        # 输入投影
        x = self.model['input_projection'](x)
        
        # 位置编码
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.model['pos_encoding'](positions)
        x = x + pos_emb
        
        # Transformer编码
        if mask is not None:
            mask = ~mask.bool()  # 转换为attention mask
        
        x = self.model['transformer'](x, src_key_padding_mask=mask)
        
        # 输出投影
        scores = self.model['output_projection'](x).squeeze(-1)
        
        return scores
    
    def _listwise_loss(self, scores: torch.Tensor, labels: torch.Tensor, 
                      mask: torch.Tensor) -> torch.Tensor:
        """ListNet损失函数"""
        # 排除padding的位置
        valid_scores = scores.masked_fill(~mask, float('-inf'))
        valid_labels = labels.masked_fill(~mask, 0)
        
        # ListNet概率分布
        score_probs = F.softmax(valid_scores, dim=-1)
        label_probs = F.softmax(valid_labels.float(), dim=-1)
        
        # KL散度损失
        loss = F.kl_div(score_probs.log(), label_probs, reduction='batchmean')
        return loss
    
    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray,
            epochs: int = 50, batch_size: int = 32):
        """训练Transformer排名模型"""
        try:
            # 数据预处理
            X_scaled = self.scaler.fit_transform(X)
            
            # 按组准备数据
            unique_groups = np.unique(groups)
            group_data = []
            
            for group_id in unique_groups:
                group_mask = groups == group_id
                group_X = X_scaled[group_mask]
                group_y = y[group_mask]
                
                if len(group_X) >= 2:  # 至少需要2个样本
                    # 截断或填充到固定长度
                    if len(group_X) > self.max_seq_len:
                        group_X = group_X[:self.max_seq_len]
                        group_y = group_y[:self.max_seq_len]
                    
                    group_data.append((group_X, group_y))
            
            if not group_data:
                raise ValueError("没有有效的组数据")
            
            self.model.train()
            
            for epoch in range(epochs):
                epoch_loss = 0.0
                num_batches = 0
                
                # 打乱数据
                np.random.shuffle(group_data)
                
                for i in range(0, len(group_data), batch_size):
                    batch_groups = group_data[i:i+batch_size]
                    
                    # 填充到相同长度
                    max_len = max(len(x) for x, _ in batch_groups)
                    batch_X = []
                    batch_y = []
                    batch_mask = []
                    
                    for group_X, group_y in batch_groups:
                        pad_len = max_len - len(group_X)
                        if pad_len > 0:
                            padded_X = np.vstack([group_X, np.zeros((pad_len, group_X.shape[1]))])
                            padded_y = np.concatenate([group_y, np.zeros(pad_len)])
                            mask = np.concatenate([np.ones(len(group_X)), np.zeros(pad_len)])
                        else:
                            padded_X = group_X
                            padded_y = group_y
                            mask = np.ones(len(group_X))
                        
                        batch_X.append(padded_X)
                        batch_y.append(padded_y)
                        batch_mask.append(mask)
                    
                    # 转换为tensor
                    X_tensor = torch.FloatTensor(np.stack(batch_X)).to(self.device)
                    y_tensor = torch.FloatTensor(np.stack(batch_y)).to(self.device)
                    mask_tensor = torch.BoolTensor(np.stack(batch_mask)).to(self.device)
                    
                    # 前向传播
                    scores = self.forward(X_tensor, mask_tensor)
                    loss = self._listwise_loss(scores, y_tensor, mask_tensor)
                    
                    # 反向传播
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                    epoch_loss += loss.item()
                    num_batches += 1
                
                if (epoch + 1) % 10 == 0:
                    avg_loss = epoch_loss / num_batches
                    self.logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            
            self.is_fitted = True
            self.logger.info("Transformer训练完成")
            
        except Exception as e:
            self.logger.error(f"Transformer训练失败: {e}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测排名分数"""
        if not self.is_fitted:
            raise ValueError("模型未训练")
        
        self.model.eval()
        with torch.no_grad():
            X_scaled = self.scaler.transform(X)
            
            # 如果输入太长，分组处理
            if len(X_scaled) > self.max_seq_len:
                # 简单分组策略
                n_chunks = (len(X_scaled) + self.max_seq_len - 1) // self.max_seq_len
                chunk_size = len(X_scaled) // n_chunks
                
                all_scores = []
                for i in range(0, len(X_scaled), chunk_size):
                    chunk_X = X_scaled[i:i+chunk_size]
                    chunk_tensor = torch.FloatTensor(chunk_X).unsqueeze(0).to(self.device)
                    chunk_mask = torch.ones(1, len(chunk_X), dtype=torch.bool, device=self.device)
                    chunk_scores = self.forward(chunk_tensor, chunk_mask).squeeze(0)
                    all_scores.append(chunk_scores.cpu().numpy())
                
                scores = np.concatenate(all_scores)
            else:
                X_tensor = torch.FloatTensor(X_scaled).unsqueeze(0).to(self.device)
                mask = torch.ones(1, len(X_scaled), dtype=torch.bool, device=self.device)
                scores = self.forward(X_tensor, mask).squeeze(0).cpu().numpy()
        
        return scores


class FlightRankingModels:
    """航班排名模型管理器"""
    
    def __init__(self, use_gpu: bool = True, logger=None):
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.logger = logger or logging.getLogger(__name__)
        self.models: Dict[str, BaseRankingModel] = {}
        self.feature_names: List[str] = []
        
        # 检查GPU状态
        if self.use_gpu:
            self.logger.info(f"GPU可用: {torch.cuda.get_device_name(0)}")
            self.logger.info(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        else:
            self.logger.info("使用CPU模式")
    
    def add_model(self, name: str, model: BaseRankingModel):
        """添加模型"""
        self.models[name] = model
        self.logger.info(f"添加模型: {name}")
    
    def create_models(self, input_dim: int, model_configs: Dict = None) -> Dict[str, BaseRankingModel]:
        """创建所有可用的模型"""
        default_configs = {
            'XGBRanker': {'n_estimators': 200, 'max_depth': 8, 'learning_rate': 0.05},
            'LGBMRanker': {'n_estimators': 200, 'max_depth': 8, 'learning_rate': 0.05},
            'RankNet': {'input_dim': input_dim, 'hidden_dims': [128, 64, 32]},
            'TransformerRanker': {'input_dim': input_dim, 'd_model': 128, 'nhead': 8}
        }
        
        if model_configs:
            default_configs.update(model_configs)
        
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
        
        # Transformer
        try:
            created_models['TransformerRanker'] = TransformerRanker(
                use_gpu=self.use_gpu,
                logger=self.logger,
                **default_configs['TransformerRanker']
            )
            self.logger.info("✓ Transformer模型创建成功")
        except Exception as e:
            self.logger.warning(f"✗ Transformer创建失败: {e}")
        
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
        """训练指定的模型"""
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
                
                # 根据模型类型设置训练参数
                if isinstance(model, (RankNet, TransformerRanker)):
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
        
        # 归一化权重
        weights = np.array(weights) / np.sum(weights)
        
        # 收集预测结果
        predictions = []
        valid_weights = []
        
        for i, name in enumerate(model_names):
            if name in self.models and self.models[name].is_fitted:
                try:
                    pred = self.models[name].predict(X)
                    predictions.append(pred)
                    valid_weights.append(weights[i])
                    self.logger.info(f"✓ {name} 预测完成")
                except Exception as e:
                    self.logger.warning(f"✗ {name} 预测失败: {e}")
        
        if not predictions:
            raise ValueError("所有模型预测都失败")
        
        # 加权平均
        valid_weights = np.array(valid_weights) / np.sum(valid_weights)
        ensemble_pred = np.average(predictions, axis=0, weights=valid_weights)
        
        self.logger.info(f"集成预测完成，使用了 {len(predictions)} 个模型")
        return ensemble_pred
    
    def save_models(self, save_dir: str):
        """保存所有模型"""
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
                    self.models[name] = BaseRankingModel.load_model(filepath)
                    self.logger.info(f"✓ 加载模型: {name}")
                except Exception as e:
                    self.logger.warning(f"✗ 加载模型失败 {name}: {e}")
        
        # 加载特征名称
        feature_path = os.path.join(save_dir, "features.pkl")
        if os.path.exists(feature_path):
            self.feature_names = joblib.load(feature_path)
        
        self.logger.info("模型加载完成")