import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional
import logging
import joblib
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import xgboost as xgb
import lightgbm as lgb



class FastRankingDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray, max_group_size: int = 50):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        unique_groups = np.unique(groups)
        self.group_data = []
        
        for group_id in unique_groups:
            mask = groups == group_id
            group_X = X[mask]
            group_y = y[mask]
            
            if len(group_X) > 1:
                if len(group_X) > max_group_size:
                    idx = np.random.choice(len(group_X), max_group_size, replace=False)
                    group_X = group_X[idx]
                    group_y = group_y[idx]
                
                group_X_tensor = torch.FloatTensor(group_X).to(self.device)
                group_y_tensor = torch.FloatTensor(group_y).to(self.device)
                self.group_data.append((group_X_tensor, group_y_tensor))
    
    def __len__(self):
        return len(self.group_data)
    
    def __getitem__(self, idx):
        return self.group_data[idx]


class XGBoostRanker:
    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.1, 
                 use_gpu=True, random_state=42, logger=None):
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.logger = logger or logging.getLogger(__name__)
        self.is_fitted = False
        
        # XGBoost参数
        self.params = {
            'objective': 'rank:pairwise',
            'eval_metric': 'ndcg',
            'eta': learning_rate,
            'max_depth': max_depth,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'seed': random_state,
            'verbosity': 0
        }
        
        # GPU配置
        if self.use_gpu:
            self.params.update({
                'tree_method': 'gpu_hist',
                'gpu_id': 0
            })
            if self.logger:
                self.logger.info("XGBoost using GPU acceleration")
        else:
            self.params['tree_method'] = 'hist'
        
        self.n_estimators = n_estimators
        self.model = None
        self.feature_names = None
    
    def _prepare_data(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray):
        """准备XGBoost训练数据"""
        unique_groups = np.unique(groups)
        group_sizes = []
        
        for group_id in unique_groups:
            group_size = np.sum(groups == group_id)
            group_sizes.append(group_size)
        
        dtrain = xgb.DMatrix(X, label=y)
        dtrain.set_group(group_sizes)
        
        return dtrain, group_sizes
    
    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray):
        """训练XGBoost排序模型"""
        try:
            self.logger.info(f"Training XGBoost with {len(X)} samples, GPU: {self.use_gpu}")
            
            dtrain, group_sizes = self._prepare_data(X, y, groups)
            
            self.model = xgb.train(
                self.params,
                dtrain,
                num_boost_round=self.n_estimators,
                verbose_eval=False
            )
            
            self.is_fitted = True
            self.logger.info("XGBoost training completed")
            
        except Exception as e:
            self.logger.error(f"XGBoost training failed: {e}")
            raise e
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测排序分数"""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model not trained")
        
        try:
            dtest = xgb.DMatrix(X)
            predictions = self.model.predict(dtest)
            return predictions
            
        except Exception as e:
            self.logger.error(f"XGBoost prediction failed: {e}")
            raise e
    
    def get_feature_importance(self) -> dict:
        """获取特征重要性"""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model not trained")
        return self.model.get_score(importance_type='weight')
    
    def save_model(self, filepath: str):
        """保存模型"""
        if not self.is_fitted:
            raise ValueError("Model not trained")
        
        model_data = {
            'model': self.model,
            'params': self.params,
            'n_estimators': self.n_estimators,
            'is_fitted': self.is_fitted,
            'use_gpu': self.use_gpu
        }
        
        joblib.dump(model_data, filepath)
    
    @classmethod
    def load_model(cls, filepath: str):
        """加载模型"""
        model_data = joblib.load(filepath)
        
        instance = cls(
            n_estimators=model_data['n_estimators'],
            use_gpu=model_data.get('use_gpu', False)
        )
        
        instance.model = model_data['model']
        instance.params = model_data['params']
        instance.is_fitted = model_data['is_fitted']
        
        return instance


class LightGBMRanker:
    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.1,
                 use_gpu=True, random_state=42, logger=None):

        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.logger = logger or logging.getLogger(__name__)
        self.is_fitted = False
        
        # LightGBM参数
        self.params = {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'boosting_type': 'gbdt',
            'num_leaves': 2 ** max_depth - 1,
            'learning_rate': learning_rate,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'seed': random_state,
            'verbosity': -1,
            'force_col_wise': True
        }
        
        # GPU配置
        if self.use_gpu:
            self.params.update({
                'device': 'gpu',
                'gpu_platform_id': 0,
                'gpu_device_id': 0
            })
            if self.logger:
                self.logger.info("LightGBM using GPU acceleration")
        
        self.n_estimators = n_estimators
        self.model = None
    
    def _prepare_data(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray):
        """准备LightGBM训练数据"""
        unique_groups = np.unique(groups)
        group_sizes = []
        
        for group_id in unique_groups:
            group_size = np.sum(groups == group_id)
            group_sizes.append(group_size)
        
        train_data = lgb.Dataset(X, label=y, group=group_sizes)
        
        return train_data, group_sizes
    
    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray):
        """训练LightGBM排序模型"""
        try:
            self.logger.info(f"Training LightGBM with {len(X)} samples, GPU: {self.use_gpu}")
            
            train_data, group_sizes = self._prepare_data(X, y, groups)
            
            self.model = lgb.train(
                self.params,
                train_data,
                num_boost_round=self.n_estimators,
                callbacks=[lgb.log_evaluation(0)]
            )
            
            self.is_fitted = True
            self.logger.info("LightGBM training completed")
            
        except Exception as e:
            self.logger.error(f"LightGBM training failed: {e}")
            raise e
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测排序分数"""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model not trained")
        
        try:
            predictions = self.model.predict(X)
            return predictions
            
        except Exception as e:
            self.logger.error(f"LightGBM prediction failed: {e}")
            raise e
    
    def get_feature_importance(self) -> dict:
        """获取特征重要性"""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model not trained")
        importances = self.model.feature_importance(importance_type='gain')
        return {f'f{i}': imp for i, imp in enumerate(importances)}
    
    def save_model(self, filepath: str):
        """保存模型"""
        if not self.is_fitted:
            raise ValueError("Model not trained")
        
        model_data = {
            'model': self.model,
            'params': self.params,
            'n_estimators': self.n_estimators,
            'is_fitted': self.is_fitted,
            'use_gpu': self.use_gpu
        }
        
        joblib.dump(model_data, filepath)
    
    @classmethod
    def load_model(cls, filepath: str):
        """加载模型"""
        model_data = joblib.load(filepath)
        
        instance = cls(
            n_estimators=model_data['n_estimators'],
            use_gpu=model_data.get('use_gpu', False)
        )
        
        instance.model = model_data['model']
        instance.params = model_data['params']
        instance.is_fitted = model_data['is_fitted']
        
        return instance


class RankNet:
    def __init__(self, input_dim: int = None, hidden_dims: List[int] = [256, 128],
                 dropout_rate: float = 0.2, learning_rate: float = 0.001,
                 use_gpu: bool = True, random_state: int = 42, logger=None):
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        self.logger = logger or logging.getLogger(__name__)
        self.is_fitted = False
        
        self.model = None
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.optimizer = None
        self.scaler_mean = None
        self.scaler_std = None
    
    def _build_model(self, input_dim):
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in self.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        return nn.Sequential(*layers).to(self.device)
    
    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray, epochs: int = 100):
        n_features = X.shape[1]
        self.model = self._build_model(n_features)
        
        self.scaler_mean = torch.FloatTensor(X.mean(axis=0)).to(self.device)
        self.scaler_std = torch.FloatTensor(X.std(axis=0) + 1e-8).to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        dataset = FastRankingDataset(X, y, groups, max_group_size=35)
        dataloader = DataLoader(dataset, batch_size=24, shuffle=True, num_workers=0)
        
        self.model.train()
        max_epochs = min(epochs, 40)
        
        for epoch in range(max_epochs):
            for batch_groups in dataloader:
                batch_loss = 0
                valid_batches = 0
                
                for group_X, group_y in batch_groups:
                    if len(group_X) < 2:
                        continue
                    
                    group_X_scaled = (group_X - self.scaler_mean) / self.scaler_std
                    scores = self.model(group_X_scaled).squeeze()
                    
                    # 简化的pairwise loss
                    indices = torch.arange(len(group_X), device=self.device)
                    if len(indices) > 20:
                        indices = indices[torch.randperm(len(indices))[:20]]
                    
                    pairs = torch.combinations(indices, 2)
                    if len(pairs) > 25:
                        pairs = pairs[torch.randperm(len(pairs))[:25]]
                    
                    if len(pairs) == 0:
                        continue
                    
                    idx1, idx2 = pairs[:, 0], pairs[:, 1]
                    scores1, scores2 = scores[idx1], scores[idx2]
                    labels1, labels2 = group_y[idx1], group_y[idx2]
                    
                    label_diff = labels1 - labels2
                    score_diff = scores1 - scores2
                    prob = torch.sigmoid(score_diff)
                    target = (label_diff > 0).float()
                    
                    mask = (label_diff != 0)
                    if mask.sum() > 0:
                        loss = F.binary_cross_entropy(prob[mask], target[mask])
                        batch_loss += loss
                        valid_batches += 1
                
                if valid_batches > 0:
                    batch_loss = batch_loss / valid_batches
                    self.optimizer.zero_grad()
                    batch_loss.backward()
                    self.optimizer.step()
        
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        batch_size = 10000
        all_scores = []
        
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch_X = X[i:i+batch_size]
                X_tensor = torch.FloatTensor(batch_X).to(self.device)
                X_scaled = (X_tensor - self.scaler_mean) / self.scaler_std
                scores = self.model(X_scaled).squeeze().cpu().numpy()
                all_scores.append(scores)
        
        return np.concatenate(all_scores)
    
    def save_model(self, filepath: str):
        torch.save({
            'model_state': self.model.state_dict(),
            'scaler_mean': self.scaler_mean,
            'scaler_std': self.scaler_std,
            'hidden_dims': self.hidden_dims,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'is_fitted': self.is_fitted
        }, filepath)
    
    @classmethod
    def load_model(cls, filepath: str):
        checkpoint = torch.load(filepath, map_location='cpu')
        instance = cls(
            hidden_dims=checkpoint['hidden_dims'],
            dropout_rate=checkpoint['dropout_rate'],
            learning_rate=checkpoint['learning_rate']
        )
        n_features = checkpoint['scaler_mean'].shape[0]
        instance.model = instance._build_model(n_features)
        instance.model.load_state_dict(checkpoint['model_state'])
        instance.scaler_mean = checkpoint['scaler_mean']
        instance.scaler_std = checkpoint['scaler_std']
        instance.is_fitted = checkpoint['is_fitted']
        return instance


class GraphRanker:
    def __init__(self, input_dim: int = None, hidden_dims: List[int] = [128, 64], 
                 num_gnn_layers: int = 2, dropout_rate: float = 0.2,
                 learning_rate: float = 0.001, use_gpu: bool = True, 
                 random_state: int = 42, logger=None):
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        self.logger = logger or logging.getLogger(__name__)
        self.is_fitted = False
        
        self.model = None
        self.hidden_dims = hidden_dims
        self.num_gnn_layers = num_gnn_layers
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.optimizer = None
        self.scaler_mean = None
        self.scaler_std = None
    
    def _build_model(self, input_dim):
        class SimpleGNN(nn.Module):
            def __init__(self, input_dim, hidden_dims, num_layers, dropout_rate):
                super().__init__()
                self.layers = nn.ModuleList()
                prev_dim = input_dim
                
                for _ in range(num_layers):
                    self.layers.append(nn.Linear(prev_dim, hidden_dims[0]))
                    prev_dim = hidden_dims[0]
                
                self.attention = nn.MultiheadAttention(prev_dim, 4, dropout=dropout_rate, batch_first=True)
                self.final = nn.Sequential(
                    nn.Linear(prev_dim, hidden_dims[-1]),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(hidden_dims[-1], 1)
                )
            
            def forward(self, x):
                for layer in self.layers:
                    x = F.relu(layer(x))
                
                x = x.unsqueeze(0)
                attn_out, _ = self.attention(x, x, x)
                x = attn_out.squeeze(0)
                
                return self.final(x).squeeze(-1)
        
        return SimpleGNN(input_dim, self.hidden_dims, self.num_gnn_layers, self.dropout_rate).to(self.device)
    
    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray):
        n_features = X.shape[1]
        self.model = self._build_model(n_features)
        
        self.scaler_mean = torch.FloatTensor(X.mean(axis=0)).to(self.device)
        self.scaler_std = torch.FloatTensor(X.std(axis=0) + 1e-8).to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        dataset = FastRankingDataset(X, y, groups, max_group_size=30)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
        
        self.model.train()
        for epoch in range(30):
            for batch_groups in dataloader:
                batch_loss = 0
                valid_batches = 0
                
                for group_X, group_y in batch_groups:
                    if len(group_X) < 2:
                        continue
                    
                    group_X_scaled = (group_X - self.scaler_mean) / self.scaler_std
                    scores = self.model(group_X_scaled)
                    
                    if torch.sum(group_y) == 1:
                        target_idx = torch.argmax(group_y)
                        loss = F.cross_entropy(scores.unsqueeze(0), target_idx.unsqueeze(0))
                    else:
                        pos_mask = group_y > 0
                        neg_mask = group_y == 0
                        if pos_mask.sum() > 0 and neg_mask.sum() > 0:
                            pos_scores = scores[pos_mask].mean()
                            neg_scores = scores[neg_mask].mean()
                            loss = F.relu(1.0 - (pos_scores - neg_scores))
                        else:
                            continue
                    
                    batch_loss += loss
                    valid_batches += 1
                
                if valid_batches > 0:
                    batch_loss = batch_loss / valid_batches
                    self.optimizer.zero_grad()
                    batch_loss.backward()
                    self.optimizer.step()
        
        self.is_fitted = True
    
    def predict(self, X: np.ndarray, groups: np.ndarray = None) -> np.ndarray:
        self.model.eval()
        batch_size = 10000
        all_scores = []
        
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch_X = X[i:i+batch_size]
                X_tensor = torch.FloatTensor(batch_X).to(self.device)
                X_scaled = (X_tensor - self.scaler_mean) / self.scaler_std
                scores = self.model(X_scaled).cpu().numpy()
                all_scores.append(scores)
        
        return np.concatenate(all_scores)
    
    def save_model(self, filepath: str):
        torch.save({
            'model_state': self.model.state_dict(),
            'scaler_mean': self.scaler_mean,
            'scaler_std': self.scaler_std,
            'hidden_dims': self.hidden_dims,
            'num_gnn_layers': self.num_gnn_layers,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'is_fitted': self.is_fitted
        }, filepath)
    
    @classmethod
    def load_model(cls, filepath: str):
        checkpoint = torch.load(filepath, map_location='cpu')
        instance = cls(
            hidden_dims=checkpoint['hidden_dims'],
            num_gnn_layers=checkpoint['num_gnn_layers'],
            dropout_rate=checkpoint['dropout_rate'],
            learning_rate=checkpoint['learning_rate']
        )
        n_features = checkpoint['scaler_mean'].shape[0]
        instance.model = instance._build_model(n_features)
        instance.model.load_state_dict(checkpoint['model_state'])
        instance.scaler_mean = checkpoint['scaler_mean']
        instance.scaler_std = checkpoint['scaler_std']
        instance.is_fitted = checkpoint['is_fitted']
        return instance


class TransformerRanker:
    def __init__(self, input_dim: int = None, d_model: int = 128, nhead: int = 4,
                 num_layers: int = 2, dim_feedforward: int = 256,
                 dropout_rate: float = 0.1, learning_rate: float = 0.001,
                 use_gpu: bool = True, random_state: int = 42, logger=None):
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        self.logger = logger or logging.getLogger(__name__)
        self.is_fitted = False
        
        self.model = None
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.optimizer = None
        self.scaler_mean = None
        self.scaler_std = None
    
    def _build_model(self, input_dim):
        class TransformerRankingModel(nn.Module):
            def __init__(self, input_dim, d_model, nhead, num_layers, 
                        dim_feedforward, dropout_rate):
                super().__init__()
                
                self.input_projection = nn.Linear(input_dim, d_model)
                
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout_rate,
                    batch_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
                
                self.output_layer = nn.Sequential(
                    nn.Linear(d_model, d_model // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(d_model // 2, 1)
                )
            
            def forward(self, x):
                if x.dim() == 2:
                    x = x.unsqueeze(0)
                
                x = self.input_projection(x)
                x = self.transformer(x)
                x = self.output_layer(x)
                
                return x.squeeze()
        
        return TransformerRankingModel(input_dim, self.d_model, self.nhead, 
                                     self.num_layers, self.dim_feedforward, 
                                     self.dropout_rate).to(self.device)
    
    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray):
        n_features = X.shape[1]
        self.model = self._build_model(n_features)
        
        self.scaler_mean = torch.FloatTensor(X.mean(axis=0)).to(self.device)
        self.scaler_std = torch.FloatTensor(X.std(axis=0) + 1e-8).to(self.device)
        
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        
        dataset = FastRankingDataset(X, y, groups, max_group_size=20)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
        
        self.model.train()
        for epoch in range(30):
            for batch_groups in dataloader:
                batch_loss = 0
                valid_batches = 0
                
                for group_X, group_y in batch_groups:
                    if len(group_X) < 2:
                        continue
                    
                    group_X_scaled = (group_X - self.scaler_mean) / self.scaler_std
                    scores = self.model(group_X_scaled)
                    if scores.dim() == 0:
                        scores = scores.unsqueeze(0)
                    
                    if torch.sum(group_y) == 1:
                        target_idx = torch.argmax(group_y)
                        loss = F.cross_entropy(scores.unsqueeze(0), target_idx.unsqueeze(0))
                    else:
                        pos_mask = group_y > 0
                        neg_mask = group_y == 0
                        if pos_mask.sum() > 0 and neg_mask.sum() > 0:
                            pos_scores = scores[pos_mask].mean()
                            neg_scores = scores[neg_mask].mean()
                            loss = F.relu(1.0 - (pos_scores - neg_scores))
                        else:
                            continue
                    
                    batch_loss += loss
                    valid_batches += 1
                
                if valid_batches > 0:
                    batch_loss = batch_loss / valid_batches
                    self.optimizer.zero_grad()
                    batch_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
        
        self.is_fitted = True
    
    def predict(self, X: np.ndarray, groups: np.ndarray = None) -> np.ndarray:
        self.model.eval()
        all_scores = np.zeros(len(X))
        
        with torch.no_grad():
            if groups is not None:
                unique_groups = np.unique(groups)
                for group_id in unique_groups:
                    group_mask = groups == group_id
                    group_X = X[group_mask]
                    
                    if len(group_X) == 0:
                        continue
                    
                    X_tensor = torch.FloatTensor(group_X).to(self.device)
                    X_scaled = (X_tensor - self.scaler_mean) / self.scaler_std
                    scores = self.model(X_scaled)
                    
                    if scores.dim() == 0:
                        scores = scores.unsqueeze(0)
                    
                    all_scores[group_mask] = scores.cpu().numpy()
            else:
                batch_size = 5000
                for i in range(0, len(X), batch_size):
                    batch_X = X[i:i+batch_size]
                    X_tensor = torch.FloatTensor(batch_X).to(self.device)
                    X_scaled = (X_tensor - self.scaler_mean) / self.scaler_std
                    scores = self.model(X_scaled)
                    all_scores[i:i+batch_size] = scores.cpu().numpy()
        
        return all_scores
    
    def save_model(self, filepath: str):
        torch.save({
            'model_state': self.model.state_dict(),
            'scaler_mean': self.scaler_mean,
            'scaler_std': self.scaler_std,
            'd_model': self.d_model,
            'nhead': self.nhead,
            'num_layers': self.num_layers,
            'dim_feedforward': self.dim_feedforward,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'is_fitted': self.is_fitted
        }, filepath)
    
    @classmethod
    def load_model(cls, filepath: str):
        checkpoint = torch.load(filepath, map_location='cpu')
        instance = cls(
            d_model=checkpoint['d_model'],
            nhead=checkpoint['nhead'],
            num_layers=checkpoint['num_layers'],
            dim_feedforward=checkpoint['dim_feedforward'],
            dropout_rate=checkpoint['dropout_rate'],
            learning_rate=checkpoint['learning_rate']
        )
        n_features = checkpoint['scaler_mean'].shape[0]
        instance.model = instance._build_model(n_features)
        instance.model.load_state_dict(checkpoint['model_state'])
        instance.scaler_mean = checkpoint['scaler_mean']
        instance.scaler_std = checkpoint['scaler_std']
        instance.is_fitted = checkpoint['is_fitted']
        return instance


# CNNRanker 添加缺失的类
class CNNRanker:
    def __init__(self, input_dim: int = None, sequence_length: int = 10,
                 conv_channels: List[int] = [64, 128], kernel_sizes: List[int] = [3, 5],
                 hidden_dims: List[int] = [256, 128], dropout_rate: float = 0.2,
                 learning_rate: float = 0.001, use_gpu: bool = True, 
                 random_state: int = 42, logger=None):
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        self.logger = logger or logging.getLogger(__name__)
        self.is_fitted = False
        
        self.model = None
        self.sequence_length = sequence_length
        self.conv_channels = conv_channels
        self.kernel_sizes = kernel_sizes
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.optimizer = None
        self.scaler_mean = None
        self.scaler_std = None
    
    def _build_model(self, input_dim):
        class CNNRankingModel(nn.Module):
            def __init__(self, input_dim, sequence_length, conv_channels, kernel_sizes, 
                        hidden_dims, dropout_rate):
                super().__init__()
                
                # 重塑输入为序列
                self.input_dim = input_dim
                self.sequence_length = sequence_length
                self.features_per_step = input_dim // sequence_length
                
                # 卷积层
                self.conv_layers = nn.ModuleList()
                in_channels = 1
                for out_channels in conv_channels:
                    self.conv_layers.append(
                        nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
                    )
                    in_channels = out_channels
                
                # 全连接层
                conv_output_size = conv_channels[-1] * self.features_per_step
                layers = []
                prev_dim = conv_output_size
                
                for hidden_dim in hidden_dims:
                    layers.extend([
                        nn.Linear(prev_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(dropout_rate)
                    ])
                    prev_dim = hidden_dim
                
                layers.append(nn.Linear(prev_dim, 1))
                self.fc_layers = nn.Sequential(*layers)
            
            def forward(self, x):
                batch_size = x.size(0)
                
                # 重塑为序列格式
                x = x.view(batch_size, 1, -1)
                
                # 卷积操作
                for conv in self.conv_layers:
                    x = F.relu(conv(x))
                
                # 展平
                x = x.view(batch_size, -1)
                
                # 全连接层
                x = self.fc_layers(x)
                
                return x.squeeze()
        
        return CNNRankingModel(input_dim, self.sequence_length, self.conv_channels, 
                              self.kernel_sizes, self.hidden_dims, self.dropout_rate).to(self.device)
    
    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray):
        n_features = X.shape[1]
        self.model = self._build_model(n_features)
        
        self.scaler_mean = torch.FloatTensor(X.mean(axis=0)).to(self.device)
        self.scaler_std = torch.FloatTensor(X.std(axis=0) + 1e-8).to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        dataset = FastRankingDataset(X, y, groups, max_group_size=30)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
        
        self.model.train()
        for epoch in range(30):
            for batch_groups in dataloader:
                batch_loss = 0
                valid_batches = 0
                
                for group_X, group_y in batch_groups:
                    if len(group_X) < 2:
                        continue
                    
                    group_X_scaled = (group_X - self.scaler_mean) / self.scaler_std
                    scores = self.model(group_X_scaled)
                    
                    if torch.sum(group_y) == 1:
                        target_idx = torch.argmax(group_y)
                        loss = F.cross_entropy(scores.unsqueeze(0), target_idx.unsqueeze(0))
                    else:
                        pos_mask = group_y > 0
                        neg_mask = group_y == 0
                        if pos_mask.sum() > 0 and neg_mask.sum() > 0:
                            pos_scores = scores[pos_mask].mean()
                            neg_scores = scores[neg_mask].mean()
                            loss = F.relu(1.0 - (pos_scores - neg_scores))
                        else:
                            continue
                    
                    batch_loss += loss
                    valid_batches += 1
                
                if valid_batches > 0:
                    batch_loss = batch_loss / valid_batches
                    self.optimizer.zero_grad()
                    batch_loss.backward()
                    self.optimizer.step()
        
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        batch_size = 10000
        all_scores = []
        
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch_X = X[i:i+batch_size]
                X_tensor = torch.FloatTensor(batch_X).to(self.device)
                X_scaled = (X_tensor - self.scaler_mean) / self.scaler_std
                scores = self.model(X_scaled).cpu().numpy()
                all_scores.append(scores)
        
        return np.concatenate(all_scores)
    
    def save_model(self, filepath: str):
        torch.save({
            'model_state': self.model.state_dict(),
            'scaler_mean': self.scaler_mean,
            'scaler_std': self.scaler_std,
            'sequence_length': self.sequence_length,
            'conv_channels': self.conv_channels,
            'kernel_sizes': self.kernel_sizes,
            'hidden_dims': self.hidden_dims,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'is_fitted': self.is_fitted
        }, filepath)
    
    @classmethod
    def load_model(cls, filepath: str):
        checkpoint = torch.load(filepath, map_location='cpu')
        instance = cls(
            sequence_length=checkpoint['sequence_length'],
            conv_channels=checkpoint['conv_channels'],
            kernel_sizes=checkpoint['kernel_sizes'],
            hidden_dims=checkpoint['hidden_dims'],
            dropout_rate=checkpoint['dropout_rate'],
            learning_rate=checkpoint['learning_rate']
        )
        n_features = checkpoint['scaler_mean'].shape[0]
        instance.model = instance._build_model(n_features)
        instance.model.load_state_dict(checkpoint['model_state'])
        instance.scaler_mean = checkpoint['scaler_mean']
        instance.scaler_std = checkpoint['scaler_std']
        instance.is_fitted = checkpoint['is_fitted']
        return instance