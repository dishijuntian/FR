"""
排序模型定义文件

该模块包含所有排序模型的定义和实现
- 基础排序模型接口
- XGBoost、LightGBM排序模型
- LambdaMART、ListNet模型
- 神经网络排序模型
- BM25排序模型

作者: Flight Ranking Team
版本: 2.1
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from xgboost import XGBRanker
from lightgbm import LGBMRanker
from rank_bm25 import BM25Okapi
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any
import warnings

warnings.filterwarnings('ignore')


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


class NeuralRanker(BaseRanker):
    """神经网络排序模型"""
    
    def __init__(self, input_dim, **params):
        default_params = {
            'hidden_units': [256, 128, 64],
            'learning_rate': 0.001,
            'epochs': 10,
            'batch_size': 32
        }
        default_params.update(params)
        
        self.input_dim = input_dim
        self.params = default_params
        self.model = self._build_model()
        
    def _build_model(self):
        inputs = keras.Input(shape=(self.input_dim,))
        
        x = layers.Dense(self.params['hidden_units'][0], activation='relu')(inputs)
        x = layers.Dropout(0.2)(x)
        
        for units in self.params['hidden_units'][1:]:
            x = layers.Dense(units, activation='relu')(x)
            x = layers.Dropout(0.2)(x)
        
        outputs = layers.Dense(1)(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.params['learning_rate']),
            loss='mse'
        )
        return model
    
    def fit(self, X, y, group, **kwargs):
        epochs = kwargs.get('epochs', self.params['epochs'])
        batch_size = kwargs.get('batch_size', self.params['batch_size'])
        
        # 将数据转换为TensorFlow Dataset
        dataset = tf.data.Dataset.from_tensor_slices((X.astype(np.float32), y.astype(np.float32)))
        dataset = dataset.batch(batch_size)
        
        self.model.fit(dataset, epochs=epochs, verbose=1)
    
    def predict(self, X):
        return self.model.predict(X.astype(np.float32)).flatten()
    
    def get_model_name(self):
        return "NeuralRanker"


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
            input_dim: 输入维度（仅NeuralRanker需要）
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
        elif model_name == 'BM25Ranker':
            return BM25Ranker(**params)
        else:
            raise ValueError(f"未知模型类型: {model_name}")
    
    @staticmethod
    def get_available_models() -> List[str]:
        """获取所有可用的模型名称"""
        return ['XGBRanker', 'LGBMRanker', 'LambdaMART', 'ListNet', 'NeuralRanker', 'BM25Ranker']