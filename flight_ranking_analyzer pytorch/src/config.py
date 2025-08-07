"""
配置管理模块 - 重构版

专注于：
- 全局配置管理
- 路径配置
- 模型参数配置
- 环境设置

作者: Flight Ranking Team
版本: 4.0 (重构版)
"""

import os
import torch
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """模型配置数据类"""
    n_estimators: int = 100
    learning_rate: float = 0.1
    max_depth: int = 6
    dropout_rate: float = 0.2
    batch_size: int = 64
    epochs: int = 15


class Config:
    """全局配置管理器"""
    
    # 基础路径配置
    DATA_BASE_PATH = Path("E:/GIT PROJECT/FR/data/aeroclub-recsys-2025")
    TRAIN_DATA_PATH = DATA_BASE_PATH / "segmented/train"
    TEST_DATA_PATH = DATA_BASE_PATH / "segmented/test"
    SUBMISSION_FILE_PATH = DATA_BASE_PATH / "sample_submission.parquet"
    OUTPUT_PATH = DATA_BASE_PATH / "results"
    
    # 训练配置
    USE_SAMPLING = True
    DEFAULT_NUM_GROUPS = 2000
    DEFAULT_MIN_GROUP_SIZE = 20
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    
    # 模型列表
    AVAILABLE_MODELS = [
        'XGBRanker', 'LGBMRanker', 'LambdaMART', 'ListNet',
        'NeuralRanker', 'RankNet', 'TransformerRanker', 'BM25Ranker'
    ]
    
    PYTORCH_MODELS = ['NeuralRanker', 'RankNet', 'TransformerRanker']
    
    # 特征配置
    EXCLUDE_FEATURES = ['Id', 'selected', 'ranker_id', 'profileId', 'companyID']
    
    # 性能配置
    MAX_SHAP_SAMPLES = 2000
    HITRATE_K = 3
    
    # PyTorch配置
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @classmethod
    def get_model_config(cls, model_name: str) -> Dict[str, Any]:
        """获取模型配置"""
        configs = {
        'XGBRanker': {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'eval_metric': 'ndcg'
        },
        'LGBMRanker': {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'objective': 'lambdarank',
            'metric': 'ndcg'
        },
        'LambdaMART': {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        },
        'ListNet': {
            'n_estimators': 100,
            'learning_rate': 0.05,
            'max_depth': 7,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        },
        'NeuralRanker': {
            'hidden_units': [256, 128, 64],
            'learning_rate': 0.001,
            'epochs': 15,                    # 增加epochs
            'batch_size': 64,                # 优化batch_size
            'dropout_rate': 0.2,
            'weight_decay': 1e-5,            # L2正则化
            'early_stopping_patience': 2     # 早停耐心
        },
        'RankNet': {
            'hidden_units': [128, 64, 32],
            'learning_rate': 0.001,
            'epochs': 20,                    # 增加epochs
            'batch_size': 128,               # 更大的batch_size
            'dropout_rate': 0.3,
            'weight_decay': 1e-4,            # L2正则化
            'early_stopping_patience': 2     # 早停耐心
        },
        'TransformerRanker': {
            'num_heads': 4,                  # 保守的头数
            'num_layers': 2,                 # 保守的层数
            'd_model': 64,                   # 适中的模型维度
            'dff': 128,                      # 前馈网络维度
            'learning_rate': 0.001,
            'epochs': 15,                    # 适中的训练轮数
            'batch_size': 64,
            'dropout_rate': 0.1,
            'max_seq_length': 16,            # 序列长度
            'weight_decay': 1e-5,            # L2正则化
            'early_stopping_patience': 2,    # 早停耐心
            'warmup_steps': 1000             # 学习率预热
        },
        'BM25Ranker': {}
    }
        return configs.get(model_name, {})
    
    @classmethod
    def get_tuning_space(cls, model_name: str) -> Dict[str, Any]:
        """获取调参空间"""
        spaces = {
        'XGBRanker': {
            'n_estimators': [50, 100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 4, 5, 6, 7, 8],
            'subsample': [0.6, 0.7, 0.8, 0.9],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9]
        },
        'LGBMRanker': {
            'n_estimators': [50, 100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 4, 5, 6, 7, 8],
            'subsample': [0.6, 0.7, 0.8, 0.9],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9]
        },
        'NeuralRanker': {
            'hidden_units': [
                [128, 64], [256, 128], [256, 128, 64], 
                [512, 256, 128], [128, 64, 32]
            ],
            'learning_rate': [0.0001, 0.0005, 0.001, 0.005],
            'epochs': [10, 15, 20, 25],              # 适当的epochs范围
            'batch_size': [32, 64, 128, 256],
            'dropout_rate': [0.1, 0.2, 0.3, 0.4],
            'weight_decay': [1e-6, 1e-5, 1e-4, 1e-3]
        },
        'RankNet': {
            'hidden_units': [
                [64, 32], [128, 64], [128, 64, 32], 
                [256, 128, 64], [64, 32, 16]
            ],
            'learning_rate': [0.0001, 0.0005, 0.001, 0.002],
            'epochs': [15, 20, 25, 30],              # 适当的epochs范围
            'batch_size': [64, 128, 256, 512],
            'dropout_rate': [0.2, 0.3, 0.4, 0.5],
            'weight_decay': [1e-6, 1e-5, 1e-4, 1e-3]
        },
        'TransformerRanker': {
            'num_heads': [2, 4, 8],                  # 保守的选择
            'num_layers': [1, 2, 3],                 # 减少层数选择
            'd_model': [32, 64, 128],                # 合适的模型维度
            'dff': [64, 128, 256],                   # 前馈网络维度
            'learning_rate': [0.0005, 0.001, 0.002], # 稳定的学习率范围
            'epochs': [10, 15, 20],                  # 适当的训练轮数
            'batch_size': [32, 64, 128],             # 合适的批次大小
            'dropout_rate': [0.1, 0.2, 0.3],        # 适中的dropout
            'max_seq_length': [8, 16, 32],           # 不同的序列长度
            'weight_decay': [1e-6, 1e-5, 1e-4]      # L2正则化
        }
    }
        return spaces.get(model_name, {})
    
    @classmethod
    def ensure_paths(cls):
        """确保必要路径存在"""
        cls.OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_device_info(cls) -> Dict[str, Any]:
        """获取设备信息"""
        info = {
            'device': str(cls.DEVICE),
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available()
        }
        
        if torch.cuda.is_available():
            info.update({
                'gpu_name': torch.cuda.get_device_name(0),
                'gpu_memory': torch.cuda.get_device_properties(0).total_memory / 1024**3
            })
        
        return info
    
    @classmethod
    def is_pytorch_model(cls, model_name: str) -> bool:
        """判断是否为PyTorch模型"""
        return model_name in cls.PYTORCH_MODELS