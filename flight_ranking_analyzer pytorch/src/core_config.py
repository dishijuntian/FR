"""
核心配置模块 - 重构版 v5.1
统一管理所有配置项，添加完善的数据处理流水线选择

作者: Flight Ranking Team
版本: 5.1 (改进版)
"""

import os
import torch
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum


class FeatureLevel(Enum):
    """特征工程级别枚举"""
    NONE = "none"           # 不进行特征工程
    BASIC = "basic"         # 基础特征工程
    ENHANCED = "enhanced"   # 增强特征工程
    ADVANCED = "advanced"   # 高级特征工程


class DataProcessMode(Enum):
    """数据处理模式枚举"""
    FULL_PROCESS = "full_process"           # 完整处理流程
    ENCODING_ONLY = "encoding_only"         # 仅数据编码
    FEATURE_ONLY = "feature_only"           # 仅特征工程
    LOAD_CACHED = "load_cached"             # 加载缓存数据
    RAW_DATA = "raw_data"                   # 使用原始数据
    COMPARE_MODES = "compare_modes"         # 比较不同处理模式


class FeatureSelectionMode(Enum):
    """特征选择模式枚举"""
    NONE = "none"                   # 不进行特征选择
    VARIANCE = "variance"           # 基于方差选择
    CORRELATION = "correlation"     # 基于相关性选择
    MUTUAL_INFO = "mutual_info"     # 基于互信息选择
    LOAD_CACHED = "load_cached"     # 加载已保存的特征列表


class ModelType(Enum):
    """模型类型枚举"""
    TRADITIONAL = "traditional"
    PYTORCH = "pytorch"


@dataclass
class PathConfig:
    """路径配置"""
    data_base: Path
    train_data: Path
    test_data: Path
    submission_file: Path
    output: Path
    models: Path
    processed_data: Path
    cache_data: Path  # 新增：缓存数据路径
    
    @classmethod
    def create_default(cls, base_path: str = "E:/GIT PROJECT/FR/data/aeroclub-recsys-2025") -> 'PathConfig':
        """创建默认路径配置"""
        base = Path(base_path)
        return cls(
            data_base=base,
            train_data=base / "segmented/train",
            test_data=base / "segmented/test", 
            submission_file=base / "sample_submission.parquet",
            output=base / "results",
            models=base / "models",
            processed_data=base / "processed",
            cache_data=base / "cache"  # 新增缓存目录
        )


@dataclass 
class TrainingConfig:
    """训练配置"""
    use_sampling: bool = True
    num_groups: int = 2000
    min_group_size: int = 20
    test_size: float = 0.2
    random_state: int = 42
    use_gpu: bool = True
    save_models: bool = True


@dataclass
class DataProcessConfig:
    """数据处理配置"""
    mode: DataProcessMode = DataProcessMode.FULL_PROCESS
    feature_level: FeatureLevel = FeatureLevel.ENHANCED
    selection_mode: FeatureSelectionMode = FeatureSelectionMode.VARIANCE
    max_features: int = 200
    cache_processed_data: bool = True      # 是否缓存处理后的数据
    auto_load_cache: bool = True           # 自动加载缓存
    compare_feature_levels: bool = False   # 是否比较不同特征工程级别
    exclude_features: List[str] = None
    
    def __post_init__(self):
        if self.exclude_features is None:
            self.exclude_features = ['Id', 'selected', 'ranker_id', 'profileId', 'companyID']


@dataclass
class FeatureConfig:
    """特征工程配置（保持向后兼容）"""
    level: FeatureLevel = FeatureLevel.ENHANCED
    enable_auto_discovery: bool = True
    enable_selection: bool = True
    max_features: int = 200
    auto_discovery_max: int = 50
    exclude_features: List[str] = None
    
    def __post_init__(self):
        if self.exclude_features is None:
            self.exclude_features = ['Id', 'selected', 'ranker_id', 'profileId', 'companyID']


@dataclass
class ModelConfig:
    """模型配置"""
    available_models: List[str] = None
    default_models: List[str] = None
    pytorch_models: List[str] = None
    
    def __post_init__(self):
        if self.available_models is None:
            self.available_models = [
                'XGBRanker', 'LGBMRanker', 'LambdaMART', 'ListNet',
                'NeuralRanker', 'RankNet', 'TransformerRanker'
            ]
        if self.default_models is None:
            self.default_models = ['XGBRanker', 'NeuralRanker']
        if self.pytorch_models is None:
            self.pytorch_models = ['NeuralRanker', 'RankNet', 'TransformerRanker']


class ConfigManager:
    """统一配置管理器"""
    
    def __init__(self, base_path: Optional[str] = None):
        """初始化配置管理器"""
        self.paths = PathConfig.create_default(base_path) if base_path else PathConfig.create_default()
        self.training = TrainingConfig()
        self.data_process = DataProcessConfig()  # 新增数据处理配置
        self.features = FeatureConfig()          # 保持向后兼容
        self.models = ModelConfig()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 模型参数配置
        self._model_params = self._init_model_params()
        self._tuning_spaces = self._init_tuning_spaces()
    
    def _init_model_params(self) -> Dict[str, Dict[str, Any]]:
        """初始化模型参数配置"""
        return {
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
            'epochs': 10,
            'batch_size': 32
        },
        'RankNet': {
            'hidden_units': [128, 64, 32],
            'learning_rate': 0.001,
            'epochs': 15,
            'batch_size': 64,
            'dropout_rate': 0.3
        },
        'TransformerRanker': {
            'num_heads': 4,
            'num_layers': 2,
            'd_model': 64,
            'dff': 128,
            'learning_rate': 0.001,
            'epochs': 10,
            'batch_size': 64,
            'dropout_rate': 0.1,
            'max_seq_length': 16
        },
        'BM25Ranker': {}
        }
    
    def _init_tuning_spaces(self) -> Dict[str, Dict[str, Any]]:
        """初始化调参空间配置"""
        return {
        'XGBRanker': {
            'n_estimators': [30, 50, 80],      # 减少选择
            'learning_rate': [0.05, 0.1, 0.2],
            'max_depth': [3, 4, 5],            # 减少深度选择
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9]
        },
        'LGBMRanker': {
            'n_estimators': [30, 50, 80],
            'learning_rate': [0.05, 0.1, 0.2],
            'max_depth': [3, 4, 5],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9]
        },
        'NeuralRanker': {
            'hidden_units': [
                [64, 32], [128, 64]             # 只保留小的网络
            ],
            'learning_rate': [0.001, 0.005],
            'epochs': [5, 8, 10],              # 减少训练轮数
            'batch_size': [32, 64]
        }
        }
    
    def get_model_params(self, model_name: str) -> Dict[str, Any]:
        """获取模型参数"""
        return self._model_params.get(model_name, {}).copy()
    
    def get_tuning_space(self, model_name: str) -> Dict[str, Any]:
        """获取调参空间"""
        return self._tuning_spaces.get(model_name, {}).copy()
    
    def is_pytorch_model(self, model_name: str) -> bool:
        """判断是否为PyTorch模型"""
        return model_name in self.models.pytorch_models
    
    def ensure_paths(self):
        """确保必要路径存在"""
        for path_attr in ['output', 'models', 'processed_data', 'cache_data']:
            path = getattr(self.paths, path_attr)
            path.mkdir(parents=True, exist_ok=True)
    
    def get_cache_info(self) -> Dict[str, Any]:
        """获取缓存信息"""
        cache_info = {
            'cache_dir': str(self.paths.cache_data),
            'cache_exists': self.paths.cache_data.exists(),
            'cached_files': []
        }
        
        if cache_info['cache_exists']:
            cache_files = list(self.paths.cache_data.glob("*.pkl"))
            cache_info['cached_files'] = [f.name for f in cache_files]
            cache_info['cache_count'] = len(cache_files)
        
        return cache_info
    
    def get_device_info(self) -> Dict[str, Any]:
        """获取设备信息"""
        info = {
            'device': str(self.device),
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available()
        }
        
        if torch.cuda.is_available():
            info.update({
                'gpu_name': torch.cuda.get_device_name(0),
                'gpu_memory': torch.cuda.get_device_properties(0).total_memory / 1024**3
            })
        
        return info
    
    def update_config(self, **kwargs):
        """更新配置"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                if isinstance(getattr(self, key), (TrainingConfig, DataProcessConfig, FeatureConfig, ModelConfig)):
                    # 更新嵌套配置
                    for sub_key, sub_value in value.items():
                        setattr(getattr(self, key), sub_key, sub_value)
                else:
                    setattr(self, key, value)
    
    def get_pipeline_summary(self) -> str:
        """获取流水线配置摘要"""
        summary_lines = []
        summary_lines.append("当前流水线配置:")
        summary_lines.append(f"  数据处理模式: {self.data_process.mode.value}")
        summary_lines.append(f"  特征工程级别: {self.data_process.feature_level.value}")
        summary_lines.append(f"  特征选择模式: {self.data_process.selection_mode.value}")
        
        if self.data_process.selection_mode != FeatureSelectionMode.NONE:
            summary_lines.append(f"  最大特征数: {self.data_process.max_features}")
        
        summary_lines.append(f"  缓存处理数据: {self.data_process.cache_processed_data}")
        summary_lines.append(f"  自动加载缓存: {self.data_process.auto_load_cache}")
        
        return "\n".join(summary_lines)


# 全局配置实例
config = ConfigManager()


# 便捷函数
def create_data_process_config(mode: str = "full_process", 
                              feature_level: str = "enhanced",
                              selection_mode: str = "variance",
                              max_features: int = 200,
                              cache_data: bool = True) -> DataProcessConfig:
    """创建数据处理配置的便捷函数"""
    return DataProcessConfig(
        mode=DataProcessMode(mode),
        feature_level=FeatureLevel(feature_level),
        selection_mode=FeatureSelectionMode(selection_mode),
        max_features=max_features,
        cache_processed_data=cache_data
    )


def get_available_modes() -> Dict[str, List[str]]:
    """获取所有可用的模式选项"""
    return {
        'data_process_modes': [mode.value for mode in DataProcessMode],
        'feature_levels': [level.value for level in FeatureLevel],
        'selection_modes': [mode.value for mode in FeatureSelectionMode]
    }