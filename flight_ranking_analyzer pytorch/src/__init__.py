"""
航班排序分析器包 - PyTorch版本

该包提供了完整的航班排序分析功能，包括：
- 多种排序模型训练和评估（支持PyTorch深度学习模型）
- 自动超参数调优
- 特征重要性分析
- SHAP可解释性分析
- 预测结果合并
- GPU加速支持

主要改进：
- 将TensorFlow/Keras模型完全迁移到PyTorch
- 增强的GPU内存管理
- 更灵活的模型定义和训练控制
- 改进的模型保存和加载机制

作者: Flight Ranking Team
版本: 3.0 (PyTorch版本)
"""

from .config import Config
from .models import ModelFactory, BaseRanker
from .data_processor import DataProcessor, PredictionMerger
from .auto_tuner import AutoTuner, create_auto_tuner
from .analyzer import FlightRankingAnalyzer
from .predictor import FlightRankingPredictor

__version__ = "3.0"
__author__ = "Flight Ranking Team"
__framework__ = "PyTorch"

__all__ = [
    'Config',
    'ModelFactory',
    'BaseRanker',
    'DataProcessor',
    'PredictionMerger',
    'AutoTuner',
    'create_auto_tuner',
    'FlightRankingAnalyzer',
    'FlightRankingPredictor'
]

# 版本和框架信息
def get_version_info():
    """获取版本和框架信息"""
    import torch
    info = {
        'package_version': __version__,
        'framework': __framework__,
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        info['cuda_version'] = torch.version.cuda
        info['gpu_count'] = torch.cuda.device_count()
        info['gpu_name'] = torch.cuda.get_device_name(0)
    
    return info

def print_version_info():
    """打印版本信息"""
    info = get_version_info()
    print(f"航班排序分析器 v{info['package_version']} ({info['framework']}版本)")
    print(f"PyTorch: {info['pytorch_version']}")
    
    if info['cuda_available']:
        print(f"CUDA: {info['cuda_version']}")
        print(f"GPU: {info['gpu_name']} (设备数: {info['gpu_count']})")
    else:
        print("CUDA: 不可用 (使用CPU)")

# 快速检查环境
def check_environment():
    """检查运行环境"""
    try:
        import torch
        import pandas as pd
        import numpy as np
        import sklearn
        import xgboost
        import lightgbm
        
        print("✅ 环境检查通过")
        print(f"  PyTorch: {torch.__version__}")
        print(f"  Pandas: {pd.__version__}")
        print(f"  NumPy: {np.__version__}")
        print(f"  Scikit-learn: {sklearn.__version__}")
        print(f"  XGBoost: {xgboost.__version__}")
        print(f"  LightGBM: {lightgbm.__version__}")
        
        if torch.cuda.is_available():
            print(f"  CUDA: {torch.version.cuda} ✅")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("  CUDA: 不可用 ⚠️")
        
        return True
        
    except ImportError as e:
        print(f"❌ 环境检查失败: {e}")
        print("请运行: pip install -r requirements.txt")
        return False

# 导入时自动检查关键依赖
try:
    import torch
    if not torch.cuda.is_available():
        print("⚠️  CUDA不可用，将使用CPU进行计算")
except ImportError:
    print("❌ PyTorch未安装，请运行: pip install torch torchvision torchaudio")