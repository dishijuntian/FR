"""
航班排序分析器包

该包提供了完整的航班排序分析功能，包括：
- 多种排序模型训练和评估
- 自动超参数调优
- 特征重要性分析
- SHAP可解释性分析
- 预测结果合并

作者: Flight Ranking Team
版本: 2.1
"""

from .config import Config
from .models import ModelFactory, BaseRanker
from .data_processor import DataProcessor, PredictionMerger
from .auto_tuner import AutoTuner, create_auto_tuner
from .analyzer import FlightRankingAnalyzer

__version__ = "2.1"
__author__ = "Flight Ranking Team"

__all__ = [
    'Config',
    'ModelFactory',
    'BaseRanker',
    'DataProcessor',
    'PredictionMerger',
    'AutoTuner',
    'create_auto_tuner',
    'FlightRankingAnalyzer'
]