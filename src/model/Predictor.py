"""
航班排名预测器
"""

import os
import time
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import pickle

import numpy as np
import pandas as pd

from .Manager import FlightRankingModelsManager

warnings.filterwarnings('ignore')


class FlightRankingPredictor:
    """航班排名预测器"""
    
    def __init__(self, config: Dict, logger=None):
        """
        初始化预测器
        
        Args:
            config: 配置字典
            logger: 日志器
        """
        self.config = config
        self.logger = logger or self._setup_logger()
        
        # 路径配置
        self.data_path = Path(config['paths']['model_input_dir'])
        self.model_save_path = Path(config['paths']['model_save_dir'])
        self.output_path = Path(config['paths']['output_dir'])
        
        # 预测配置
        prediction_config = config['prediction']
        self.segments = prediction_config['segments']
        self.model_names = prediction_config['model_names']
        self.use_gpu = prediction_config['use_gpu']
        self.enable_business_rules = prediction_config['enable_business_rules']
        self.ensemble_weights = prediction_config.get('ensemble_weights', {})
        
        # 确保输出目录存在
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("预测器初始化完成")
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s | %(levelname)8s | %(name)s | %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def load_segment_models(self, segment_id: int) -> Dict:
        """加载数据段的模型"""
        segment_dir = self.model_save_path / f"segment_{segment_id}"
        
        if not segment_dir.exists():
            raise FileNotFoundError(f"段模型目录不存在: {segment_dir}")
        
        models_manager = FlightRankingModelsManager(use_gpu=self.use_gpu, logger=self.logger)
        models_manager.load_models(str(segment_dir), self.model_names)
        
        if not models_manager.models:
            raise ValueError(f"没有成功加载 segment_{segment_id} 的模型")
        
        self.logger.info(f"成功加载 {len(models_manager.models)} 个模型")
        return models_manager
    
    def load_segment_features(self, segment_id: int) -> List[str]:
        """加载数据段的特征名称"""
        feature_path = self.model_save_path / f"segment_{segment_id}" / "features.pkl"
        
        if not feature_path.exists():
            raise FileNotFoundError(f"特征文件不存在: {feature_path}")
        
        with open(feature_path, 'rb') as f:
            features = pickle.load(f)
        
        self.logger.info(f"加载特征: {len(features)} 个")
        return features
    
    def generate_rankings(self, scores: np.ndarray, groups: np.ndarray) -> np.ndarray:
        """根据分数生成排名"""
        unique_groups = np.unique(groups)
        rankings = np.zeros(len(scores), dtype=int)
        
        for group_id in unique_groups:
            group_mask = groups == group_id
            group_scores = scores[group_mask]
            group_indices = np.where(group_mask)[0]
            
            # 按分数降序排序
            sort_indices = np.argsort(-group_scores)
            
            # 生成排名 (1-based)
            group_rankings = np.arange(1, len(group_scores) + 1)
            rankings[group_indices[sort_indices]] = group_rankings
        
        return rankings
    
    def validate_rankings(self, rankings: np.ndarray, groups: np.ndarray) -> bool:
        """验证排名的有效性"""
        unique_groups = np.unique(groups)
        
        for group_id in unique_groups:
            group_mask = groups == group_id
            group_rankings = rankings[group_mask]
            
            # 检查排名是否是1到N的连续整数
            expected_rankings = set(range(1, len(group_rankings) + 1))
            actual_rankings = set(group_rankings)
            
            if expected_rankings != actual_rankings:
                self.logger.error(f"组 {group_id} 排名无效")
                return False
        
        return True
    
    def apply_business_rules(self, scores: np.ndarray, features: pd.DataFrame) -> np.ndarray:
        """应用业务规则调整分数"""
        if not self.enable_business_rules:
            return scores
        
        adjusted_scores = scores.copy()
        
        # 示例业务规则
        if 'total_price' in features.columns:
            high_price_mask = features['total_price'] > features['total_price'].quantile(0.9)
            adjusted_scores[high_price_mask] *= 0.9
        
        if 'departure_hour' in features.columns:
            early_mask = features['departure_hour'] < 6
            late_mask = features['departure_hour'] > 22
            adjusted_scores[early_mask | late_mask] *= 0.95
        
        self.logger.info("业务规则调整完成")
        return adjusted_scores
    
    def predict_segment(self, segment_id: int, save_individual: bool = False) -> pd.DataFrame:
        """预测单个数据段"""
        self.logger.info(f"开始预测 segment_{segment_id}")
        start_time = time.time()
        
        # 加载测试数据
        test_file = self.data_path / "test" / f"test_segment_{segment_id}.parquet"
        if not test_file.exists():
            raise FileNotFoundError(f"测试文件不存在: {test_file}")
        
        df = pd.read_parquet(test_file)
        self.logger.info(f"加载测试数据: {df.shape}")
        
        # 加载模型
        models_manager = self.load_segment_models(segment_id)
        
        # 数据预处理
        X, _, groups, _, processed_df = models_manager.prepare_data(df, target_col='selected')
        
        # 获取权重
        weights = []
        available_models = []
        for model_name in self.model_names:
            if model_name in models_manager.models:
                available_models.append(model_name)
                weights.append(self.ensemble_weights.get(model_name, 1.0))
        
        # 集成预测
        predictions = models_manager.predict_ensemble(X, available_models, weights)
        
        # 应用业务规则
        predictions = self.apply_business_rules(predictions, processed_df)
        
        # 生成排名
        rankings = self.generate_rankings(predictions, groups)
        
        # 验证排名
        if not self.validate_rankings(rankings, groups):
            raise ValueError(f"segment_{segment_id} 排名验证失败")
        
        # 生成结果
        results = df[['Id', 'ranker_id']].copy()
        results['selected'] = rankings
        
        # 保存结果
        prediction_time = time.time() - start_time
        
        if save_individual:
            output_file = self.output_path / f"predictions_segment_{segment_id}.csv"
            results.to_csv(output_file, index=False)
            self.logger.info(f"✓ 保存到: {output_file}")
        
        self.logger.info(f"✓ segment_{segment_id} 预测完成 (时间: {prediction_time:.1f}s)")
        
        return results
    
    def predict_all_segments(self) -> pd.DataFrame:
        """预测所有数据段"""
        self.logger.info(f"开始预测所有段: {self.segments}")
        
        all_results = []
        total_start_time = time.time()
        
        for segment_id in self.segments:
            try:
                result = self.predict_segment(segment_id, save_individual=True)
                all_results.append(result)
            except Exception as e:
                self.logger.error(f"✗ segment_{segment_id} 预测失败: {e}")
                continue
        
        if not all_results:
            raise ValueError("所有段的预测都失败")
        
        # 合并结果
        final_submission = pd.concat(all_results, ignore_index=True)
        final_submission = final_submission.sort_values('Id').reset_index(drop=True)
        
        # 保存最终结果
        total_time = time.time() - total_start_time
        output_file = self.output_path / "final_submission.csv"
        final_submission.to_csv(output_file, index=False)
        
        # 生成预测报告
        self._generate_prediction_report(final_submission, total_time)
        
        self.logger.info(f"✓ 所有预测完成 (总时间: {total_time:.1f}s)")
        self.logger.info(f"✓ 最终结果: {output_file}")
        self.logger.info(f"✓ 总记录数: {len(final_submission)}")
        
        return final_submission
    
    def _generate_prediction_report(self, results: pd.DataFrame, total_time: float):
        """生成预测报告"""
        report = {
            'prediction_summary': {
                'segments': self.segments,
                'total_samples': len(results),
                'total_time': total_time,
                'avg_time_per_sample': total_time / len(results) * 1000,  # ms
                'models_used': self.model_names,
                'ensemble_weights': self.ensemble_weights
            },
            'data_statistics': {
                'total_rankers': results['ranker_id'].nunique(),
                'avg_options_per_ranker': len(results) / results['ranker_id'].nunique(),
                'min_options': results.groupby('ranker_id').size().min(),
                'max_options': results.groupby('ranker_id').size().max()
            }
        }
        
        # 保存报告
        report_path = self.output_path / "prediction_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"预测报告已保存: {report_path}")