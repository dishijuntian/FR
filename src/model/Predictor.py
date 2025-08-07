"""
航班排名预测器 - 修复版本
适配现有的模型文件结构: XGBRanker_segment_X.pkl
"""

import os
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import joblib

import numpy as np
import pandas as pd

from .Manager import FlightRankingModelsManager


class FlightRankingPredictor:
    """航班排名预测器 - 修复版本，适配现有模型文件结构"""
    
    def __init__(self, config: Dict, logger=None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # 路径配置
        self.data_path = Path(config['paths']['model_input_dir'])
        self.model_save_path = Path(config['paths']['model_save_dir'])
        self.output_path = Path(config['paths']['output_dir'])
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # 预测配置
        prediction_config = config.get('prediction', {})
        self.segments = prediction_config.get('segments', [0, 1, 2])
        self.model_names = prediction_config.get('model_names', ['XGBRanker', 'LGBMRanker'])
        self.use_gpu = prediction_config.get('use_gpu', True)
        self.use_full_data = config.get('training', {}).get('use_full_data', False)
        
        self.logger.info("预测器初始化完成")
    
    def predict_segment(self, segment_id: int, save_individual: bool = False) -> pd.DataFrame:
        """预测单个数据段 - 适配现有文件结构"""
        self.logger.info(f"开始预测 segment_{segment_id}")
        start_time = time.time()
        
        # 加载测试数据
        test_file = self.data_path / "test" / f"test_segment_{segment_id}.parquet"
        if not test_file.exists():
            raise FileNotFoundError(f"测试文件不存在: {test_file}")
        
        df = pd.read_parquet(test_file)
        self.logger.info(f"加载测试数据: {df.shape}")
        
        # 加载模型 - 适配现有文件结构
        models_manager, validation_scores = self._load_segment_models_legacy(segment_id)
        
        # 数据预处理
        X, _, groups, _, _ = models_manager.prepare_data(df, target_col='selected')
        
        # 使用加权预测（基于验证得分）
        predictions = models_manager.predict_model(X, validation_scores, self.model_names)
        
        # 生成排名
        rankings = self._generate_rankings(predictions, groups)
        
        # 验证排名
        if not self._validate_rankings(rankings, groups):
            raise ValueError(f"segment_{segment_id} 排名验证失败")
        
        # 生成结果
        results = df[['Id', 'ranker_id']].copy()
        results['selected'] = rankings
        
        if save_individual:
            output_file = self.output_path / f"predictions_segment_{segment_id}.csv"
            results.to_csv(output_file, index=False)
            self.logger.info(f"✓ 保存到: {output_file}")
        
        prediction_time = time.time() - start_time
        self.logger.info(f"✓ segment_{segment_id} 预测完成 (时间: {prediction_time:.1f}s)")
        
        return results
    
    def _load_segment_models_legacy(self, segment_id: int) -> Tuple[FlightRankingModelsManager, Dict]:
        """加载模型 - 适配现有的文件命名格式: XGBRanker_segment_X.pkl"""
        
        # 检查现有的模型文件格式
        available_models = {}
        validation_scores = {}
        
        for model_name in self.model_names:
            # 查找格式: ModelName_segment_X.pkl
            model_file = self.model_save_path / f"{model_name}_segment_{segment_id}.pkl"
            
            if model_file.exists():
                try:
                    # 直接加载模型文件
                    model = joblib.load(model_file)
                    available_models[model_name] = model
                    
                    # 设置默认验证得分（因为没有training_report.json）
                    validation_scores[model_name] = 0.8  # 默认得分
                    
                    self.logger.info(f"成功加载模型: {model_file}")
                    
                except Exception as e:
                    self.logger.warning(f"加载模型失败: {model_file}, 错误: {e}")
                    continue
            else:
                self.logger.warning(f"模型文件不存在: {model_file}")
        
        if not available_models:
            raise FileNotFoundError(f"segment_{segment_id} 没有找到任何可用的模型文件")
        
        # 创建模型管理器并设置加载的模型
        models_manager = FlightRankingModelsManager(use_gpu=self.use_gpu, logger=self.logger)
        models_manager.models = available_models
        
        # 设置模型为已训练状态
        for model_name, model in available_models.items():
            if hasattr(model, 'is_fitted'):
                model.is_fitted = True
        
        self.logger.info(f"成功加载 {len(available_models)} 个模型: {list(available_models.keys())}")
        
        return models_manager, validation_scores
    
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
        self.logger.info(f"✓ 最终结果: {output_file}, 总记录数: {len(final_submission)}")
        
        return final_submission
    
    def _generate_rankings(self, scores: np.ndarray, groups: np.ndarray) -> np.ndarray:
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
    
    def _validate_rankings(self, rankings: np.ndarray, groups: np.ndarray) -> bool:
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
    
    def _generate_prediction_report(self, results: pd.DataFrame, total_time: float):
        """生成预测报告"""
        report = {
            'prediction_summary': {
                'segments': self.segments,
                'total_samples': len(results),
                'total_time': total_time,
                'models_used': self.model_names,
                'use_full_data': self.use_full_data,
                'file_structure': 'legacy'  # 标记使用了传统文件结构
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