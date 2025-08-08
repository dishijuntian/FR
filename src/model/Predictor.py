"""
航班排名预测器 - 修复版本
主要修复：
1. 修复模型加载路径问题，支持全量数据模式
2. 添加回退机制，兼容不同的保存模式
3. 改进错误处理和日志记录
"""

import os
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

import numpy as np
import pandas as pd

# 修复导入问题 - 使用正确的类名
from .Manager import FlightRankingModelsManager


class FlightRankingPredictor:
    """航班排名预测器 - 修复版本"""
    
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
        self.logger.info(f"全量数据模式: {self.use_full_data}")
    
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
        
        # 加载模型和验证得分
        models_manager, validation_scores = self._load_segment_models_with_scores(segment_id)
        
        # 数据预处理 - 使用优化的方法
        X, _, groups, _, _ = models_manager.prepare_data_simple(df, target_col='selected')
        
        # 使用加权预测（基于验证得分）
        predictions = models_manager.predict_simple(X, self.model_names)
        
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

    def _load_segment_models_with_scores(self, segment_id: int) -> Tuple[FlightRankingModelsManager, Dict]:
        """加载数据段的模型和验证得分 - 修复版"""
        # 修复：智能路径选择，支持全量数据模式和回退机制
        candidate_dirs = []
        
        if self.use_full_data:
            # 全量数据模式：优先从统一目录加载
            full_data_dir = self.model_save_path / "full_data"
            candidate_dirs.append(("full_data", full_data_dir))
            
            # 回退到分段目录
            segment_dir = self.model_save_path / f"segment_{segment_id}"
            candidate_dirs.append(("segment", segment_dir))
        else:
            # 分段模式：直接从分段目录加载
            segment_dir = self.model_save_path / f"segment_{segment_id}"
            candidate_dirs.append(("segment", segment_dir))
            
            # 回退到全量数据目录（如果存在）
            full_data_dir = self.model_save_path / "full_data"
            candidate_dirs.append(("full_data", full_data_dir))
        
        # 尝试从候选目录加载模型
        for mode, candidate_dir in candidate_dirs:
            if candidate_dir.exists():
                self.logger.info(f"尝试从 {mode} 目录加载模型: {candidate_dir}")
                
                # 检查是否有模型文件
                model_files = list(candidate_dir.glob("*.pkl"))
                model_files = [f for f in model_files if f.name != "features.pkl"]
                
                if model_files:
                    try:
                        # 创建模型管理器并加载模型
                        models_manager = FlightRankingModelsManager(use_gpu=self.use_gpu, logger=self.logger)
                        
                        # 使用优化的加载方法
                        success = models_manager.load_models_simple(str(candidate_dir), self.model_names)
                        
                        if success and models_manager.trained_models:
                            self.logger.info(f"✓ 成功从 {mode} 目录加载 {len(models_manager.trained_models)} 个模型")
                            
                            # 加载验证得分
                            validation_scores = self._load_validation_scores(candidate_dir)
                            
                            return models_manager, validation_scores
                        else:
                            self.logger.warning(f"从 {mode} 目录加载模型失败，尝试下一个目录")
                            continue
                            
                    except Exception as e:
                        self.logger.warning(f"从 {mode} 目录加载模型时出错: {e}")
                        continue
                else:
                    self.logger.warning(f"{mode} 目录中没有找到模型文件")
                    continue
            else:
                self.logger.info(f"{mode} 目录不存在: {candidate_dir}")
        
        # 所有尝试都失败
        tried_dirs = [str(d) for _, d in candidate_dirs]
        raise FileNotFoundError(f"无法从以下任何目录加载模型: {tried_dirs}")
    
    def _load_validation_scores(self, model_dir: Path) -> Dict:
        """加载验证得分"""
        validation_scores = {}
        report_file = model_dir / "training_report.json"
        
        if report_file.exists():
            try:
                with open(report_file, 'r') as f:
                    report = json.load(f)
                    validation_scores = report.get('validation_scores', {})
                self.logger.info(f"加载验证得分: {validation_scores}")
            except Exception as e:
                self.logger.warning(f"无法加载验证得分: {e}")
        else:
            self.logger.info("训练报告文件不存在，使用默认验证得分")
        
        return validation_scores
    
    def predict_all_segments(self) -> pd.DataFrame:
        """预测所有数据段"""
        self.logger.info(f"开始预测所有段: {self.segments}")
        
        all_results = []
        total_start_time = time.time()
        successful_segments = []
        failed_segments = []
        
        for segment_id in self.segments:
            try:
                result = self.predict_segment(segment_id, save_individual=True)
                all_results.append(result)
                successful_segments.append(segment_id)
                self.logger.info(f"✓ segment_{segment_id} 预测成功")
            except Exception as e:
                self.logger.error(f"✗ segment_{segment_id} 预测失败: {e}")
                failed_segments.append(segment_id)
                continue
        
        # 预测结果统计
        total_segments = len(self.segments)
        success_count = len(successful_segments)
        fail_count = len(failed_segments)
        
        self.logger.info(f"预测完成统计: 成功 {success_count}/{total_segments}, 失败 {fail_count}")
        
        if successful_segments:
            self.logger.info(f"成功的段: {successful_segments}")
        if failed_segments:
            self.logger.warning(f"失败的段: {failed_segments}")
        
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
        self._generate_prediction_report(final_submission, total_time, successful_segments, failed_segments)
        
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
                self.logger.error(f"组 {group_id} 排名无效: 期望 {expected_rankings}, 实际 {actual_rankings}")
                return False
        
        return True
    
    def _generate_prediction_report(self, results: pd.DataFrame, total_time: float, 
                                  successful_segments: List[int], failed_segments: List[int]):
        """生成预测报告"""
        report = {
            'prediction_summary': {
                'segments': self.segments,
                'successful_segments': successful_segments,
                'failed_segments': failed_segments,
                'success_rate': len(successful_segments) / len(self.segments),
                'total_samples': len(results),
                'total_time': total_time,
                'models_used': self.model_names,
                'use_full_data': self.use_full_data
            },
            'data_statistics': {
                'total_rankers': results['ranker_id'].nunique(),
                'avg_options_per_ranker': len(results) / results['ranker_id'].nunique(),
                'min_options': results.groupby('ranker_id').size().min(),
                'max_options': results.groupby('ranker_id').size().max()
            },
            'ranking_validation': {
                'all_groups_valid': self._validate_rankings(results['selected'].values, results['ranker_id'].values),
                'ranking_distribution': results['selected'].value_counts().head(10).to_dict()
            }
        }
        
        # 保存报告
        report_path = self.output_path / "prediction_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"预测报告已保存: {report_path}")
        
        # 打印关键统计信息
        self.logger.info(f"预测成功率: {report['prediction_summary']['success_rate']:.2%}")
        self.logger.info(f"总排名组数: {report['data_statistics']['total_rankers']}")
        self.logger.info(f"平均每组选项数: {report['data_statistics']['avg_options_per_ranker']:.1f}")