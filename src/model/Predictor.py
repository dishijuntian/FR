"""
智能预测器
"""

import os
import time
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

import numpy as np
import pandas as pd

from .Manager import FlightRankingModelsManager

warnings.filterwarnings('ignore')


class FlightRankingPredictor:
    """智能预测器"""
    
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
        self.use_gpu = prediction_config['use_gpu']
        
        # 加载最佳模型配置
        self.best_models_config = self._load_best_models_config()
        
        # 确保输出目录存在
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("智能预测器初始化完成")
        self.logger.info(f"最佳模型配置: {self.best_models_config}")
    
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
    
    def _load_best_models_config(self) -> Dict:
        """加载最佳模型配置"""
        config_path = self.model_save_path / "best_models_config.json"
        
        if not config_path.exists():
            self.logger.warning("最佳模型配置文件不存在，将尝试从训练报告中推断")
            return self._infer_best_models_from_reports()
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        self.logger.info(f"加载最佳模型配置: {config}")
        return config
    
    def _infer_best_models_from_reports(self) -> Dict:
        """从训练报告中推断最佳模型"""
        best_models = {}
        
        for segment_id in self.segments:
            report_path = self.model_save_path / f"segment_{segment_id}" / "training_report.json"
            
            if report_path.exists():
                try:
                    with open(report_path, 'r') as f:
                        report = json.load(f)
                    
                    best_model = report.get('best_model_name')
                    if best_model:
                        best_models[str(segment_id)] = best_model
                        self.logger.info(f"推断 segment_{segment_id} 最佳模型: {best_model}")
                
                except Exception as e:
                    self.logger.warning(f"读取 segment_{segment_id} 训练报告失败: {e}")
        
        return best_models
    
    def load_segment_best_model(self, segment_id: int) -> Tuple[FlightRankingModelsManager, str]:
        """加载数据段的最佳模型"""
        segment_dir = self.model_save_path / f"segment_{segment_id}"
        
        if not segment_dir.exists():
            raise FileNotFoundError(f"段模型目录不存在: {segment_dir}")
        
        # 获取最佳模型名称
        best_model_name = self.best_models_config.get(str(segment_id))
        if not best_model_name:
            raise ValueError(f"未找到 segment_{segment_id} 的最佳模型配置")
        
        # 检查模型文件是否存在
        model_file = segment_dir / f"{best_model_name}.pkl"
        if not model_file.exists():
            raise FileNotFoundError(f"最佳模型文件不存在: {model_file}")
        
        # 加载模型
        models_manager = FlightRankingModelsManager(use_gpu=self.use_gpu, logger=self.logger)
        success = models_manager.load_model(best_model_name, str(segment_dir))
        
        if not success:
            raise ValueError(f"加载 segment_{segment_id} 最佳模型 {best_model_name} 失败")
        
        # 验证模型
        if not models_manager.validate_model(best_model_name):
            raise ValueError(f"segment_{segment_id} 最佳模型 {best_model_name} 验证失败")
        
        self.logger.info(f"成功加载 segment_{segment_id} 最佳模型: {best_model_name}")
        return models_manager, best_model_name
    
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
        """简单验证排名的有效性"""
        unique_groups = np.unique(groups)
        
        for group_id in unique_groups:
            group_mask = groups == group_id
            group_rankings = rankings[group_mask]
            
            # 检查排名是否是1到N的连续整数
            expected_rankings = set(range(1, len(group_rankings) + 1))
            actual_rankings = set(group_rankings)
            
            if expected_rankings != actual_rankings:
                return False
        
        return True
    
    def predict_segment(self, segment_id: int, save_individual: bool = False) -> pd.DataFrame:
        """预测单个数据段"""
        self.logger.info(f"开始预测 segment_{segment_id}")
        start_time = time.time()
        
        # 加载测试数据
        test_file = self.data_path / "test" / f"test_segment_{segment_id}.parquet"
        if not test_file.exists():
            raise FileNotFoundError(f"测试文件不存在: {test_file}")
        
        df = pd.read_parquet(test_file)
        
        # 加载最佳模型
        models_manager, best_model_name = self.load_segment_best_model(segment_id)
        
        # 数据预处理
        X, _, groups, _, _ = models_manager.prepare_data(df, target_col='selected')
        
        # 模型预测
        try:
            if best_model_name in ['GraphRanker', 'TransformerRanker']:
                predictions = models_manager.predict_model(best_model_name, X, groups)
            else:
                predictions = models_manager.predict_model(best_model_name, X)
            
        except Exception as e:
            self.logger.error(f"✗ {best_model_name} 预测失败: {e}")
            raise
        
        # 生成排名
        rankings = self.generate_rankings(predictions, groups)
        
        # 验证排名
        if not self.validate_rankings(rankings, groups):
            raise ValueError(f"segment_{segment_id} 排名验证失败")
        
        # 生成结果
        results = df[['Id', 'ranker_id']].copy()
        results['selected'] = rankings
        
        prediction_time = time.time() - start_time
        
        if save_individual:
            output_file = self.output_path / f"predictions_segment_{segment_id}.csv"
            results.to_csv(output_file, index=False)
        
        self.logger.info(f"✓ segment_{segment_id} 预测完成 (模型: {best_model_name}, 时间: {prediction_time:.1f}s)")
        
        return results, best_model_name, prediction_time
    
    def predict_all_segments(self) -> pd.DataFrame:
        """预测所有数据段"""
        self.logger.info(f"开始预测所有段: {self.segments}")
        
        all_results = []
        prediction_summary = {}
        total_start_time = time.time()
        successful_predictions = 0
        failed_predictions = 0
        
        for segment_id in self.segments:
            try:
                result, best_model_name, prediction_time = self.predict_segment(segment_id, save_individual=True)
                all_results.append(result)
                successful_predictions += 1
                
                prediction_summary[f'segment_{segment_id}'] = {
                    'best_model': best_model_name,
                    'prediction_time': prediction_time,
                    'samples_count': len(result),
                    'rankers_count': result['ranker_id'].nunique(),
                    'status': 'success'
                }
                
            except Exception as e:
                self.logger.error(f"✗ segment_{segment_id} 预测失败: {e}")
                failed_predictions += 1
                prediction_summary[f'segment_{segment_id}'] = {
                    'status': 'failed',
                    'error': str(e)
                }
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
        self._generate_prediction_report(
            final_submission, prediction_summary, total_time, 
            successful_predictions, failed_predictions
        )
        
        self.logger.info(f"✓ 所有预测完成 (总时间: {total_time:.1f}s)")
        self.logger.info(f"✓ 最终结果: {output_file}")
        self.logger.info(f"✓ 总记录数: {len(final_submission):,}")
        self.logger.info(f"✓ 成功预测段数: {successful_predictions}/{len(self.segments)}")
        
        return final_submission
    
    def _generate_prediction_report(self, results: pd.DataFrame, prediction_summary: Dict,
                                   total_time: float, successful_predictions: int, 
                                   failed_predictions: int):
        """生成预测报告"""
        total_segments = successful_predictions + failed_predictions
        
        # 统计使用的模型分布
        model_usage = {}
        for segment_info in prediction_summary.values():
            if segment_info['status'] == 'success':
                model_name = segment_info['best_model']
                model_usage[model_name] = model_usage.get(model_name, 0) + 1
        
        # 计算基础统计
        total_samples = len(results)
        total_rankers = int(results['ranker_id'].nunique())
        group_sizes = results.groupby('ranker_id').size()
        
        report = {
            'prediction_summary': {
                'total_segments': total_segments,
                'successful_segments': successful_predictions,
                'failed_segments': failed_predictions,
                'success_rate': successful_predictions / total_segments if total_segments > 0 else 0,
                'total_time': total_time,
                'avg_time_per_segment': total_time / successful_predictions if successful_predictions > 0 else 0
            },
            'data_statistics': {
                'total_samples': total_samples,
                'total_rankers': total_rankers,
                'avg_options_per_ranker': total_samples / total_rankers if total_rankers > 0 else 0,
                'min_options': int(group_sizes.min()) if len(group_sizes) > 0 else 0,
                'max_options': int(group_sizes.max()) if len(group_sizes) > 0 else 0
            },
            'model_usage': model_usage,
            'segment_details': prediction_summary,
            'best_models_config': self.best_models_config
        }
        
        # 保存报告
        report_path = self.output_path / "intelligent_prediction_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # 打印总结
        self.logger.info(f"\n预测总结:")
        self.logger.info(f"  - 成功率: {report['prediction_summary']['success_rate']:.1%}")
        self.logger.info(f"  - 总样本数: {total_samples:,}")
        self.logger.info(f"  - Ranker数: {total_rankers:,}")
        
        if model_usage:
            self.logger.info(f"\n使用的模型分布:")
            for model_name, count in sorted(model_usage.items(), key=lambda x: x[1], reverse=True):
                self.logger.info(f"  - {model_name}: {count} 段")
        
        self.logger.info(f"\n详细报告已保存: {report_path}")
    
    def get_prediction_status(self) -> Dict:
        """获取预测状态信息"""
        status = {
            'target_segments': self.segments,
            'best_models_config': self.best_models_config,
            'available_models': {},
            'completed_predictions': [],
            'missing_models': []
        }
        
        for segment_id in self.segments:
            segment_dir = self.model_save_path / f"segment_{segment_id}"
            best_model_name = self.best_models_config.get(str(segment_id))
            
            if best_model_name:
                model_file = segment_dir / f"{best_model_name}.pkl"
                prediction_file = self.output_path / f"predictions_segment_{segment_id}.csv"
                
                status['available_models'][segment_id] = {
                    'model_name': best_model_name,
                    'model_exists': model_file.exists(),
                    'prediction_exists': prediction_file.exists()
                }
                
                if model_file.exists():
                    if prediction_file.exists():
                        status['completed_predictions'].append(segment_id)
                else:
                    status['missing_models'].append(segment_id)
            else:
                status['missing_models'].append(segment_id)
        
        return status