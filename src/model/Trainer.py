"""
多模型训练器
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import ndcg_score

from .Manager import FlightRankingModelsManager

warnings.filterwarnings('ignore')


class FlightRankingTrainer:
    """多模型训练器"""
    
    def __init__(self, config: Dict, logger=None):
        """
        初始化训练器
        
        Args:
            config: 配置字典
            logger: 日志器
        """
        self.config = config
        self.logger = logger or self._setup_logger()
        
        # 路径配置
        self.data_path = Path(config['paths']['model_input_dir'])
        self.model_save_path = Path(config['paths']['model_save_dir'])
        
        # 训练配置
        training_config = config['training']
        self.segments = training_config['segments']
        self.model_names = training_config['model_names']  # 要训练的所有模型列表
        self.use_gpu = training_config['use_gpu']
        self.random_state = training_config['random_state']
        self.model_configs = training_config.get('model_configs', {})
        self.epochs = training_config.get('epochs', 100)
        self.validation_split = training_config.get('validation_split', 0.2)
        
        # 确保目录存在
        self.model_save_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("多模型训练器初始化完成")
        self.logger.info(f"将训练模型: {self.model_names}")
    
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
    
    def load_segment_data(self, segment_id: int) -> pd.DataFrame:
        """加载数据段"""
        train_file = self.data_path / "train" / f"train_segment_{segment_id}.parquet"
        if not train_file.exists():
            raise FileNotFoundError(f"训练文件不存在: {train_file}")
        
        df = pd.read_parquet(train_file)
        self.logger.info(f"加载 segment_{segment_id}: {df.shape}")
        return df
    
    def create_validation_split(self, groups: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """创建验证集划分（按组）"""
        unique_groups = np.unique(groups)
        train_groups, val_groups = train_test_split(
            unique_groups, 
            test_size=self.validation_split, 
            random_state=self.random_state
        )
        
        train_mask = np.isin(groups, train_groups)
        val_mask = np.isin(groups, val_groups)
        
        train_idx = np.where(train_mask)[0]
        val_idx = np.where(val_mask)[0]
        
        self.logger.info(f"训练集: {len(train_groups)} 组 ({len(train_idx)} 样本)")
        self.logger.info(f"验证集: {len(val_groups)} 组 ({len(val_idx)} 样本)")
        
        return train_idx, val_idx
        
    def evaluate_model(self, model, model_name: str, X_val: np.ndarray, 
                    y_val: np.ndarray, groups_val: np.ndarray) -> float:
        """评估模型性能 - 使用HitRate@3"""
        try:
            if model_name in ['GraphRanker', 'TransformerRanker']:
                val_pred = model.predict(X_val, groups_val)
            else:
                val_pred = model.predict(X_val)
            
            # 计算每个组的HitRate@3，然后取平均
            unique_groups = np.unique(groups_val)
            hitrate_scores = []
            
            for group_id in unique_groups:
                group_mask = groups_val == group_id
                group_y_true = y_val[group_mask]
                group_y_pred = val_pred[group_mask]
                
                if len(group_y_true) > 1 and np.sum(group_y_true) > 0:
                    try:
                        # 计算HitRate@3
                        hitrate = self._calculate_hitrate_at_k(group_y_true, group_y_pred, k=3)
                        hitrate_scores.append(hitrate)
                    except:
                        continue
            
            if hitrate_scores:
                avg_hitrate = np.mean(hitrate_scores)
                return avg_hitrate
            else:
                return 0.0
                
        except Exception as e:
            self.logger.warning(f"评估 {model_name} 失败: {e}")
            return 0.0

    def _calculate_hitrate_at_k(self, y_true: np.ndarray, y_pred: np.ndarray, k: int = 3) -> float:
        """
        计算HitRate@k
        
        Args:
            y_true: 真实标签 (0或1)
            y_pred: 预测分数
            k: 取前k个预测结果
        
        Returns:
            HitRate@k 分数
        """
        # 按预测分数降序排序，获取top-k的索引
        top_k_indices = np.argsort(y_pred)[::-1][:k]
        
        # 检查top-k中是否有正样本
        hit = np.any(y_true[top_k_indices] > 0)
        
        return float(hit)
    
    def train_segment(self, segment_id: int) -> Dict:
        """训练单个数据段的所有模型"""
        self.logger.info(f"开始训练 segment_{segment_id}")
        start_time = time.time()
        
        # 加载数据
        df = self.load_segment_data(segment_id)
        
        # 数据预处理
        models_manager = FlightRankingModelsManager(use_gpu=self.use_gpu, logger=self.logger)
        X, y, groups, feature_cols, _ = models_manager.prepare_data(df)
        
        # 数据统计
        n_samples = len(X)
        n_rankers = len(np.unique(groups))
        n_features = X.shape[1]
        
        self.logger.info(f"数据统计: {n_samples} 样本, {n_rankers} 组, {n_features} 特征")
        
        # 创建验证集
        train_idx, val_idx = self.create_validation_split(groups)
        X_train, y_train, groups_train = X[train_idx], y[train_idx], groups[train_idx]
        X_val, y_val, groups_val = X[val_idx], y[val_idx], groups[val_idx]
        
        # 训练所有模型
        model_results = {}
        best_model_name = None
        best_score = -1
        
        for model_name in self.model_names:
            self.logger.info(f"训练 {model_name}...")
            model_start_time = time.time()
            
            try:
                # 创建模型
                model_config = self.model_configs.get(model_name, {})
                model = models_manager.create_model(model_name, n_features, model_config)
                
                if model is None:
                    self.logger.warning(f"创建 {model_name} 失败")
                    continue
                
                # 训练模型
                training_kwargs = {}
                if model_name in ['RankNet', 'GraphRanker', 'CNNRanker', 'TransformerRanker']:
                    training_kwargs['epochs'] = self.epochs
                
                success = models_manager.train_model(
                    model_name, X_train, y_train, groups_train, **training_kwargs
                )
                
                if not success:
                    self.logger.warning(f"训练 {model_name} 失败")
                    continue
                
                # 评估模型
                score = self.evaluate_model(model, model_name, X_val, y_val, groups_val)
                training_time = time.time() - model_start_time
                
                model_results[model_name] = {
                    'model': model,
                    'score': score,
                    'training_time': training_time,
                    'success': True
                }
                
                # 更新最佳模型
                if score > best_score:
                    best_score = score
                    best_model_name = model_name
                
                self.logger.info(f"✓ {model_name} - NDCG: {score:.4f} (时间: {training_time:.1f}s)")
                
            except Exception as e:
                self.logger.error(f"✗ {model_name} 训练失败: {e}")
                model_results[model_name] = {
                    'model': None,
                    'score': 0.0,
                    'training_time': time.time() - model_start_time,
                    'success': False,
                    'error': str(e)
                }
        
        # 如果有成功的模型，用全量数据重新训练最佳模型
        final_best_model = None
        if best_model_name and model_results[best_model_name]['success']:
            self.logger.info(f"最佳模型: {best_model_name} (NDCG: {best_score:.4f})")
            self.logger.info(f"用全量数据重新训练 {best_model_name}...")
            
            try:
                # 创建新的模型实例用于全量训练
                model_config = self.model_configs.get(best_model_name, {})
                final_model = models_manager.create_model(best_model_name, n_features, model_config)
                
                training_kwargs = {}
                if best_model_name in ['RankNet', 'GraphRanker', 'CNNRanker', 'TransformerRanker']:
                    training_kwargs['epochs'] = self.epochs
                
                success = models_manager.train_model(
                    best_model_name, X, y, groups, **training_kwargs
                )
                
                if success:
                    final_best_model = final_model
                    self.logger.info(f"✓ 全量训练 {best_model_name} 完成")
                
            except Exception as e:
                self.logger.error(f"✗ 全量训练 {best_model_name} 失败: {e}")
        
        # 准备结果
        total_time = time.time() - start_time
        results = {
            'segment_id': segment_id,
            'n_samples': n_samples,
            'n_rankers': n_rankers,
            'n_features': n_features,
            'model_results': {name: {
                'score': result['score'],
                'training_time': result['training_time'],
                'success': result['success']
            } for name, result in model_results.items()},
            'best_model_name': best_model_name,
            'best_score': best_score,
            'final_model': final_best_model,
            'total_time': total_time,
            'feature_names': feature_cols
        }
        
        # 保存结果
        self._save_segment_results(segment_id, results, models_manager)
        
        self.logger.info(f"✓ segment_{segment_id} 训练完成 (总时间: {total_time:.1f}s)")
        return results
    
    def _save_segment_results(self, segment_id: int, results: Dict, 
                             models_manager: FlightRankingModelsManager):
        """保存训练结果"""
        segment_dir = self.model_save_path / f"segment_{segment_id}"
        segment_dir.mkdir(exist_ok=True)
        
        # 只保存最佳模型
        if results['final_model'] and results['best_model_name']:
            try:
                # 临时替换管理器中的模型
                original_models = models_manager.models.copy()
                models_manager.models = {results['best_model_name']: results['final_model']}
                
                # 保存最佳模型
                models_manager.save_model(results['best_model_name'], str(segment_dir))
                
                # 恢复管理器状态
                models_manager.models = original_models
                
                self.logger.info(f"最佳模型 {results['best_model_name']} 已保存")
                
            except Exception as e:
                self.logger.error(f"保存模型失败: {e}")
        
        # 保存训练报告（包含所有模型的性能）
        report = {
            'segment_id': results['segment_id'],
            'n_samples': results['n_samples'],
            'n_rankers': results['n_rankers'],
            'n_features': results['n_features'],
            'model_results': results['model_results'],
            'best_model_name': results['best_model_name'],
            'best_score': results['best_score'],
            'total_time': results['total_time'],
            'feature_count': len(results['feature_names'])
        }
        
        report_path = segment_dir / "training_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"训练报告已保存: {report_path}")
    
    def train_all_segments(self) -> Dict:
        """训练所有数据段"""
        self.logger.info(f"开始训练所有段: {self.segments}")
        self.logger.info(f"候选模型: {self.model_names}")
        
        all_results = {}
        total_start_time = time.time()
        successful_segments = 0
        failed_segments = 0
        
        for segment_id in self.segments:
            try:
                self.logger.info(f"\n{'='*50}")
                self.logger.info(f"训练 segment_{segment_id}")
                self.logger.info(f"{'='*50}")
                
                results = self.train_segment(segment_id)
                all_results[f'segment_{segment_id}'] = results
                
                if results['best_model_name']:
                    successful_segments += 1
                else:
                    failed_segments += 1
                
            except Exception as e:
                self.logger.error(f"✗ segment_{segment_id} 训练失败: {e}")
                failed_segments += 1
                continue
        
        # 生成总体报告
        total_time = time.time() - total_start_time
        self._generate_final_report(all_results, total_time, successful_segments, failed_segments)
        
        return all_results
    
    def _generate_final_report(self, all_results: Dict, total_time: float,
                              successful_segments: int, failed_segments: int):
        """生成最终训练报告"""
        total_segments = successful_segments + failed_segments
        
        # 统计最佳模型分布
        best_model_counts = {}
        segment_best_models = {}
        
        for segment_name, results in all_results.items():
            if results.get('best_model_name'):
                best_model = results['best_model_name']
                best_model_counts[best_model] = best_model_counts.get(best_model, 0) + 1
                segment_best_models[segment_name] = {
                    'best_model': best_model,
                    'score': results['best_score'],
                    'training_time': results['total_time']
                }
        
        # 计算平均性能
        model_avg_scores = {}
        for segment_name, results in all_results.items():
            for model_name, model_result in results.get('model_results', {}).items():
                if model_result['success']:
                    if model_name not in model_avg_scores:
                        model_avg_scores[model_name] = []
                    model_avg_scores[model_name].append(model_result['score'])
        
        # 计算平均分数
        for model_name in model_avg_scores:
            scores = model_avg_scores[model_name]
            model_avg_scores[model_name] = {
                'avg_score': np.mean(scores),
                'std_score': np.std(scores),
                'segments_count': len(scores)
            }
        
        report = {
            'training_summary': {
                'total_segments': total_segments,
                'successful_segments': successful_segments,
                'failed_segments': failed_segments,
                'success_rate': successful_segments / total_segments if total_segments > 0 else 0,
                'total_time': total_time,
                'candidate_models': self.model_names,
                'epochs_used': self.epochs
            },
            'best_model_selection': segment_best_models,
            'best_model_distribution': best_model_counts,
            'model_average_performance': model_avg_scores
        }
        
        # 保存报告
        report_path = self.model_save_path / "multi_model_training_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # 保存预测配置
        prediction_config = {}
        for segment_name, best_info in segment_best_models.items():
            segment_id = segment_name.split('_')[1]
            prediction_config[segment_id] = best_info['best_model']
        
        pred_config_path = self.model_save_path / "best_models_config.json"
        with open(pred_config_path, 'w') as f:
            json.dump(prediction_config, f, indent=2)
        
        # 打印总结
        self.logger.info(f"\n{'='*60}")
        self.logger.info("多模型训练总结")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"总训练时间: {total_time:.1f}s")
        self.logger.info(f"成功训练段数: {successful_segments}/{total_segments}")
        self.logger.info(f"成功率: {report['training_summary']['success_rate']:.1%}")
        
        if best_model_counts:
            self.logger.info(f"\n最佳模型分布:")
            for model_name, count in sorted(best_model_counts.items(), key=lambda x: x[1], reverse=True):
                self.logger.info(f"  {model_name}: {count} 段")
        
        if model_avg_scores:
            self.logger.info(f"\n模型平均性能排名:")
            sorted_models = sorted(
                model_avg_scores.items(),
                key=lambda x: x[1]['avg_score'],
                reverse=True
            )
            for i, (model_name, perf) in enumerate(sorted_models, 1):
                self.logger.info(f"  {i}. {model_name}: {perf['avg_score']:.4f}±{perf['std_score']:.4f} "
                               f"({perf['segments_count']} 段)")
        
        self.logger.info(f"\n详细报告已保存到: {report_path}")
        self.logger.info(f"预测配置已保存到: {pred_config_path}")