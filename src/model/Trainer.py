"""
优化后的航班排名模型训练器
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
    """优化后的航班排名训练器"""
    
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
        self.model_names = training_config['model_names']
        self.use_gpu = training_config['use_gpu']
        self.random_state = training_config['random_state']
        self.model_configs = training_config.get('model_configs', {})
        
        # 自动划分策略配置
        self.auto_split_thresholds = {
            1000: 2,    # ≤1000个ranker_id: 2折
            2000: 3,    # ≤2000个ranker_id: 3折
            5000: 5,    # ≤5000个ranker_id: 5折
            10000: 8,   # ≤10000个ranker_id: 8折
            float('inf'): 10  # >10000个ranker_id: 10折
        }
        
        # 确保目录存在
        self.model_save_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("优化训练器初始化完成")
    
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
    
    def determine_n_folds(self, n_rankers: int) -> int:
        """根据ranker_id数量自动确定折数"""
        for threshold, n_folds in self.auto_split_thresholds.items():
            if n_rankers <= threshold:
                return n_folds
        return 10  # 默认值
    
    def create_train_val_split(self, groups: np.ndarray, test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
        """创建训练验证集划分（基于组）"""
        unique_groups = np.unique(groups)
        n_groups = len(unique_groups)
        
        self.logger.info(f"总共 {n_groups} 个ranker组")
        
        # 自动确定验证集大小
        if n_groups < 100:
            test_size = max(0.3, test_size)  # 小数据集用更大验证集
        elif n_groups > 10000:
            test_size = min(0.1, test_size)  # 大数据集用更小验证集
        
        # 按组划分
        train_groups, val_groups = train_test_split(
            unique_groups, 
            test_size=test_size, 
            random_state=self.random_state
        )
        
        # 转换为样本索引
        train_mask = np.isin(groups, train_groups)
        val_mask = np.isin(groups, val_groups)
        
        train_idx = np.where(train_mask)[0]
        val_idx = np.where(val_mask)[0]
        
        self.logger.info(f"训练集: {len(train_groups)} 组 ({len(train_idx)} 样本)")
        self.logger.info(f"验证集: {len(val_groups)} 组 ({len(val_idx)} 样本)")
        
        return train_idx, val_idx
    
    def load_segment_data(self, segment_id: int) -> pd.DataFrame:
        """加载数据段"""
        train_file = self.data_path / "train" / f"train_segment_{segment_id}.parquet"
        if not train_file.exists():
            raise FileNotFoundError(f"训练文件不存在: {train_file}")
        
        df = pd.read_parquet(train_file)
        self.logger.info(f"加载 segment_{segment_id}: {df.shape}")
        return df
    
    def train_and_validate_model(self, model, model_name: str, X_train: np.ndarray, 
                                 y_train: np.ndarray, groups_train: np.ndarray,
                                 X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """训练并验证单个模型"""
        self.logger.info(f"训练 {model_name}...")
        start_time = time.time()
        
        try:
            # 训练模型
            if model_name == 'RankNet':
                model.fit(X_train, y_train, groups_train, epochs=50)
            else:
                model.fit(X_train, y_train, groups_train)
            
            # 验证
            val_pred = model.predict(X_val)
            val_score = ndcg_score([y_val], [val_pred], k=10)
            
            training_time = time.time() - start_time
            
            self.logger.info(f"✓ {model_name} - NDCG@10: {val_score:.4f} (时间: {training_time:.1f}s)")
            
            return {
                'model': model,
                'val_score': val_score,
                'training_time': training_time,
                'status': 'success'
            }
            
        except Exception as e:
            self.logger.error(f"✗ {model_name} 训练失败: {e}")
            return {
                'model': None,
                'val_score': 0.0,
                'training_time': time.time() - start_time,
                'status': 'failed',
                'error': str(e)
            }
    
    def train_final_model(self, best_model_info: Dict, model_name: str, 
                         X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> object:
        """使用全部数据训练最终模型"""
        self.logger.info(f"训练最终 {model_name} 模型...")
        
        # 创建新的模型实例（使用相同配置）
        models_manager = FlightRankingModelsManager(use_gpu=self.use_gpu, logger=self.logger)
        input_dim = X.shape[1]
        final_models = models_manager.create_models(input_dim, self.model_configs)
        
        if model_name not in final_models:
            raise ValueError(f"无法创建最终模型: {model_name}")
        
        final_model = final_models[model_name]
        
        # 训练最终模型
        if model_name == 'RankNet':
            final_model.fit(X, y, groups, epochs=100)  # 更多轮次
        else:
            final_model.fit(X, y, groups)
        
        return final_model
    
    def train_segment(self, segment_id: int) -> Dict:
        """训练单个数据段"""
        self.logger.info(f"开始训练 segment_{segment_id}")
        start_time = time.time()
        
        # 加载数据
        df = self.load_segment_data(segment_id)
        
        # 数据预处理（只创建一次管理器）
        models_manager = FlightRankingModelsManager(use_gpu=self.use_gpu, logger=self.logger)
        X, y, groups, feature_cols, _ = models_manager.prepare_data(df)
        
        # 自动确定数据划分策略
        n_rankers = len(np.unique(groups))
        n_folds = self.determine_n_folds(n_rankers)
        self.logger.info(f"Ranker数量: {n_rankers}, 使用 {n_folds} 折交叉验证策略")
        
        # 创建训练验证集划分
        train_idx, val_idx = self.create_train_val_split(groups)
        
        X_train, y_train, groups_train = X[train_idx], y[train_idx], groups[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        # 创建模型（只创建一次）
        input_dim = X.shape[1]
        created_models = models_manager.create_models(input_dim, self.model_configs)
        
        # 训练结果
        results = {
            'segment_id': segment_id,
            'n_rankers': n_rankers,
            'n_folds_used': n_folds,
            'models': {},
            'validation_scores': {},
            'training_time': 0,
            'feature_names': feature_cols
        }
        
        # 训练和验证每个模型
        model_results = {}
        
        for model_name in self.model_names:
            if model_name not in created_models:
                self.logger.warning(f"模型 {model_name} 创建失败，跳过")
                continue
            
            model = created_models[model_name]
            model_result = self.train_and_validate_model(
                model, model_name, X_train, y_train, groups_train, X_val, y_val
            )
            
            model_results[model_name] = model_result
            
            if model_result['status'] == 'success':
                results['validation_scores'][model_name] = model_result['val_score']
        
        # 选择最佳模型并训练最终版本
        if model_results:
            # 按验证分数排序
            successful_models = {k: v for k, v in model_results.items() 
                               if v['status'] == 'success'}
            
            if successful_models:
                best_model_name = max(successful_models.keys(), 
                                    key=lambda x: successful_models[x]['val_score'])
                
                self.logger.info(f"最佳模型: {best_model_name} "
                               f"(NDCG@10: {successful_models[best_model_name]['val_score']:.4f})")
                
                # 为每个成功的模型训练最终版本
                for model_name, model_result in successful_models.items():
                    if model_result['status'] == 'success':
                        final_model = self.train_final_model(
                            model_result, model_name, X, y, groups
                        )
                        results['models'][model_name] = final_model
        
        # 记录训练时间
        total_time = time.time() - start_time
        results['training_time'] = total_time
        
        # 保存结果
        self._save_segment_results(segment_id, results)
        
        self.logger.info(f"✓ segment_{segment_id} 训练完成 (总时间: {total_time:.1f}s)")
        return results
    
    def _save_segment_results(self, segment_id: int, results: Dict):
        """保存训练结果"""
        segment_dir = self.model_save_path / f"segment_{segment_id}"
        segment_dir.mkdir(exist_ok=True)
        
        # 保存模型
        for model_name, model in results['models'].items():
            model_path = segment_dir / f"{model_name}.pkl"
            model.save_model(str(model_path))
        
        # 保存特征名称
        import joblib
        feature_path = segment_dir / "features.pkl"
        joblib.dump(results['feature_names'], str(feature_path))
        
        # 保存训练报告
        report = {
            'segment_id': results['segment_id'],
            'n_rankers': results['n_rankers'],
            'n_folds_used': results['n_folds_used'],
            'validation_scores': results['validation_scores'],
            'training_time': results['training_time'],
            'model_count': len(results['models'])
        }
        
        report_path = segment_dir / "training_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"结果已保存到: {segment_dir}")
    
    def train_all_segments(self) -> Dict:
        """训练所有数据段"""
        self.logger.info(f"开始训练所有段: {self.segments}")
        self.logger.info(f"使用模型: {self.model_names}")
        self.logger.info(f"自动划分策略: {self.auto_split_thresholds}")
        
        all_results = {}
        total_start_time = time.time()
        
        for segment_id in self.segments:
            try:
                self.logger.info(f"\n{'='*50}")
                self.logger.info(f"训练 segment_{segment_id}")
                self.logger.info(f"{'='*50}")
                
                results = self.train_segment(segment_id)
                all_results[f'segment_{segment_id}'] = results
                
            except Exception as e:
                self.logger.error(f"✗ segment_{segment_id} 训练失败: {e}")
                continue
        
        # 生成总体报告
        total_time = time.time() - total_start_time
        self._generate_final_report(all_results, total_time)
        
        return all_results
    
    def _generate_final_report(self, all_results: Dict, total_time: float):
        """生成最终训练报告"""
        report = {
            'total_segments': len(all_results),
            'total_training_time': total_time,
            'auto_split_strategy': self.auto_split_thresholds,
            'segment_summary': {},
            'model_performance': {}
        }
        
        # 汇总每个segment的结果
        all_model_scores = {}
        
        for segment_name, results in all_results.items():
            validation_scores = results.get('validation_scores', {})
            training_time = results.get('training_time', 0)
            n_rankers = results.get('n_rankers', 0)
            n_folds_used = results.get('n_folds_used', 0)
            
            report['segment_summary'][segment_name] = {
                'n_rankers': n_rankers,
                'n_folds_used': n_folds_used,
                'training_time': training_time,
                'models_trained': len(validation_scores),
                'best_model': max(validation_scores.keys(), 
                                key=lambda x: validation_scores[x]) if validation_scores else None,
                'best_score': max(validation_scores.values()) if validation_scores else 0
            }
            
            # 收集模型分数
            for model_name, score in validation_scores.items():
                if model_name not in all_model_scores:
                    all_model_scores[model_name] = []
                all_model_scores[model_name].append(score)
        
        # 计算模型平均性能
        for model_name, scores in all_model_scores.items():
            report['model_performance'][model_name] = {
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'min_score': np.min(scores),
                'max_score': np.max(scores),
                'segments_count': len(scores)
            }
        
        # 保存报告
        report_path = self.model_save_path / "optimized_training_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # 打印总结
        self.logger.info(f"\n{'='*60}")
        self.logger.info("优化训练总结")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"总训练时间: {total_time:.1f}s")
        self.logger.info(f"成功训练段数: {len(all_results)}")
        
        if report['model_performance']:
            self.logger.info("\n模型性能排名 (基于验证集):")
            sorted_models = sorted(
                report['model_performance'].items(),
                key=lambda x: x[1]['mean_score'],
                reverse=True
            )
            for i, (model_name, perf) in enumerate(sorted_models, 1):
                self.logger.info(f"{i}. {model_name}: {perf['mean_score']:.4f}±{perf['std_score']:.4f} "
                               f"({perf['segments_count']} 段)")
        
        # 显示自动划分策略效果
        self.logger.info("\n自动数据划分策略效果:")
        for segment_name, summary in report['segment_summary'].items():
            self.logger.info(f"{segment_name}: {summary['n_rankers']} rankers → "
                           f"{summary['n_folds_used']} 折验证")
        
        self.logger.info(f"\n详细报告已保存到: {report_path}")