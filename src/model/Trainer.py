"""
航班排名模型训练器
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
from sklearn.model_selection import GroupKFold
from sklearn.metrics import ndcg_score

from .Manager import FlightRankingModelsManager

warnings.filterwarnings('ignore')


class FlightRankingTrainer:
    """航班排名训练器"""
    
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
        self.n_folds = training_config['n_folds']
        self.random_state = training_config['random_state']
        self.model_configs = training_config.get('model_configs', {})
        
        # 确保目录存在
        self.model_save_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("训练器初始化完成")
    
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
    
    def create_cv_folds(self, groups: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """创建交叉验证fold"""
        unique_groups = np.unique(groups)
        n_groups = len(unique_groups)
        
        if n_groups < self.n_folds:
            self.logger.warning(f"组数量({n_groups})少于fold数({self.n_folds})，调整为{n_groups}fold")
            n_folds = n_groups
        else:
            n_folds = self.n_folds
        
        gkf = GroupKFold(n_splits=n_folds)
        dummy_data = np.zeros(len(groups))
        folds = list(gkf.split(dummy_data, groups=groups))
        
        self.logger.info(f"创建 {len(folds)} 个CV fold")
        return folds
    
    def train_segment(self, segment_id: int) -> Dict:
        """训练单个数据段"""
        self.logger.info(f"开始训练 segment_{segment_id}")
        start_time = time.time()
        
        # 加载数据
        df = self.load_segment_data(segment_id)
        
        # 数据预处理
        models_manager = FlightRankingModelsManager(use_gpu=self.use_gpu, logger=self.logger)
        X, y, groups, feature_cols, _ = models_manager.prepare_data(df)
        
        # 创建模型
        input_dim = X.shape[1]
        created_models = models_manager.create_models(input_dim, self.model_configs)
        
        # 训练结果
        results = {
            'segment_id': segment_id,
            'models': {},
            'cv_scores': {},
            'training_time': 0,
            'feature_names': feature_cols
        }
        
        # 创建CV fold
        folds = self.create_cv_folds(groups)
        
        # 训练每个模型
        for model_name in self.model_names:
            if model_name not in created_models:
                self.logger.warning(f"模型 {model_name} 创建失败，跳过")
                continue
            
            self.logger.info(f"训练 {model_name}")
            model_start_time = time.time()
            
            try:
                # CV评估
                fold_scores = []
                best_model = None
                best_score = -1
                
                for fold_idx, (train_idx, val_idx) in enumerate(folds):
                    X_train, y_train = X[train_idx], y[train_idx]
                    X_val, y_val = X[val_idx], y[val_idx]
                    groups_train = groups[train_idx]
                    
                    # 创建新模型实例
                    model = models_manager.create_models(input_dim, self.model_configs)[model_name]
                    
                    # 训练
                    if model_name == 'RankNet':
                        model.fit(X_train, y_train, groups_train, epochs=50)
                    else:
                        model.fit(X_train, y_train, groups_train)
                    
                    # 验证
                    pred = model.predict(X_val)
                    score = ndcg_score([y_val], [pred], k=10)
                    fold_scores.append(score)
                    
                    # 保存最佳模型
                    if score > best_score:
                        best_score = score
                        best_model = model
                    
                    self.logger.info(f"Fold {fold_idx+1}/{len(folds)}, NDCG@10: {score:.4f}")
                
                # 记录CV结果
                results['cv_scores'][model_name] = {
                    'mean': np.mean(fold_scores),
                    'std': np.std(fold_scores),
                    'scores': fold_scores,
                    'best_score': best_score
                }
                
                # 训练最终模型（使用全部数据）
                final_model = models_manager.create_models(input_dim, self.model_configs)[model_name]
                if model_name == 'RankNet':
                    final_model.fit(X, y, groups, epochs=100)
                else:
                    final_model.fit(X, y, groups)
                
                results['models'][model_name] = final_model
                
                model_time = time.time() - model_start_time
                self.logger.info(f"✓ {model_name} 训练完成: CV={np.mean(fold_scores):.4f}±{np.std(fold_scores):.4f} "
                               f"(时间: {model_time:.1f}s)")
                
            except Exception as e:
                self.logger.error(f"✗ {model_name} 训练失败: {e}")
                continue
        
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
            'cv_scores': results['cv_scores'],
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
            'segment_summary': {},
            'model_performance': {}
        }
        
        # 汇总每个segment的结果
        all_model_scores = {}
        
        for segment_name, results in all_results.items():
            cv_scores = results.get('cv_scores', {})
            training_time = results.get('training_time', 0)
            
            report['segment_summary'][segment_name] = {
                'training_time': training_time,
                'models_trained': len(cv_scores),
                'best_model': max(cv_scores.keys(), key=lambda x: cv_scores[x]['mean']) if cv_scores else None,
                'best_score': max(cv_scores[x]['mean'] for x in cv_scores.keys()) if cv_scores else 0
            }
            
            # 收集模型分数
            for model_name, score_info in cv_scores.items():
                if model_name not in all_model_scores:
                    all_model_scores[model_name] = []
                all_model_scores[model_name].append(score_info['mean'])
        
        # 计算模型平均性能
        for model_name, scores in all_model_scores.items():
            report['model_performance'][model_name] = {
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'min_score': np.min(scores),
                'max_score': np.max(scores)
            }
        
        # 保存报告
        report_path = self.model_save_path / "final_training_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # 打印总结
        self.logger.info(f"\n{'='*60}")
        self.logger.info("训练总结")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"总训练时间: {total_time:.1f}s")
        self.logger.info(f"成功训练段数: {len(all_results)}")
        
        if report['model_performance']:
            self.logger.info("\n模型性能排名:")
            sorted_models = sorted(
                report['model_performance'].items(),
                key=lambda x: x[1]['mean_score'],
                reverse=True
            )
            for i, (model_name, perf) in enumerate(sorted_models, 1):
                self.logger.info(f"{i}. {model_name}: {perf['mean_score']:.4f}±{perf['std_score']:.4f}")
        
        self.logger.info(f"\n详细报告已保存到: {report_path}")