<<<<<<< HEAD
"""
高效GPU加速航班排名预测器
支持集成预测、批量处理、内存优化和多进程加速
"""

import os
import time
import logging
import warnings
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pickle
import json

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler

# 导入模型
from .Models import FlightRankingModels, BaseRankingModel

warnings.filterwarnings('ignore')


class GPUInferenceOptimizer:
    """GPU推理优化器"""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gpu_available = torch.cuda.is_available()
        
        if self.gpu_available:
            self.logger.info(f"GPU推理可用: {torch.cuda.get_device_name(0)}")
            self._optimize_inference_settings()
        else:
            self.logger.info("使用CPU推理")
    
    def _optimize_inference_settings(self):
        """优化GPU推理设置"""
        try:
            # 启用推理优化
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            # 设置推理模式
            torch.backends.cudnn.enabled = True
            
            # 清理GPU缓存
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            
            self.logger.info("GPU推理设置已优化")
        except Exception as e:
            self.logger.warning(f"GPU推理优化失败: {e}")
    
    def get_optimal_batch_size(self, model_complexity: str = 'medium') -> int:
        """获取最优推理批次大小"""
        if not self.gpu_available:
            return 2048
        
        try:
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            # 推理通常比训练需要更少内存
            complexity_multipliers = {'simple': 2.0, 'medium': 1.5, 'complex': 1.0}
            multiplier = complexity_multipliers.get(model_complexity, 1.5)
            
            if gpu_memory_gb >= 12:
                base_batch_size = 4096
            elif gpu_memory_gb >= 8:
                base_batch_size = 2048
            elif gpu_memory_gb >= 6:
                base_batch_size = 1024
            else:
                base_batch_size = 512
            
            optimal_size = int(base_batch_size * multiplier)
            self.logger.info(f"推理批次大小: {optimal_size}")
            return optimal_size
            
        except Exception as e:
            self.logger.warning(f"批次大小优化失败: {e}")
            return 2048


class BatchProcessor:
    """批量处理器"""
    
    def __init__(self, batch_size: int = 2048, logger=None):
        self.batch_size = batch_size
        self.logger = logger or logging.getLogger(__name__)
    
    def process_in_batches(self, data: np.ndarray, predict_func, 
                          desc: str = "Processing") -> np.ndarray:
        """批量处理数据"""
        n_samples = len(data)
        n_batches = (n_samples + self.batch_size - 1) // self.batch_size
        
        self.logger.info(f"{desc}: {n_samples} 样本, {n_batches} 批次")
        
        results = []
        for i in range(0, n_samples, self.batch_size):
            batch_data = data[i:i + self.batch_size]
            
            try:
                batch_result = predict_func(batch_data)
                results.append(batch_result)
            except Exception as e:
                self.logger.error(f"批次 {i//self.batch_size + 1} 处理失败: {e}")
                # 使用零填充作为fallback
                results.append(np.zeros(len(batch_data)))
        
        return np.concatenate(results)


class EnsemblePredictor:
    """集成预测器"""
    
    def __init__(self, models: Dict[str, BaseRankingModel], 
                 weights: Optional[Dict[str, float]] = None,
                 logger=None):
        self.models = models
        self.logger = logger or logging.getLogger(__name__)
        
        # 设置权重
        if weights is None:
            self.weights = {name: 1.0 for name in models.keys()}
        else:
            self.weights = weights
        
        # 归一化权重
        total_weight = sum(self.weights.values())
        self.weights = {name: w/total_weight for name, w in self.weights.items()}
        
        self.logger.info(f"集成预测器: {len(models)} 个模型")
        self.logger.info(f"权重: {self.weights}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """集成预测"""
        predictions = []
        valid_weights = []
        
        for name, model in self.models.items():
            if not model.is_fitted:
                self.logger.warning(f"模型 {name} 未训练，跳过")
                continue
            
            try:
                pred = model.predict(X)
                predictions.append(pred)
                valid_weights.append(self.weights[name])
                self.logger.debug(f"✓ {name} 预测完成")
            except Exception as e:
                self.logger.warning(f"✗ {name} 预测失败: {e}")
                continue
        
        if not predictions:
            raise ValueError("所有模型预测都失败")
        
        # 加权平均
        valid_weights = np.array(valid_weights)
        valid_weights = valid_weights / valid_weights.sum()
        
        ensemble_pred = np.average(predictions, axis=0, weights=valid_weights)
        
        self.logger.info(f"集成预测完成，使用 {len(predictions)} 个模型")
        return ensemble_pred
    
    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """带不确定性的集成预测"""
        predictions = []
        
        for name, model in self.models.items():
            if not model.is_fitted:
                continue
            
            try:
                pred = model.predict(X)
                predictions.append(pred)
            except Exception as e:
                self.logger.warning(f"模型 {name} 预测失败: {e}")
                continue
        
        if not predictions:
            raise ValueError("所有模型预测都失败")
        
        predictions = np.array(predictions)
        
        # 计算加权平均和标准差
        weights = np.array([self.weights.get(name, 1.0) for name in self.models.keys() 
                           if name in self.models and self.models[name].is_fitted])
        weights = weights / weights.sum()
        
        mean_pred = np.average(predictions, axis=0, weights=weights)
        std_pred = np.std(predictions, axis=0)
        
        return mean_pred, std_pred


class RankingPostProcessor:
    """排名后处理器"""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
    
    def generate_rankings(self, scores: np.ndarray, groups: np.ndarray,
                         tie_breaker: Optional[np.ndarray] = None) -> np.ndarray:
        """根据分数生成排名"""
        unique_groups = np.unique(groups)
        rankings = np.zeros(len(scores), dtype=int)
        
        for group_id in unique_groups:
            group_mask = groups == group_id
            group_scores = scores[group_mask]
            group_indices = np.where(group_mask)[0]
            
            if tie_breaker is not None:
                group_tie_breaker = tie_breaker[group_mask]
                # 按分数降序，tie_breaker升序排序
                sort_indices = np.lexsort((group_tie_breaker, -group_scores))
            else:
                # 只按分数降序排序
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
                self.logger.error(f"组 {group_id} 排名无效: 期望 {expected_rankings}, 实际 {actual_rankings}")
                return False
        
        return True
    
    def apply_business_rules(self, scores: np.ndarray, features: pd.DataFrame,
                           groups: np.ndarray) -> np.ndarray:
        """应用业务规则调整分数"""
        adjusted_scores = scores.copy()
        
        # 示例业务规则：
        # 1. 价格太高的航班降低分数
        if 'total_price' in features.columns:
            high_price_mask = features['total_price'] > features['total_price'].quantile(0.9)
            adjusted_scores[high_price_mask] *= 0.9
        
        # 2. 时间太早或太晚的航班降低分数
        if 'departure_hour' in features.columns:
            early_mask = features['departure_hour'] < 6
            late_mask = features['departure_hour'] > 22
            adjusted_scores[early_mask | late_mask] *= 0.95
        
        # 3. 提升商务舱或头等舱分数
        if 'cabin_class' in features.columns:
            premium_mask = features['cabin_class'].isin(['Business', 'First'])
            adjusted_scores[premium_mask] *= 1.1
        
        self.logger.info("业务规则调整完成")
        return adjusted_scores


class FlightRankingPredictor:
    """航班排名高效预测器"""
    
    def __init__(self, 
                 data_path: str = "data/aeroclub-recsys-2025",
                 model_save_path: str = "models", 
                 output_path: str = "submissions",
                 use_gpu: bool = True,
                 enable_parallel: bool = True,
                 enable_business_rules: bool = False,
                 logger=None):
        
        self.data_path = Path(data_path + "/segmented")
        self.model_save_path = Path(model_save_path)
        self.output_path = Path(output_path)
        self.use_gpu = use_gpu
        self.enable_parallel = enable_parallel
        self.enable_business_rules = enable_business_rules
        
        # 设置logger
        self.logger = logger or self._setup_logger()
        
        # 初始化组件
        self.gpu_optimizer = GPUInferenceOptimizer(self.logger)
        self.post_processor = RankingPostProcessor(self.logger)
        
        # 获取最优批次大小
        self.batch_size = self.gpu_optimizer.get_optimal_batch_size()
        self.batch_processor = BatchProcessor(self.batch_size, self.logger)
        
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
    
    def load_segment_models(self, segment_id: int, 
                           model_names: List[str] = None) -> Dict[str, BaseRankingModel]:
        """加载数据段的模型"""
        segment_dir = self.model_save_path / f"segment_{segment_id}"
        
        if not segment_dir.exists():
            raise FileNotFoundError(f"段模型目录不存在: {segment_dir}")
        
        models = {}
        
        # 获取可用的模型文件
        available_models = [f.stem for f in segment_dir.glob("*.pkl") 
                           if f.stem != "features"]
        
        if model_names is None:
            model_names = available_models
        else:
            model_names = [name for name in model_names if name in available_models]
        
        # 加载模型
        for model_name in model_names:
            model_path = segment_dir / f"{model_name}.pkl"
            if model_path.exists():
                try:
                    model = BaseRankingModel.load_model(str(model_path))
                    models[model_name] = model
                    self.logger.info(f"✓ 加载模型: {model_name}")
                except Exception as e:
                    self.logger.warning(f"✗ 加载模型失败 {model_name}: {e}")
        
        if not models:
            raise ValueError(f"没有成功加载 segment_{segment_id} 的模型")
        
        return models
    
    def load_segment_features(self, segment_id: int) -> List[str]:
        """加载数据段的特征名称"""
        feature_path = self.model_save_path / f"segment_{segment_id}" / "features.pkl"
        
        if not feature_path.exists():
            raise FileNotFoundError(f"特征文件不存在: {feature_path}")
        
        with open(feature_path, 'rb') as f:
            features = pickle.load(f)
        
        self.logger.info(f"加载特征: {len(features)} 个")
        return features
    
    def predict_segment(self, segment_id: int, 
                       model_names: List[str] = None,
                       ensemble_weights: Dict[str, float] = None,
                       save_individual: bool = False) -> pd.DataFrame:
        """预测单个数据段"""
        self.logger.info(f"开始预测 segment_{segment_id}")
        start_time = time.time()
        
        # 加载测试数据
        test_file = self.data_path / "test" / f"test_segment_{segment_id}.parquet"
        if not test_file.exists():
            raise FileNotFoundError(f"测试文件不存在: {test_file}")
        
        df = pd.read_parquet(test_file)
        self.logger.info(f"加载测试数据: {df.shape}")
        
        # 加载模型和特征
        models = self.load_segment_models(segment_id, model_names)
        feature_names = self.load_segment_features(segment_id)
        
        # 数据预处理
        models_manager = FlightRankingModels(use_gpu=self.use_gpu, logger=self.logger)
        X, _, groups, _, processed_df = models_manager.prepare_data(df, target_col='selected')
        
        # 创建集成预测器
        ensemble = EnsemblePredictor(models, ensemble_weights, self.logger)
        
        # 批量预测
        predictions = self.batch_processor.process_in_batches(
            X, ensemble.predict, f"预测 segment_{segment_id}"
        )
        
        # 应用业务规则（如果启用）
        if self.enable_business_rules:
            predictions = self.post_processor.apply_business_rules(
                predictions, processed_df[feature_names], groups
            )
        
        # 生成排名
        rankings = self.post_processor.generate_rankings(
            predictions, groups, processed_df['Id'].values
        )
        
        # 验证排名
        if not self.post_processor.validate_rankings(rankings, groups):
            raise ValueError(f"segment_{segment_id} 排名验证失败")
        
        # 生成结果
        results = df[['Id', 'ranker_id']].copy()
        results['prediction_score'] = predictions
        results['selected'] = rankings
        
        # 保存结果
        prediction_time = time.time() - start_time
        
        if save_individual:
            output_file = self.output_path / f"predictions_segment_{segment_id}.csv"
            final_results = results[['Id', 'ranker_id', 'selected']]
            final_results.to_csv(output_file, index=False)
            self.logger.info(f"✓ 保存到: {output_file}")
        
        self.logger.info(f"✓ segment_{segment_id} 预测完成 (时间: {prediction_time:.1f}s)")
        
        return results[['Id', 'ranker_id', 'selected']]
    
    def predict_with_uncertainty(self, segment_id: int,
                                model_names: List[str] = None) -> pd.DataFrame:
        """带不确定性的预测"""
        self.logger.info(f"不确定性预测 segment_{segment_id}")
        
        # 加载数据和模型
        test_file = self.data_path / "test" / f"test_segment_{segment_id}.parquet"
        df = pd.read_parquet(test_file)
        
        models = self.load_segment_models(segment_id, model_names)
        
        # 数据预处理
        models_manager = FlightRankingModels(use_gpu=self.use_gpu, logger=self.logger)
        X, _, groups, _, _ = models_manager.prepare_data(df, target_col='selected')
        
        # 创建集成预测器
        ensemble = EnsemblePredictor(models, logger=self.logger)
        
        # 预测均值和标准差
        mean_pred, std_pred = ensemble.predict_with_uncertainty(X)
        
        # 生成结果
        results = df[['Id', 'ranker_id']].copy()
        results['prediction_mean'] = mean_pred
        results['prediction_std'] = std_pred
        results['confidence'] = 1.0 / (1.0 + std_pred)  # 简单的置信度度量
        
        # 根据均值生成排名
        rankings = self.post_processor.generate_rankings(
            mean_pred, groups, df['Id'].values
        )
        results['selected'] = rankings
        
        return results
    
    def predict_all_segments(self, segments: List[int] = None,
                           model_names: List[str] = None,
                           ensemble_weights: Dict[str, float] = None,
                           method: str = 'ensemble') -> pd.DataFrame:
        """预测所有数据段"""
        if segments is None:
            segments = [0, 1, 2]
        
        self.logger.info(f"开始预测所有段: {segments}")
        self.logger.info(f"预测方法: {method}")
        
        all_results = []
        total_start_time = time.time()
        
        def predict_single_segment(segment_id):
            """预测单个段的函数"""
            try:
                if method == 'uncertainty':
                    return self.predict_with_uncertainty(segment_id, model_names)
                else:
                    return self.predict_segment(
                        segment_id, model_names, ensemble_weights, save_individual=True
                    )
            except Exception as e:
                self.logger.error(f"✗ segment_{segment_id} 预测失败: {e}")
                return None
        
        # 并行或串行预测
        if self.enable_parallel and len(segments) > 1:
            self.logger.info("使用并行预测")
            results = Parallel(n_jobs=-1, backend='threading')(
                delayed(predict_single_segment)(segment_id) for segment_id in segments
            )
            all_results = [r for r in results if r is not None]
        else:
            for segment_id in segments:
                result = predict_single_segment(segment_id)
                if result is not None:
                    all_results.append(result)
        
        if not all_results:
            raise ValueError("所有段的预测都失败")
        
        # 合并结果
        final_submission = pd.concat(all_results, ignore_index=True)
        final_submission = final_submission.sort_values('Id').reset_index(drop=True)
        
        # 保存最终结果
        total_time = time.time() - total_start_time
        
        if method == 'uncertainty':
            output_file = self.output_path / "uncertainty_predictions.csv"
        else:
            output_file = self.output_path / "ensemble_final_submission.csv"
        
        final_submission.to_csv(output_file, index=False)
        
        # 生成预测报告
        self._generate_prediction_report(final_submission, segments, total_time, method)
        
        self.logger.info(f"✓ 所有预测完成 (总时间: {total_time:.1f}s)")
        self.logger.info(f"✓ 最终结果: {output_file}")
        self.logger.info(f"✓ 总记录数: {len(final_submission)}")
        
        return final_submission
    
    def _generate_prediction_report(self, results: pd.DataFrame, segments: List[int],
                                  total_time: float, method: str):
        """生成预测报告"""
        report = {
            'prediction_summary': {
                'method': method,
                'segments': segments,
                'total_samples': len(results),
                'total_time': total_time,
                'avg_time_per_sample': total_time / len(results) * 1000,  # ms
                'use_gpu': self.use_gpu,
                'batch_size': self.batch_size
            },
            'data_statistics': {
                'total_rankers': results['ranker_id'].nunique(),
                'avg_options_per_ranker': len(results) / results['ranker_id'].nunique(),
                'min_options': results.groupby('ranker_id').size().min(),
                'max_options': results.groupby('ranker_id').size().max()
            }
        }
        
        # 添加预测统计
        if 'prediction_score' in results.columns:
            report['prediction_statistics'] = {
                'score_mean': float(results['prediction_score'].mean()),
                'score_std': float(results['prediction_score'].std()),
                'score_min': float(results['prediction_score'].min()),
                'score_max': float(results['prediction_score'].max())
            }
        
        if 'prediction_std' in results.columns:
            report['uncertainty_statistics'] = {
                'uncertainty_mean': float(results['prediction_std'].mean()),
                'uncertainty_std': float(results['prediction_std'].std()),
                'avg_confidence': float(results['confidence'].mean())
            }
        
        # 保存报告
        report_path = self.output_path / f"prediction_report_{method}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"预测报告已保存: {report_path}")
    
    def benchmark_models(self, segment_id: int, 
                        model_names: List[str] = None) -> Dict:
        """模型性能基准测试"""
        self.logger.info(f"开始模型基准测试 segment_{segment_id}")
        
        # 加载数据和模型
        test_file = self.data_path / "test" / f"test_segment_{segment_id}.parquet"
        df = pd.read_parquet(test_file)
        
        models = self.load_segment_models(segment_id, model_names)
        
        # 数据预处理
        models_manager = FlightRankingModels(use_gpu=self.use_gpu, logger=self.logger)
        X, _, groups, _, _ = models_manager.prepare_data(df, target_col='selected')
        
        benchmark_results = {}
        
        # 测试每个模型
        for model_name, model in models.items():
            self.logger.info(f"基准测试: {model_name}")
            
            # 预热
            try:
                _ = model.predict(X[:100])
            except:
                pass
            
            # 计时测试
            times = []
            for _ in range(3):
                start_time = time.time()
                _ = model.predict(X)
                times.append(time.time() - start_time)
            
            benchmark_results[model_name] = {
                'avg_time': np.mean(times),
                'std_time': np.std(times),
                'throughput': len(X) / np.mean(times),  # samples/sec
                'model_type': type(model).__name__
            }
        
        # 保存基准测试结果
        benchmark_path = self.output_path / f"benchmark_segment_{segment_id}.json"
        with open(benchmark_path, 'w') as f:
            json.dump(benchmark_results, f, indent=2)
        
        # 打印结果
        self.logger.info("\n基准测试结果:")
        for name, stats in benchmark_results.items():
            self.logger.info(f"{name}: {stats['avg_time']:.3f}s, "
                           f"{stats['throughput']:.0f} samples/sec")
        
        return benchmark_results
    
    # 保持原有接口兼容性
    def predict_all(self, segments: List[int] = None, model_name: str = 'ensemble'):
        """兼容原有接口"""
        if model_name == 'ensemble':
            return self.predict_all_segments(segments)
        else:
            return self.predict_all_segments(segments, model_names=[model_name])
    
    def predict_all_with_ensemble(self, segments: List[int] = None, 
                                 models: List[str] = None):
        """兼容原有接口"""
        return self.predict_all_segments(segments, model_names=models)
=======
import pandas as pd
import numpy as np
from src.model.FlightRankingAnalyzer import FlightRankingAnalyzer
import joblib
from pathlib import Path

class FlightRankingPredictor:
    def __init__(self, data_path="data/aeroclub-recsys-2025/segmented", 
                 model_save_path="models", output_path="submissions",
                 use_gpu=False, random_state=42):
        self.data_path = Path(data_path)
        self.model_save_path = self.data_path / model_save_path
        self.output_path = self.data_path / output_path
        self.use_gpu = use_gpu
        self.random_state = random_state
        
        # 确保输出目录存在
        self.output_path.mkdir(parents=True, exist_ok=True)
    
    def predict_segment(self, segment_id, model_name='XGBRanker'):
        """预测单个数据段"""
        print(f"开始预测 segment_{segment_id}")
        
        # 检查模型文件是否存在
        model_path = self.model_save_path / f"{model_name}_segment_{segment_id}.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # 加载测试数据
        test_file = self.data_path / "test" / f"test_segment_{segment_id}.parquet"
        df = pd.read_parquet(test_file)
        print(f"测试数据形状: {df.shape}")
        
        # 初始化分析器
        analyzer = FlightRankingAnalyzer(use_gpu=self.use_gpu, random_state=self.random_state)
        
        # 加载特征名称
        feature_path = self.model_save_path / f"features_segment_{segment_id}.pkl"
        feature_cols = joblib.load(feature_path)
        analyzer.feature_names = feature_cols
        
        # 准备测试数据
        X, _, groups, _, df_processed = analyzer.prepare_data(df, target_col='selected')
        
        # 加载模型
        analyzer.load_model(str(model_path), model_name)
        print(f"已加载模型: {model_path}")
        
        # 预测排名
        ranks = analyzer.predict_ranks(X, groups, model_name)
        
        # 准备提交结果
        submission = pd.DataFrame({
            'Id': df['Id'],
            'ranker_id': df['ranker_id'],
            'selected': ranks
        })
        
        print(f"预测完成，结果形状: {submission.shape}")
        return submission
    
    def predict_all(self, segments=[0, 1, 2], model_name='XGBRanker'):
        """预测所有指定数据段并生成最终提交文件"""
        all_predictions = []
        
        for segment_id in segments:
            try:
                print(f"\n{'='*50}")
                prediction = self.predict_segment(segment_id, model_name)
                all_predictions.append(prediction)
                
                # 保存单个段的预测结果
                segment_output = self.output_path / f"{model_name}_segment_{segment_id}_prediction.csv"
                prediction.to_csv(segment_output, index=False)
                print(f"已保存预测结果: {segment_output}")
                
            except Exception as e:
                print(f"预测 segment_{segment_id} 失败: {e}")
                continue
        
        # 合并所有预测结果
        if not all_predictions:
            print("没有成功的预测结果")
            return None
        
        final_submission = pd.concat(all_predictions, ignore_index=True)
        
        # 按Id排序
        final_submission = final_submission.sort_values('Id').reset_index(drop=True)
        
        # 保存最终结果
        final_output = self.output_path / f"{model_name}_final_submission.csv"
        final_submission.to_csv(final_output, index=False)
        
        # 结果验证
        print(f"\n{'='*50}")
        print(f"预测完成!")
        print(f"最终提交文件: {final_output}")
        print(f"总记录数: {len(final_submission)}")
        print(f"唯一ranker_id数量: {final_submission['ranker_id'].nunique()}")
        
        self.validate_predictions(final_submission)
        return final_submission
    
    def validate_predictions(self, submission, sample_size=5):
        """验证预测结果的有效性"""
        print("\n验证预测结果:")
        unique_rankers = submission['ranker_id'].unique()
        
        # 随机抽样检查
        sample_rankers = np.random.choice(unique_rankers, min(sample_size, len(unique_rankers)), replace=False)
        
        for ranker_id in sample_rankers:
            group_data = submission[submission['ranker_id'] == ranker_id]
            ranks = sorted(group_data['selected'].values)
            expected_ranks = list(range(1, len(group_data) + 1))
            is_valid = ranks == expected_ranks
            print(f"ranker_id {ranker_id}: 排名{'有效' if is_valid else '无效'}")
>>>>>>> e9addff7041905ba228e279124f71a4a54b1d4e3
