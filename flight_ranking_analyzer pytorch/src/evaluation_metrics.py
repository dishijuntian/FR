"""
评估指标模块 - 重构版
统一管理所有评估指标计算，移除重复定义

作者: Flight Ranking Team
版本: 5.0 (重构版)
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod


class BaseMetric(ABC):
    """评估指标基类"""
    
    @abstractmethod
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray, 
                 group_sizes: List[int], **kwargs) -> float:
        """计算指标"""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """指标名称"""
        pass


class HitRateMetric(BaseMetric):
    """Hit Rate@K指标"""
    
    def __init__(self, k: int = 3):
        self.k = k
    
    @property
    def name(self) -> str:
        return f"HitRate@{self.k}"
    
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray, 
                 group_sizes: List[int], **kwargs) -> float:
        """计算Hit Rate@K"""
        hits = 0
        total_groups = 0
        start_idx = 0
        
        for group_size in group_sizes:
            end_idx = start_idx + group_size
            
            if end_idx > len(y_true):
                break
                
            group_pred = y_pred[start_idx:end_idx]
            group_true = y_true[start_idx:end_idx]
            
            # 找到正样本的索引
            positive_indices = np.where(group_true == 1)[0]
            
            if len(positive_indices) > 0:
                # 获取预测分数最高的K个样本的索引
                top_k_indices = np.argsort(group_pred)[-self.k:]
                # 检查top-K中是否包含正样本
                if any(idx in positive_indices for idx in top_k_indices):
                    hits += 1
            
            total_groups += 1
            start_idx = end_idx
        
        return hits / total_groups if total_groups > 0 else 0.0


class NDCGMetric(BaseMetric):
    """NDCG@K指标"""
    
    def __init__(self, k: int = 5):
        self.k = k
    
    @property
    def name(self) -> str:
        return f"NDCG@{self.k}"
    
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray, 
                 group_sizes: List[int], **kwargs) -> float:
        """计算NDCG@K"""
        try:
            from sklearn.metrics import ndcg_score
        except ImportError:
            # 如果sklearn不可用，返回简化版本
            return self._simple_ndcg(y_true, y_pred, group_sizes)
        
        ndcg_scores = []
        start_idx = 0
        
        for group_size in group_sizes:
            if group_size < self.k:
                start_idx += group_size
                continue
                
            end_idx = start_idx + group_size
            
            if end_idx > len(y_true):
                break
                
            group_true = y_true[start_idx:end_idx].reshape(1, -1)
            group_pred = y_pred[start_idx:end_idx].reshape(1, -1)
            
            try:
                ndcg = ndcg_score(group_true, group_pred, k=self.k)
                ndcg_scores.append(ndcg)
            except Exception:
                # 如果计算失败，跳过这个组
                pass
            
            start_idx = end_idx
        
        return np.mean(ndcg_scores) if ndcg_scores else 0.0
    
    def _simple_ndcg(self, y_true: np.ndarray, y_pred: np.ndarray, 
                    group_sizes: List[int]) -> float:
        """简化版NDCG计算（不依赖sklearn）"""
        def dcg_at_k(r, k):
            r = np.asfarray(r)[:k]
            if r.size:
                return np.sum(r / np.log2(np.arange(2, r.size + 2)))
            return 0.0
        
        ndcg_scores = []
        start_idx = 0
        
        for group_size in group_sizes:
            if group_size < self.k:
                start_idx += group_size
                continue
                
            end_idx = start_idx + group_size
            
            if end_idx > len(y_true):
                break
                
            group_true = y_true[start_idx:end_idx]
            group_pred = y_pred[start_idx:end_idx]
            
            # 按预测分数排序
            sorted_indices = np.argsort(group_pred)[::-1]
            sorted_relevance = group_true[sorted_indices]
            
            # 计算DCG和IDCG
            dcg = dcg_at_k(sorted_relevance, self.k)
            idcg = dcg_at_k(sorted(group_true, reverse=True), self.k)
            
            if idcg > 0:
                ndcg_scores.append(dcg / idcg)
            
            start_idx = end_idx
        
        return np.mean(ndcg_scores) if ndcg_scores else 0.0


class PrecisionMetric(BaseMetric):
    """Precision@K指标"""
    
    def __init__(self, k: int = 3):
        self.k = k
    
    @property
    def name(self) -> str:
        return f"Precision@{self.k}"
    
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray, 
                 group_sizes: List[int], **kwargs) -> float:
        """计算Precision@K"""
        precisions = []
        start_idx = 0
        
        for group_size in group_sizes:
            end_idx = start_idx + group_size
            
            if end_idx > len(y_true):
                break
                
            group_pred = y_pred[start_idx:end_idx]
            group_true = y_true[start_idx:end_idx]
            
            # 获取top-K预测
            top_k_indices = np.argsort(group_pred)[-self.k:]
            top_k_true = group_true[top_k_indices]
            
            # 计算精确率
            precision = np.sum(top_k_true) / self.k
            precisions.append(precision)
            
            start_idx = end_idx
        
        return np.mean(precisions) if precisions else 0.0


class MAPMetric(BaseMetric):
    """Mean Average Precision指标"""
    
    @property
    def name(self) -> str:
        return "MAP"
    
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray, 
                 group_sizes: List[int], **kwargs) -> float:
        """计算MAP"""
        average_precisions = []
        start_idx = 0
        
        for group_size in group_sizes:
            end_idx = start_idx + group_size
            
            if end_idx > len(y_true):
                break
                
            group_pred = y_pred[start_idx:end_idx]
            group_true = y_true[start_idx:end_idx]
            
            # 按预测分数排序
            sorted_indices = np.argsort(group_pred)[::-1]
            sorted_true = group_true[sorted_indices]
            
            # 计算Average Precision
            num_relevant = np.sum(sorted_true)
            if num_relevant == 0:
                start_idx = end_idx
                continue
            
            precision_at_k = []
            num_relevant_seen = 0
            
            for i, relevance in enumerate(sorted_true):
                if relevance == 1:
                    num_relevant_seen += 1
                    precision_at_k.append(num_relevant_seen / (i + 1))
            
            if precision_at_k:
                average_precisions.append(np.mean(precision_at_k))
            
            start_idx = end_idx
        
        return np.mean(average_precisions) if average_precisions else 0.0


class MetricsCalculator:
    """指标计算器"""
    
    def __init__(self):
        self.metrics = {
            'hit_rate_3': HitRateMetric(k=3),
            'hit_rate_5': HitRateMetric(k=5),
            'ndcg_3': NDCGMetric(k=3),
            'ndcg_5': NDCGMetric(k=5),
            'precision_3': PrecisionMetric(k=3),
            'precision_5': PrecisionMetric(k=5),
            'map': MAPMetric()
        }
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         group_sizes: List[int], 
                         metrics: Optional[List[str]] = None) -> Dict[str, float]:
        """计算多个指标"""
        if metrics is None:
            metrics = ['hit_rate_3', 'ndcg_5']
        
        results = {}
        for metric_name in metrics:
            if metric_name in self.metrics:
                try:
                    value = self.metrics[metric_name].calculate(y_true, y_pred, group_sizes)
                    results[metric_name] = value
                except Exception as e:
                    print(f"计算指标 {metric_name} 失败: {e}")
                    results[metric_name] = 0.0
            else:
                print(f"未知指标: {metric_name}")
        
        return results
    
    def add_custom_metric(self, name: str, metric: BaseMetric):
        """添加自定义指标"""
        self.metrics[name] = metric
    
    def get_available_metrics(self) -> List[str]:
        """获取可用指标列表"""
        return list(self.metrics.keys())


class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, metrics: Optional[List[str]] = None):
        self.calculator = MetricsCalculator()
        self.default_metrics = metrics or ['hit_rate_3', 'ndcg_5']
    
    def evaluate_model(self, model, X_test: np.ndarray, y_test: np.ndarray,
                      group_sizes: List[int], test_info: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """评估单个模型"""
        try:
            # 预测
            if hasattr(model, 'predict'):
                y_pred = model.predict(X_test)
            else:
                raise ValueError("模型没有predict方法")
            
            # 计算指标
            results = self.calculator.calculate_metrics(y_test, y_pred, group_sizes, self.default_metrics)
            
            return results
            
        except Exception as e:
            print(f"模型评估失败: {e}")
            return {metric: 0.0 for metric in self.default_metrics}
    
    def evaluate_models(self, models: Dict[str, Any], X_test: np.ndarray, 
                       y_test: np.ndarray, group_sizes: List[int],
                       test_info: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """评估多个模型"""
        results = []
        
        for model_name, model in models.items():
            print(f"评估模型: {model_name}")
            model_results = self.evaluate_model(model, X_test, y_test, group_sizes, test_info)
            
            result_row = {'Model': model_name}
            result_row.update(model_results)
            results.append(result_row)
        
        return pd.DataFrame(results)
    
    def compare_models(self, results_df: pd.DataFrame, 
                      primary_metric: str = 'hit_rate_3') -> Dict[str, Any]:
        """比较模型性能"""
        if results_df.empty:
            return {}
        
        # 找到最佳模型
        best_idx = results_df[primary_metric].idxmax()
        best_model = results_df.loc[best_idx]
        
        # 计算统计信息
        comparison = {
            'best_model': best_model['Model'],
            'best_score': best_model[primary_metric],
            'mean_score': results_df[primary_metric].mean(),
            'std_score': results_df[primary_metric].std(),
            'score_range': results_df[primary_metric].max() - results_df[primary_metric].min()
        }
        
        return comparison


# 便捷函数
def calculate_hit_rate(y_true: np.ndarray, y_pred: np.ndarray, 
                      group_sizes: List[int], k: int = 3) -> float:
    """计算Hit Rate@K的便捷函数"""
    metric = HitRateMetric(k=k)
    return metric.calculate(y_true, y_pred, group_sizes)


def calculate_ndcg(y_true: np.ndarray, y_pred: np.ndarray, 
                  group_sizes: List[int], k: int = 5) -> float:
    """计算NDCG@K的便捷函数"""
    metric = NDCGMetric(k=k)
    return metric.calculate(y_true, y_pred, group_sizes)