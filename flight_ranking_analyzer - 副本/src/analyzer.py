"""
主分析器模块 - 修复排名重复问题版本

该模块整合所有组件，实现完整的航班排序分析功能
- 修复了排名重复问题
- 改进了排名唯一性保证机制
- 加强了结果验证

作者: Flight Ranking Team
版本: 3.1 (修复排名问题)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import gc
import os
import joblib
import tensorflow as tf
from typing import Dict, List, Optional, Union, Any, Tuple
import warnings
from collections import defaultdict

# 尝试相对导入，如果失败则使用绝对导入
try:
    from .models import ModelFactory, BaseRanker
    from .data_processor import DataProcessor, PredictionMerger
    from .auto_tuner import AutoTuner, create_auto_tuner
    from .config import Config
    from .progress_utils import (
        ProgressTracker, ModelTrainingProgress, 
        create_data_loading_progress, show_completion_summary,
        progress_bar
    )
    from .predictor import FlightRankingPredictor
except ImportError:
    from models import ModelFactory, BaseRanker
    from data_processor import DataProcessor, PredictionMerger
    from auto_tuner import AutoTuner, create_auto_tuner
    from config import Config
    from progress_utils import (
        ProgressTracker, ModelTrainingProgress, 
        create_data_loading_progress, show_completion_summary,
        progress_bar
    )
    from predictor import FlightRankingPredictor

warnings.filterwarnings('ignore')


class FlightRankingAnalyzer:
    """航班排序分析器 - 修复排名重复问题版本"""
    
    def __init__(self, 
                 use_gpu: bool = True, 
                 logger=None, 
                 selected_models: Optional[List[str]] = None,
                 enable_auto_tuning: bool = False,
                 auto_tuning_trials: int = 50,
                 save_models: bool = True):
        """
        初始化分析器
        
        Args:
            use_gpu: 是否使用GPU加速
            logger: 日志记录器
            selected_models: 要运行的模型名称列表
            enable_auto_tuning: 是否启用自动调参
            auto_tuning_trials: 自动调参试验次数
            save_models: 是否保存训练好的模型
        """
        self.logger = logger
        self.use_gpu = use_gpu
        self.selected_models = selected_models or Config.AVAILABLE_MODELS
        self.enable_auto_tuning = enable_auto_tuning
        self.auto_tuning_trials = auto_tuning_trials
        self.save_models = save_models
        
        # 初始化组件
        self.data_processor = DataProcessor(logger=logger)
        self.prediction_merger = PredictionMerger(logger=logger)
        
        # 初始化预测器
        self.predictor = FlightRankingPredictor(
            data_path=Config.DATA_BASE_PATH,
            use_gpu=use_gpu,
            logger=logger
        )
        
        # 存储训练结果
        self.feature_importance_results = {}
        self.shap_values = {}
        self.trained_models = {}
        self.model_performances = {}
        
        # 自动调参器（延迟初始化）
        self.auto_tuner = None
        
        self._log(f"初始化分析器完成, 选择的模型: {self.selected_models}")
        self._log(f"自动调参: {'启用' if enable_auto_tuning else '关闭'}")
        self._log(f"模型保存: {'启用' if save_models else '关闭'}")
    
    def _log(self, message):
        """记录日志"""
        if self.logger:
            self.logger.info(message)
        else:
            print(message)
    
    def calculate_hitrate_at_k(self, y_true: pd.DataFrame, y_pred_ranks: np.ndarray, k: int = 3) -> float:
        """
        计算HitRate@K指标
        
        Args:
            y_true: 包含ranker_id和selected列的真实标签
            y_pred_ranks: 预测排名
            k: Top-K参数
            
        Returns:
            float: HitRate@K分数
        """
        hits = 0
        total_queries = 0
        
        # 按ranker_id分组计算
        for ranker_id in y_true['ranker_id'].unique():
            group_mask = y_true['ranker_id'] == ranker_id
            group_true = y_true[group_mask]['selected'].values
            group_ranks = y_pred_ranks[group_mask]
            
            # 找到真实选择的航班位置
            true_idx = np.where(group_true == 1)[0]
            if len(true_idx) > 0:
                true_rank = group_ranks[true_idx[0]]
                if true_rank <= k:
                    hits += 1
            total_queries += 1
        
        return hits / total_queries if total_queries > 0 else 0
    
    def _calculate_group_ranks(self, scores: np.ndarray, group_sizes: List[int]) -> np.ndarray:
        """
        修复版本：确保排名唯一且连续
        
        Args:
            scores: 预测分数
            group_sizes: 每组的大小
            
        Returns:
            np.ndarray: 唯一且连续的排名
        """
        ranks = np.zeros_like(scores, dtype=int)
        start_idx = 0
        
        for group_idx, group_size in enumerate(group_sizes):
            end_idx = start_idx + group_size
            group_scores = scores[start_idx:end_idx]
            
            if group_size == 1:
                # 单个元素的组，排名直接为1
                ranks[start_idx:end_idx] = 1
            else:
                # 多个元素的组，需要确保排名唯一
                # 使用组索引和位置索引创建唯一的随机种子
                unique_seed = (group_idx * 12345 + start_idx) % 2147483647
                np.random.seed(unique_seed)
                
                # 添加微小但足够的随机噪声
                noise_scale = 1e-8  # 增加噪声强度
                noise = np.random.random(len(group_scores)) * noise_scale
                noisy_scores = group_scores + noise
                
                # 计算排名：分数越高，排名越靠前（rank=1最好）
                sorted_indices = np.argsort(-noisy_scores)  # 降序排列的索引
                group_ranks = np.zeros(group_size, dtype=int)
                
                # 分配唯一且连续的排名
                for rank, idx in enumerate(sorted_indices):
                    group_ranks[idx] = rank + 1
                
                ranks[start_idx:end_idx] = group_ranks
                
                # 验证排名的唯一性和连续性
                unique_ranks = set(group_ranks)
                expected_ranks = set(range(1, group_size + 1))
                if unique_ranks != expected_ranks:
                    # 如果仍有问题，强制修复
                    self._log(f"警告：组{group_idx}排名不唯一，强制修复")
                    ranks[start_idx:end_idx] = np.arange(1, group_size + 1)
            
            start_idx = end_idx
        
        return ranks
    
    def _validate_rankings(self, ranks: np.ndarray, group_sizes: List[int], 
                          context: str = "") -> bool:
        """
        验证排名的有效性
        
        Args:
            ranks: 排名数组
            group_sizes: 组大小列表
            context: 上下文信息用于日志
            
        Returns:
            bool: 排名是否有效
        """
        start_idx = 0
        valid = True
        
        for group_idx, group_size in enumerate(group_sizes):
            end_idx = start_idx + group_size
            group_ranks = ranks[start_idx:end_idx]
            
            # 检查排名是否唯一且连续
            unique_ranks = set(group_ranks)
            expected_ranks = set(range(1, group_size + 1))
            
            if unique_ranks != expected_ranks:
                self._log(f"排名验证失败 - {context} 组{group_idx}: "
                         f"期望{sorted(expected_ranks)}, 实际{sorted(unique_ranks)}")
                valid = False
            
            start_idx = end_idx
        
        if valid:
            self._log(f"排名验证通过 - {context}")
        
        return valid
    
    def _force_fix_rankings(self, ranks: np.ndarray, group_sizes: List[int]) -> np.ndarray:
        """
        强制修复排名问题
        
        Args:
            ranks: 原始排名
            group_sizes: 组大小列表
            
        Returns:
            np.ndarray: 修复后的排名
        """
        fixed_ranks = ranks.copy()
        start_idx = 0
        
        for group_idx, group_size in enumerate(group_sizes):
            end_idx = start_idx + group_size
            group_ranks = fixed_ranks[start_idx:end_idx]
            
            # 检查是否需要修复
            unique_ranks = set(group_ranks)
            expected_ranks = set(range(1, group_size + 1))
            
            if unique_ranks != expected_ranks:
                # 强制分配连续排名
                # 使用组ID作为随机种子确保可重复性
                np.random.seed(group_idx * 54321)
                new_ranks = np.random.permutation(range(1, group_size + 1))
                fixed_ranks[start_idx:end_idx] = new_ranks
                
                self._log(f"强制修复组{group_idx}的排名")
            
            start_idx = end_idx
        
        return fixed_ranks
        
    def train_models(self, X_train: np.ndarray, X_test: np.ndarray, 
                    y_train: np.ndarray, y_test: np.ndarray,
                    train_group_sizes: List[int], test_group_sizes: List[int],
                    feature_names: List[str], test_info: pd.DataFrame,
                    segment_name: str = "", segment_id: int = None) -> pd.DataFrame:
        """
        训练所有选择的模型（修复排名问题版本）
        
        Args:
            X_train, X_test: 训练和测试特征
            y_train, y_test: 训练和测试标签
            train_group_sizes, test_group_sizes: 组大小信息
            feature_names: 特征名称列表
            test_info: 测试集的ranker_id和selected信息
            segment_name: 数据段名称
            segment_id: 数据段ID（用于保存模型）
            
        Returns:
            pd.DataFrame: 模型性能比较结果
        """
        self._log(f"\n开始训练模型 - {segment_name}")
        
        # 如果启用自动调参，先进行参数优化
        best_params_dict = {}
        if self.enable_auto_tuning:
            self._log("🎯 执行自动调参...")
            with create_data_loading_progress("自动调参") as pbar:
                best_params_dict = self._perform_auto_tuning(
                    X_train, y_train, train_group_sizes, 
                    X_test, y_test, test_info, 
                    input_dim=X_train.shape[1]
                )
                pbar.update(1)
        
        model_results = []
        
        # 创建模型训练进度显示
        training_progress = ModelTrainingProgress(self.selected_models)
        
        with training_progress.training_session() as trainer:
            for model_name in progress_bar(self.selected_models, desc="训练模型"):
                self._log(f"\n🔧 训练 {model_name}...")
                
                try:
                    # 开始当前模型训练
                    trainer.start_model(model_name, steps=7)  # 增加验证步骤
                    
                    # 步骤1: 获取模型参数
                    trainer.update_model_progress(1, f"获取 {model_name} 参数")
                    if model_name in best_params_dict:
                        model_params = best_params_dict[model_name]
                        self._log(f"📋 使用调优参数: {model_params}")
                    else:
                        model_params = Config.DEFAULT_MODEL_PARAMS.get(model_name, {})
                        self._log(f"📋 使用默认参数")
                    
                    # 步骤2: 创建模型
                    trainer.update_model_progress(1, f"创建 {model_name} 模型")
                    input_dim = X_train.shape[1] if model_name == 'NeuralRanker' else None
                    model = ModelFactory.create_model(
                        model_name=model_name,
                        use_gpu=self.use_gpu,
                        input_dim=input_dim,
                        **model_params
                    )
                    
                    # 步骤3: 训练模型
                    trainer.update_model_progress(1, f"训练 {model_name} 模型")
                    if model_name == 'NeuralRanker':
                        epochs = model_params.get('epochs', 10)
                        batch_size = model_params.get('batch_size', 32)
                        model.fit(X_train, y_train, group=train_group_sizes, 
                                 epochs=epochs, batch_size=batch_size)
                    else:
                        model.fit(X_train, y_train, group=train_group_sizes)
                    
                    # 步骤4: 预测
                    trainer.update_model_progress(1, f"执行 {model_name} 预测")
                    y_pred_scores = model.predict(X_test)
                    
                    # 步骤5: 计算排名（关键修复）
                    trainer.update_model_progress(1, f"计算 {model_name} 排名")
                    y_pred_ranks = self._calculate_group_ranks(y_pred_scores, test_group_sizes)
                    
                    # 步骤6: 验证排名（新增）
                    trainer.update_model_progress(1, f"验证 {model_name} 排名")
                    is_valid = self._validate_rankings(
                        y_pred_ranks, test_group_sizes, f"{model_name}-{segment_name}"
                    )
                    
                    if not is_valid:
                        self._log(f"⚠️ {model_name} 排名验证失败，强制修复...")
                        y_pred_ranks = self._force_fix_rankings(y_pred_ranks, test_group_sizes)
                        # 再次验证
                        self._validate_rankings(
                            y_pred_ranks, test_group_sizes, f"{model_name}-{segment_name}-修复后"
                        )
                    
                    # 计算性能
                    hitrate_3 = self.calculate_hitrate_at_k(test_info, y_pred_ranks, k=3)
                    
                    # 步骤7: 保存模型和特征
                    if self.save_models and segment_id is not None:
                        trainer.update_model_progress(1, f"保存 {model_name} 模型")
                        try:
                            self.predictor.save_model_and_features(
                                model=model.model if hasattr(model, 'model') else model,
                                model_name=model_name,
                                segment_id=segment_id,
                                feature_names=feature_names,
                                performance=hitrate_3
                            )
                            self._log(f"✅ 已保存模型: {model_name}_segment_{segment_id}")
                        except Exception as e:
                            self._log(f"⚠️ 保存模型失败: {str(e)}")
                    else:
                        trainer.update_model_progress(1, f"跳过保存 {model_name}")
                    
                    # 存储结果
                    model_results.append({
                        'Model': model_name,
                        'HitRate@3': hitrate_3,
                        'Segment': segment_name
                    })
                    
                    # 存储训练好的模型（内存中）
                    model_key = f"{segment_name}_{model_name}" if segment_name else model_name
                    self.trained_models[model_key] = {
                        'model': model,
                        'params': model_params,
                        'performance': hitrate_3,
                        'feature_names': feature_names
                    }
                    
                    # 获取特征重要性
                    self._extract_feature_importance(model, model_name, X_test, feature_names)
                    
                    # 计算SHAP值
                    self._compute_shap_values(model, model_name, X_test, feature_names)
                    
                    # 完成当前模型
                    trainer.finish_model(model_name, hitrate_3)
                    self._log(f"✅ {model_name} 训练完成, HitRate@3: {hitrate_3:.4f}")
                    
                except Exception as e:
                    self._log(f"❌ 训练 {model_name} 时出错: {str(e)}")
                    trainer.finish_model(model_name, 0.0)
                    continue
        
        # 显示模型性能比较
        results_df = pd.DataFrame(model_results)
        if not results_df.empty:
            self._log("\n📊 模型性能比较:")
            print(results_df.to_string(index=False))
            
            # 显示完成总结
            summary = {
                "训练模型数": len(results_df),
                "最佳模型": results_df.loc[results_df['HitRate@3'].idxmax(), 'Model'],
                "最佳性能": results_df['HitRate@3'].max(),
                "平均性能": results_df['HitRate@3'].mean()
            }
            show_completion_summary(summary, f"{segment_name} 训练完成")
        
        return results_df
    
    def _perform_auto_tuning(self, X_train: np.ndarray, y_train: np.ndarray, 
                           train_group_sizes: List[int], X_val: np.ndarray, 
                           y_val: np.ndarray, val_info: pd.DataFrame,
                           input_dim: int) -> Dict[str, Dict[str, Any]]:
        """执行自动调参"""
        self._log("\n开始自动调参...")
        
        # 初始化自动调参器
        if self.auto_tuner is None:
            self.auto_tuner = create_auto_tuner(
                model_factory=ModelFactory.create_model,
                hitrate_calculator=self.calculate_hitrate_at_k,
                n_trials=self.auto_tuning_trials,
                timeout=Config.AUTO_TUNING_TIMEOUT
            )
        
        # 为每个模型进行调参
        best_params = {}
        for model_name in self.selected_models:
            if model_name == 'BM25Ranker':
                continue  # BM25不需要调参
            
            try:
                result = self.auto_tuner.optimize(
                    model_name=model_name,
                    X_train=X_train,
                    y_train=y_train,
                    train_groups=train_group_sizes,
                    X_val=X_val,
                    y_val=y_val,
                    val_info=val_info,
                    use_gpu=self.use_gpu,
                    input_dim=input_dim
                )
                best_params[model_name] = result['best_params']
                self._log(f"{model_name} 调参完成, 最佳HitRate@3: {result['best_score']:.4f}")
            except Exception as e:
                self._log(f"{model_name} 调参失败: {str(e)}")
        
        return best_params
    
    def _extract_feature_importance(self, model: BaseRanker, model_name: str, 
                                  X_test: np.ndarray, feature_names: List[str]):
        """提取特征重要性"""
        try:
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                self.feature_importance_results[model_name] = {
                    'importance': importance,
                    'feature_names': feature_names
                }
            elif model_name == 'NeuralRanker':
                # 对于神经网络，使用梯度作为特征重要性
                importance = self._compute_neural_importance(model, X_test)
                self.feature_importance_results[model_name] = {
                    'importance': importance,
                    'feature_names': feature_names
                }
        except Exception as e:
            self._log(f"提取 {model_name} 特征重要性失败: {str(e)}")
    
    def _compute_neural_importance(self, model: BaseRanker, X: np.ndarray, 
                                 samples: int = 1000) -> np.ndarray:
        """计算神经网络的特征重要性"""
        if samples < len(X):
            X_sample = X[np.random.choice(len(X), samples, replace=False)]
        else:
            X_sample = X
        
        X_tensor = tf.convert_to_tensor(X_sample, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            tape.watch(X_tensor)
            predictions = model.model(X_tensor)
        
        grads = tape.gradient(predictions, X_tensor)
        importance = np.mean(np.abs(grads.numpy()), axis=0)
        
        return importance
    
    def _compute_shap_values(self, model: BaseRanker, model_name: str, 
                           X_test: np.ndarray, feature_names: List[str]):
        """计算SHAP值"""
        if len(X_test) > Config.MAX_SHAP_SAMPLES or model_name == 'BM25Ranker':
            return
        
        try:
            self._log(f"计算 {model_name} 的SHAP值...")
            sample_idx = np.random.choice(
                len(X_test), 
                min(Config.MAX_SHAP_SAMPLES, len(X_test)), 
                replace=False
            )
            X_sample = X_test[sample_idx]
            
            if model_name in ['XGBRanker', 'LGBMRanker', 'LambdaMART', 'ListNet']:
                explainer = shap.TreeExplainer(model.model)
                shap_values = explainer.shap_values(X_sample)
            elif model_name == 'NeuralRanker':
                explainer = shap.DeepExplainer(model.model, X_sample)
                shap_values = explainer.shap_values(X_sample)[0]
            else:
                return
            
            self.shap_values[model_name] = {
                'values': shap_values,
                'data': X_sample,
                'feature_names': feature_names
            }
            
            # 可视化SHAP摘要图
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
            plt.title(f'{model_name} SHAP Feature Importance')
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            self._log(f"计算 {model_name} SHAP值失败: {str(e)}")
    
    def predict_with_saved_models(self, segments: List[int] = None, 
                                 model_names: List[str] = None,
                                 ensemble_method: str = 'average') -> Optional[pd.DataFrame]:
        """
        使用保存的模型进行预测（修复排名问题版本）
        
        Args:
            segments: 要预测的数据段列表
            model_names: 要使用的模型名称列表
            ensemble_method: 集成方法
            
        Returns:
            Optional[pd.DataFrame]: 预测结果
        """
        self._log("使用保存的模型进行预测...")
        
        # 检查可用模型
        available_models = self.predictor.get_available_models()
        self._log(f"可用模型: {available_models}")
        
        if not available_models:
            self._log("没有找到保存的模型，请先训练模型")
            return None
        
        # 使用预测器进行预测
        result = self.predictor.predict_all(
            segments=segments,
            model_names=model_names or list(available_models.keys()),
            ensemble_method=ensemble_method
        )
        
        # 验证最终结果的排名唯一性
        if result is not None:
            self._log("验证最终预测结果的排名唯一性...")
            fixed_result = self.predictor._validate_and_fix_rankings(result)
            return fixed_result
        
        return result
    
    def analyze_feature_importance(self) -> Optional[pd.Series]:
        """
        分析特征重要性
        
        Returns:
            Optional[pd.Series]: 平均特征重要性（按重要性降序排列）
        """
        if not self.feature_importance_results:
            self._log("没有可用的特征重要性数据")
            return None
        
        self._log("\n分析特征重要性...")
        
        # 收集所有模型的特征重要性
        importance_dfs = []
        
        for model_name, result in self.feature_importance_results.items():
            importance = result['importance']
            feature_names = result['feature_names']
            
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance,
                'Model': model_name
            })
            importance_dfs.append(importance_df)
        
        all_importance = pd.concat(importance_dfs)
        
        # 计算平均重要性
        avg_importance = all_importance.groupby('Feature')['Importance'].mean().sort_values(ascending=False)
        
        # 可视化特征重要性
        plt.figure(figsize=(12, 10))
        top_30 = all_importance.nlargest(30, 'Importance')
        sns.barplot(data=top_30, x='Importance', y='Feature', hue='Model')
        plt.title('Top 30 Feature Importance Across Models')
        plt.tight_layout()
        plt.show()
        
        # 可视化平均特征重要性
        plt.figure(figsize=(12, 10))
        avg_importance.head(30).sort_values().plot(kind='barh')
        plt.title('Top 30 Average Feature Importance Across Models')
        plt.tight_layout()
        plt.show()
        
        return avg_importance
    
    def full_analysis(self, file_path: str, use_sampling: bool = True,
                     num_groups: int = 2000, min_group_size: int = 20) -> Dict[str, Any]:
        """
        完整的分析流程（修复排名问题版本）
        
        Args:
            file_path: 数据文件路径
            use_sampling: 是否使用抽样
            num_groups: 抽样组数
            min_group_size: 最小组大小
            
        Returns:
            Dict: 包含所有分析结果的字典
        """
        # 1. 加载数据
        df = self.data_processor.load_data(
            file_path, 
            use_sampling=use_sampling,
            num_groups=num_groups, 
            min_group_size=min_group_size
        )
        
        # 2. 准备排序数据
        (X_train, X_test, y_train, y_test, 
         train_group_sizes, test_group_sizes, 
         feature_cols, test_info) = self.data_processor.prepare_ranking_data(df)
        
        # 3. 生成一致的segment名称和ID
        file_basename = os.path.basename(file_path)
        segment_id = None
        
        # 从文件名中提取segment ID
        import re
        match = re.search(r'segment[_\s]*(\d+)', file_basename)
        if match:
            segment_id = int(match.group(1))
        
        # 移除文件扩展名，统一格式
        if file_basename.endswith('.parquet'):
            segment_name = file_basename[:-8]  # 移除 .parquet
        else:
            segment_name = file_basename
        
        # 确保segment名称格式一致
        if not segment_name.startswith('train_segment_'):
            if segment_id is not None:
                segment_name = f'train_segment_{segment_id}'
            else:
                segment_name = f'train_segment_unknown'
        
        self._log(f"使用segment名称: {segment_name}, ID: {segment_id}")
        
        # 4. 训练排序模型（修复排名问题版本）
        model_results = self.train_models(
            X_train, X_test, y_train, y_test,
            train_group_sizes, test_group_sizes,
            feature_cols, test_info, segment_name, segment_id
        )
        
        # 5. 分析特征重要性
        avg_importance = self.analyze_feature_importance()
        
        # 6. 清理内存
        gc.collect()
        
        return {
            'model_results': model_results,
            'feature_importance': avg_importance,
            'trained_models': self.trained_models,
            'segment_name': segment_name,
            'segment_id': segment_id
        }
    
    # 保留原有的合并方法以兼容旧代码
    def merge_all_predictions(self, prediction_files: List[str], 
                            submission_file: str, 
                            output_file: str) -> str:
        """
        合并所有预测文件（兼容方法）
        
        Args:
            prediction_files: 预测文件路径列表
            submission_file: submission模板文件路径
            output_file: 输出文件路径
            
        Returns:
            str: 输出文件路径
        """
        return self.prediction_merger.merge_predictions(
            prediction_files=prediction_files,
            submission_file=submission_file,
            output_file=output_file,
            ensemble_method='average'
        )