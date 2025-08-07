"""
航班排序分析器 - 完整版本

该模块提供完整的航班排序分析功能，包括：
- 多种排序模型训练和评估
- 自动超参数调优
- 特征重要性分析
- SHAP可解释性分析
- 预测结果合并

作者: Flight Ranking Team
版本: 2.2 (修复版)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import tensorflow as tf
import gc
import os
import warnings
from typing import Dict, List, Optional, Tuple, Any
from sklearn.metrics import ndcg_score
import joblib

# 导入本地模块
try:
    from .config import Config
    from .models import ModelFactory, BaseRanker
    from .data_processor import DataProcessor, PredictionMerger
    from .auto_tuner import AutoTuner, create_auto_tuner
    from .predictor import FlightRankingPredictor
    from .progress_utils import progress_bar, ModelTrainingProgress, show_completion_summary
except ImportError:
    from config import Config
    from models import ModelFactory, BaseRanker
    from data_processor import DataProcessor, PredictionMerger
    from auto_tuner import AutoTuner, create_auto_tuner
    from predictor import FlightRankingPredictor
    from progress_utils import progress_bar, ModelTrainingProgress, show_completion_summary

warnings.filterwarnings('ignore')


class FlightRankingAnalyzer:
    """航班排序分析器主类"""
    
    def __init__(self, 
                 use_gpu: bool = True,
                 logger=None,
                 selected_models: List[str] = None,
                 enable_auto_tuning: bool = False,
                 auto_tuning_trials: int = 50,
                 save_models: bool = True):
        """
        初始化分析器
        
        Args:
            use_gpu: 是否使用GPU
            logger: 日志记录器
            selected_models: 选择的模型列表
            enable_auto_tuning: 是否启用自动调参
            auto_tuning_trials: 自动调参试验次数
            save_models: 是否保存模型
        """
        self.use_gpu = use_gpu
        self.logger = logger
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
        
        # 存储结果
        self.model_results = {}
        self.feature_importance_results = {}
        self.shap_values = {}
        self.trained_models = {}
        
        # 当前段信息
        self.current_segment_id = 0
        
        self._log("FlightRankingAnalyzer 初始化完成")
    
    def _log(self, message: str):
        """记录日志"""
        if self.logger:
            self.logger.info(message)
        else:
            print(message)
    
    def full_analysis(self, train_file_path: str, 
                     use_sampling: bool = True,
                     num_groups: int = 2000, 
                     min_group_size: int = 20) -> Dict[str, Any]:
        """
        执行完整的分析流程
        
        Args:
            train_file_path: 训练文件路径
            use_sampling: 是否使用抽样
            num_groups: 抽样组数
            min_group_size: 最小组大小
            
        Returns:
            Dict: 分析结果
        """
        self._log(f"开始完整分析: {os.path.basename(train_file_path)}")
        
        try:
            # 1. 加载和准备数据
            self._log("步骤1: 加载和准备数据")
            df = self.data_processor.load_data(
                train_file_path, 
                use_sampling=use_sampling,
                num_groups=num_groups,
                min_group_size=min_group_size
            )
            
            # 2. 准备排序数据
            self._log("步骤2: 准备排序数据")
            data_tuple = self.data_processor.prepare_ranking_data(df)
            (X_train, X_test, y_train, y_test, 
             train_group_sizes, test_group_sizes, 
             feature_cols, test_info) = data_tuple
            
            # 3. 训练和评估模型
            self._log("步骤3: 训练和评估模型")
            model_results = self._train_and_evaluate_models(
                X_train, X_test, y_train, y_test,
                train_group_sizes, test_group_sizes,
                feature_cols, test_info
            )
            
            # 4. 特征重要性分析
            self._log("步骤4: 特征重要性分析")
            feature_importance = self._analyze_feature_importance(feature_cols)
            
            # 5. 生成可视化
            self._log("步骤5: 生成可视化")
            self._generate_visualizations(model_results, feature_importance)
            
            # 6. 保存结果
            self._log("步骤6: 保存结果")
            results = {
                'model_results': model_results,
                'feature_importance': feature_importance,
                'data_shape': df.shape,
                'feature_count': len(feature_cols)
            }
            
            self._save_analysis_results(results)
            
            self._log("完整分析完成")
            return results
            
        except Exception as e:
            self._log(f"分析过程中出错: {str(e)}")
            raise
        finally:
            # 清理内存
            gc.collect()
    
    def _train_and_evaluate_models(self, X_train: np.ndarray, X_test: np.ndarray,
                                  y_train: np.ndarray, y_test: np.ndarray,
                                  train_group_sizes: List[int], 
                                  test_group_sizes: List[int],
                                  feature_cols: List[str],
                                  test_info: pd.DataFrame) -> pd.DataFrame:
        """训练和评估所有模型"""
        results = []
        
        # 创建训练进度显示
        with ModelTrainingProgress(self.selected_models).training_session() as training_ctx:
            
            for model_name in self.selected_models:
                try:
                    self._log(f"训练模型: {model_name}")
                    training_ctx.start_model(model_name)
                    
                    # 创建模型
                    model_params = Config.DEFAULT_MODEL_PARAMS.get(model_name, {}).copy()
                    
                    # 自动调参
                    if self.enable_auto_tuning and model_name in Config.TUNING_SEARCH_SPACES:
                        self._log(f"对 {model_name} 执行自动调参...")
                        tuner = create_auto_tuner(
                            model_name=model_name,
                            search_space=Config.TUNING_SEARCH_SPACES[model_name],
                            n_trials=self.auto_tuning_trials
                        )
                        
                        best_params = tuner.optimize(
                            X_train, y_train, train_group_sizes,
                            X_test, y_test, test_group_sizes
                        )
                        model_params.update(best_params)
                        training_ctx.update_model_progress(1, f"自动调参完成: {model_name}")
                    
                    # 创建模型实例
                    if model_name in ['NeuralRanker', 'RankNet', 'TransformerRanker']:
                        model = ModelFactory.create_model(
                            model_name, 
                            use_gpu=self.use_gpu,
                            input_dim=X_train.shape[1],
                            **model_params
                        )
                    else:
                        model = ModelFactory.create_model(
                            model_name, 
                            use_gpu=self.use_gpu,
                            **model_params
                        )
                    
                    # 训练模型
                    model.fit(X_train, y_train, train_group_sizes)
                    training_ctx.update_model_progress(1, f"训练完成: {model_name}")
                    
                    # 预测
                    y_pred = model.predict(X_test)
                    
                    # 评估
                    hit_rate = self._calculate_hit_rate(y_test, y_pred, test_group_sizes, test_info)
                    ndcg = self._calculate_ndcg(y_test, y_pred, test_group_sizes)
                    
                    # 存储模型和结果
                    self.trained_models[model_name] = model
                    
                    results.append({
                        'Model': model_name,
                        'HitRate@3': hit_rate,
                        'NDCG': ndcg,
                        'Parameters': str(model_params)
                    })
                    
                    # 提取特征重要性
                    self._extract_feature_importance(model, model_name, X_test, feature_cols)
                    
                    # 计算SHAP值（如果适用）
                    self._compute_shap_values(model, model_name, X_test, feature_cols)
                    
                    # 保存模型（如果启用）
                    if self.save_models:
                        self.predictor.save_model_and_features(
                            model, model_name, self.current_segment_id,
                            feature_cols, hit_rate
                        )
                    
                    training_ctx.finish_model(model_name, hit_rate)
                    self._log(f"模型 {model_name} HitRate@3: {hit_rate:.4f}")
                    
                except Exception as e:
                    self._log(f"训练模型 {model_name} 时出错: {str(e)}")
                    training_ctx.finish_model(model_name, 0.0)
                    continue
        
        return pd.DataFrame(results)
    
    def _calculate_hit_rate(self, y_true: np.ndarray, y_pred: np.ndarray,
                          group_sizes: List[int], test_info: pd.DataFrame,
                          k: int = 3) -> float:
        """计算Hit Rate@K"""
        try:
            hits = 0
            total_groups = 0
            start_idx = 0
            
            for group_size in group_sizes:
                end_idx = start_idx + group_size
                
                # 获取该组的预测分数和真实标签
                group_pred = y_pred[start_idx:end_idx]
                group_true = y_true[start_idx:end_idx]
                
                # 找到真实的正样本
                positive_indices = np.where(group_true == 1)[0]
                
                if len(positive_indices) > 0:
                    # 按预测分数排序，获取前k个
                    top_k_indices = np.argsort(group_pred)[-k:]
                    
                    # 检查前k个中是否包含正样本
                    if any(idx in positive_indices for idx in top_k_indices):
                        hits += 1
                
                total_groups += 1
                start_idx = end_idx
            
            return hits / total_groups if total_groups > 0 else 0.0
            
        except Exception as e:
            self._log(f"计算Hit Rate时出错: {str(e)}")
            return 0.0
    
    def _calculate_ndcg(self, y_true: np.ndarray, y_pred: np.ndarray,
                       group_sizes: List[int], k: int = 5) -> float:
        """计算NDCG@K"""
        try:
            ndcg_scores = []
            start_idx = 0
            
            for group_size in group_sizes:
                end_idx = start_idx + group_size
                
                if group_size >= k:
                    group_true = y_true[start_idx:end_idx].reshape(1, -1)
                    group_pred = y_pred[start_idx:end_idx].reshape(1, -1)
                    
                    try:
                        ndcg = ndcg_score(group_true, group_pred, k=k)
                        ndcg_scores.append(ndcg)
                    except:
                        continue
                
                start_idx = end_idx
            
            return np.mean(ndcg_scores) if ndcg_scores else 0.0
            
        except Exception as e:
            self._log(f"计算NDCG时出错: {str(e)}")
            return 0.0
    
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
            elif model_name in ['NeuralRanker', 'RankNet', 'TransformerRanker']:
                # 对于神经网络模型，使用梯度作为特征重要性
                importance = self._compute_neural_importance(model, X_test)
                self.feature_importance_results[model_name] = {
                    'importance': importance,
                    'feature_names': feature_names
                }
            elif model_name == 'BM25Ranker':
                # BM25模型使用统一的重要性（因为它基于文本匹配）
                importance = np.ones(len(feature_names)) / len(feature_names)
                self.feature_importance_results[model_name] = {
                    'importance': importance,
                    'feature_names': feature_names
                }
            else:
                # 对于其他模型，使用统一的重要性作为后备方案
                self._log(f"警告: {model_name} 没有特征重要性属性，使用均匀分布")
                importance = np.ones(len(feature_names)) / len(feature_names)
                self.feature_importance_results[model_name] = {
                    'importance': importance,
                    'feature_names': feature_names
                }
        except Exception as e:
            self._log(f"提取 {model_name} 特征重要性失败: {str(e)}")
            # 使用均匀分布作为后备方案
            importance = np.ones(len(feature_names)) / len(feature_names)
            self.feature_importance_results[model_name] = {
                'importance': importance,
                'feature_names': feature_names
            }
    
    def _compute_neural_importance(self, model: BaseRanker, X: np.ndarray, 
                                 samples: int = 1000) -> np.ndarray:
        """计算神经网络的特征重要性"""
        try:
            # 优先使用模型自己的特征重要性
            if hasattr(model, 'feature_importances_'):
                return model.feature_importances_
            
            # 对于样本量大的情况进行采样
            if samples < len(X):
                sample_indices = np.random.choice(len(X), samples, replace=False)
                X_sample = X[sample_indices]
            else:
                X_sample = X
            
            X_tensor = tf.convert_to_tensor(X_sample, dtype=tf.float32)
            
            with tf.GradientTape() as tape:
                tape.watch(X_tensor)
                if hasattr(model, 'model'):
                    predictions = model.model(X_tensor)
                else:
                    predictions = model.predict(X_sample)
                    predictions = tf.convert_to_tensor(predictions, dtype=tf.float32)
            
            grads = tape.gradient(predictions, X_tensor)
            
            if grads is not None:
                importance = np.mean(np.abs(grads.numpy()), axis=0)
                # 归一化
                importance = importance / (np.sum(importance) + 1e-8)
                return importance
            else:
                # 如果无法计算梯度，使用均匀分布
                return np.ones(X.shape[1]) / X.shape[1]
                
        except Exception as e:
            self._log(f"计算神经网络特征重要性时出错: {str(e)}")
            # 返回均匀分布作为后备方案
            return np.ones(X.shape[1]) / X.shape[1]
    
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
            elif model_name in ['NeuralRanker', 'RankNet', 'TransformerRanker']:
                # 对于神经网络模型，使用DeepExplainer或GradientExplainer
                try:
                    explainer = shap.DeepExplainer(model.model, X_sample[:100])  # 使用小样本作为背景
                    shap_values = explainer.shap_values(X_sample)[0]
                except Exception as deep_e:
                    self._log(f"DeepExplainer失败，尝试GradientExplainer: {str(deep_e)}")
                    try:
                        explainer = shap.GradientExplainer(model.model, X_sample[:100])
                        shap_values = explainer.shap_values(X_sample)[0]
                    except Exception as grad_e:
                        self._log(f"GradientExplainer也失败，跳过SHAP计算: {str(grad_e)}")
                        return
            else:
                self._log(f"模型 {model_name} 不支持SHAP分析，跳过")
                return
            
            self.shap_values[model_name] = {
                'values': shap_values,
                'data': X_sample,
                'feature_names': feature_names
            }
            
            # 可视化SHAP摘要图
            try:
                plt.figure(figsize=(10, 8))
                shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
                plt.title(f'{model_name} SHAP Feature Importance')
                plt.tight_layout()
                plt.show()
            except Exception as plot_e:
                self._log(f"SHAP图形绘制失败: {str(plot_e)}")
            
        except Exception as e:
            self._log(f"计算 {model_name} SHAP值失败: {str(e)}")
    
    def _analyze_feature_importance(self, feature_cols: List[str]) -> pd.DataFrame:
        """分析特征重要性"""
        if not self.feature_importance_results:
            return pd.DataFrame()
        
        # 汇总特征重要性
        importance_data = {}
        for model_name, result in self.feature_importance_results.items():
            importance = result['importance']
            feature_names = result['feature_names']
            
            for i, feature_name in enumerate(feature_names[:len(importance)]):
                if feature_name not in importance_data:
                    importance_data[feature_name] = {}
                importance_data[feature_name][model_name] = importance[i]
        
        # 创建DataFrame
        importance_df = pd.DataFrame(importance_data).T
        importance_df = importance_df.fillna(0)
        
        # 计算平均重要性
        importance_df['Average'] = importance_df.mean(axis=1)
        importance_df = importance_df.sort_values('Average', ascending=False)
        
        return importance_df
    
    def _generate_visualizations(self, model_results: pd.DataFrame, 
                               feature_importance: pd.DataFrame):
        """生成可视化图表"""
        try:
            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            
            # 1. 模型性能对比
            if not model_results.empty:
                plt.figure(figsize=(12, 6))
                
                plt.subplot(1, 2, 1)
                plt.bar(model_results['Model'], model_results['HitRate@3'])
                plt.title('模型HitRate@3性能对比')
                plt.xlabel('模型')
                plt.ylabel('HitRate@3')
                plt.xticks(rotation=45)
                
                plt.subplot(1, 2, 2)
                plt.bar(model_results['Model'], model_results['NDCG'])
                plt.title('模型NDCG性能对比')
                plt.xlabel('模型')
                plt.ylabel('NDCG')
                plt.xticks(rotation=45)
                
                plt.tight_layout()
                plt.savefig(os.path.join(Config.OUTPUT_PATH, f'model_performance_segment_{self.current_segment_id}.png'))
                plt.show()
            
            # 2. 特征重要性可视化
            if not feature_importance.empty:
                plt.figure(figsize=(12, 8))
                
                # 显示前20个最重要的特征
                top_features = feature_importance.head(20)
                
                plt.barh(range(len(top_features)), top_features['Average'])
                plt.yticks(range(len(top_features)), top_features.index)
                plt.xlabel('平均重要性')
                plt.title('前20个最重要特征')
                plt.gca().invert_yaxis()
                
                plt.tight_layout()
                plt.savefig(os.path.join(Config.OUTPUT_PATH, f'feature_importance_segment_{self.current_segment_id}.png'))
                plt.show()
                
        except Exception as e:
            self._log(f"生成可视化时出错: {str(e)}")
    
    def _save_analysis_results(self, results: Dict[str, Any]):
        """保存分析结果"""
        try:
            # 保存模型结果
            if 'model_results' in results and not results['model_results'].empty:
                model_results_path = os.path.join(
                    Config.OUTPUT_PATH, 
                    f'model_results_segment_{self.current_segment_id}.csv'
                )
                results['model_results'].to_csv(model_results_path, index=False)
                self._log(f"模型结果已保存到: {model_results_path}")
            
            # 保存特征重要性
            if 'feature_importance' in results and not results['feature_importance'].empty:
                feature_importance_path = os.path.join(
                    Config.OUTPUT_PATH, 
                    f'feature_importance_segment_{self.current_segment_id}.csv'
                )
                results['feature_importance'].to_csv(feature_importance_path, index=True)
                self._log(f"特征重要性已保存到: {feature_importance_path}")
            
        except Exception as e:
            self._log(f"保存分析结果时出错: {str(e)}")
    
    def predict_test_data(self, test_file_path: str, segment_id: int) -> Optional[str]:
        """预测测试数据"""
        try:
            self.current_segment_id = segment_id
            self._log(f"开始预测测试数据: {os.path.basename(test_file_path)}")
            
            # 加载测试数据
            test_df = self.data_processor.load_test_data(test_file_path)
            
            # 准备测试特征
            X_test, group_sizes = self.data_processor.prepare_test_features(test_df)
            
            # 对每个模型进行预测
            all_predictions = {}
            for model_name, model in self.trained_models.items():
                try:
                    self._log(f"使用 {model_name} 进行预测...")
                    scores = model.predict(X_test)
                    
                    # 分配排名
                    ranks = self._assign_rankings(scores, group_sizes)
                    
                    all_predictions[model_name] = {
                        'scores': scores,
                        'ranks': ranks
                    }
                    
                except Exception as e:
                    self._log(f"模型 {model_name} 预测失败: {str(e)}")
                    continue
            
            if not all_predictions:
                self._log("没有成功的预测结果")
                return None
            
            # 保存预测结果
            output_path = os.path.join(
                Config.OUTPUT_PATH, 
                f'predictions_segment_{segment_id}.parquet'
            )
            
            result_path = self.data_processor.save_predictions(
                test_df, all_predictions, output_path
            )
            
            return result_path
            
        except Exception as e:
            self._log(f"预测测试数据时出错: {str(e)}")
            return None
    
    def _assign_rankings(self, scores: np.ndarray, group_sizes: List[int]) -> np.ndarray:
        """分配排名"""
        ranks = np.zeros_like(scores, dtype=int)
        start_idx = 0
        
        for group_size in group_sizes:
            end_idx = start_idx + group_size
            group_scores = scores[start_idx:end_idx]
            
            # 按分数排序，分数越高排名越靠前
            group_ranks = np.argsort(-group_scores) + 1
            ranks[start_idx:end_idx] = group_ranks
            
            start_idx = end_idx
        
        return ranks
    
    def merge_all_predictions(self, prediction_files: List[str], 
                            submission_file: str, output_file: str) -> str:
        """合并所有预测结果"""
        return self.prediction_merger.merge_predictions(
            prediction_files=prediction_files,
            submission_file=submission_file,
            output_file=output_file,
            ensemble_method='average'
        )
    
    def predict_with_saved_models(self, segments: List[int], 
                                model_names: List[str], 
                                ensemble_method: str = 'average') -> Optional[str]:
        """使用保存的模型进行预测"""
        try:
            result = self.predictor.predict_all(
                segments=segments,
                model_names=model_names,
                ensemble_method=ensemble_method
            )
            
            if result is not None:
                # 返回保存的文件路径
                model_suffix = "_".join(model_names)
                final_output = self.predictor.output_path / f"{model_suffix}_final_submission.csv"
                return str(final_output)
            
            return None
            
        except Exception as e:
            self._log(f"使用保存的模型预测失败: {str(e)}")
            return None
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """获取分析总结"""
        summary = {
            'total_models_trained': len(self.trained_models),
            'models_with_feature_importance': len(self.feature_importance_results),
            'models_with_shap': len(self.shap_values),
        }
        
        if self.model_results:
            best_model = max(self.model_results.items(), 
                           key=lambda x: x[1].get('HitRate@3', 0))
            summary['best_model'] = best_model[0]
            summary['best_hit_rate'] = best_model[1].get('HitRate@3', 0)
        
        return summary