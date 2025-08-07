"""
主分析器 - 重构版

专注于：
- 整体分析流程控制
- 模型训练和评估
- 特征重要性分析
- 结果可视化

作者: Flight Ranking Team
版本: 4.0 (重构版)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from pathlib import Path
from typing import Dict, List, Optional, Any
import warnings
import gc

warnings.filterwarnings('ignore')


class ModelEvaluator:
    """模型评估器"""
    
    @staticmethod
    def calculate_hit_rate(y_true: np.ndarray, y_pred: np.ndarray,
                          group_sizes: List[int], test_info: pd.DataFrame, k: int = 3) -> float:
        """计算Hit Rate@K"""
        hits = 0
        total_groups = 0
        start_idx = 0
        
        for group_size in group_sizes:
            end_idx = start_idx + group_size
            
            group_pred = y_pred[start_idx:end_idx]
            group_true = y_true[start_idx:end_idx]
            
            positive_indices = np.where(group_true == 1)[0]
            
            if len(positive_indices) > 0:
                top_k_indices = np.argsort(group_pred)[-k:]
                if any(idx in positive_indices for idx in top_k_indices):
                    hits += 1
            
            total_groups += 1
            start_idx = end_idx
        
        return hits / total_groups if total_groups > 0 else 0.0
    
    @staticmethod
    def calculate_ndcg(y_true: np.ndarray, y_pred: np.ndarray,
                      group_sizes: List[int], k: int = 5) -> float:
        """计算NDCG@K"""
        from sklearn.metrics import ndcg_score
        
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
                    pass
            
            start_idx = end_idx
        
        return np.mean(ndcg_scores) if ndcg_scores else 0.0


class FeatureAnalyzer:
    """特征分析器"""
    
    def __init__(self):
        self.feature_importance_results = {}
    
    def extract_feature_importance(self, model, model_name: str, 
                                 X_test: np.ndarray, feature_names: List[str]):
        """提取特征重要性"""
        try:
            if hasattr(model, 'feature_importances_') and model.feature_importances_ is not None:
                importance = model.feature_importances_
            elif hasattr(model, '_feature_importance') and model._feature_importance is not None:
                importance = model._feature_importance
            else:
                # 使用均匀分布作为后备
                importance = np.ones(len(feature_names)) / len(feature_names)
            
            self.feature_importance_results[model_name] = {
                'importance': importance,
                'feature_names': feature_names
            }
            
        except Exception as e:
            print(f"提取{model_name}特征重要性失败: {e}")
            # 使用均匀分布
            importance = np.ones(len(feature_names)) / len(feature_names)
            self.feature_importance_results[model_name] = {
                'importance': importance,
                'feature_names': feature_names
            }
    
    def analyze_feature_importance(self, feature_cols: List[str]) -> pd.DataFrame:
        """分析特征重要性"""
        if not self.feature_importance_results:
            return pd.DataFrame()
        
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


class Visualizer:
    """可视化器"""
    
    def __init__(self, output_path: Path):
        self.output_path = output_path
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    
    def plot_model_performance(self, model_results: pd.DataFrame, segment_id: int):
        """绘制模型性能对比"""
        if model_results.empty:
            return
        
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
        plt.savefig(self.output_path / f'model_performance_segment_{segment_id}.png')
        plt.show()
    
    def plot_feature_importance(self, feature_importance: pd.DataFrame, segment_id: int):
        """绘制特征重要性"""
        if feature_importance.empty:
            return
        
        plt.figure(figsize=(12, 8))
        
        top_features = feature_importance.head(20)
        
        plt.barh(range(len(top_features)), top_features['Average'])
        plt.yticks(range(len(top_features)), top_features.index)
        plt.xlabel('平均重要性')
        plt.title('前20个最重要特征')
        plt.gca().invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(self.output_path / f'feature_importance_segment_{segment_id}.png')
        plt.show()


class FlightRankingAnalyzer:
    """航班排序分析器 - 主控制器"""
    
    def __init__(self, use_gpu: bool = True, selected_models: List[str] = None,
                 enable_auto_tuning: bool = False, auto_tuning_trials: int = 50,
                 save_models: bool = True, output_path: Path = None):
        """
        初始化分析器
        
        Args:
            use_gpu: 是否使用GPU
            selected_models: 选择的模型列表
            enable_auto_tuning: 是否启用自动调参
            auto_tuning_trials: 自动调参试验次数
            save_models: 是否保存模型
            output_path: 输出路径
        """
        self.use_gpu = use_gpu
        self.selected_models = selected_models or ['XGBRanker', 'NeuralRanker']
        self.enable_auto_tuning = enable_auto_tuning
        self.auto_tuning_trials = auto_tuning_trials
        self.save_models = save_models
        
        # 初始化组件
        from data_processor import DataProcessor
        from predictor import FlightRankingPredictor
        from config import Config
        
        self.data_processor = DataProcessor()
        self.model_evaluator = ModelEvaluator()
        self.feature_analyzer = FeatureAnalyzer()
        
        # 输出路径
        self.output_path = output_path or Config.OUTPUT_PATH
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.visualizer = Visualizer(self.output_path)
        
        # 预测器
        self.predictor = FlightRankingPredictor(
            Config.DATA_BASE_PATH, use_gpu=use_gpu
        )
        
        # 结果存储
        self.trained_models = {}
        self.current_segment_id = 0
        
        print(f"分析器初始化完成，使用设备: {'GPU' if use_gpu else 'CPU'}")
    
    def full_analysis(self, train_file_path: Path, use_sampling: bool = True,
                     num_groups: int = 2000, min_group_size: int = 20) -> Dict[str, Any]:
        """
        执行完整分析流程
        
        Args:
            train_file_path: 训练文件路径
            use_sampling: 是否使用抽样
            num_groups: 抽样组数
            min_group_size: 最小组大小
            
        Returns:
            Dict: 分析结果
        """
        print(f"开始完整分析: {train_file_path.name}")
        
        try:
            # 1. 加载和准备数据
            print("步骤1: 加载和准备数据")
            df = self.data_processor.load_data(
                train_file_path, use_sampling, num_groups, min_group_size
            )
            
            # 2. 准备排序数据
            print("步骤2: 准备排序数据")
            data_tuple = self.data_processor.split_ranking_data(df)
            (X_train, X_test, y_train, y_test, 
             train_group_sizes, test_group_sizes, 
             feature_cols, test_info) = data_tuple
            
            # 3. 训练和评估模型
            print("步骤3: 训练和评估模型")
            model_results = self._train_and_evaluate_models(
                X_train, X_test, y_train, y_test,
                train_group_sizes, test_group_sizes,
                feature_cols, test_info
            )
            
            # 4. 特征重要性分析
            print("步骤4: 特征重要性分析")
            feature_importance = self.feature_analyzer.analyze_feature_importance(feature_cols)
            
            # 5. 生成可视化
            print("步骤5: 生成可视化")
            self.visualizer.plot_model_performance(model_results, self.current_segment_id)
            self.visualizer.plot_feature_importance(feature_importance, self.current_segment_id)
            
            # 6. 保存结果
            print("步骤6: 保存结果")
            results = {
                'model_results': model_results,
                'feature_importance': feature_importance,
                'data_shape': df.shape,
                'feature_count': len(feature_cols)
            }
            
            self._save_results(results)
            
            print("完整分析完成")
            return results
            
        except Exception as e:
            print(f"分析过程出错: {e}")
            raise
        finally:
            # 清理内存
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def _train_and_evaluate_models(self, X_train: np.ndarray, X_test: np.ndarray,
                                  y_train: np.ndarray, y_test: np.ndarray,
                                  train_group_sizes: List[int], 
                                  test_group_sizes: List[int],
                                  feature_cols: List[str],
                                  test_info: pd.DataFrame) -> pd.DataFrame:
        """训练和评估所有模型"""
        results = []
        
        from models import ModelFactory
        from config import Config
        from auto_tuner import create_auto_tuner
        
        for model_name in self.selected_models:
            try:
                print(f"训练模型: {model_name}")
                
                # 获取模型参数
                model_params = Config.get_model_config(model_name).copy()
                
                # 自动调参
                if self.enable_auto_tuning:
                    print(f"执行自动调参: {model_name}")
                    tuner = create_auto_tuner(
                        model_name=model_name,
                        search_space=Config.get_tuning_space(model_name),
                        n_trials=self.auto_tuning_trials
                    )
                    
                    best_params = tuner.optimize(
                        X_train, y_train, train_group_sizes,
                        X_test, y_test, test_group_sizes
                    )
                    model_params.update(best_params)
                
                # 创建和训练模型
                if Config.is_pytorch_model(model_name):
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
                
                model.fit(X_train, y_train, train_group_sizes)
                
                # 预测和评估
                y_pred = model.predict(X_test)
                
                hit_rate = self.model_evaluator.calculate_hit_rate(
                    y_test, y_pred, test_group_sizes, test_info
                )
                ndcg = self.model_evaluator.calculate_ndcg(
                    y_test, y_pred, test_group_sizes
                )
                
                # 存储结果
                self.trained_models[model_name] = model
                
                results.append({
                    'Model': model_name,
                    'HitRate@3': hit_rate,
                    'NDCG': ndcg,
                    'Parameters': str(model_params)
                })
                
                # 提取特征重要性
                self.feature_analyzer.extract_feature_importance(
                    model, model_name, X_test, feature_cols
                )
                
                # 保存模型
                if self.save_models:
                    self.predictor.save_model_and_features(
                        model, model_name, self.current_segment_id,
                        feature_cols, hit_rate
                    )
                
                print(f"模型 {model_name} HitRate@3: {hit_rate:.4f}")
                
                # 清理GPU内存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"训练模型 {model_name} 失败: {e}")
                continue
        
        return pd.DataFrame(results)
    
    def predict_test_data(self, test_file_path: Path, segment_id: int) -> Optional[Path]:
        """预测测试数据"""
        try:
            self.current_segment_id = segment_id
            print(f"开始预测测试数据: {test_file_path.name}")
            
            # 使用预测器进行预测
            result = self.predictor.predict_all(
                segments=[segment_id],
                model_names=list(self.trained_models.keys()),
                ensemble_method='average'
            )
            
            if result is not None:
                output_file = self.output_path / f'predictions_segment_{segment_id}.csv'
                result.to_csv(output_file, index=False)
                return output_file
            
            return None
            
        except Exception as e:
            print(f"预测测试数据失败: {e}")
            return None
    
    def merge_all_predictions(self, prediction_files: List[Path], 
                            submission_file: Path, output_file: Path) -> Path:
        """合并所有预测结果"""
        from data_processor import PredictionMerger
        
        merger = PredictionMerger()
        return merger.merge_predictions(
            prediction_files=prediction_files,
            submission_file=submission_file,
            output_file=output_file,
            ensemble_method='average'
        )
    
    def _save_results(self, results: Dict[str, Any]):
        """保存分析结果"""
        try:
            # 保存模型结果
            if 'model_results' in results and not results['model_results'].empty:
                model_results_path = self.output_path / f'model_results_segment_{self.current_segment_id}.csv'
                results['model_results'].to_csv(model_results_path, index=False)
                print(f"模型结果已保存: {model_results_path}")
            
            # 保存特征重要性
            if 'feature_importance' in results and not results['feature_importance'].empty:
                feature_importance_path = self.output_path / f'feature_importance_segment_{self.current_segment_id}.csv'
                results['feature_importance'].to_csv(feature_importance_path, index=True)
                print(f"特征重要性已保存: {feature_importance_path}")
            
        except Exception as e:
            print(f"保存结果失败: {e}")
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """获取分析总结"""
        summary = {
            'total_models_trained': len(self.trained_models),
            'pytorch_version': torch.__version__,
            'device_used': 'GPU' if self.use_gpu else 'CPU'
        }
        
        return summary