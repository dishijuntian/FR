"""
航班排名模型管理器 - 修复测试数据处理版
主要修复：
1. 修复测试数据没有'selected'列的问题
2. 区分训练和预测时的数据处理逻辑
3. 改进错误处理和日志记录
"""

import os
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
import warnings
import gc

from .Models import create_model_fast

warnings.filterwarnings('ignore')


class FlightRankingModelsManager:
    """简化的航班排名模型管理器 - 修复版"""
    
    def __init__(self, use_gpu: bool = True, logger=None):
        self.use_gpu = use_gpu
        self.logger = logger or logging.getLogger(__name__)
        self.trained_models: Dict[str, object] = {}
        self.feature_names: List[str] = []
        
        if self.use_gpu:
            try:
                import torch
                if torch.cuda.is_available():
                    self.logger.info(f"[GPU] 使用GPU: {torch.cuda.get_device_name(0)}")
                else:
                    self.use_gpu = False
                    self.logger.info("[GPU] GPU不可用，使用CPU")
            except ImportError:
                self.use_gpu = False
                self.logger.info("[GPU] PyTorch未安装，使用CPU")
    
    def prepare_data_simple(self, df: pd.DataFrame, target_col: str = 'selected') -> Tuple:
        """简化的数据预处理 - 修复测试数据处理"""
        start_time = time.time()
        
        # 检查是否为测试数据（没有target_col）
        is_test_data = target_col not in df.columns
        
        if is_test_data:
            self.logger.info(f"[PREP] 检测到测试数据，没有'{target_col}'列")
        else:
            self.logger.info(f"[PREP] 检测到训练数据，包含'{target_col}'列")
            # 1. 数据清理 - 只对训练数据进行
            selected_per_group = df.groupby('ranker_id')[target_col].sum()
            invalid_groups = selected_per_group[selected_per_group != 1].index
            if len(invalid_groups) > 0:
                df = df[~df['ranker_id'].isin(invalid_groups)]
                self.logger.info(f"[PREP] 移除了 {len(invalid_groups)} 个无效组")
        
        # 2. 特征选择 - 统一处理
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = {'Id', target_col, 'ranker_id', 'profileId', 'companyID'}
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        self.logger.info(f"[PREP] 选择了 {len(feature_cols)} 个特征")
        
        # 3. 处理缺失值 - 简单填充
        if is_test_data:
            # 测试数据：只处理特征列和必要的ID列
            required_cols = feature_cols + ['ranker_id']
            df_work = df[required_cols].copy()
        else:
            # 训练数据：包含目标列
            required_cols = feature_cols + [target_col, 'ranker_id']
            df_work = df[required_cols].copy()
        
        # 填充缺失值
        df_work[feature_cols] = df_work[feature_cols].fillna(0)
        
        # 4. 转换数据类型
        X = df_work[feature_cols].values.astype(np.float32)
        groups = df_work['ranker_id'].values
        
        if is_test_data:
            # 测试数据：创建虚拟的y数组
            y = np.zeros(len(df_work), dtype=np.float32)
            self.logger.info(f"[PREP] 测试数据处理完成: shape={X.shape}")
        else:
            # 训练数据：使用真实的目标值
            y = df_work[target_col].values.astype(np.float32)
            self.logger.info(f"[PREP] 训练数据处理完成: shape={X.shape}")
        
        self.feature_names = feature_cols
        
        prep_time = time.time() - start_time
        self.logger.info(f"[PREP] 数据预处理完成: 耗时={prep_time:.2f}s")
        
        return X, y, groups, feature_cols, df_work
    
    def train_models_simple(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                           model_names: List[str], model_configs: Dict = None) -> Dict:
        """简化的模型训练"""
        if model_configs is None:
            model_configs = {}
        
        trained_models = {}
        
        for model_name in model_names:
            try:
                self.logger.info(f"[TRAIN] 训练模型: {model_name}")
                train_start = time.time()
                
                # 获取模型参数
                model_params = model_configs.get(model_name, {})
                
                # 直接创建模型 - 避免工厂模式开销
                if model_name in ['RankNet', 'NeuralRanker', 'TransformerRanker']:
                    model = create_model_fast(
                        model_name, 
                        use_gpu=self.use_gpu,
                        input_dim=X.shape[1],
                        **model_params
                    )
                else:
                    model = create_model_fast(
                        model_name, 
                        use_gpu=self.use_gpu,
                        **model_params
                    )
                
                # 训练模型
                model.fit(X, y, groups)
                trained_models[model_name] = model
                
                train_time = time.time() - train_start
                self.logger.info(f"[TRAIN] {model_name} 完成: {train_time:.2f}s")
                
                # 清理GPU内存
                if self.use_gpu:
                    try:
                        import torch
                        torch.cuda.empty_cache()
                    except ImportError:
                        pass
                
            except Exception as e:
                train_time = time.time() - train_start if 'train_start' in locals() else 0
                self.logger.error(f"[TRAIN] {model_name} 失败: {e} ({train_time:.2f}s)")
                continue
        
        self.trained_models.update(trained_models)
        return trained_models
    
    def predict_simple(self, X: np.ndarray, model_names: List[str] = None) -> np.ndarray:
        """简化的预测 - 使用第一个可用模型或平均预测"""
        if model_names is None:
            model_names = list(self.trained_models.keys())
        
        available_models = [name for name in model_names if name in self.trained_models]
        
        if not available_models:
            raise ValueError("没有可用的已训练模型")
        
        self.logger.info(f"[PREDICT] 使用模型: {available_models}")
        
        predictions = []
        for model_name in available_models:
            try:
                model = self.trained_models[model_name]
                pred = model.predict(X)
                predictions.append(pred)
                self.logger.info(f"[PREDICT] {model_name} 预测完成: shape={pred.shape}")
            except Exception as e:
                self.logger.warning(f"[PREDICT] {model_name} 预测失败: {e}")
                continue
        
        if not predictions:
            raise ValueError("所有模型预测都失败")
        
        # 简单平均或使用单个模型结果
        if len(predictions) == 1:
            final_prediction = predictions[0]
        else:
            final_prediction = np.mean(predictions, axis=0)
            self.logger.info(f"[PREDICT] 使用 {len(predictions)} 个模型的平均预测")
        
        return final_prediction
    
    def save_models_simple(self, save_dir: str):
        """简化的模型保存"""
        os.makedirs(save_dir, exist_ok=True)
        
        import joblib
        
        for name, model in self.trained_models.items():
            try:
                filepath = os.path.join(save_dir, f"{name}.pkl")
                model.save_model(filepath)
                self.logger.info(f"[SAVE] {name} 保存完成")
            except Exception as e:
                self.logger.error(f"[SAVE] {name} 保存失败: {e}")
        
        # 保存特征名称
        if self.feature_names:
            joblib.dump(self.feature_names, os.path.join(save_dir, "features.pkl"))
    
    def load_models_simple(self, save_dir: str, model_names: List[str] = None):
        """简化的模型加载"""
        import joblib
        
        if not os.path.exists(save_dir):
            raise FileNotFoundError(f"模型目录不存在: {save_dir}")
        
        if model_names is None:
            model_files = [f for f in os.listdir(save_dir) 
                          if f.endswith('.pkl') and f != 'features.pkl']
            model_names = [f.replace('.pkl', '') for f in model_files]
        
        loaded_count = 0
        for name in model_names:
            filepath = os.path.join(save_dir, f"{name}.pkl")
            if os.path.exists(filepath):
                try:
                    model = joblib.load(filepath)
                    self.trained_models[name] = model
                    loaded_count += 1
                    self.logger.info(f"[LOAD] {name} 加载完成")
                except Exception as e:
                    self.logger.warning(f"[LOAD] {name} 加载失败: {e}")
        
        # 加载特征名称
        features_path = os.path.join(save_dir, "features.pkl")
        if os.path.exists(features_path):
            try:
                self.feature_names = joblib.load(features_path)
                self.logger.info(f"[LOAD] 特征名称加载完成: {len(self.feature_names)} 个特征")
            except Exception as e:
                self.logger.warning(f"[LOAD] 特征名称加载失败: {e}")
        
        return loaded_count > 0


def calculate_hit_rate_fast(y_true: np.ndarray, y_pred: np.ndarray, 
                           groups: np.ndarray, k: int = 3) -> float:
    """快速计算HitRate@K"""
    unique_groups = np.unique(groups)
    hits = 0
    total_groups = 0
    
    for group in unique_groups:
        group_mask = groups == group
        group_y_true = y_true[group_mask]
        group_scores = y_pred[group_mask]
        
        true_selected_indices = np.where(group_y_true == 1)[0]
        if len(true_selected_indices) > 0:
            sorted_indices = np.argsort(group_scores)[::-1]
            top_k_indices = sorted_indices[:k]
            if any(idx in top_k_indices for idx in true_selected_indices):
                hits += 1
            total_groups += 1
    
    return hits / total_groups if total_groups > 0 else 0


def evaluate_models_fast(models: Dict, X_test: np.ndarray, y_test: np.ndarray, 
                        groups_test: np.ndarray) -> pd.DataFrame:
    """快速模型评估"""
    results = []
    
    for model_name, model in models.items():
        try:
            y_pred = model.predict(X_test)
            hit_rate = calculate_hit_rate_fast(y_test, y_pred, groups_test, k=3)
            
            results.append({
                'Model': model_name,
                'HitRate@3': f'{hit_rate:.4f}'
            })
            
        except Exception as e:
            print(f"评估{model_name}失败: {e}")
            continue
    
    return pd.DataFrame(results)


# 快速分析器类
class FastFlightRankingAnalyzer:
    """快速航班排序分析器 - 简化版本"""
    
    def __init__(self, use_gpu: bool = True, selected_models: List[str] = None):
        self.use_gpu = use_gpu
        self.selected_models = selected_models or ['XGBRanker', 'LGBMRanker']
        self.manager = FlightRankingModelsManager(use_gpu=use_gpu)
        self.trained_models = {}
    
    def analyze_segment(self, train_file_path: str, use_sampling: bool = True,
                       num_groups: int = 2000, min_group_size: int = 20) -> Dict:
        """分析单个数据段 - 高效版本"""
        
        # 1. 加载数据
        df = self.load_and_sample_data(train_file_path, use_sampling, num_groups, min_group_size)
        
        # 2. 准备数据
        X, y, groups, feature_cols, _ = self.manager.prepare_data_simple(df)
        
        # 3. 数据划分
        from sklearn.model_selection import train_test_split
        unique_groups = np.unique(groups)
        train_groups, test_groups = train_test_split(unique_groups, test_size=0.2, random_state=42)
        
        train_mask = np.isin(groups, train_groups)
        test_mask = np.isin(groups, test_groups)
        
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        groups_test = groups[test_mask]
        
        # 4. 训练模型
        trained_models = self.manager.train_models_simple(X_train, y_train, groups[train_mask], self.selected_models)
        
        # 5. 评估模型
        results_df = evaluate_models_fast(trained_models, X_test, y_test, groups_test)
        
        self.trained_models.update(trained_models)
        
        return {
            'model_results': results_df,
            'trained_models': trained_models,
            'feature_names': feature_cols
        }
    
    def load_and_sample_data(self, file_path: str, use_sampling: bool, num_groups: int, min_group_size: int) -> pd.DataFrame:
        """加载和采样数据"""
        df = pd.read_parquet(file_path)
        
        if use_sampling:
            group_counts = df['ranker_id'].value_counts()
            valid_groups = group_counts[group_counts >= min_group_size].index
            
            if len(valid_groups) > num_groups:
                np.random.seed(42)
                selected_groups = np.random.choice(valid_groups, size=num_groups, replace=False)
                df = df[df['ranker_id'].isin(selected_groups)]
        
        return df
    
    def save_models(self, save_dir: str):
        """保存训练的模型"""
        self.manager.save_models_simple(save_dir)