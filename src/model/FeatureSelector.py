import pandas as pd
import numpy as np
import torch
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

from .Models import XGBoostRanker, LightGBMRanker

class FeatureSelector:
    def __init__(self, config: Dict = None, logger=None):
        self.logger = logger
        self.config = config or {'max_features': 80, 'sample_size': 20000}
        self.selected_features = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def remove_constant_features(self, X: pd.DataFrame) -> List[str]:
        """快速移除常数特征"""
        valid_features = []
        for col in X.columns:
            nunique = X[col].nunique()
            if nunique > 1:
                # 快速检查：如果唯一值数量足够多，直接保留
                if nunique > max(10, len(X) * 0.01):
                    valid_features.append(col)
                else:
                    # 只对可疑特征进行详细检查
                    mode_ratio = X[col].value_counts().iloc[0] / len(X)
                    if mode_ratio < 0.995:
                        valid_features.append(col)
        return valid_features
    
    def simple_correlation_filter(self, X: pd.DataFrame, threshold: float = 0.95) -> List[str]:
        """简化的相关性过滤，按组采样"""
        try:
            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) < 2:
                return X.columns.tolist()
            
            # 按组采样处理大数据集
            if 'ranker_id' in X.columns and len(X) > 10000:
                unique_groups = X['ranker_id'].unique()
                if len(unique_groups) > 500:
                    # 随机选择500个组进行相关性分析
                    selected_groups = np.random.choice(unique_groups, size=500, replace=False)
                    X_sample = X[X['ranker_id'].isin(selected_groups)]
                else:
                    X_sample = X
            else:
                # 如果没有ranker_id或数据量较小，直接使用
                X_sample = X
            
            # 计算相关矩阵
            corr_matrix = X_sample[numeric_cols].fillna(0).corr().abs()
            
            # 找出高相关特征对
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            
            # 移除高相关特征
            to_drop = [column for column in upper_triangle.columns 
                      if any(upper_triangle[column] > threshold)]
            
            remaining_features = [col for col in X.columns if col not in to_drop]
            
            if self.logger:
                self.logger.info(f"Correlation filter: {len(X.columns)} -> {len(remaining_features)}")
            
            return remaining_features
            
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Correlation filter failed: {e}, using all features")
            return X.columns.tolist()
    
    def tree_model_importance_selection(self, X: pd.DataFrame, y: pd.Series, top_k: int) -> List[str]:
        """直接调用树模型进行特征重要性选择"""
        if 'ranker_id' not in X.columns:
            raise ValueError("ranker_id column not found in the input data")
        
        # 按组采样数据以提高速度
        unique_ranker_ids = X['ranker_id'].unique()
        max_groups = min(500, len(unique_ranker_ids))
        selected_ranker_ids = np.random.choice(unique_ranker_ids, 
                                            size=max_groups, 
                                            replace=False)
        
        # 按组提取数据，保持组的完整性
        mask = X['ranker_id'].isin(selected_ranker_ids)
        X_sample = X[mask].copy()
        y_sample = y[mask].copy()
        
        # 如果采样后的数据仍然太大，进一步按组减少
        if len(X_sample) > self.config.get('sample_size', 20000):
            current_groups = X_sample['ranker_id'].unique()
            target_groups = min(300, len(current_groups))  # 进一步减少组数
            selected_groups = np.random.choice(current_groups, size=target_groups, replace=False)
            
            group_mask = X_sample['ranker_id'].isin(selected_groups)
            X_sample = X_sample[group_mask]
            y_sample = y_sample[group_mask]
        
        # 移除ranker_id列进行训练
        feature_cols = [col for col in X_sample.columns if col != 'ranker_id']
        X_features = X_sample[feature_cols].fillna(-1).values
        y_values = y_sample.values
        groups = X_sample['ranker_id'].values
        
        try:
            # 优先使用XGBoost
            try:
                if self.logger:
                    self.logger.info("Using XGBoost for feature importance")
                
                model = XGBoostRanker(
                    n_estimators=50,  # 减少树的数量以提高速度
                    max_depth=4,
                    learning_rate=0.3,
                    use_gpu=torch.cuda.is_available(),
                    logger=self.logger
                )
                
                model.fit(X_features, y_values, groups)
                importance_dict = model.get_feature_importance()
                
                # 转换特征重要性字典
                feature_importance = {}
                for feature_idx_str, importance in importance_dict.items():
                    # XGBoost返回的特征索引格式为'f0', 'f1'等
                    if feature_idx_str.startswith('f'):
                        feature_idx = int(feature_idx_str[1:])
                        if feature_idx < len(feature_cols):
                            feature_importance[feature_cols[feature_idx]] = importance
                
                # 如果没有获得重要性，使用所有特征
                if not feature_importance:
                    for i, col in enumerate(feature_cols):
                        feature_importance[col] = 1.0
                
            except Exception as xgb_e:
                if self.logger:
                    self.logger.warning(f"XGBoost failed: {xgb_e}, trying LightGBM")
                
                # 降级到LightGBM
                model = LightGBMRanker(
                    n_estimators=50,
                    max_depth=4,
                    learning_rate=0.3,
                    use_gpu=torch.cuda.is_available(),
                    logger=self.logger
                )
                
                model.fit(X_features, y_values, groups)
                importance_dict = model.get_feature_importance()
                
                # 转换特征重要性字典
                feature_importance = {}
                for feature_idx_str, importance in importance_dict.items():
                    if feature_idx_str.startswith('f'):
                        feature_idx = int(feature_idx_str[1:])
                        if feature_idx < len(feature_cols):
                            feature_importance[feature_cols[feature_idx]] = importance
                
                if not feature_importance:
                    for i, col in enumerate(feature_cols):
                        feature_importance[col] = 1.0
            
            # 选择top特征
            if len(feature_importance) == 0:
                if self.logger:
                    self.logger.warning("No feature importance obtained, using first features")
                return feature_cols[:top_k]
            
            selected_features = sorted(feature_importance.items(), 
                                     key=lambda x: x[1], 
                                     reverse=True)[:top_k]
            
            result = [f[0] for f in selected_features]
            
            if self.logger:
                self.logger.info(f"Tree model selection: {len(feature_cols)} -> {len(result)}")
            
            return result
            
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Tree model importance selection failed: {e}, using first {top_k} features")
            return feature_cols[:top_k]
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, target_features: Optional[int] = None) -> List[str]:
        """主特征选择方法"""
        target_features = target_features or self.config.get('max_features', 80)
        
        if self.logger:
            self.logger.info(f"Starting feature selection: {X.shape[1]} features")
        
        # 移除常数特征
        valid_features = self.remove_constant_features(X)
        X_filtered = X[valid_features]
        
        if len(valid_features) <= target_features:
            self.selected_features = valid_features
            if self.logger:
                self.logger.info(f"Selected {len(valid_features)} features (constant filter only)")
            return valid_features
        
        # 移除相关特征
        valid_features = self.simple_correlation_filter(X_filtered)
        X_filtered = X_filtered[valid_features]
        
        if len(valid_features) <= target_features:
            self.selected_features = valid_features
            if self.logger:
                self.logger.info(f"Selected {len(valid_features)} features (correlation filter)")
            return valid_features
        
        # 使用树模型选择重要特征
        final_features = self.tree_model_importance_selection(X_filtered, y, target_features)
        self.selected_features = final_features
        
        if self.logger:
            self.logger.info(f"Feature selection completed: {X.shape[1]} -> {len(final_features)}")
        
        to_remove = ['selected', 'Id', 'profileId', 'companyID', 'isVip']
        return [x for x in final_features if x not in to_remove and x in X.columns]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series, target_features: Optional[int] = None) -> pd.DataFrame:
        """拟合并转换特征"""
        self.select_features(X, y, target_features)
        if self.selected_features is None:
            raise ValueError("Call select_features first")
        
        available_features = [f for f in self.selected_features if f in X.columns]
        if len(available_features) < len(self.selected_features) and self.logger:
            missing = len(self.selected_features) - len(available_features)
            self.logger.warning(f"Missing {missing} features, using {len(available_features)} available")
        
        return X[available_features].copy()
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """转换特征（仅转换，不重新选择）"""
        if self.selected_features is None:
            raise ValueError("FeatureSelector not fitted. Call fit_transform first.")
        
        available_features = [f for f in self.selected_features if f in X.columns]
        if len(available_features) < len(self.selected_features) and self.logger:
            missing = len(self.selected_features) - len(available_features)
            self.logger.warning(f"Missing {missing} features during transform, using {len(available_features)} available")
        
        return X[available_features].copy()