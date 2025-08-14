import pandas as pd
import numpy as np
import torch
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


class FeatureSelector:
    """优化的特征选择器 - 仅使用常数过滤和相关性过滤"""
    
    def __init__(self, config: Dict = None, logger=None):
        self.logger = logger
        self.config = config or {
            'enabled': True,
            'max_features': 80, 
            'variance_threshold': 0.0,
            'correlation_threshold': 0.95,
            'sample_size': 20000
        }
        self.selected_features = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 从配置中读取参数
        self.enabled = self.config.get('enabled', True)
        self.max_features = self.config.get('max_features', 80)
        self.variance_threshold = self.config.get('variance_threshold', 0.0)
        self.correlation_threshold = self.config.get('correlation_threshold', 0.95)
        self.sample_size = self.config.get('sample_size', 20000)
        
        if not self.enabled:
            if self.logger:
                self.logger.info("特征选择已禁用")
        else:
            if self.logger:
                self.logger.info(f"特征选择器初始化: max_features={self.max_features}, "
                               f"correlation_threshold={self.correlation_threshold}")
        
    def remove_constant_features(self, X: pd.DataFrame) -> List[str]:
        """快速移除常数特征和低方差特征"""
        if not self.enabled:
            return X.columns.tolist()
            
        valid_features = []
        
        for col in X.columns:
            # 跳过非数值列
            if not pd.api.types.is_numeric_dtype(X[col]):
                continue
                
            # 计算唯一值数量
            nunique = X[col].nunique()
            
            # 如果只有一个唯一值，直接跳过
            if nunique <= 1:
                continue
            
            # 快速检查：如果唯一值数量足够多，直接保留
            if nunique > max(10, len(X) * 0.01):
                valid_features.append(col)
            else:
                # 对可疑特征进行详细检查
                try:
                    # 检查最频繁值的比例
                    mode_ratio = X[col].value_counts().iloc[0] / len(X)
                    if mode_ratio < 0.995:  # 如果最频繁值占比小于99.5%，保留
                        # 额外检查方差
                        if self.variance_threshold > 0:
                            variance = X[col].var()
                            if pd.notna(variance) and variance > self.variance_threshold:
                                valid_features.append(col)
                        else:
                            valid_features.append(col)
                except:
                    # 如果计算出错，保守地保留该特征
                    valid_features.append(col)
        
        if self.logger:
            removed_count = len(X.columns) - len(valid_features)
            self.logger.info(f"常数特征过滤: {len(X.columns)} -> {len(valid_features)} "
                           f"(移除 {removed_count} 个常数/低方差特征)")
        
        return valid_features
    
    def simple_correlation_filter(self, X: pd.DataFrame, threshold: float = None) -> List[str]:
        """简化的相关性过滤，按组采样处理大数据集"""
        if not self.enabled:
            return X.columns.tolist()
            
        threshold = threshold or self.correlation_threshold
        
        try:
            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) < 2:
                return X.columns.tolist()
            
            # 固定随机种子确保一致性
            X_sample = X
            np.random.seed(42)
            
            if 'ranker_id' in X.columns and len(X) > self.sample_size:
                unique_groups = X['ranker_id'].unique()
                if len(unique_groups) > 5000:
                    max_groups = min(5000, len(unique_groups))
                    selected_groups = np.random.choice(unique_groups, 
                                                     size=max_groups, 
                                                     replace=False)
                    X_sample = X[X['ranker_id'].isin(selected_groups)]
            
            if len(X_sample) > self.sample_size:
                X_sample = X_sample.sample(n=self.sample_size, random_state=42)
            
            # 计算相关矩阵
            corr_matrix = X_sample[numeric_cols].fillna(0).corr().abs()
            
            # 找出高相关特征对
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            
            # 移除高相关特征 - 保留每对中的第一个
            to_drop = [column for column in upper_triangle.columns 
                      if any(upper_triangle[column] > threshold)]
            
            # 构建最终特征列表
            remaining_features = [col for col in X.columns if col not in to_drop]
            
            if self.logger:
                removed_count = len(X.columns) - len(remaining_features)
                self.logger.info(f"相关性过滤: {len(X.columns)} -> {len(remaining_features)} "
                               f"(移除 {removed_count} 个高相关特征)")
            
            return remaining_features
            
        except Exception as e:
            if self.logger:
                self.logger.warning(f"相关性过滤失败: {e}, 使用所有特征")
            return X.columns.tolist()

    def select_features(self, X: pd.DataFrame, y: pd.Series, target_features: Optional[int] = None) -> List[str]:
        """主特征选择方法 - 仅使用常数过滤和相关性过滤"""
        if not self.enabled:
            # 如果特征选择被禁用，返回所有数值特征（排除特殊列）
            to_remove = {'selected', 'Id', 'profileId', 'companyID', 'isVip', 'ranker_id'}
            all_features = [col for col in X.columns if col not in to_remove]
            self.selected_features = all_features
            if self.logger:
                self.logger.info(f"特征选择已禁用，使用所有特征: {len(all_features)}")
            return all_features
        
        target_features = target_features or self.max_features
        
        if self.logger:
            self.logger.info(f"开始特征选择: {X.shape[1]} 个特征，目标: {target_features}")
        
        # 第一步：移除常数特征和低方差特征
        valid_features = self.remove_constant_features(X)
        if not valid_features:
            if self.logger:
                self.logger.warning("没有有效的数值特征")
            return []
        
        X_filtered = X[valid_features]
        
        # 如果特征数已经满足要求，直接返回
        if len(valid_features) <= target_features:
            # 移除特殊列
            to_remove = {'selected', 'Id', 'profileId', 'companyID', 'isVip', 'ranker_id'}
            final_features = [col for col in valid_features if col not in to_remove]
            self.selected_features = final_features
            if self.logger:
                self.logger.info(f"常数过滤后特征数已满足要求: {len(final_features)}")
            return final_features
        
        # 第二步：移除高相关特征
        final_features = self.simple_correlation_filter(X_filtered, self.correlation_threshold)
        
        # 移除特殊列
        to_remove = {'selected', 'Id', 'profileId', 'companyID', 'isVip', 'ranker_id'}
        final_features = [col for col in final_features if col not in to_remove and col in X.columns]
        
        # 如果特征数仍然超过目标，随机选择（保持稳定性）
        if len(final_features) > target_features:
            # 使用固定随机种子确保可重复性
            np.random.seed(42)
            final_features = sorted(final_features)  # 排序确保稳定性
            selected_indices = np.random.choice(len(final_features), 
                                              size=target_features, 
                                              replace=False)
            final_features = [final_features[i] for i in sorted(selected_indices)]
            
            if self.logger:
                self.logger.info(f"随机选择达到目标特征数: {len(final_features)}")
        
        self.selected_features = final_features
        
        if self.logger:
            self.logger.info(f"特征选择完成: {X.shape[1]} -> {len(final_features)}")
        
        return final_features

    def fit_transform(self, X: pd.DataFrame, y: pd.Series, target_features: Optional[int] = None) -> pd.DataFrame:
        """拟合并转换特征"""
        selected_features = self.select_features(X, y, target_features)
        
        if not selected_features:
            if self.logger:
                self.logger.warning("没有选择到任何特征")
            return pd.DataFrame()
        
        available_features = [f for f in selected_features if f in X.columns]
        
        if len(available_features) < len(selected_features) and self.logger:
            missing = len(selected_features) - len(available_features)
            self.logger.warning(f"缺少 {missing} 个特征，使用 {len(available_features)} 个可用特征")
        
        return X[available_features].copy()
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """转换特征（仅转换，不重新选择）"""
        if not self.enabled:
            # 如果特征选择被禁用，返回所有数值特征
            to_remove = {'selected', 'Id', 'profileId', 'companyID', 'isVip', 'ranker_id'}
            available_features = [col for col in X.columns if col not in to_remove]
            return X[available_features].copy()
        
        if self.selected_features is None:
            raise ValueError("特征选择器未拟合，请先调用 fit_transform")
        
        available_features = [f for f in self.selected_features if f in X.columns]
        
        if len(available_features) < len(self.selected_features) and self.logger:
            missing = len(self.selected_features) - len(available_features)
            self.logger.warning(f"转换时缺少 {missing} 个特征，使用 {len(available_features)} 个可用特征")
        
        return X[available_features].copy()
    
    def get_selected_features(self) -> List[str]:
        """获取选择的特征列表"""
        return self.selected_features.copy() if self.selected_features else []
    
    def save_selector(self, filepath: str):
        """保存特征选择器"""
        import joblib
        
        selector_data = {
            'config': self.config,
            'selected_features': self.selected_features,
            'enabled': self.enabled
        }
        
        joblib.dump(selector_data, filepath)
        if self.logger:
            self.logger.info(f"特征选择器已保存: {filepath}")
    
    @classmethod
    def load_selector(cls, filepath: str, logger=None):
        """加载特征选择器"""
        import joblib
        
        selector_data = joblib.load(filepath)
        
        instance = cls(config=selector_data['config'], logger=logger)
        instance.selected_features = selector_data['selected_features']
        
        if logger:
            logger.info(f"特征选择器已加载: {filepath}")
        
        return instance