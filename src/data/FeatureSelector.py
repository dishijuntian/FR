"""
特征选择器 - 多种筛选机制的集成
支持配置驱动的特征筛选策略
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from sklearn.feature_selection import (
    SelectKBest, SelectPercentile, f_classif, f_regression, chi2,
    mutual_info_classif, mutual_info_regression, VarianceThreshold,
    SelectFromModel, RFE, RFECV
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.metrics import mutual_info_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False


class FeatureSelector:
    """特征选择器主类"""
    
    def __init__(self, config: Dict = None, logger=None):
        """
        初始化特征选择器
        
        Args:
            config: 特征选择配置
            logger: 日志记录器
        """
        self.logger = logger
        self.config = config or self._get_default_config()
        self.selected_features = None
        self.selection_stats = {}
        
    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            'enabled': True,
            'max_features': 200,
            'selection_methods': ['variance', 'correlation', 'importance', 'statistical'],
            'variance_threshold': 0.01,
            'correlation_threshold': 0.95,
            'importance_method': 'lightgbm',  # lightgbm, random_forest, lasso
            'statistical_method': 'mutual_info',  # mutual_info, f_test, chi2
            'sample_size': 50000,
            'early_stopping': True,
            'save_feature_importance': True
        }
    
    def remove_constant_features(self, X: pd.DataFrame) -> List[str]:
        """移除常量特征和近似常量特征"""
        self.logger.info("移除常量特征...")
        valid_features = []
        
        variance_threshold = self.config.get('variance_threshold', 0.01)
        
        for col in tqdm(X.columns, desc="检查常量特征", leave=False):
            try:
                # 检查唯一值数量
                unique_count = X[col].nunique()
                if unique_count <= 1:
                    continue
                
                # 检查唯一值比例
                unique_ratio = unique_count / len(X)
                if unique_ratio < 0.01:
                    continue
                
                # 检查方差
                if X[col].dtype in ['int8', 'int16', 'int32', 'float32', 'float64']:
                    variance = X[col].var()
                    if variance < variance_threshold:
                        continue
                
                # 检查众数占比
                mode_ratio = X[col].value_counts().iloc[0] / len(X)
                if mode_ratio > 0.95:
                    continue
                
                valid_features.append(col)
                
            except Exception as e:
                self.logger.info(f"检查特征 {col} 时出错: {str(e)}")
                continue
        
        removed_count = len(X.columns) - len(valid_features)
        self.logger.info(f"移除了 {removed_count} 个常量/近似常量特征")
        self.selection_stats['constant_removed'] = removed_count
        
        return valid_features
    
    def remove_correlated_features(self, X: pd.DataFrame) -> List[str]:
        """移除高相关性特征"""
        self.logger.info("移除高相关性特征...")
        correlation_threshold = self.config.get('correlation_threshold', 0.95)
        
        # 计算相关性矩阵
        try:
            corr_matrix = X.corr().abs()
            
            # 获取上三角矩阵
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            
            # 找到高相关性的特征对
            to_drop = []
            for col in upper_triangle.columns:
                if any(upper_triangle[col] > correlation_threshold):
                    # 保留方差更大的特征
                    correlated_features = upper_triangle.index[upper_triangle[col] > correlation_threshold].tolist()
                    correlated_features.append(col)
                    
                    # 计算方差并保留方差最大的
                    variances = X[correlated_features].var()
                    features_to_drop = variances.index[variances != variances.max()].tolist()
                    to_drop.extend(features_to_drop)
            
            # 去重
            to_drop = list(set(to_drop))
            remaining_features = [col for col in X.columns if col not in to_drop]
            
            self.logger.info(f"移除了 {len(to_drop)} 个高相关性特征")
            self.selection_stats['correlation_removed'] = len(to_drop)
            
            return remaining_features
            
        except Exception as e:
            self.logger.info(f"相关性分析失败: {str(e)}")
            return X.columns.tolist()
    
    def select_by_importance(self, X: pd.DataFrame, y: pd.Series, top_k: int) -> Tuple[List[str], Dict]:
        """基于模型重要性选择特征"""
        self.logger.info("基于模型重要性选择特征...")
        importance_method = self.config.get('importance_method', 'lightgbm')
        
        # 采样以提高效率
        sample_size = min(self.config.get('sample_size', 50000), len(X))
        if len(X) > sample_size:
            sample_idx = np.random.choice(len(X), sample_size, replace=False)
            X_sample, y_sample = X.iloc[sample_idx], y.iloc[sample_idx]
        else:
            X_sample, y_sample = X, y
        
        # 确定任务类型
        is_classification = len(y.unique()) <= 10  # 简单启发式判断
        y_processed = y_sample if not is_classification else (y_sample == 1).astype(int)
        
        importance_dict = {}
        
        try:
            if importance_method == 'lightgbm' and HAS_LGB:
                if is_classification:
                    model = lgb.LGBMClassifier(
                        n_estimators=100, max_depth=6, verbose=-1, 
                        random_state=42, force_col_wise=True
                    )
                else:
                    model = lgb.LGBMRegressor(
                        n_estimators=100, max_depth=6, verbose=-1, 
                        random_state=42, force_col_wise=True
                    )
                model.fit(X_sample, y_processed)
                importance_dict = dict(zip(X.columns, model.feature_importances_))
                
            elif importance_method == 'random_forest':
                if is_classification:
                    model = RandomForestClassifier(
                        n_estimators=100, max_depth=8, random_state=42, n_jobs=4
                    )
                else:
                    model = RandomForestRegressor(
                        n_estimators=100, max_depth=8, random_state=42, n_jobs=4
                    )
                model.fit(X_sample, y_processed)
                importance_dict = dict(zip(X.columns, model.feature_importances_))
                
            elif importance_method == 'lasso':
                if is_classification:
                    # 对于分类任务，使用随机森林作为备选
                    model = RandomForestClassifier(
                        n_estimators=50, max_depth=6, random_state=42, n_jobs=4
                    )
                    model.fit(X_sample, y_processed)
                    importance_dict = dict(zip(X.columns, model.feature_importances_))
                else:
                    model = LassoCV(cv=3, random_state=42)
                    model.fit(X_sample, y_processed)
                    importance_dict = dict(zip(X.columns, np.abs(model.coef_)))
                    
            else:
                # 默认使用随机森林
                if is_classification:
                    model = RandomForestClassifier(
                        n_estimators=50, max_depth=6, random_state=42, n_jobs=4
                    )
                else:
                    model = RandomForestRegressor(
                        n_estimators=50, max_depth=6, random_state=42, n_jobs=4
                    )
                model.fit(X_sample, y_processed)
                importance_dict = dict(zip(X.columns, model.feature_importances_))
                
        except Exception as e:
            self.logger.info(f"重要性模型训练失败: {str(e)}, 使用随机森林备选方案")
            # 备选方案
            try:
                if is_classification:
                    model = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42, n_jobs=2)
                else:
                    model = RandomForestRegressor(n_estimators=50, max_depth=6, random_state=42, n_jobs=2)
                model.fit(X_sample, y_processed)
                importance_dict = dict(zip(X.columns, model.feature_importances_))
            except Exception as e2:
                self.logger.info(f"备选方案也失败: {str(e2)}, 返回随机重要性")
                importance_dict = dict(zip(X.columns, np.random.random(len(X.columns))))
        
        # 选择top_k特征
        selected_features = [f[0] for f in sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_k]]
        
        self.logger.info(f"基于重要性选择了 {len(selected_features)} 个特征")
        return selected_features, importance_dict
    
    def select_by_statistical(self, X: pd.DataFrame, y: pd.Series, top_k: int) -> List[str]:
        """基于统计方法选择特征"""
        self.logger.info("基于统计方法选择特征...")
        statistical_method = self.config.get('statistical_method', 'mutual_info')
        
        # 采样以提高效率
        sample_size = min(self.config.get('sample_size', 30000), len(X))
        if len(X) > sample_size:
            sample_idx = np.random.choice(len(X), sample_size, replace=False)
            X_sample, y_sample = X.iloc[sample_idx], y.iloc[sample_idx]
        else:
            X_sample, y_sample = X, y
        
        # 确定任务类型
        is_classification = len(y.unique()) <= 10
        y_processed = y_sample if not is_classification else (y_sample == 1).astype(int)
        
        try:
            if statistical_method == 'mutual_info':
                if is_classification:
                    scores = mutual_info_classif(X_sample, y_processed, random_state=42)
                else:
                    scores = mutual_info_regression(X_sample, y_processed, random_state=42)
                    
            elif statistical_method == 'f_test':
                if is_classification:
                    selector = SelectKBest(score_func=f_classif, k=min(top_k, X_sample.shape[1]))
                else:
                    selector = SelectKBest(score_func=f_regression, k=min(top_k, X_sample.shape[1]))
                selector.fit(X_sample, y_processed)
                scores = selector.scores_
                
            elif statistical_method == 'chi2' and is_classification:
                # Chi2要求非负特征值
                X_sample_non_neg = X_sample.copy()
                X_sample_non_neg[X_sample_non_neg < 0] = 0
                selector = SelectKBest(score_func=chi2, k=min(top_k, X_sample.shape[1]))
                selector.fit(X_sample_non_neg, y_processed)
                scores = selector.scores_
                
            else:
                # 默认使用互信息
                if is_classification:
                    scores = mutual_info_classif(X_sample, y_processed, random_state=42)
                else:
                    scores = mutual_info_regression(X_sample, y_processed, random_state=42)
            
            # 选择top_k特征
            scores_dict = dict(zip(X.columns, scores))
            selected_features = [f[0] for f in sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)[:top_k]]
            
            self.logger.info(f"基于统计方法选择了 {len(selected_features)} 个特征")
            return selected_features
            
        except Exception as e:
            self.logger.info(f"统计方法选择失败: {str(e)}, 返回前{top_k}个特征")
            return X.columns.tolist()[:top_k]
    
    def select_by_univariate(self, X: pd.DataFrame, y: pd.Series, top_k: int) -> List[str]:
        """单变量特征选择"""
        self.logger.info("单变量特征选择...")
        
        # 确定任务类型
        is_classification = len(y.unique()) <= 10
        y_processed = y if not is_classification else (y == 1).astype(int)
        
        try:
            if is_classification:
                selector = SelectKBest(score_func=f_classif, k=min(top_k, X.shape[1]))
            else:
                selector = SelectKBest(score_func=f_regression, k=min(top_k, X.shape[1]))
            
            selector.fit(X, y_processed)
            selected_features = X.columns[selector.get_support()].tolist()
            
            self.logger.info(f"单变量选择了 {len(selected_features)} 个特征")
            return selected_features
            
        except Exception as e:
            self.logger.info(f"单变量选择失败: {str(e)}")
            return X.columns.tolist()[:top_k]
    
    def select_by_model_selection(self, X: pd.DataFrame, y: pd.Series, top_k: int) -> List[str]:
        """基于模型的特征选择（RFE等）"""
        self.logger.info("基于模型的递归特征选择...")
        
        # 采样以提高效率
        sample_size = min(self.config.get('sample_size', 20000), len(X))
        if len(X) > sample_size:
            sample_idx = np.random.choice(len(X), sample_size, replace=False)
            X_sample, y_sample = X.iloc[sample_idx], y.iloc[sample_idx]
        else:
            X_sample, y_sample = X, y
        
        # 确定任务类型
        is_classification = len(y.unique()) <= 10
        y_processed = y_sample if not is_classification else (y_sample == 1).astype(int)
        
        try:
            # 使用轻量级模型进行RFE
            if is_classification:
                estimator = RandomForestClassifier(n_estimators=20, max_depth=5, random_state=42, n_jobs=2)
            else:
                estimator = RandomForestRegressor(n_estimators=20, max_depth=5, random_state=42, n_jobs=2)
            
            # 递归特征消除
            selector = RFE(estimator=estimator, n_features_to_select=min(top_k, X_sample.shape[1]), step=0.1)
            selector.fit(X_sample, y_processed)
            
            selected_features = X.columns[selector.support_].tolist()
            
            self.logger.info(f"RFE选择了 {len(selected_features)} 个特征")
            return selected_features
            
        except Exception as e:
            self.logger.info(f"RFE选择失败: {str(e)}")
            return X.columns.tolist()[:top_k]
    
    def ensemble_selection(self, X: pd.DataFrame, y: pd.Series, 
                          methods_results: Dict[str, List[str]], 
                          target_count: int) -> List[str]:
        """集成多种选择方法的结果"""
        self.logger.info("集成多种特征选择方法...")
        
        # 统计每个特征被选中的次数
        feature_votes = {}
        for method, features in methods_results.items():
            for feature in features:
                feature_votes[feature] = feature_votes.get(feature, 0) + 1
        
        # 按投票数排序
        sorted_features = sorted(feature_votes.items(), key=lambda x: x[1], reverse=True)
        
        # 选择得票最多的特征
        selected_features = [f[0] for f in sorted_features[:target_count]]
        
        # 如果特征数不够，补充高方差特征
        if len(selected_features) < target_count:
            remaining_features = [col for col in X.columns if col not in selected_features]
            if remaining_features:
                # 按方差排序，选择高方差特征
                variances = X[remaining_features].var().sort_values(ascending=False)
                additional_features = variances.index[:target_count - len(selected_features)].tolist()
                selected_features.extend(additional_features)
        
        self.logger.info(f"集成选择了 {len(selected_features)} 个特征")
        
        # 记录投票统计
        self.selection_stats['voting_results'] = {
            'total_methods': len(methods_results),
            'feature_votes': feature_votes,
            'consensus_features': [f for f, votes in feature_votes.items() if votes >= len(methods_results) // 2]
        }
        
        return selected_features
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, 
                       target_features: Optional[int] = None) -> List[str]:
        """主要的特征选择入口"""
        if not self.config.get('enabled', True):
            self.logger.info("特征选择已禁用")
            return X.columns.tolist()
        
        target_features = target_features or self.config.get('max_features', 200)
        self.logger.info(f"开始特征选择: {X.shape[1]} -> {target_features}")
        
        # 记录原始特征数
        self.selection_stats['original_features'] = X.shape[1]
        self.selection_stats['target_features'] = target_features
        
        # 第一步：移除常量特征
        valid_features = self.remove_constant_features(X)
        X_filtered = X[valid_features]
        
        if len(valid_features) <= target_features:
            self.selected_features = valid_features
            self.selection_stats['final_features'] = len(valid_features)
            self.logger.info(f"常量过滤后特征数({len(valid_features)})已满足要求")
            return valid_features
        
        # 第二步：移除高相关性特征
        if 'correlation' in self.config.get('selection_methods', []):
            valid_features = self.remove_correlated_features(X_filtered)
            X_filtered = X_filtered[valid_features]
            
            if len(valid_features) <= target_features:
                self.selected_features = valid_features
                self.selection_stats['final_features'] = len(valid_features)
                self.logger.info(f"相关性过滤后特征数({len(valid_features)})已满足要求")
                return valid_features
        
        # 第三步：应用多种选择方法
        selection_methods = self.config.get('selection_methods', ['importance', 'statistical'])
        methods_results = {}
        
        # 计算每种方法应该选择的特征数
        method_count = len([m for m in selection_methods if m in ['importance', 'statistical', 'univariate', 'rfe']])
        features_per_method = min(target_features, max(target_features // method_count, target_features // 2))
        
        # 重要性方法
        if 'importance' in selection_methods:
            try:
                importance_features, importance_dict = self.select_by_importance(
                    X_filtered, y, features_per_method
                )
                methods_results['importance'] = importance_features
                if self.config.get('save_feature_importance', True):
                    self.selection_stats['feature_importance'] = importance_dict
            except Exception as e:
                self.logger.info(f"重要性选择失败: {str(e)}")
        
        # 统计方法
        if 'statistical' in selection_methods:
            try:
                statistical_features = self.select_by_statistical(X_filtered, y, features_per_method)
                methods_results['statistical'] = statistical_features
            except Exception as e:
                self.logger.info(f"统计方法选择失败: {str(e)}")
        
        # 单变量方法
        if 'univariate' in selection_methods:
            try:
                univariate_features = self.select_by_univariate(X_filtered, y, features_per_method)
                methods_results['univariate'] = univariate_features
            except Exception as e:
                self.logger.info(f"单变量选择失败: {str(e)}")
        
        # RFE方法
        if 'rfe' in selection_methods:
            try:
                rfe_features = self.select_by_model_selection(X_filtered, y, features_per_method)
                methods_results['rfe'] = rfe_features
            except Exception as e:
                self.logger.info(f"RFE选择失败: {str(e)}")
        
        # 如果有多种方法，进行集成
        if len(methods_results) > 1:
            final_features = self.ensemble_selection(X_filtered, y, methods_results, target_features)
        elif len(methods_results) == 1:
            method_name = list(methods_results.keys())[0]
            final_features = methods_results[method_name][:target_features]
        else:
            # 如果所有方法都失败，使用方差选择
            self.logger.info("所有选择方法都失败，使用方差选择")
            variances = X_filtered.var().sort_values(ascending=False)
            final_features = variances.index[:target_features].tolist()
        
        self.selected_features = final_features
        self.selection_stats['final_features'] = len(final_features)
        self.selection_stats['methods_used'] = list(methods_results.keys())
        
        self.logger.info(f"特征选择完成: {X.shape[1]} -> {len(final_features)}")
        return final_features
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """使用选中的特征转换数据"""
        if self.selected_features is None:
            raise ValueError("请先调用select_features方法")
        
        # 确保所有选中的特征都存在于数据中
        available_features = [f for f in self.selected_features if f in X.columns]
        if len(available_features) < len(self.selected_features):
            missing_features = set(self.selected_features) - set(available_features)
            self.logger.info(f"警告: {len(missing_features)} 个选中的特征在数据中不存在")
        
        return X[available_features].copy()
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series, 
                     target_features: Optional[int] = None) -> pd.DataFrame:
        """拟合并转换数据"""
        self.select_features(X, y, target_features)
        return self.transform(X)
    
    def get_feature_importance(self) -> Dict:
        """获取特征重要性"""
        return self.selection_stats.get('feature_importance', {})
    
    def get_selection_stats(self) -> Dict:
        """获取选择统计信息"""
        return self.selection_stats.copy()
    
    def save_selection_report(self, filepath: str):
        """保存特征选择报告"""
        report = {
            'config': self.config,
            'selected_features': self.selected_features,
            'stats': self.selection_stats
        }
        
        try:
            import json
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)
            self.logger.info(f"特征选择报告已保存: {filepath}")
        except Exception as e:
            self.logger.info(f"保存报告失败: {str(e)}")


class MultiTargetFeatureSelector:
    """多目标特征选择器"""
    
    def __init__(self, config: Dict = None, logger=None):
        self.logger = logger
        self.config = config or {}
        self.selectors = {}
        
    def select_for_multiple_targets(self, X: pd.DataFrame, 
                                   targets: Dict[str, pd.Series], 
                                   strategy: str = 'union') -> Dict[str, List[str]]:
        """
        为多个目标变量选择特征
        
        Args:
            X: 特征数据
            targets: 目标变量字典 {target_name: target_series}
            strategy: 合并策略 ('union', 'intersection', 'weighted')
        """
        results = {}
        
        for target_name, target_series in targets.items():
            selector = FeatureSelector(self.config, self.logger)
            selected_features = selector.select_features(X, target_series)
            self.selectors[target_name] = selector
            results[target_name] = selected_features
        
        # 根据策略合并结果
        if strategy == 'union':
            # 并集：所有目标选中的特征
            all_features = set()
            for features in results.values():
                all_features.update(features)
            results['combined'] = list(all_features)
            
        elif strategy == 'intersection':
            # 交集：所有目标都选中的特征
            common_features = set(list(results.values())[0])
            for features in list(results.values())[1:]:
                common_features.intersection_update(features)
            results['combined'] = list(common_features)
            
        elif strategy == 'weighted':
            # 加权：根据特征重要性加权合并
            feature_weights = {}
            for target_name, features in results.items():
                importance = self.selectors[target_name].get_feature_importance()
                for feature in features:
                    weight = importance.get(feature, 0.5)  # 默认权重
                    feature_weights[feature] = feature_weights.get(feature, 0) + weight
            
            # 按权重排序选择
            max_features = self.config.get('max_features', 200)
            sorted_features = sorted(feature_weights.items(), key=lambda x: x[1], reverse=True)
            results['combined'] = [f[0] for f in sorted_features[:max_features]]
        
        return results


class AdaptiveFeatureSelector:
    """自适应特征选择器 - 根据数据特性自动调整策略"""
    
    def __init__(self, logger=None):
        self.logger = logger
        
    def _analyze_data_characteristics(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """分析数据特性"""
        characteristics = {
            'n_samples': len(X),
            'n_features': X.shape[1],
            'feature_to_sample_ratio': X.shape[1] / len(X),
            'target_type': 'classification' if len(y.unique()) <= 10 else 'regression',
            'target_cardinality': len(y.unique()),
            'missing_rate': X.isnull().sum().sum() / (X.shape[0] * X.shape[1]),
            'numeric_features': len(X.select_dtypes(include=[np.number]).columns),
            'categorical_features': len(X.select_dtypes(exclude=[np.number]).columns)
        }
        
        return characteristics
    
    def _get_adaptive_config(self, characteristics: Dict) -> Dict:
        """根据数据特性生成自适应配置"""
        config = {
            'enabled': True,
            'variance_threshold': 0.01,
            'correlation_threshold': 0.95,
            'sample_size': min(50000, characteristics['n_samples']),
            'save_feature_importance': True
        }
        
        # 根据特征数量调整策略
        if characteristics['n_features'] > 1000:
            # 高维数据：使用多种快速方法
            config['selection_methods'] = ['variance', 'correlation', 'importance', 'statistical']
            config['max_features'] = min(500, characteristics['n_features'] // 4)
            config['importance_method'] = 'lightgbm'  # 更快
            
        elif characteristics['n_features'] > 200:
            # 中等维度：平衡方法
            config['selection_methods'] = ['variance', 'correlation', 'importance', 'statistical', 'univariate']
            config['max_features'] = min(200, characteristics['n_features'] // 2)
            config['importance_method'] = 'random_forest'
            
        else:
            # 低维数据：使用所有方法
            config['selection_methods'] = ['variance', 'correlation', 'importance', 'statistical', 'univariate', 'rfe']
            config['max_features'] = characteristics['n_features']
            config['importance_method'] = 'random_forest'
        
        # 根据样本量调整
        if characteristics['n_samples'] < 1000:
            config['sample_size'] = characteristics['n_samples']
            config['selection_methods'] = ['variance', 'correlation', 'statistical']  # 简化方法
        
        # 根据特征-样本比调整
        if characteristics['feature_to_sample_ratio'] > 0.5:
            # 特征多样本少：更激进的特征选择
            config['max_features'] = min(config['max_features'], characteristics['n_samples'] // 2)
            config['correlation_threshold'] = 0.9  # 更严格的相关性阈值
        
        return config
    
    def select_features(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """自适应特征选择"""
        # 分析数据特性
        characteristics = self._analyze_data_characteristics(X, y)
        
        if self.logger:
            self.logger.info(f"数据特性分析: {characteristics}")
        
        # 生成自适应配置
        config = self._get_adaptive_config(characteristics)
        
        if self.logger:
            self.logger.info(f"自适应配置: {config}")
        
        # 执行特征选择
        selector = FeatureSelector(config, self.logger)
        selected_features = selector.select_features(X, y)
        
        return selected_features