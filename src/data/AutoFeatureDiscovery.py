import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


class AutoFeatureDiscovery:
    """自动特征发现类 - 使用深度学习思想的探索式特征生成"""
    
    def __init__(self, logger=None):
        self.logger = logger
        self.discovered_features = {}
        self.feature_generators = []
    
    def discover_arithmetic_features(self, df: pd.DataFrame, 
                                   top_k: int = 20) -> pd.DataFrame:
        """发现算术组合特征"""
        df = df.copy()
        self.logger.info("发现算术组合特征...")
        
        # 选择数值特征
        numeric_cols = []
        for col in df.columns:
            if (df[col].dtype in ['int8', 'int16', 'int32', 'float32', 'float64'] and 
                col not in ['Id', 'ranker_id', 'selected']):
                numeric_cols.append(col)
        
        if len(numeric_cols) < 2:
            return df
        
        # 限制特征数量以避免组合爆炸
        selected_numeric = numeric_cols[:min(10, len(numeric_cols))]
        discovered_count = 0
        
        # 生成二元算术特征
        operations = [
            ('add', lambda x, y: x + y),
            ('subtract', lambda x, y: np.abs(x - y)),
            ('multiply', lambda x, y: x * y),
            ('divide', lambda x, y: x / (y + 1)),
            ('max', lambda x, y: np.maximum(x, y)),
            ('min', lambda x, y: np.minimum(x, y))
        ]
        
        candidate_features = []
        
        for i, col1 in enumerate(selected_numeric):
            for j, col2 in enumerate(selected_numeric[i+1:], i+1):
                for op_name, op_func in operations:
                    if discovered_count >= top_k:
                        break
                    
                    try:
                        feature_name = f'auto_{op_name}_{col1}_{col2}'
                        feature_values = op_func(df[col1].fillna(0), df[col2].fillna(0))
                        
                        # 检查特征质量
                        if self._is_good_feature(feature_values):
                            candidate_features.append((feature_name, feature_values))
                            discovered_count += 1
                            
                    except Exception:
                        continue
                        
                if discovered_count >= top_k:
                    break
            if discovered_count >= top_k:
                break
        
        # 添加发现的特征
        for feature_name, feature_values in candidate_features:
            df[feature_name] = feature_values.astype('int32')
        
        self.logger.info(f"发现 {len(candidate_features)} 个算术特征")
        return df
    
    def discover_conditional_features(self, df: pd.DataFrame, 
                                    top_k: int = 15) -> pd.DataFrame:
        """发现条件特征"""
        df = df.copy()
        self.logger.info("发现条件特征...")
        
        # 选择特征
        numeric_cols = [col for col in df.columns 
                       if df[col].dtype in ['int8', 'int16', 'int32', 'float32'] 
                       and col not in ['Id', 'ranker_id', 'selected']]
        
        if len(numeric_cols) < 2:
            return df
        
        discovered_count = 0
        
        # 生成条件特征
        for i, col1 in enumerate(numeric_cols[:8]):
            for j, col2 in enumerate(numeric_cols[:8]):
                if i == j or discovered_count >= top_k:
                    continue
                
                try:
                    # 条件1: col1 > median(col1) AND col2 > median(col2)
                    med1 = df[col1].median()
                    med2 = df[col2].median()
                    
                    feature_name1 = f'auto_both_high_{col1}_{col2}'
                    feature_values1 = ((df[col1] > med1) & (df[col2] > med2)).astype('int8')
                    
                    if self._is_good_feature(feature_values1):
                        df[feature_name1] = feature_values1
                        discovered_count += 1
                    
                    # 条件2: col1 > 75th percentile OR col2 > 75th percentile
                    p75_1 = df[col1].quantile(0.75)
                    p75_2 = df[col2].quantile(0.75)
                    
                    feature_name2 = f'auto_either_high_{col1}_{col2}'
                    feature_values2 = ((df[col1] > p75_1) | (df[col2] > p75_2)).astype('int8')
                    
                    if self._is_good_feature(feature_values2) and discovered_count < top_k:
                        df[feature_name2] = feature_values2
                        discovered_count += 1
                        
                except Exception:
                    continue
        
        self.logger.info(f"发现 {discovered_count} 个条件特征")
        return df
    
    def discover_ranking_features(self, df: pd.DataFrame, 
                                top_k: int = 10) -> pd.DataFrame:
        """发现排名特征"""
        df = df.copy()
        self.logger.info("发现排名特征...")
        
        if 'ranker_id' not in df.columns:
            return df
        
        # 选择数值特征
        numeric_cols = [col for col in df.columns 
                       if df[col].dtype in ['int8', 'int16', 'int32', 'float32'] 
                       and col not in ['Id', 'ranker_id', 'selected']]
        
        discovered_count = 0
        
        for col in numeric_cols[:top_k]:
            try:
                # 组内排名
                feature_name1 = f'auto_rank_{col}'
                df[feature_name1] = df.groupby('ranker_id')[col].rank().astype('int16')
                
                # 组内百分位排名
                feature_name2 = f'auto_pct_rank_{col}'
                df[feature_name2] = (df.groupby('ranker_id')[col].rank(pct=True) * 100).astype('int8')
                
                # 与组内最值的比较
                feature_name3 = f'auto_vs_max_{col}'
                group_max = df.groupby('ranker_id')[col].transform('max')
                df[feature_name3] = (df[col] / (group_max + 1) * 100).astype('int8')
                
                discovered_count += 3
                
            except Exception:
                continue
        
        self.logger.info(f"发现 {discovered_count} 个排名特征")
        return df
    
    def discover_aggregation_features(self, df: pd.DataFrame, 
                                    top_k: int = 15) -> pd.DataFrame:
        """发现聚合特征"""
        df = df.copy()
        self.logger.info("发现聚合特征...")
        
        if 'ranker_id' not in df.columns:
            return df
        
        # 选择数值特征
        numeric_cols = [col for col in df.columns 
                       if df[col].dtype in ['int8', 'int16', 'int32', 'float32'] 
                       and col not in ['Id', 'ranker_id', 'selected']]
        
        # 聚合函数
        agg_functions = {
            'mean': 'mean',
            'std': 'std', 
            'range': lambda x: x.max() - x.min(),
            'q75_q25': lambda x: x.quantile(0.75) - x.quantile(0.25)
        }
        
        discovered_count = 0
        
        for col in numeric_cols[:min(5, len(numeric_cols))]:
            for agg_name, agg_func in agg_functions.items():
                if discovered_count >= top_k:
                    break
                
                try:
                    feature_name = f'auto_group_{agg_name}_{col}'
                    
                    if agg_name in ['mean', 'std']:
                        feature_values = df.groupby('ranker_id')[col].transform(agg_func).fillna(0)
                    else:
                        feature_values = df.groupby('ranker_id')[col].transform(agg_func).fillna(0)
                    
                    if self._is_good_feature(feature_values):
                        df[feature_name] = feature_values.astype('float32')
                        discovered_count += 1
                        
                except Exception:
                    continue
        
        self.logger.info(f"发现 {discovered_count} 个聚合特征")
        return df
    
    def _is_good_feature(self, feature_values: np.ndarray, 
                        min_variance: float = 0.01, 
                        max_missing_rate: float = 0.8) -> bool:
        """判断特征质量"""
        try:
            # 检查方差
            if np.var(feature_values) < min_variance:
                return False
            
            # 检查缺失率
            missing_rate = np.isnan(feature_values).mean()
            if missing_rate > max_missing_rate:
                return False
            
            # 检查唯一值数量
            unique_values = len(np.unique(feature_values[~np.isnan(feature_values)]))
            if unique_values < 2:
                return False
            
            return True
            
        except Exception:
            return False
    
    def auto_discover_features(self, df: pd.DataFrame, 
                             max_total_features: int = 50) -> pd.DataFrame:
        """自动特征发现主入口"""
        df_processed = df.copy()
        original_features = df_processed.shape[1]
        
        self.logger.info(f"开始自动特征发现，原始特征数：{original_features}")
        
        # 按重要性顺序应用特征发现方法
        discovery_methods = [
            ('arithmetic', self.discover_arithmetic_features, max_total_features // 3),
            ('ranking', self.discover_ranking_features, max_total_features // 5),
            ('conditional', self.discover_conditional_features, max_total_features // 4),
            ('aggregation', self.discover_aggregation_features, max_total_features // 3)
        ]
        
        total_discovered = 0
        
        for method_name, method_func, max_features in discovery_methods:
            if total_discovered >= max_total_features:
                break
                
            try:
                before_count = df_processed.shape[1]
                df_processed = method_func(df_processed, top_k=max_features)
                after_count = df_processed.shape[1]
                
                method_discovered = after_count - before_count
                total_discovered += method_discovered
                
                self.logger.info(f"{method_name} 方法发现 {method_discovered} 个特征")
                
            except Exception as e:
                self.logger.info(f"{method_name} 特征发现失败: {str(e)}")
                continue
        
        final_features = df_processed.shape[1]
        self.discovered_features['auto_discovery'] = {
            'original_features': original_features,
            'final_features': final_features,
            'discovered_features': total_discovered
        }
        
        self.logger.info(f"自动特征发现完成：{original_features} -> {final_features} 特征")
        return df_processed

