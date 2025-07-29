"""
数据工程模块 - 特征工程和数据预处理
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import gc

from src.utils.Common import timer, memory_monitor
from src.utils.DataUtils import DataUtils
from src.utils.MemoryUtils import MemoryUtils


class FeatureEngineering:
    """特征工程类"""
    
    def __init__(self, logger=None):
        self.logger = logger
        self.feature_configs = {}
    
    @timer
    @memory_monitor
    def create_flight_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建航班相关特征"""
        df = df.copy()
        
        # 航班连接数特征
        for leg in [0, 1]:
            segments_cols = [col for col in df.columns if f'legs{leg}_segments' in col and 'flightNumber' in col]
            df[f'legs{leg}_segments_count'] = (df[segments_cols] != -1).sum(axis=1)
        
        # 航班时间特征
        for leg in [0, 1]:
            dep_col = f'legs{leg}_departureAt'
            arr_col = f'legs{leg}_arrivalAt'
            
            if dep_col in df.columns and arr_col in df.columns:
                # 转换为小时
                df[f'legs{leg}_departure_hour'] = (df[dep_col] % 86400) // 3600
                df[f'legs{leg}_arrival_hour'] = (df[arr_col] % 86400) // 3600
                
                # 是否为红眼航班（深夜起飞或早晨到达）
                df[f'legs{leg}_is_redeye'] = (
                    (df[f'legs{leg}_departure_hour'] >= 22) | 
                    (df[f'legs{leg}_arrival_hour'] <= 6)
                ).astype(int)
        
        return df
    
    @timer
    def create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建价格相关特征"""
        df = df.copy()
        
        # 价格分箱特征已在DataEncode中处理
        
        # 价格相对特征（如果有多个选项的话）
        if 'ranker_id' in df.columns:
            # 组内价格排名（按ranker_id分组）
            price_cols = [col for col in df.columns if 'price' in col.lower() and '_bin' in col]
            
            for price_col in price_cols:
                df[f'{price_col}_rank'] = df.groupby('ranker_id')[price_col].rank(method='dense').astype('int8')
                df[f'{price_col}_is_cheapest'] = (df[f'{price_col}_rank'] == 1).astype('int8')
        
        return df
    
    @timer
    def create_route_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建路线相关特征"""
        df = df.copy()
        
        # 机场特征
        airport_cols = [col for col in df.columns if 'airport' in col and 'iata' in col]
        
        # 机场代码频次编码
        for col in airport_cols:
            if col in df.columns:
                # 计算机场出现频次
                airport_counts = df[col].value_counts()
                df[f'{col}_frequency'] = df[col].map(airport_counts).fillna(0).astype('int16')
        
        # 航线类型特征
        for leg in [0, 1]:
            dep_col = f'legs{leg}_segments0_departureFrom_airport_city_iata'
            arr_col = f'legs{leg}_segments0_arrivalTo_airport_city_iata'
            
            if dep_col in df.columns and arr_col in df.columns:
                # 是否为同城航线
                df[f'legs{leg}_is_same_city'] = (df[dep_col] == df[arr_col]).astype('int8')
        
        return df
    
    @timer
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建时间相关特征"""
        df = df.copy()
        
        # 请求时间特征
        if 'requestDate' in df.columns:
            # 转换为时间戳再转回时间特征
            request_ts = df['requestDate']
            
            # 提取时间特征
            df['request_hour'] = ((request_ts % 86400) // 3600).astype('int8')
            df['request_day_of_week'] = ((request_ts // 86400 + 4) % 7).astype('int8')  # 1970-01-01是周四
            df['request_is_weekend'] = (df['request_day_of_week'] >= 5).astype('int8')
        
        # 提前预订天数
        for leg in [0, 1]:
            dep_col = f'legs{leg}_departureAt'
            if dep_col in df.columns and 'requestDate' in df.columns:
                days_ahead = (df[dep_col] - df['requestDate']) // 86400
                df[f'legs{leg}_days_ahead'] = np.clip(days_ahead, 0, 365).astype('int16')
                
                # 分类：当天、近期、远期
                df[f'legs{leg}_booking_type'] = pd.cut(
                    df[f'legs{leg}_days_ahead'], 
                    bins=[-1, 0, 7, 30, float('inf')], 
                    labels=[0, 1, 2, 3]
                ).astype('int8')
        
        return df
    
    @timer
    def create_passenger_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建乘客相关特征"""
        df = df.copy()
        
        # 乘客数量特征
        if 'pricingInfo_passengerCount' in df.columns:
            df['is_single_passenger'] = (df['pricingInfo_passengerCount'] == 1).astype('int8')
            df['is_group_travel'] = (df['pricingInfo_passengerCount'] >= 3).astype('int8')
        
        # VIP特征交互
        if 'isVip' in df.columns and 'corporateTariffCode' in df.columns:
            df['vip_corporate'] = (df['isVip'] & (df['corporateTariffCode'] != -1)).astype('int8')
        
        return df
    
    @timer
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建交互特征"""
        df = df.copy()
        
        # 价格-时间交互
        price_cols = [col for col in df.columns if 'price' in col.lower() and '_bin' in col]
        time_cols = [col for col in df.columns if 'hour' in col]
        
        for price_col in price_cols[:2]:  # 限制特征数量
            for time_col in time_cols[:2]:
                if price_col in df.columns and time_col in df.columns:
                    interaction_name = f'{price_col}_x_{time_col}'
                    df[interaction_name] = (df[price_col] * df[time_col]).astype('int16')
        
        return df

class DataQuality:
    """数据质量管理类"""
    
    def __init__(self, logger=None):
        self.logger = logger
    
    @timer
    def detect_outliers(self, df: pd.DataFrame, method: str = 'iqr') -> pd.DataFrame:
        """检测异常值"""
        outlier_info = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in ['Id', 'ranker_id', 'profileId', 'companyID']:
                continue
                
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
                outlier_info[col] = {
                    'count': outliers.sum(),
                    'percentage': outliers.mean() * 100,
                    'bounds': (lower_bound, upper_bound)
                }
        
        if self.logger:
            total_outliers = sum(info['count'] for info in outlier_info.values())
            self.logger.info(f"检测到 {total_outliers} 个异常值")
        
        return outlier_info
    
    @timer
    def handle_missing_values(self, df: pd.DataFrame, strategy: Dict[str, str] = None) -> pd.DataFrame:
        """处理缺失值"""
        df = df.copy()
        
        default_strategy = {
            'numeric': 'median',
            'categorical': 'mode',
            'boolean': 'mode'
        }
        
        strategy = strategy or default_strategy
        
        for col in df.columns:
            if df[col].isnull().any():
                dtype = df[col].dtype
                
                if pd.api.types.is_numeric_dtype(dtype):
                    if strategy.get('numeric') == 'median':
                        df[col].fillna(df[col].median(), inplace=True)
                    elif strategy.get('numeric') == 'mean':
                        df[col].fillna(df[col].mean(), inplace=True)
                    else:
                        df[col].fillna(-1, inplace=True)
                
                elif pd.api.types.is_bool_dtype(dtype):
                    df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else False, inplace=True)
                
                else:
                    df[col].fillna('unknown', inplace=True)
        
        return df

class DataEngineering:
    """数据工程主类，整合所有数据处理功能"""
    
    def __init__(self, logger=None):
        self.logger = logger
        self.feature_eng = FeatureEngineering(logger)
        self.data_quality = DataQuality(logger)
        self.processing_stats = {}
    
    @timer
    @memory_monitor
    def process_features(self, df: pd.DataFrame, 
                        feature_types: List[str] = None) -> pd.DataFrame:
        """
        处理特征工程
        
        Args:
            df: 输入数据
            feature_types: 要创建的特征类型列表
            
        Returns:
            处理后的数据
        """
        if feature_types is None:
            feature_types = ['flight', 'price', 'route', 'temporal', 'passenger']
        
        df_processed = df.copy()
        original_shape = df_processed.shape
        
        # 记录处理前状态
        self.processing_stats['original_shape'] = original_shape
        self.processing_stats['original_memory'] = df_processed.memory_usage(deep=True).sum() / 1024**2
        
        # 按类型创建特征
        if 'flight' in feature_types:
            df_processed = self.feature_eng.create_flight_features(df_processed)
        
        if 'price' in feature_types:
            df_processed = self.feature_eng.create_price_features(df_processed)
        
        if 'route' in feature_types:
            df_processed = self.feature_eng.create_route_features(df_processed)
        
        if 'temporal' in feature_types:
            df_processed = self.feature_eng.create_temporal_features(df_processed)
        
        if 'passenger' in feature_types:
            df_processed = self.feature_eng.create_passenger_features(df_processed)
        
        if 'interaction' in feature_types:
            df_processed = self.feature_eng.create_interaction_features(df_processed)
        
        # 内存优化
        df_processed = MemoryUtils.optimize_dataframe_memory(df_processed)
        
        # 记录处理后状态
        final_shape = df_processed.shape
        self.processing_stats['final_shape'] = final_shape
        self.processing_stats['final_memory'] = df_processed.memory_usage(deep=True).sum() / 1024**2
        self.processing_stats['features_added'] = final_shape[1] - original_shape[1]
        
        if self.logger:
            self.logger.info(f"特征工程完成: {original_shape} -> {final_shape}")
            self.logger.info(f"新增特征: {self.processing_stats['features_added']}")
            memory_saved = self.processing_stats['original_memory'] - self.processing_stats['final_memory']
            if memory_saved > 0:
                self.logger.info(f"内存优化节省: {memory_saved:.1f}MB")
        
        # 清理内存
        del df
        gc.collect()
        
        return df_processed
    
    @timer
    def validate_and_clean(self, df: pd.DataFrame, 
                          clean_outliers: bool = False,
                          outlier_method: str = 'iqr') -> Tuple[pd.DataFrame, Dict]:
        """
        验证和清理数据
        
        Args:
            df: 输入数据
            clean_outliers: 是否清理异常值
            outlier_method: 异常值检测方法
            
        Returns:
            (清理后的数据, 质量报告)
        """
        df_clean = df.copy()
        
        # 检测异常值
        outlier_info = self.data_quality.detect_outliers(df_clean, method=outlier_method)
        
        # 处理异常值（如果需要）
        if clean_outliers:
            for col, info in outlier_info.items():
                if info['percentage'] > 5:  # 如果异常值超过5%，只做截尾处理
                    lower_bound, upper_bound = info['bounds']
                    df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)
        
        # 处理缺失值
        df_clean = self.data_quality.handle_missing_values(df_clean)
        
        # 生成质量报告
        quality_report = {
            'outliers': outlier_info,
            'missing_values_before': df.isnull().sum().to_dict(),
            'missing_values_after': df_clean.isnull().sum().to_dict(),
            'shape_before': df.shape,
            'shape_after': df_clean.shape
        }
        
        return df_clean, quality_report
    
    def get_processing_summary(self) -> Dict:
        """获取处理总结"""
        return self.processing_stats
    
    @timer
    def create_feature_importance_map(self, df: pd.DataFrame) -> Dict[str, int]:
        """创建特征重要性映射（基于非缺失值比例和方差）"""
        feature_importance = {}
        
        for col in df.select_dtypes(include=[np.number]).columns:
            if col in ['Id', 'ranker_id', 'profileId', 'companyID']:
                continue
            
            # 计算非缺失值比例
            non_missing_ratio = df[col].notna().mean()
            
            # 计算标准化方差
            if df[col].var() > 0:
                normalized_var = df[col].var() / (df[col].mean() + 1e-8)
            else:
                normalized_var = 0
            
            # 综合分数
            importance_score = non_missing_ratio * normalized_var
            feature_importance[col] = importance_score
        
        # 排序
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        if self.logger:
            self.logger.info(f"计算了 {len(sorted_features)} 个特征的重要性")
            self.logger.info(f"Top 5 特征: {[f[0] for f in sorted_features[:5]]}")
        
        return dict(sorted_features)