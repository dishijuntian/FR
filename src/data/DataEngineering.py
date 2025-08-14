"""
合并版数据工程模块 - 集成深度特征工程和自动特征发现
适配编码后数据，包含经济学、行为学、运营等多维度特征
"""
import pandas as pd
import numpy as np
import os
import gc
from typing import Dict, List, Tuple
from pathlib import Path
import pyarrow.parquet as pq
import warnings
warnings.filterwarnings('ignore')

from utils.Common import timer
from utils.MemoryUtils import MemoryUtils


class DataEngineering:
    """完整数据工程类"""
    
    def __init__(self, logger=None):
        self.logger = logger
        self.processing_stats = {}
        self.feature_importance_cache = {}
        self.cluster_models = {}

    # ==================== 核心特征工程 ====================
    
    @timer
    def create_core_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建核心特征（基于编码后数据优化）"""
        df = df.copy()
        
        # 1. 价格特征
        if 'taxes_bin' in df.columns and 'totalPrice_bin' in df.columns:
            df['tax_rate'] = df['taxes_bin'] / (df['totalPrice_bin'] + 1)
            df['log_price_proxy'] = np.log1p(df['totalPrice_bin'])
        
        # 2. 持续时间特征
        duration_cols = ['legs0_duration', 'legs1_duration']
        existing_duration_cols = [col for col in duration_cols if col in df.columns]
        
        if len(existing_duration_cols) >= 2:
            df['total_duration'] = df['legs0_duration'].fillna(0) + df['legs1_duration'].fillna(0)
            df['duration_ratio'] = np.where(
                df['legs1_duration'].fillna(0) > 0,
                df['legs0_duration'] / (df['legs1_duration'] + 1),
                1.0
            )
        elif 'legs0_duration' in df.columns:
            df['total_duration'] = df['legs0_duration'].fillna(0)
            df['duration_ratio'] = 1.0
        
        # 3. 行程类型特征
        if 'legs1_duration' in df.columns:
            df['is_one_way'] = (
                (df['legs1_duration'] == -1) | 
                (df['legs1_duration'] == 0) |
                (df['legs1_duration'].isna())
            ).astype('int8')
        else:
            df['is_one_way'] = 1
        
        # 4. 航班段数统计
        for leg in [0, 1]:
            segment_cols = [col for col in df.columns 
                           if f'legs{leg}_segments' in col and 'flightNumber' in col]
            if segment_cols:
                df[f'n_segments_leg{leg}'] = (df[segment_cols] != -1).sum(axis=1).astype('int8')
            else:
                df[f'n_segments_leg{leg}'] = 0
        
        # 5. 常旅客特征
        if 'frequentFlyer' in df.columns:
            df['has_frequent_flyer'] = (df['frequentFlyer'] != -1).astype('int8')
        
        # 6. 二值特征
        binary_features = [
            ('corporateTariffCode', 'has_corporate_tariff'),
            ('pricingInfo_isAccessTP', 'has_access_tp'),
            ('isVip', 'is_vip'),
        ]
        
        for source_col, target_col in binary_features:
            if source_col in df.columns:
                df[target_col] = ((df[source_col] == 1) | (df[source_col] != -1)).astype('int8')
        
        # 7. 取消/改签规则特征
        if 'miniRules0_monetaryAmount' in df.columns and 'miniRules0_statusInfos' in df.columns:
            df['free_cancel'] = (
                (df['miniRules0_monetaryAmount'] == 0) & 
                (df['miniRules0_statusInfos'] == 1)
            ).astype('int8')
        
        if 'miniRules1_monetaryAmount' in df.columns and 'miniRules1_statusInfos' in df.columns:
            df['free_exchange'] = (
                (df['miniRules1_monetaryAmount'] == 0) & 
                (df['miniRules1_statusInfos'] == 1)
            ).astype('int8')
        
        return df
    
    @timer
    def create_route_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建路线相关特征"""
        df = df.copy()
        
        # 舱位等级特征
        cabin_cols = [col for col in df.columns if 'cabinClass' in col]
        if len(cabin_cols) >= 2:
            df['avg_cabin_class'] = df[cabin_cols].replace(-1, np.nan).mean(axis=1, skipna=True).fillna(0)
        
        # 直飞特征
        if 'n_segments_leg0' in df.columns:
            df['is_direct_leg0'] = (df['n_segments_leg0'] == 1).astype('int8')
        
        if 'n_segments_leg1' in df.columns and 'is_one_way' in df.columns:
            df['is_direct_leg1'] = np.where(
                df['is_one_way'] == 1, 
                0, 
                (df['n_segments_leg1'] == 1).astype('int8')
            )
        
        # 总段数
        if 'n_segments_leg0' in df.columns and 'n_segments_leg1' in df.columns:
            df['total_segments'] = df['n_segments_leg0'] + df['n_segments_leg1']
            
            # 都是直飞
            if 'is_direct_leg0' in df.columns and 'is_direct_leg1' in df.columns:
                df['both_direct'] = (df['is_direct_leg0'] & df['is_direct_leg1']).astype('int8')
        
        return df
    
    @timer
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建时间相关特征"""
        df = df.copy()
        
        # 处理时间戳格式的时间列
        time_cols = ['legs0_departureAt', 'legs0_arrivalAt', 'legs1_departureAt', 'legs1_arrivalAt']
        
        for col in time_cols:
            if col in df.columns:
                # 时间戳转小时（假设-1表示缺失）
                valid_mask = df[col] != -1
                df[f'{col}_hour'] = np.where(
                    valid_mask,
                    (df[col] % 86400) // 3600,  # Unix时间戳转小时
                    12  # 默认值
                ).astype('int8')
                
                # 工作日（简化处理，基于时间戳）
                df[f'{col}_weekday'] = np.where(
                    valid_mask,
                    ((df[col] // 86400 + 4) % 7),  # 1970-01-01是周四
                    0  # 默认值
                ).astype('int8')
                
                # 商务时间（早高峰 6-9点，晚高峰 17-20点）
                hour_col = f'{col}_hour'
                df[f'{col}_business_time'] = (
                    ((df[hour_col] >= 6) & (df[hour_col] <= 9)) |
                    ((df[hour_col] >= 17) & (df[hour_col] <= 20))
                ).astype('int8')
                
                # 红眼航班特征
                df[f'{col}_is_redeye'] = (
                    (df[hour_col] >= 22) | (df[hour_col] <= 6)
                ).astype('int8')
        
        # 提前预订天数
        if 'requestDate' in df.columns:
            for leg in [0, 1]:
                dep_col = f'legs{leg}_departureAt'
                if dep_col in df.columns:
                    # 计算提前预订天数
                    valid_mask = (df[dep_col] != -1) & (df['requestDate'] != -1)
                    df[f'legs{leg}_days_ahead'] = np.where(
                        valid_mask,
                        np.clip((df[dep_col] - df['requestDate']) // 86400, 0, 365),
                        7  # 默认一周
                    ).astype('int16')
                    
                    # 预订类型分类
                    days_ahead = df[f'legs{leg}_days_ahead']
                    df[f'legs{leg}_booking_type'] = np.select(
                        [days_ahead <= 0, days_ahead <= 7, days_ahead <= 30],
                        [0, 1, 2],
                        default=3
                    ).astype('int8')
        
        return df
    
    @timer
    def create_ranker_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建基于ranker_id的特征"""
        df = df.copy()
        
        if 'ranker_id' not in df.columns:
            return df
        
        # 组大小特征
        df['group_size'] = df.groupby('ranker_id')['Id'].transform('count').astype('int16')
        df['group_size_log'] = np.log1p(df['group_size'])
        
        # 价格排名特征（基于价格分箱）
        if 'totalPrice_bin' in df.columns:
            df['price_rank'] = df.groupby('ranker_id')['totalPrice_bin'].rank().astype('int8')
            df['price_pct_rank'] = df.groupby('ranker_id')['totalPrice_bin'].rank(pct=True)
            df['is_cheapest'] = (df['price_rank'] == 1).astype('int8')
            
            # 价格距离中位数
            median_price = df.groupby('ranker_id')['totalPrice_bin'].transform('median')
            std_price = df.groupby('ranker_id')['totalPrice_bin'].transform('std')
            df['price_from_median'] = (df['totalPrice_bin'] - median_price) / (std_price + 1)
        
        # 持续时间排名
        if 'total_duration' in df.columns:
            df['duration_rank'] = df.groupby('ranker_id')['total_duration'].rank().astype('int8')
        
        # 最少段数
        if 'total_segments' in df.columns:
            df['is_min_segments'] = (
                df['total_segments'] == df.groupby('ranker_id')['total_segments'].transform('min')
            ).astype('int8')
        
        # 直飞最便宜
        if 'is_direct_leg0' in df.columns and 'totalPrice_bin' in df.columns:
            # 计算每个ranker_id中直飞航班的最低价格
            direct_cheapest = (
                df[df['is_direct_leg0'] == 1]
                .groupby('ranker_id')['totalPrice_bin']
                .min()
                .reset_index()
                .rename(columns={'totalPrice_bin': 'min_direct_price'})
            )
            
            df = df.merge(direct_cheapest, on='ranker_id', how='left')
            df['is_direct_cheapest'] = (
                (df['is_direct_leg0'] == 1) & 
                (df['totalPrice_bin'] == df['min_direct_price'])
            ).astype('int8').fillna(0)
            df.drop('min_direct_price', axis=1, inplace=True)
        
        return df
    
    @timer
    def create_carrier_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建航空公司相关特征"""
        df = df.copy()
        
        # 主要航空公司特征（基于哈希值，需要预先知道SU和S7的哈希值）
        carrier_col = 'legs0_segments0_marketingCarrier_code'
        if carrier_col in df.columns:
            # 由于已经哈希编码，无法直接判断，使用频次作为代理
            carrier_counts = df[carrier_col].value_counts()
            top_carriers = carrier_counts.head(5).index.tolist()  # 前5大航空公司
            df['is_major_carrier'] = df[carrier_col].isin(top_carriers).astype('int8')
        
        # 航空公司一致性（往返程是否同一航空公司）
        outbound_carrier = 'legs0_segments0_marketingCarrier_code'
        return_carrier = 'legs1_segments0_marketingCarrier_code'
        
        if outbound_carrier in df.columns and return_carrier in df.columns:
            df['same_carrier_roundtrip'] = (
                (df[outbound_carrier] == df[return_carrier]) & 
                (df[outbound_carrier] != -1) & 
                (df[return_carrier] != -1)
            ).astype('int8')
        
        return df
    
    @timer
    def create_passenger_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建乘客相关特征"""
        df = df.copy()
        
        # VIP和常旅客组合特征
        vip_col = 'is_vip' if 'is_vip' in df.columns else 'isVip'
        ff_col = 'has_frequent_flyer'
        
        if vip_col in df.columns and ff_col in df.columns:
            df['is_vip_or_freq'] = (
                (df[vip_col] == 1) | (df[ff_col] == 1)
            ).astype('int8')
        
        # 企业用户特征
        if 'has_corporate_tariff' in df.columns and vip_col in df.columns:
            df['vip_corporate'] = (
                (df[vip_col] == 1) & (df['has_corporate_tariff'] == 1)
            ).astype('int8')
        
        # 乘客数量特征
        if 'pricingInfo_passengerCount' in df.columns:
            df['is_single_passenger'] = (df['pricingInfo_passengerCount'] == 1).astype('int8')
            df['is_group_travel'] = (df['pricingInfo_passengerCount'] >= 3).astype('int8')
        
        return df
    
    @timer
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建交互特征（选择性创建，避免特征爆炸）"""
        df = df.copy()
        
        # 价格-时间交互（只选择最重要的）
        if 'totalPrice_bin' in df.columns and 'legs0_departureAt_hour' in df.columns:
            df['price_hour_interaction'] = (
                df['totalPrice_bin'] * df['legs0_departureAt_hour']
            ).astype('int16')
        
        # 直飞-价格交互
        if 'is_direct_leg0' in df.columns and 'totalPrice_bin' in df.columns:
            df['direct_price_interaction'] = (
                df['is_direct_leg0'] * df['totalPrice_bin']
            ).astype('int16')
        
        # VIP-舱位交互
        if 'is_vip' in df.columns and 'avg_cabin_class' in df.columns:
            df['vip_cabin_interaction'] = (
                df['is_vip'] * df['avg_cabin_class'].fillna(0)
            ).astype('int16')
        
        return df
    
    # ==================== 经济学特征 ====================
    
    @timer
    def create_economic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建具有经济学意义的特征"""
        df = df.copy()
                
        # 1. 价格弹性和需求理论特征
        if 'totalPrice_bin' in df.columns and 'ranker_id' in df.columns:
            # 价格弹性指标（组内价格敏感度）
            group_price_stats = df.groupby('ranker_id')['totalPrice_bin'].agg([
                'mean', 'std', 'min', 'max', 'count'
            ]).reset_index()
            group_price_stats.columns = ['ranker_id', 'group_price_mean', 'group_price_std', 
                                       'group_price_min', 'group_price_max', 'group_size_price']
            
            df = df.merge(group_price_stats, on='ranker_id', how='left')
            
            # 相对价格位置（价格锚定效应）
            df['price_relative_position'] = (df['totalPrice_bin'] - df['group_price_min']) / (
                df['group_price_max'] - df['group_price_min'] + 1)
            
            # 价格离散系数（选择复杂度）
            df['price_coefficient_variation'] = df['group_price_std'] / (df['group_price_mean'] + 1)
            
            # 是否为极值选项（锚定效应）
            df['is_price_anchor_high'] = (df['totalPrice_bin'] == df['group_price_max']).astype('int8')
            df['is_price_anchor_low'] = (df['totalPrice_bin'] == df['group_price_min']).astype('int8')
            
            # 价格-质量感知比（基于舱位和价格）
            if 'avg_cabin_class' in df.columns:
                df['price_quality_ratio'] = df['totalPrice_bin'] / (df['avg_cabin_class'] + 1)
        
        # 2. 时间价值特征（Time Value of Money）
        if 'legs0_departureAt_hour' in df.columns:
            # 商务时间溢价（Business Time Premium）
            business_hours = [7, 8, 9, 17, 18, 19, 20]
            df['is_business_prime_time'] = df['legs0_departureAt_hour'].isin(business_hours).astype('int8')
            
            # 红眼航班折扣效应
            redeye_hours = [22, 23, 0, 1, 2, 3, 4, 5, 6]
            df['is_redeye_discount'] = df['legs0_departureAt_hour'].isin(redeye_hours).astype('int8')
            
            # 黄金时间（8-10点，18-20点）
            golden_hours = [8, 9, 18, 19]
            df['is_golden_time'] = df['legs0_departureAt_hour'].isin(golden_hours).astype('int8')
        
        # 3. 预订行为经济学特征
        if 'legs0_days_ahead' in df.columns:
            # 预订时机分类（行为经济学）
            df['booking_urgency'] = np.select([
                df['legs0_days_ahead'] <= 1,    # 紧急预订
                df['legs0_days_ahead'] <= 7,    # 临时预订
                df['legs0_days_ahead'] <= 14,   # 正常预订
                df['legs0_days_ahead'] <= 30,   # 提前预订
                df['legs0_days_ahead'] <= 60,   # 早期预订
            ], [4, 3, 2, 1, 0], default=0).astype('int8')
            
            # 最佳预订窗口（21-60天通常价格最优）
            df['in_optimal_booking_window'] = (
                (df['legs0_days_ahead'] >= 21) & (df['legs0_days_ahead'] <= 60)
            ).astype('int8')
            
            # 冲动购买指标（<=3天）
            df['is_impulse_booking'] = (df['legs0_days_ahead'] <= 3).astype('int8')
        
        # 4. 网络效应和市场集中度
        if 'legs0_segments0_marketingCarrier_code' in df.columns:
            # 航空公司市场份额（网络效应）
            carrier_market_share = df['legs0_segments0_marketingCarrier_code'].value_counts(normalize=True)
            df['carrier_market_share'] = df['legs0_segments0_marketingCarrier_code'].map(carrier_market_share)
            
            # 是否为市场领导者（>20%市场份额）
            df['is_market_leader'] = (df['carrier_market_share'] > 0.2).astype('int8')
        
        # 5. 服务质量感知特征
        if 'legs0_segments0_cabinClass' in df.columns and 'totalPrice_bin' in df.columns:
            # 性价比指标
            cabin_price_ratio = df.groupby('legs0_segments0_cabinClass')['totalPrice_bin'].mean()
            df['cabin_price_expectation'] = df['legs0_segments0_cabinClass'].map(cabin_price_ratio)
            df['price_expectation_gap'] = df['totalPrice_bin'] - df['cabin_price_expectation']
        
        # 6. 便利性溢价特征
        if 'total_segments' in df.columns:
            # 直飞溢价（便利性价值）
            df['convenience_score'] = np.select([
                df['total_segments'] == 2,  # 往返都直飞
                df['total_segments'] == 3,  # 一程直飞
                df['total_segments'] >= 4,  # 都需转机
            ], [2, 1, 0], default=0).astype('int8')
        
        return df
    
    # ==================== 行为学特征 ====================
    
    @timer
    def create_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建行为经济学特征"""
        df = df.copy()
        
        # 1. 选择过载理论（Choice Overload）
        if 'ranker_id' in df.columns:
            # 选择复杂度
            choice_complexity = df.groupby('ranker_id').agg({
                'Id': 'count',  # 选项数量
                'totalPrice_bin': ['std', 'max', 'min'] if 'totalPrice_bin' in df.columns else ['count', 'count', 'count'],
                'total_duration': ['std', 'mean'] if 'total_duration' in df.columns else ['count', 'count']
            }).reset_index()
            
            choice_complexity.columns = ['ranker_id', 'choice_set_size_beh', 'price_variance_beh', 
                                       'price_range_max_beh', 'price_range_min_beh', 'duration_variance', 'duration_mean']
            
            df = df.merge(choice_complexity, on='ranker_id', how='left')
            
            # 选择过载指标
            df['choice_overload_score'] = np.select([
                df['choice_set_size_beh'] >= 20,  # 高过载
                df['choice_set_size_beh'] >= 10,  # 中等过载
                df['choice_set_size_beh'] >= 5,   # 轻微过载
            ], [3, 2, 1], default=0).astype('int8')
            
            # 选择集中度（基于价格分布）
            df['choice_concentration'] = 1 / (df['price_variance_beh'] + 1)
        
        # 2. 损失厌恶特征（Loss Aversion）
        if 'miniRules0_monetaryAmount' in df.columns and 'miniRules1_monetaryAmount' in df.columns:
            # 潜在损失风险
            df['max_cancellation_loss'] = np.maximum(
                df['miniRules0_monetaryAmount'].fillna(0),
                df['miniRules1_monetaryAmount'].fillna(0)
            )
            
            # 损失厌恶阈值（相对于票价的百分比）
            if 'totalPrice_bin' in df.columns:
                df['loss_aversion_ratio'] = df['max_cancellation_loss'] / (df['totalPrice_bin'] + 1)
            
            # 低风险选项（可免费取消/改签）
            df['is_low_risk_option'] = (
                (df['miniRules0_monetaryAmount'] == 0) | 
                (df['miniRules1_monetaryAmount'] == 0)
            ).astype('int8')
        
        # 3. 社会认同特征（Social Proof）
        if 'legs0_segments0_marketingCarrier_code' in df.columns and 'ranker_id' in df.columns:
            # 同组最受欢迎航空公司
            popular_carrier_in_group = df.groupby('ranker_id')['legs0_segments0_marketingCarrier_code'].agg(
                lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else -1
            ).reset_index()
            popular_carrier_in_group.columns = ['ranker_id', 'popular_carrier']
            
            df = df.merge(popular_carrier_in_group, on='ranker_id', how='left')
            df['is_popular_carrier_choice'] = (
                df['legs0_segments0_marketingCarrier_code'] == df['popular_carrier']
            ).astype('int8')
        
        # 4. 认知偏差特征
        if 'legs0_departureAt_hour' in df.columns and 'legs0_arrivalAt_hour' in df.columns:
            # 到达时间偏好（认知便利性）
            df['arrival_time_preference'] = np.select([
                df['legs0_arrivalAt_hour'].between(8, 12),   # 上午到达（便于安排）
                df['legs0_arrivalAt_hour'].between(13, 18),  # 下午到达
                df['legs0_arrivalAt_hour'].between(19, 22),  # 晚上到达
            ], [2, 1, 0], default=0).astype('int8')
        
        # 5. 框架效应特征（Framing Effect）
        if 'totalPrice_bin' in df.columns and 'taxes_bin' in df.columns:
            # 价格透明度（税费占比）
            df['price_transparency'] = df['taxes_bin'] / (df['totalPrice_bin'] + 1)
            
            # 隐藏成本感知
            df['hidden_cost_perception'] = np.select([
                df['price_transparency'] > 0.3,  # 高税费比例
                df['price_transparency'] > 0.15, # 中等税费比例
            ], [2, 1], default=0).astype('int8')
        
        return df
    
    # ==================== 自动特征发现 ====================
    
    @timer
    def auto_discover_features(self, df: pd.DataFrame, max_total_features: int = 50) -> pd.DataFrame:
        """自动特征发现"""
        df = df.copy()
        
        original_cols = set(df.columns)
        
        # 1. 基于统计的特征
        df = self._create_statistical_features(df)
        
        # 2. 基于聚类的特征
        df = self._create_clustering_features(df)
        
        # 3. 基于分箱的特征
        df = self._create_binning_features(df)
        
        # 4. 基于组合的特征
        df = self._create_combination_features(df)
        
        # 5. 控制特征数量
        new_cols = [col for col in df.columns if col not in original_cols]
        if len(new_cols) > max_total_features:
            # 基于方差选择特征
            selected_cols = self._select_features_by_variance(df[new_cols], max_total_features)
            df = df[list(original_cols) + selected_cols]
            self.logger.info(f"自动特征发现: 创建了{len(selected_cols)}个特征（从{len(new_cols)}中选择）")
        else:
            self.logger.info(f"自动特征发现: 创建了{len(new_cols)}个特征")
        
        return df
    
    def _create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建统计特征"""
        if 'ranker_id' not in df.columns:
            return df
        
        # 数值列统计特征
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        exclude_cols = ['Id', 'ranker_id', 'profileId', 'companyID']
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        for col in numeric_cols[:5]:  # 限制数量
            if df[col].nunique() > 1:
                # 组内统计
                df[f'{col}_group_mean'] = df.groupby('ranker_id')[col].transform('mean')
                df[f'{col}_group_std'] = df.groupby('ranker_id')[col].transform('std').fillna(0)
                df[f'{col}_zscore'] = (df[col] - df[f'{col}_group_mean']) / (df[f'{col}_group_std'] + 1e-6)
        
        return df
    
    def _create_clustering_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建聚类特征"""
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            
            # 选择聚类特征
            cluster_cols = []
            if 'totalPrice_bin' in df.columns:
                cluster_cols.append('totalPrice_bin')
            if 'total_duration' in df.columns:
                cluster_cols.append('total_duration')
            if 'legs0_days_ahead' in df.columns:
                cluster_cols.append('legs0_days_ahead')
            
            if len(cluster_cols) >= 2:
                # 数据预处理
                X = df[cluster_cols].fillna(0)
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # K-means聚类
                kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
                df['cluster_label'] = kmeans.fit_predict(X_scaled)
                
                # 到聚类中心的距离
                distances = kmeans.transform(X_scaled)
                df['cluster_distance'] = np.min(distances, axis=1)
                
        except ImportError:
            self.logger.info("sklearn不可用，跳过聚类特征")
        except Exception as e:
            self.logger.info(f"聚类特征创建失败: {e}")
        
        return df
    
    def _create_binning_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建分箱特征"""
        # 价格分箱（如果还没有分箱特征）
        if 'totalPrice_bin' in df.columns and 'price_quartile' not in df.columns:
            df['price_quartile'] = pd.qcut(df['totalPrice_bin'], q=4, labels=False, duplicates='drop').astype('int8')
        
        # 持续时间分箱
        if 'total_duration' in df.columns and df['total_duration'].nunique() > 10:
            df['duration_bin'] = pd.cut(df['total_duration'], bins=5, labels=False).astype('int8')
        
        # 提前预订天数分箱
        if 'legs0_days_ahead' in df.columns and df['legs0_days_ahead'].nunique() > 10:
            df['days_ahead_bin'] = pd.cut(df['legs0_days_ahead'], 
                                        bins=[0, 1, 7, 14, 30, 365], 
                                        labels=False, include_lowest=True).astype('int8')
        
        return df
    
    def _create_combination_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建组合特征"""
        # 价格效率比
        if 'totalPrice_bin' in df.columns and 'total_duration' in df.columns:
            df['price_per_hour'] = df['totalPrice_bin'] / (df['total_duration'] / 60 + 1)
        
        # 便利性得分
        if 'is_direct_leg0' in df.columns and 'legs0_departureAt_hour' in df.columns:
            business_hours = [7, 8, 9, 17, 18, 19, 20]
            df['convenience_score_auto'] = (
                df['is_direct_leg0'] * 2 + 
                df['legs0_departureAt_hour'].isin(business_hours).astype(int)
            ).astype('int8')
        
        # 风险得分（基于取消政策和价格）
        if 'free_cancel' in df.columns and 'totalPrice_bin' in df.columns:
            df['risk_adjusted_price'] = df['totalPrice_bin'] * (2 - df['free_cancel'])
        
        return df
    
    def _select_features_by_variance(self, df: pd.DataFrame, max_features: int) -> List[str]:
        """基于方差选择特征"""
        # 计算每个特征的方差
        variances = df.var(numeric_only=True).sort_values(ascending=False)
        
        # 移除方差为0的特征
        variances = variances[variances > 0]
        
        # 选择前max_features个特征
        selected_features = variances.head(max_features).index.tolist()
        
        return selected_features
    
    # ==================== 主处理方法 ====================
    
    @timer
    def process_enhanced_features(self, df: pd.DataFrame, 
                                feature_types: List[str] = None,
                                apply_selection: bool = False,
                                max_features: int = 200) -> pd.DataFrame:
        """
        处理增强特征
        
        Args:
            df: 输入数据
            feature_types: 特征类型 ['economic', 'behavioral', 'auto_discovery']
            apply_selection: 是否应用特征选择
            max_features: 最大特征数量
        """
        if feature_types is None:
            feature_types = ['economic', 'behavioral']
        
        df_processed = df.copy()
        original_shape = df_processed.shape
        
        # 经济学特征
        if 'economic' in feature_types:
            df_processed = self.create_economic_features(df_processed)
        
        # 行为学特征
        if 'behavioral' in feature_types:
            df_processed = self.create_behavioral_features(df_processed)
        
        # 自动特征发现
        if 'auto_discovery' in feature_types:
            max_auto = max_features // 4  # 自动特征占总数的1/4
            df_processed = self.auto_discover_features(df_processed, max_auto)
        
        # 特征选择（如果需要）
        if apply_selection:
            df_processed = self._apply_feature_selection(df_processed, max_features)
        
        final_shape = df_processed.shape
        self.logger.info(f"增强特征工程完成: {original_shape} -> {final_shape}")
        
        return df_processed
    
    def _apply_feature_selection(self, df: pd.DataFrame, max_features: int) -> pd.DataFrame:
        """应用特征选择"""
        # 保留重要的原始列
        important_cols = ['Id', 'ranker_id', 'label'] if 'label' in df.columns else ['Id', 'ranker_id']
        if 'profileId' in df.columns:
            important_cols.append('profileId')
        
        # 数值特征
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col not in important_cols]
        
        if len(feature_cols) <= max_features:
            return df
        
        # 基于方差和相关性进行特征选择
        selected_features = self._select_features_by_variance(df[feature_cols], max_features)
        
        return df[important_cols + selected_features]
    
    @timer
    def process_features(self, df: pd.DataFrame, 
                        feature_types: List[str] = None,
                        config: Dict = None) -> pd.DataFrame:
        """
        主特征工程入口
        
        Args:
            df: 输入数据（已编码）
            feature_types: 要创建的特征类型
            config: 配置参数
        """
        if feature_types is None:
            feature_types = ['core', 'route', 'time', 'ranker', 'carrier', 'passenger']
        
        if config is None:
            config = {}
        
        df_processed = df.copy()
        original_shape = df_processed.shape
        
        self.processing_stats['original_shape'] = original_shape
        
        if self.logger:
            self.logger.info(f"开始特征工程: {original_shape}")
        
        # 基础特征工程
        if 'core' in feature_types:
            df_processed = self.create_core_features(df_processed)
        
        if 'route' in feature_types:
            df_processed = self.create_route_features(df_processed)
        
        if 'time' in feature_types:
            df_processed = self.create_time_features(df_processed)
        
        if 'ranker' in feature_types:
            df_processed = self.create_ranker_features(df_processed)
        
        if 'carrier' in feature_types:
            df_processed = self.create_carrier_features(df_processed)
        
        if 'passenger' in feature_types:
            df_processed = self.create_passenger_features(df_processed)
        
        if 'interaction' in feature_types:
            df_processed = self.create_interaction_features(df_processed)
        
        # 增强特征工程
        enhanced_config = config.get('enhanced_features', {})
        if enhanced_config.get('enabled', True):
            enhanced_types = enhanced_config.get('types', ['economic', 'behavioral'])
            df_processed = self.process_enhanced_features(
                df_processed, 
                feature_types=enhanced_types,
                apply_selection=True,  # 特征选择在后面统一处理
                max_features=enhanced_config.get('max_features', 200)
            )
        
        # 自动特征发现
        auto_discovery_config = config.get('auto_discovery', {})
        if auto_discovery_config.get('enabled', True):
            max_auto_features = auto_discovery_config.get('max_features', 50)
            df_processed = self.auto_discover_features(
                df_processed, max_total_features=max_auto_features
            )
        
        # 内存优化
        df_processed = MemoryUtils.optimize_dataframe_memory(df_processed)
        
        # 记录处理后状态
        final_shape = df_processed.shape
        self.processing_stats['final_shape'] = final_shape
        self.processing_stats['features_added'] = final_shape[1] - original_shape[1]
        
        if self.logger:
            self.logger.info(f"特征工程完成: {original_shape} -> {final_shape}")
            self.logger.info(f"新增特征: {self.processing_stats['features_added']}")
        
        del df
        gc.collect()
        
        return df_processed
    
    # ==================== 数据验证和质量控制 ====================
    
    @timer
    def validate_and_clean(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """数据验证和清理"""
        df_clean = df.copy()
        
        # 检查缺失值
        missing_info = df_clean.isnull().sum()
        missing_cols = missing_info[missing_info > 0].to_dict()
        
        # 检查异常值（基于数值列）
        outlier_info = {}
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in ['Id', 'ranker_id', 'profileId', 'companyID']:
                continue
            
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
            if outliers.sum() > 0:
                outlier_info[col] = {
                    'count': outliers.sum(),
                    'percentage': outliers.mean() * 100
                }
        
        # 生成质量报告
        quality_report = {
            'missing_values': missing_cols,
            'outliers': outlier_info,
            'shape': df_clean.shape
        }
        
        if self.logger:
            if missing_cols:
                self.logger.warning(f"发现缺失值: {len(missing_cols)} 列")
            if outlier_info:
                total_outliers = sum(info['count'] for info in outlier_info.values())
                self.logger.warning(f"发现异常值: {total_outliers} 个")
        
        return df_clean, quality_report
    
    def get_processing_summary(self) -> Dict:
        """获取处理总结"""
        summary = self.processing_stats.copy()
        return summary
    
    # ==================== 文件处理接口 ====================
    
    @timer
    def process_segment_file(self, input_file: str, output_file: str, 
                           feature_types: List[str] = None,
                           config: Dict = None) -> bool:
        """处理单个segment文件"""
        try:
            if not os.path.exists(input_file):
                if self.logger:
                    self.logger.warning(f"输入文件不存在: {input_file}")
                return False
            
            # 检查文件是否为空
            pf = pq.ParquetFile(input_file)
            if pf.metadata.num_rows == 0:
                if self.logger:
                    self.logger.info(f"空文件，跳过: {input_file}")
                # 创建空的输出文件
                empty_df = pd.DataFrame()
                empty_df.to_parquet(output_file, index=False)
                return True
            
            # 读取数据
            df = pd.read_parquet(input_file)
            
            if len(df) == 0:
                if self.logger:
                    self.logger.info(f"空数据，跳过: {input_file}")
                df.to_parquet(output_file, index=False)
                return True
            
            # 特征工程
            df_featured = self.process_features(df, feature_types, config)
            
            # 数据验证
            df_clean, quality_report = self.validate_and_clean(df_featured)
            
            # 保存结果
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            df_clean.to_parquet(output_file, index=False)
            
            if self.logger:
                self.logger.info(f"处理完成: {input_file} -> {output_file}")
                self.logger.info(f"  形状: {df.shape} -> {df_clean.shape}")
            
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"处理文件失败 {input_file}: {str(e)}")
            return False
    
    @timer
    def process_all_segments(self, input_dir: str, output_dir: str, 
                           data_type: str, feature_types: List[str] = None,
                           config: Dict = None) -> bool:
        """处理所有segment文件，保持按组大小分割的格式"""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        
        # 确保输出目录存在
        output_dir.mkdir(parents=True, exist_ok=True)
        
        success_count = 0
        total_count = 0
        
        for segment_level in [0, 1, 2, 3]:
            for group_category in ['small', 'medium', 'big']:        
                input_file = input_dir / f"{data_type}_segment_{segment_level}_{group_category}.parquet"
                output_file = output_dir / f"{data_type}_segment_{segment_level}_{group_category}.parquet"
                
                total_count += 1
                
                if self.process_segment_file(str(input_file), str(output_file), feature_types, config):
                    success_count += 1
        
        if self.logger:
            self.logger.info(f"批处理完成: {success_count}/{total_count} 个文件成功")
        
        return success_count == total_count
    
    # ==================== 配置和工具方法 ====================
    
    def get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            'enhanced_features': {
                'enabled': True,
                'types': ['economic', 'behavioral'],
                'max_features': 200
            },
            'auto_discovery': {
                'enabled': False,
                'max_features': 50
            },
            'feature_selection': {
                'enabled': False,
                'max_features': 500,
                'method': 'variance'
            },
            'memory_optimization': {
                'enabled': True,
                'compress_strings': True
            }
        }