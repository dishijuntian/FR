"""
数据工程模块 - 特征工程和数据预处理
"""
import pandas as pd
import numpy as np
import os
import gc
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import pyarrow.parquet as pq
from datetime import datetime

from src.utils.Common import timer
from src.utils.MemoryUtils import MemoryUtils


class DataEngineering:
    """数据工程主类"""
    
    def __init__(self, logger=None):
        self.logger = logger
        self.processing_stats = {}
    
    @timer
    def create_core_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建核心特征（基于d.py逻辑优化）"""
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
                           if f'legs{leg}_segments' in col and 'flightNumber' in col and col in df.columns]
            if segment_cols:
                df[f'n_segments_leg{leg}'] = (df[segment_cols] != -1).sum(axis=1).astype('int8')
            else:
                df[f'n_segments_leg{leg}'] = 0
        
        # 5. 常旅客特征
        if 'frequentFlyer' in df.columns:
            # 注意：原始数据已被哈希编码，这里基于是否为-1判断
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
        
        # 航线特征（基于哈希值无法直接判断，使用编码后的值）
        if 'is_round_trip' in df.columns:
            # 这个特征在DataEncode中已经创建
            pass
        elif 'searchRoute' in df.columns:
            # 如果searchRoute还存在（未被删除）
            popular_routes = ['MOWLED/LEDMOW', 'LEDMOW/MOWLED', 'MOWLED', 'LEDMOW']
            df['is_popular_route'] = df['searchRoute'].isin(popular_routes).astype('int8')
        
        # 舱位等级特征
        cabin_cols = [col for col in df.columns if 'cabinClass' in col and col in df.columns]
        if len(cabin_cols) >= 2:
            # 计算平均舱位等级
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
            
            # 价格偏离中位数
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
    
    @timer
    def process_features(self, df: pd.DataFrame, 
                        feature_types: List[str] = None) -> pd.DataFrame:
        """
        主特征工程入口
        
        Args:
            df: 输入数据（已编码）
            feature_types: 要创建的特征类型
        """
        if feature_types is None:
            feature_types = ['core', 'route', 'time', 'ranker', 'carrier', 'passenger']
        
        df_processed = df.copy()
        original_shape = df_processed.shape
        
        self.processing_stats['original_shape'] = original_shape
        
        if self.logger:
            self.logger.info(f"开始特征工程: {original_shape}")
        
        # 按类型创建特征
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
        return self.processing_stats
    
    @timer
    def process_segment_file(self, input_file: str, output_file: str, 
                           feature_types: List[str] = None) -> bool:
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
            df_featured = self.process_features(df, feature_types)
            
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
                           data_type: str, feature_types: List[str] = None) -> bool:
        """处理所有segment文件"""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        
        # 确保输出目录存在
        output_dir.mkdir(parents=True, exist_ok=True)
        
        success_count = 0
        total_count = 0
        
        for segment_level in [0, 1, 2, 3]:
            input_file = input_dir / f"{data_type}_segment_{segment_level}.parquet"
            output_file = output_dir / f"{data_type}_segment_{segment_level}.parquet"
            
            total_count += 1
            
            if self.process_segment_file(str(input_file), str(output_file), feature_types):
                success_count += 1
        
        if self.logger:
            self.logger.info(f"批处理完成: {success_count}/{total_count} 个文件成功")
        
        return success_count == total_count