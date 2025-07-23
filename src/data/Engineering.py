import pandas as pd
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

class FlightFeatureEngineering:
    """
    重构版航班特征工程类
    专注于保留的核心业务特征构建
    """
    
    def __init__(self):
        self.global_stats = {}
    
    def _safe_numeric_convert(self, series: pd.Series, default_value: float = 0.0) -> pd.Series:
        """安全数值转换"""
        if pd.api.types.is_numeric_dtype(series):
            return series
        
        # 处理布尔字符串
        bool_mapping = {
            'true': 1, 'false': 0, 'True': 1, 'False': 0,
            'yes': 1, 'no': 0, 'Y': 1, 'N': 0, '1': 1, '0': 0
        }
        
        numeric_series = pd.to_numeric(series, errors='coerce')
        if numeric_series.isna().sum() > len(series) * 0.5:
            mapped_series = series.astype(str).str.lower().map(bool_mapping)
            numeric_series = numeric_series.fillna(mapped_series)
        
        return numeric_series.fillna(default_value)
    
    def _safe_datetime_convert(self, series: pd.Series) -> pd.Series:
        """安全时间转换"""
        if pd.api.types.is_datetime64_any_dtype(series):
            return series
        try:
            return pd.to_datetime(series, unit='s', errors='coerce')
        except:
            return pd.to_datetime(series, errors='coerce')
    
    def compute_global_statistics(self, df: pd.DataFrame):
        """计算关键全局统计信息 - 仅保留必要统计"""
        # 价格统计
        if 'totalPrice' in df.columns:
            price_series = self._safe_numeric_convert(df['totalPrice'])
            self.global_stats['price'] = {
                'mean': price_series.mean(),
                'std': max(price_series.std(), 1e-6),
                'q25': price_series.quantile(0.25),
                'q75': price_series.quantile(0.75)
            }
        
        # 航空公司统计 (仅考虑首段航司)
        if 'legs0_segments0_marketingCarrier_code' in df.columns:
            airlines = df['legs0_segments0_marketingCarrier_code'].dropna().astype(str).tolist()
            if airlines:
                airline_freq = pd.Series(airlines).value_counts()
                self.global_stats['top_airlines'] = airline_freq.head(10).index.tolist()
        
        # 机场统计 (仅考虑首段出发/到达机场)
        airport_cols = [
            'legs0_segments0_departureFrom_airport_iata',
            'legs0_segments0_arrivalTo_airport_iata'
        ]
        if all(col in df.columns for col in airport_cols):
            all_airports = []
            for col in airport_cols:
                all_airports.extend(df[col].dropna().astype(str).tolist())
            
            if all_airports:
                airport_freq = pd.Series(all_airports).value_counts()
                self.global_stats['top_airports'] = airport_freq.head(20).index.tolist()
    
    def create_user_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """构建用户特征 - 仅保留核心特征"""
        # 基础用户特征
        user_cols = ['bySelf', 'isVip', 'isAccess3D']
        for col in user_cols:
            if col in df.columns:
                df[col] = self._safe_numeric_convert(df[col])
        
        # 性别编码
        if 'sex' in df.columns:
            df['sex'] = self._safe_numeric_convert(df['sex'], -1)
        
        return df
    
    def create_corporate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """构建企业特征 - 简化版"""
        # 企业政策特征
        if 'pricingInfo_isAccessTP' in df.columns:
            df['pricingInfo_isAccessTP'] = self._safe_numeric_convert(df['pricingInfo_isAccessTP'])
        
        # 乘客数量
        if 'pricingInfo_passengerCount' in df.columns:
            df['passenger_count'] = self._safe_numeric_convert(df['pricingInfo_passengerCount'], 1)
            df['is_group_booking'] = (df['passenger_count'] > 1).astype(int)
        
        return df
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """构建时间特征 - 仅保留必要特征"""
        # 时间转换
        time_cols = ['requestDate', 'legs0_departureAt']
        for col in time_cols:
            if col in df.columns:
                df[col] = self._safe_datetime_convert(df[col])
        
        # 预订提前期
        if all(col in df.columns for col in ['requestDate', 'legs0_departureAt']):
            df['booking_advance_days'] = (
                df['legs0_departureAt'] - df['requestDate']
            ).dt.days.fillna(0).clip(0, 365)
            
            # 预订时机分类
            advance_days = df['booking_advance_days']
            booking_category = pd.cut(
                advance_days,
                bins=[-1, 7, 30, 90, 365],
                labels=['LastMinute', 'Weekly', 'Monthly', 'Advanced']
            )
            df['booking_timing_category'] = booking_category.astype(str).replace('nan', 'Unknown')
            
            # 预订紧急度
            df['booking_urgency'] = (df['booking_advance_days'] < 7).astype(int)
            df['advance_booking'] = (df['booking_advance_days'] > 30).astype(int)
        
        # 出发时间特征
        if 'legs0_departureAt' in df.columns:
            departure_dt = df['legs0_departureAt']
            df['departure_hour'] = departure_dt.dt.hour.fillna(12)
            df['is_weekend_departure'] = (departure_dt.dt.dayofweek >= 5).astype(int)
            
            # 商务友好时间
            business_hours = [7, 8, 9, 17, 18, 19, 20]
            df['business_friendly_departure'] = df['departure_hour'].isin(business_hours).astype(int)
        
        # 飞行时长特征
        if 'legs0_duration' in df.columns:
            df['flight_duration_hours'] = self._safe_numeric_convert(df['legs0_duration']) / 3600
            df['is_long_flight'] = (df['flight_duration_hours'] > 6).astype(int)
        
        return df
    
    def create_pricing_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """构建价格特征 - 简化版"""
        # 价格特征
        if 'totalPrice' in df.columns:
            df['total_price'] = self._safe_numeric_convert(df['totalPrice'])
            
            # 税费特征
            if 'taxes' in df.columns:
                df['taxes_amount'] = self._safe_numeric_convert(df['taxes'])
                df['tax_ratio'] = df['taxes_amount'] / (df['total_price'] + 1e-6)
        
        return df
    
    def create_route_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """构建航线特征 - 仅考虑首段"""
        # 航段信息
        if 'searchRoute' in df.columns:
            df['is_round_trip'] = (df['searchRoute'] == 'round-trip').astype(int)
        
        # 航空公司特征
        if 'legs0_segments0_marketingCarrier_code' in df.columns and 'top_airlines' in self.global_stats:
            top_airlines = self.global_stats['top_airlines']
            df['uses_major_airline'] = df['legs0_segments0_marketingCarrier_code'].astype(str).isin(top_airlines).astype(int)
        
        # 机场特征
        departure_col = 'legs0_segments0_departureFrom_airport_iata'
        arrival_col = 'legs0_segments0_arrivalTo_airport_iata'
        
        if 'top_airports' in self.global_stats:
            top_airports = self.global_stats['top_airports']
            
            if departure_col in df.columns:
                df['departure_major_airport'] = df[departure_col].astype(str).isin(top_airports).astype(int)
            
            if arrival_col in df.columns:
                df['arrival_major_airport'] = df[arrival_col].astype(str).isin(top_airports).astype(int)
        
        return df
    
    def create_service_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """构建服务特征 - 仅考虑首段"""
        # 舱位特征
        cabin_col = 'legs0_segments0_cabinClass'
        if cabin_col in df.columns:
            df['cabin_class'] = self._safe_numeric_convert(df[cabin_col])
            df['is_premium_cabin'] = (df['cabin_class'] > 2).astype(int)
        
        # 座位可用性
        seats_col = 'legs0_segments0_seatsAvailable'
        if seats_col in df.columns:
            df['seats_available'] = self._safe_numeric_convert(df[seats_col])
            df['seat_scarcity'] = (df['seats_available'] <= 5).astype(int)
        
        # 行李特征
        baggage_col = 'legs0_segments0_baggageAllowance_quantity'
        if baggage_col in df.columns:
            df['baggage_pieces'] = self._safe_numeric_convert(df[baggage_col])
            df['generous_baggage'] = (df['baggage_pieces'] >= 2).astype(int)
        
        return df
    
    def create_policy_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """构建政策特征 - 仅保留改签政策"""
        # 改签费用特征
        change_fee_col = 'miniRules1_monetaryAmount'
        if change_fee_col in df.columns:
            df['change_fee'] = self._safe_numeric_convert(df[change_fee_col])
            df['has_free_changes'] = (df['change_fee'] == 0).astype(int)
        
        return df
    
    def create_composite_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """构建复合特征 - 简化版"""
        # 价值评分
        value_factors = []
        if 'total_price' in df.columns:
            # 价格越低越好（需标准化）
            price_factor = 1 - (df['total_price'] / df['total_price'].max())
            value_factors.append(price_factor)
        
        if 'business_friendly_departure' in df.columns:
            value_factors.append(df['business_friendly_departure'])
        
        if value_factors:
            df['value_score'] = np.mean(value_factors, axis=0)
        
        return df
    
    def process_features(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """主要特征处理流程"""
        print("开始特征工程...")
        
        # 移除无效特征
        invalid_features = [
            'legs0_segments0_duration', 'legs1_segments0_duration', 'legs1_arrivalAt',
            'is_round_trip', 'legs0_arrivalAt', 'legs1_segments0_operatingCarrier_code',
            'legs0_segments0_baggageAllowance_weightMeasurementType', 'corporateTariffCode',
            'miniRules0_monetaryAmount', 'legs0_segments0_operatingCarrier_code',
            'legs1_duration_hours', 'price_global_zscore', 'bySelf_num', 'frequentFlyer_num',
            'isAccess3D_num', 'isVip_num', 'corporate_pricing_access', 'gender_encoded',
            'taxes_amount', 'totalPrice_bin', 'total_price', 'business_friendliness_score',
            'company_service_match', 'legs0_duration_hours', 'user_maturity_score',
            'frequentFlyer_label', 'frequentFlyer_num_label', 
            'Id', 'ranker_id', 'profileId', 'companyID', 'nationality',
            'miniRules0_statusInfos', 'miniRules1_statusInfos', 'miniRules0_percentage',
            'legs1_segments0_marketingCarrier_code', 
            'legs1_segments0_baggageAllowance_quantity',
            'legs1_segments0_baggageAllowance_weightMeasurementType',
            'legs1_segments0_operatingCarrier_code',
            'frequentFlyer'
        ]
        
        # 添加中转航段特征
        invalid_features += [col for col in df.columns if 'segments1_' in col]
        invalid_features += [col for col in df.columns if 'segments2_' in col]
        invalid_features += [col for col in df.columns if 'segments3_' in col]
        
        # 移除无效特征
        valid_features = [col for col in df.columns if col not in invalid_features]
        df = df[valid_features].copy()
        
        # 计算全局统计（仅训练时）
        if is_training:
            print("计算全局统计...")
            self.compute_global_statistics(df)
        
        # 创建各类特征
        print("创建用户特征...")
        df = self.create_user_features(df)
        
        print("创建企业特征...")
        df = self.create_corporate_features(df)
        
        print("创建时间特征...")
        df = self.create_temporal_features(df)
        
        print("创建价格特征...")
        df = self.create_pricing_features(df)
        
        print("创建航线特征...")
        df = self.create_route_features(df)
        
        print("创建服务特征...")
        df = self.create_service_features(df)
        
        print("创建政策特征...")
        df = self.create_policy_features(df)
        
        print("创建复合特征...")
        df = self.create_composite_features(df)
        
        # 处理缺失值
        print("处理缺失值...")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        exclude_cols = ['selected'] if 'selected' in df.columns else []
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
        
        # 处理无限值
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], 0)
        
        print(f"特征工程完成，最终形状: {df.shape}")
        return df


# 使用示例
if __name__ == "__main__":
    # 初始化特征工程器
    engineer = FlightFeatureEngineering()
    
    # 加载数据
    print("加载数据...")
    train_data = pd.read_parquet("D:/kaggle/filght/data/aeroclub-recsys-2025/encode/test/test_segment_2_encoded.parquet")
    
    # 训练模式特征工程
    processed_data = engineer.process_features(train_data, is_training=True)
    
    print(f"原始数据形状: {train_data.shape}")
    print(f"处理后数据形状: {processed_data.shape}")
    print(f"有效特征数量: {processed_data.shape[1]}")
    