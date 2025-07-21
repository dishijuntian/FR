import pandas as pd
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

class FlightFeatureEngineering:
    """
    简化版航班排名特征工程类
    专注于核心业务特征构建
    """
    
    def __init__(self):
        self.global_stats = {}
    
    def _safe_numeric_convert(self, series: pd.Series, default_value: float = 0.0) -> pd.Series:
        """安全数值转换"""
        if pd.api.types.is_numeric_dtype(series):
            return pd.to_numeric(series, errors='coerce').fillna(default_value)
        
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
    
    def _safe_qcut(self, series: pd.Series, q: int, labels: list = None) -> pd.Series:
        """安全分位数切分"""
        try:
            # 检查唯一值数量
            unique_vals = series.nunique()
            if unique_vals < q:
                q = max(1, unique_vals)
                if labels:
                    labels = labels[:q]
            
            return pd.qcut(series, q=q, labels=labels, duplicates='drop')
        except:
            # 如果分位数切分失败，使用简单分类
            if labels:
                return pd.Series([labels[0]] * len(series), index=series.index)
            else:
                return pd.Series([0] * len(series), index=series.index)
    
    def compute_global_statistics(self, df: pd.DataFrame):
        """计算关键全局统计信息"""
        # 价格统计
        if 'totalPrice_bin' in df.columns:
            price_series = self._safe_numeric_convert(df['totalPrice_bin'])
            self.global_stats['price'] = {
                'mean': price_series.mean(),
                'std': max(price_series.std(), 1e-6),
                'q25': price_series.quantile(0.25),
                'q75': price_series.quantile(0.75)
            }
        
        # 航空公司统计
        airline_cols = [col for col in df.columns if 'Carrier_code' in col]
        if airline_cols:
            all_airlines = []
            for col in airline_cols:
                all_airlines.extend(df[col].dropna().astype(str).tolist())
            
            if all_airlines:
                airline_freq = pd.Series(all_airlines).value_counts()
                self.global_stats['top_airlines'] = airline_freq.head(10).index.tolist()
        
        # 机场统计
        airport_cols = [col for col in df.columns if 'airport_iata' in col]
        if airport_cols:
            all_airports = []
            for col in airport_cols:
                all_airports.extend(df[col].dropna().astype(str).tolist())
            
            if all_airports:
                airport_freq = pd.Series(all_airports).value_counts()
                self.global_stats['top_airports'] = airport_freq.head(20).index.tolist()
    
    def create_user_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """构建用户特征"""
        # 基础用户特征
        user_cols = ['bySelf', 'frequentFlyer', 'isVip', 'isAccess3D']
        for col in user_cols:
            if col in df.columns:
                df[f'{col}_num'] = self._safe_numeric_convert(df[col])
        
        # 用户成熟度评分
        if all(col in df.columns for col in ['frequentFlyer', 'isVip']):
            freq_num = self._safe_numeric_convert(df['frequentFlyer'])
            vip_num = self._safe_numeric_convert(df['isVip'])
            df['user_maturity_score'] = freq_num * 0.4 + vip_num * 0.6
        
        # 性别编码
        if 'sex' in df.columns:
            df['gender_encoded'] = self._safe_numeric_convert(df['sex'], -1)
        
        # 国籍特征
        if 'nationality' in df.columns:
            # 简单编码：是否为主要国籍
            if 'top_airlines' in self.global_stats:  # 作为代理指标
                df['is_major_nationality'] = df['nationality'].notna().astype(int)
        
        return df
    
    def create_corporate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """构建企业特征"""
        # 企业特征
        if 'companyID' in df.columns:
            df['has_company'] = df['companyID'].notna().astype(int)
            # 企业活跃度（基于出现频率）
            if df['companyID'].notna().sum() > 0:
                company_freq = df['companyID'].value_counts()
                df['company_frequency'] = df['companyID'].map(company_freq).fillna(1)
                df['is_frequent_company'] = (df['company_frequency'] > 
                                           df['company_frequency'].median()).astype(int)
        
        # 企业政策特征
        if 'corporateTariffCode' in df.columns:
            df['has_corporate_tariff'] = df['corporateTariffCode'].notna().astype(int)
        
        if 'pricingInfo_isAccessTP' in df.columns:
            df['corporate_pricing_access'] = self._safe_numeric_convert(df['pricingInfo_isAccessTP'])
        
        # 乘客数量
        if 'pricingInfo_passengerCount' in df.columns:
            passenger_count = self._safe_numeric_convert(df['pricingInfo_passengerCount'], 1)
            df['passenger_count'] = passenger_count
            df['is_group_booking'] = (passenger_count > 1).astype(int)
        
        return df
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """构建时间特征"""
        # 时间转换
        time_cols = ['requestDate', 'legs0_departureAt', 'legs0_arrivalAt', 
                    'legs1_departureAt', 'legs1_arrivalAt']
        for col in time_cols:
            if col in df.columns:
                df[col] = self._safe_datetime_convert(df[col])
        
        # 预订提前期
        if all(col in df.columns for col in ['requestDate', 'legs0_departureAt']):
            df['booking_advance_days'] = (
                df['legs0_departureAt'] - df['requestDate']
            ).dt.days.fillna(0).clip(0, 365)
            
            # 预订时机分类（修复原错误）
            advance_days = df['booking_advance_days']
            booking_category = pd.cut(
                advance_days,
                bins=[-1, 7, 30, 90, 365],
                labels=['LastMinute', 'Weekly', 'Monthly', 'Advanced']
            )
            # 转换为字符串后再填充缺失值
            df['booking_timing_category'] = booking_category.astype(str).replace('nan', 'Unknown')
            
            # 预订紧急度
            df['booking_urgency'] = (df['booking_advance_days'] < 7).astype(int)
            df['advance_booking'] = (df['booking_advance_days'] > 30).astype(int)
        
        # 出发时间特征
        if 'legs0_departureAt' in df.columns:
            departure_dt = df['legs0_departureAt']
            df['departure_hour'] = departure_dt.dt.hour.fillna(12)
            df['departure_day_of_week'] = departure_dt.dt.dayofweek.fillna(0)
            df['departure_month'] = departure_dt.dt.month.fillna(1)
            df['is_weekend_departure'] = (df['departure_day_of_week'] >= 5).astype(int)
            
            # 商务友好时间
            business_hours = [7, 8, 9, 17, 18, 19, 20]
            df['business_friendly_departure'] = df['departure_hour'].isin(business_hours).astype(int)
            
            # 时间段分类
            def categorize_time(hour):
                if pd.isna(hour):
                    return 'Unknown'
                elif 5 <= hour < 12:
                    return 'Morning'
                elif 12 <= hour < 18:
                    return 'Afternoon'
                elif 18 <= hour < 22:
                    return 'Evening'
                else:
                    return 'Night'
            
            df['departure_period'] = df['departure_hour'].apply(categorize_time)
        
        # 飞行时长特征
        duration_cols = ['legs0_duration', 'legs1_duration']
        for col in duration_cols:
            if col in df.columns:
                duration_hours = self._safe_numeric_convert(df[col]) / 3600
                df[f'{col}_hours'] = duration_hours
                df[f'{col}_is_long'] = (duration_hours > 6).astype(int)
        
        return df
    
    def create_pricing_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """构建价格特征"""
        if 'totalPrice_bin' in df.columns:
            price_numeric = self._safe_numeric_convert(df['totalPrice_bin'])
            df['total_price'] = price_numeric
            
            # 组内价格特征
            if 'ranker_id' in df.columns:
                # 组内价格排名
                df['price_rank_in_group'] = df.groupby('ranker_id')['total_price'].rank(method='min')
                df['price_percentile_in_group'] = df.groupby('ranker_id')['total_price'].rank(pct=True)
                
                # 组内统计
                group_stats = df.groupby('ranker_id')['total_price'].agg(['mean', 'std', 'min', 'max'])
                group_stats['std'] = group_stats['std'].fillna(1e-6)
                group_stats.columns = ['group_price_mean', 'group_price_std', 'group_price_min', 'group_price_max']
                
                df = df.merge(group_stats, left_on='ranker_id', right_index=True, how='left')
                
                # 价格相对位置
                df['price_above_group_mean'] = (df['total_price'] > df['group_price_mean']).astype(int)
                df['is_cheapest_option'] = (df['price_rank_in_group'] == 1).astype(int)
                df['is_premium_option'] = (df['price_percentile_in_group'] >= 0.8).astype(int)
        
        # 全局价格特征
        if 'price' in self.global_stats and 'total_price' in df.columns:
            stats = self.global_stats['price']
            df['price_above_global_mean'] = (df['total_price'] > stats['mean']).astype(int)
            df['price_global_zscore'] = (df['total_price'] - stats['mean']) / stats['std']
        
        # 税费特征
        if 'taxes_bin' in df.columns and 'total_price' in df.columns:
            taxes_numeric = self._safe_numeric_convert(df['taxes_bin'])
            df['taxes_amount'] = taxes_numeric
            df['tax_ratio'] = taxes_numeric / (df['total_price'] + 1e-6)
        
        return df
    
    def create_route_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """构建航线特征"""
        # 航段数量特征
        legs_counts = []
        for leg in ['legs0', 'legs1']:
            max_segments = 0
            for i in range(5):  # 检查最多5个航段
                segment_col = f'{leg}_segments{i}_aircraft_code'
                if segment_col in df.columns and not df[segment_col].isna().all():
                    max_segments = i + 1
            legs_counts.append(max_segments)
            df[f'{leg}_segment_count'] = max_segments
        
        # 总航段数和复杂度
        df['total_segments'] = sum(legs_counts)
        df['is_direct_flight'] = (df['total_segments'] == 2).astype(int)
        df['has_connections'] = (df['total_segments'] > 2).astype(int)
        
        # 航空公司特征
        marketing_cols = [col for col in df.columns if 'marketingCarrier_code' in col]
        if marketing_cols and 'top_airlines' in self.global_stats:
            top_airlines = self.global_stats['top_airlines']
            df['uses_major_airline'] = df[marketing_cols[0]].astype(str).isin(top_airlines).astype(int)
        
        # 机场特征
        departure_airports = [col for col in df.columns if 'departureFrom_airport_iata' in col]
        arrival_airports = [col for col in df.columns if 'arrivalTo_airport_iata' in col]
        
        if departure_airports and 'top_airports' in self.global_stats:
            top_airports = self.global_stats['top_airports']
            df['departure_major_airport'] = df[departure_airports[0]].astype(str).isin(top_airports).astype(int)
        
        if arrival_airports and 'top_airports' in self.global_stats:
            top_airports = self.global_stats['top_airports']
            df['arrival_major_airport'] = df[arrival_airports[0]].astype(str).isin(top_airports).astype(int)
        
        return df
    
    def create_service_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """构建服务特征"""
        # 舱位特征
        cabin_cols = [col for col in df.columns if 'cabinClass' in col]
        if cabin_cols:
            cabin_values = []
            for col in cabin_cols:
                cabin_values.append(self._safe_numeric_convert(df[col]))
            
            if cabin_values:
                cabin_df = pd.DataFrame(cabin_values).T
                df['highest_cabin_class'] = cabin_df.max(axis=1)
                df['lowest_cabin_class'] = cabin_df.min(axis=1)
                df['cabin_consistency'] = (cabin_df.max(axis=1) == cabin_df.min(axis=1)).astype(int)
        
        # 座位可用性
        seat_cols = [col for col in df.columns if 'seatsAvailable' in col]
        if seat_cols:
            seat_values = []
            for col in seat_cols:
                seat_values.append(self._safe_numeric_convert(df[col]))
            
            if seat_values:
                seat_df = pd.DataFrame(seat_values).T
                df['min_seats_available'] = seat_df.min(axis=1)
                df['seat_scarcity'] = (df['min_seats_available'] <= 5).astype(int)
        
        # 行李特征
        baggage_cols = [col for col in df.columns if 'baggageAllowance_quantity' in col]
        if baggage_cols:
            baggage_values = []
            for col in baggage_cols:
                baggage_values.append(self._safe_numeric_convert(df[col]))
            
            if baggage_values:
                baggage_df = pd.DataFrame(baggage_values).T
                df['min_baggage_pieces'] = baggage_df.min(axis=1)
                df['generous_baggage_policy'] = (df['min_baggage_pieces'] >= 2).astype(int)
        
        return df
    
    def create_policy_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """构建政策特征"""
        # 改签费用特征
        amount_cols = [col for col in df.columns if 'miniRules' in col and 'monetaryAmount' in col]
        if amount_cols:
            amount_values = []
            for col in amount_cols:
                amount_values.append(self._safe_numeric_convert(df[col]))
            
            if amount_values:
                amount_df = pd.DataFrame(amount_values).T
                df['max_change_fee'] = amount_df.max(axis=1)
                df['has_free_changes'] = (amount_df.min(axis=1) == 0).astype(int)
        
        # 政策存在性
        policy_cols = [col for col in df.columns if 'miniRules' in col]
        if policy_cols:
            df['policy_count'] = df[policy_cols].count(axis=1)
            df['has_flexible_policy'] = (df['policy_count'] > 0).astype(int)
        
        return df
    
    def create_composite_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """构建复合特征"""
        # 商务友好度评分
        business_factors = []
        if 'business_friendly_departure' in df.columns:
            business_factors.append(df['business_friendly_departure'])
        if 'is_direct_flight' in df.columns:
            business_factors.append(df['is_direct_flight'])
        if 'departure_major_airport' in df.columns:
            business_factors.append(df['departure_major_airport'])
        
        if business_factors:
            df['business_friendliness_score'] = np.mean(business_factors, axis=0)
        
        # 性价比评分
        value_factors = []
        if 'price_percentile_in_group' in df.columns:
            value_factors.append(1 - df['price_percentile_in_group'])  # 价格越低越好
        if 'business_friendliness_score' in df.columns:
            value_factors.append(df['business_friendliness_score'])
        
        if value_factors:
            df['value_for_money_score'] = np.mean(value_factors, axis=0)
        
        # 用户匹配度
        if 'user_maturity_score' in df.columns and 'business_friendliness_score' in df.columns:
            df['user_match_score'] = df['user_maturity_score'] * df['business_friendliness_score']
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """构建交互特征"""
        # 用户-价格交互
        if all(col in df.columns for col in ['user_maturity_score', 'price_percentile_in_group']):
            df['user_price_sensitivity'] = df['user_maturity_score'] * (1 - df['price_percentile_in_group'])
        
        # 时间-价格交互
        if all(col in df.columns for col in ['booking_urgency', 'price_percentile_in_group']):
            df['urgency_price_tolerance'] = df['booking_urgency'] * df['price_percentile_in_group']
        
        # 企业-服务交互
        if all(col in df.columns for col in ['has_company', 'business_friendliness_score']):
            df['company_service_match'] = df['has_company'] * df['business_friendliness_score']
        
        return df
    
    def process_features(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """主要特征处理流程"""
        print("开始特征工程...")
        
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
        
        print("创建交互特征...")
        df = self.create_interaction_features(df)
        
        # 处理缺失值
        print("处理缺失值...")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        exclude_cols = ['ranker_id', 'Id'] + (['selected'] if 'selected' in df.columns else [])
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
    train_data = pd.read_parquet("D:/kaggle/filght/data/aeroclub-recsys-2025/segment/train/train_segment_0.parquet")
    
    # 训练模式特征工程
    processed_data = engineer.process_features(train_data, is_training=True)
    
    print(f"原始数据形状: {train_data.shape}")
    print(f"处理后数据形状: {processed_data.shape}")
    print(f"新增特征数量: {processed_data.shape[1] - train_data.shape[1]}")
    
    # 显示部分新特征
    new_features = [col for col in processed_data.columns if col not in train_data.columns]
    print(f"\n新增特征示例: {new_features[:10]}")