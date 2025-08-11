"""
数据处理模块 - 重构版 v5.1
统一数据加载、编码、特征工程和选择功能，添加缓存管理和灵活的处理模式

作者: Flight Ranking Team
版本: 5.1 (改进版)
"""

import pandas as pd
import numpy as np
import warnings
import pickle
import hashlib
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any, Union
from sklearn.model_selection import train_test_split
from abc import ABC, abstractmethod
import gc
import time

warnings.filterwarnings('ignore')


class CacheManager:
    """缓存管理器"""
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_index_file = self.cache_dir / "cache_index.pkl"
        self.cache_index = self._load_cache_index()
    
    def _load_cache_index(self) -> Dict[str, Any]:
        """加载缓存索引"""
        if self.cache_index_file.exists():
            try:
                with open(self.cache_index_file, 'rb') as f:
                    return pickle.load(f)
            except:
                return {}
        return {}
    
    def _save_cache_index(self):
        """保存缓存索引"""
        try:
            with open(self.cache_index_file, 'wb') as f:
                pickle.dump(self.cache_index, f)
        except Exception as e:
            print(f"保存缓存索引失败: {e}")
    
    def _generate_cache_key(self, file_path: Path, config: Dict[str, Any]) -> str:
        """生成缓存键"""
        # 基于文件路径、文件修改时间和配置生成唯一键
        file_stat = file_path.stat()
        key_data = {
            'file_path': str(file_path),
            'file_size': file_stat.st_size,
            'file_mtime': file_stat.st_mtime,
            'config': sorted(config.items())
        }
        key_str = str(key_data)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get_cached_data(self, file_path: Path, config: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """获取缓存数据"""
        cache_key = self._generate_cache_key(file_path, config)
        
        if cache_key in self.cache_index:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            if cache_file.exists():
                try:
                    df = pd.read_pickle(cache_file)
                    print(f"✅ 加载缓存: {file_path.name}")
                    return df
                except Exception as e:
                    print(f"加载缓存失败: {e}")
                    # 删除损坏的缓存
                    self._remove_cache(cache_key)
        
        return None
    
    def save_cached_data(self, df: pd.DataFrame, file_path: Path, config: Dict[str, Any]):
        """保存数据到缓存"""
        cache_key = self._generate_cache_key(file_path, config)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            df.to_pickle(cache_file)
            
            # 更新缓存索引
            self.cache_index[cache_key] = {
                'file_path': str(file_path),
                'cache_file': str(cache_file),
                'config': config,
                'created_time': time.time(),
                'data_shape': df.shape
            }
            self._save_cache_index()
            
            print(f"💾 保存缓存: {file_path.name} -> {cache_file.name}")
            
        except Exception as e:
            print(f"保存缓存失败: {e}")
    
    def _remove_cache(self, cache_key: str):
        """删除指定缓存"""
        if cache_key in self.cache_index:
            cache_file = Path(self.cache_index[cache_key]['cache_file'])
            if cache_file.exists():
                cache_file.unlink()
            del self.cache_index[cache_key]
            self._save_cache_index()
    
    def clear_all_cache(self):
        """清理所有缓存"""
        for cache_file in self.cache_dir.glob("*.pkl"):
            if cache_file.name != "cache_index.pkl":
                cache_file.unlink()
        
        self.cache_index = {}
        self._save_cache_index()
        print("🗑️ 已清理所有缓存")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """获取缓存信息"""
        cache_files = list(self.cache_dir.glob("*.pkl"))
        cache_files = [f for f in cache_files if f.name != "cache_index.pkl"]
        
        total_size = sum(f.stat().st_size for f in cache_files)
        
        return {
            'cache_count': len(cache_files),
            'total_size_mb': total_size / (1024 * 1024),
            'cache_index_count': len(self.cache_index),
            'cache_files': [f.name for f in cache_files]
        }


class BaseDataProcessor(ABC):
    """数据处理器基类"""
    
    @abstractmethod
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理数据"""
        pass


class DataEncoder(BaseDataProcessor):
    """数据编码器"""
    
    def __init__(self, logger=None):
        self.logger = logger
    
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """数据编码主流程"""
        df = df.copy()
        
        if self.logger:
            original_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
            self.logger.info(f"开始数据编码，原始内存: {original_memory:.1f}MB")
        
        # 编码各类型数据
        df = self._encode_integers(df)
        df = self._encode_times(df)
        df = self._encode_durations(df)
        df = self._encode_booleans(df)
        df = self._encode_categoricals(df)
        df = self._encode_special_features(df)
        
        if self.logger:
            new_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
            reduction = (1 - new_memory / original_memory) * 100
            self.logger.info(f"数据编码完成，新内存: {new_memory:.1f}MB，减少: {reduction:.1f}%")
        
        return df
    
    def _encode_integers(self, df: pd.DataFrame) -> pd.DataFrame:
        """编码整数类型字段"""
        int_cols = [
            'legs1_segments3_baggageAllowance_quantity', 'legs1_segments3_cabinClass',
            'legs1_segments3_seatsAvailable', 'legs0_segments3_baggageAllowance_quantity',
            'legs0_segments3_cabinClass', 'legs0_segments3_seatsAvailable',
            'legs1_segments2_baggageAllowance_quantity', 'legs1_segments2_cabinClass',
            'legs1_segments2_seatsAvailable', 'legs0_segments2_baggageAllowance_quantity',
            'legs0_segments2_cabinClass', 'legs0_segments2_seatsAvailable',
            'miniRules1_percentage', 'miniRules0_percentage', 'legs1_segments1_seatsAvailable',
            'legs1_segments1_baggageAllowance_quantity', 'legs1_segments1_cabinClass',
            'legs0_segments1_seatsAvailable', 'legs0_segments1_baggageAllowance_quantity',
            'legs0_segments1_cabinClass', 'corporateTariffCode', 'legs1_segments0_seatsAvailable',
            'legs1_segments0_baggageAllowance_quantity', 'legs1_segments0_cabinClass',
            'miniRules1_statusInfos', 'miniRules0_statusInfos', 'miniRules1_monetaryAmount',
            'miniRules0_monetaryAmount', 'pricingInfo_isAccessTP', 'legs0_segments0_seatsAvailable',
            'legs0_segments0_baggageAllowance_quantity', 'legs0_segments0_cabinClass',
            'nationality', 'Id', 'pricingInfo_passengerCount', 'profileId', 'companyID'
        ]
        
        for col in int_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(-1).astype('int32')
        
        return df
    
    def _encode_times(self, df: pd.DataFrame) -> pd.DataFrame:
        """编码时间类型字段"""
        time_cols = ['legs1_departureAt', 'legs1_arrivalAt', 'legs0_departureAt', 'legs0_arrivalAt', 'requestDate']
        
        for col in time_cols:
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    df[col] = (df[col] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
                    df[col] = df[col].fillna(-1).astype('int32')
                except Exception:
                    df[col] = -1
                    df[col] = df[col].astype('int32')
        
        return df
    
    def _encode_durations(self, df: pd.DataFrame) -> pd.DataFrame:
        """编码持续时间字段"""
        duration_cols = [
            'legs1_duration', 'legs1_segments0_duration', 'legs0_segments0_duration',
            'legs0_duration', 'legs1_segments3_duration', 'legs0_segments3_duration',
            'legs1_segments2_duration', 'legs0_segments2_duration', 'legs1_segments1_duration',
            'legs0_segments1_duration'
        ]
        
        for col in duration_cols:
            if col in df.columns:
                df[col] = df[col].apply(self._parse_duration).astype('int32')
        
        return df
    
    def _encode_booleans(self, df: pd.DataFrame) -> pd.DataFrame:
        """编码布尔类型字段"""
        bool_cols = ['isAccess3D', 'isVip', 'bySelf', 'sex']
        
        for col in bool_cols:
            if col in df.columns:
                df[col] = df[col].astype('bool')
        
        return df
    
    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """编码分类变量"""
        str_cat_cols = [
            'frequentFlyer', 'legs1_segments3_marketingCarrier_code', 'legs1_segments3_operatingCarrier_code',
            'legs1_segments3_flightNumber', 'legs1_segments3_arrivalTo_airport_iata',
            'legs1_segments3_departureFrom_airport_iata', 'legs1_segments3_aircraft_code',
            'legs0_segments3_marketingCarrier_code', 'legs0_segments3_operatingCarrier_code',
            'legs0_segments3_flightNumber', 'legs0_segments3_arrivalTo_airport_iata',
            'legs0_segments3_departureFrom_airport_iata', 'legs0_segments3_aircraft_code',
            'legs1_segments2_marketingCarrier_code', 'legs1_segments2_operatingCarrier_code',
            'legs1_segments2_flightNumber', 'legs1_segments2_arrivalTo_airport_iata',
            'legs1_segments2_departureFrom_airport_iata', 'legs1_segments2_aircraft_code',
            'legs0_segments2_marketingCarrier_code', 'legs0_segments2_operatingCarrier_code',
            'legs0_segments2_flightNumber', 'legs0_segments2_arrivalTo_airport_iata',
            'legs0_segments2_departureFrom_airport_iata', 'legs0_segments2_aircraft_code',
            'legs1_segments1_marketingCarrier_code', 'legs1_segments1_operatingCarrier_code',
            'legs1_segments1_flightNumber', 'legs1_segments1_arrivalTo_airport_iata',
            'legs1_segments1_departureFrom_airport_iata', 'legs1_segments1_aircraft_code',
            'legs0_segments1_marketingCarrier_code', 'legs0_segments1_operatingCarrier_code',
            'legs0_segments1_flightNumber', 'legs0_segments1_arrivalTo_airport_iata',
            'legs0_segments1_departureFrom_airport_iata', 'legs0_segments1_aircraft_code',
            'legs1_segments0_marketingCarrier_code', 'legs1_segments0_operatingCarrier_code',
            'legs1_segments0_flightNumber', 'legs1_segments0_arrivalTo_airport_iata',
            'legs1_segments0_departureFrom_airport_iata', 'legs1_segments0_aircraft_code',
            'legs0_segments0_marketingCarrier_code', 'legs0_segments0_operatingCarrier_code',
            'legs0_segments0_flightNumber', 'legs0_segments0_arrivalTo_airport_iata',
            'legs0_segments0_departureFrom_airport_iata', 'legs0_segments0_aircraft_code'
        ]
        
        for col in str_cat_cols:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: hash(x) & 0x7FFFFFFF if pd.notna(x) else -1
                ).astype('int32')
        
        return df
    
    def _encode_special_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """编码特殊特征"""
        # 处理搜索路线
        if 'searchRoute' in df.columns:
            df['is_round_trip'] = df['searchRoute'].str.contains('/', na=False).astype('int8')
            df.drop(columns=['searchRoute'], inplace=True, errors='ignore')
        
        # 处理价格字段（分箱）
        for col in ['taxes', 'totalPrice']:
            if col in df.columns:
                try:
                    df[f'{col}_bin'] = pd.qcut(df[col], q=10, labels=False, duplicates='drop').fillna(-1).astype('int8')
                    df.drop(columns=[col], inplace=True, errors='ignore')
                except:
                    df[f'{col}_bin'] = 0
        
        # 目标变量
        if 'selected' in df.columns:
            df['selected'] = df['selected'].fillna(-1).astype('int8')
        
        return df
    
    @staticmethod
    def _parse_duration(duration_str) -> int:
        """解析持续时间字符串为秒数"""
        if pd.isna(duration_str):
            return -1
        
        try:
            duration_str = str(duration_str)
            if ':' in duration_str:
                parts = duration_str.split(':')
                if len(parts) == 3:
                    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
                elif len(parts) == 2:
                    return int(parts[0]) * 60 + int(parts[1])
            return int(float(duration_str) * 3600)
        except:
            return 0


class FeatureEngineer(BaseDataProcessor):
    """特征工程器"""
    
    def __init__(self, level: str = 'enhanced', logger=None):
        self.level = level
        self.logger = logger
    
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """特征工程主流程"""
        if self.level == 'none':
            return df
        
        df = df.copy()
        
        if self.logger:
            self.logger.info(f"开始特征工程，级别: {self.level}")
        
        # 核心特征
        if self.level in ['basic', 'enhanced', 'advanced']:
            df = self._create_core_features(df)
        
        if self.level in ['enhanced', 'advanced']:
            df = self._create_enhanced_features(df)
        
        if self.level == 'advanced':
            df = self._create_advanced_features(df)
        
        return df
    
    def _create_core_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建核心特征"""
        # 价格特征
        if 'taxes_bin' in df.columns and 'totalPrice_bin' in df.columns:
            df['tax_rate'] = df['taxes_bin'] / (df['totalPrice_bin'] + 1)
            df['log_price_proxy'] = np.log1p(df['totalPrice_bin'])
        
        # 持续时间特征
        if 'legs0_duration' in df.columns and 'legs1_duration' in df.columns:
            df['total_duration'] = df['legs0_duration'].fillna(0) + df['legs1_duration'].fillna(0)
            df['duration_ratio'] = np.where(
                df['legs1_duration'].fillna(0) > 0,
                df['legs0_duration'] / (df['legs1_duration'] + 1),
                1.0
            )
        
        # 行程类型特征
        if 'legs1_duration' in df.columns:
            df['is_one_way'] = (
                (df['legs1_duration'] == -1) | 
                (df['legs1_duration'] == 0) |
                (df['legs1_duration'].isna())
            ).astype('int8')
        
        # 时间特征
        time_cols = ['legs0_departureAt', 'legs0_arrivalAt', 'legs1_departureAt', 'legs1_arrivalAt']
        for col in time_cols:
            if col in df.columns:
                valid_mask = df[col] != -1
                df[f'{col}_hour'] = np.where(
                    valid_mask,
                    (df[col] % 86400) // 3600,
                    12
                ).astype('int8')
                
                df[f'{col}_business_time'] = (
                    ((df[f'{col}_hour'] >= 6) & (df[f'{col}_hour'] <= 9)) |
                    ((df[f'{col}_hour'] >= 17) & (df[f'{col}_hour'] <= 20))
                ).astype('int8')
        
        # 组相关特征
        if 'ranker_id' in df.columns:
            df['group_size'] = df.groupby('ranker_id')['Id'].transform('count').astype('int16')
            
            if 'totalPrice_bin' in df.columns:
                df['price_rank'] = df.groupby('ranker_id')['totalPrice_bin'].rank().astype('int8')
                df['is_cheapest'] = (df['price_rank'] == 1).astype('int8')
        
        return df
    
    def _create_enhanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建增强特征"""
        # 预订时机特征
        if 'requestDate' in df.columns and 'legs0_departureAt' in df.columns:
            valid_mask = (df['legs0_departureAt'] != -1) & (df['requestDate'] != -1)
            df['days_ahead'] = np.where(
                valid_mask,
                np.clip((df['legs0_departureAt'] - df['requestDate']) // 86400, 0, 365),
                7
            ).astype('int16')
            
            df['booking_urgency'] = np.select([
                df['days_ahead'] <= 1,
                df['days_ahead'] <= 7,
                df['days_ahead'] <= 14,
                df['days_ahead'] <= 30,
            ], [4, 3, 2, 1], default=0).astype('int8')
        
        # 舱位等级特征
        cabin_cols = [col for col in df.columns if 'cabinClass' in col]
        if cabin_cols:
            df['avg_cabin_class'] = df[cabin_cols].replace(-1, np.nan).mean(axis=1, skipna=True).fillna(0)
        
        # VIP和企业用户特征组合
        if 'isVip' in df.columns and 'corporateTariffCode' in df.columns:
            df['vip_or_corporate'] = (
                (df['isVip'] == 1) | (df['corporateTariffCode'] != -1)
            ).astype('int8')
        
        return df
    
    def _create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建高级特征"""
        # 经济学特征
        if 'ranker_id' in df.columns and 'totalPrice_bin' in df.columns:
            # 价格分散度
            group_price_stats = df.groupby('ranker_id')['totalPrice_bin'].agg(['std', 'mean'])
            group_price_stats.columns = ['price_std', 'price_mean']
            df = df.merge(group_price_stats, left_on='ranker_id', right_index=True, how='left')
            
            df['price_coefficient_variation'] = df['price_std'] / (df['price_mean'] + 1)
        
        # 选择复杂度特征
        if 'ranker_id' in df.columns:
            choice_set_size = df.groupby('ranker_id')['Id'].transform('count')
            df['choice_overload_score'] = np.select([
                choice_set_size >= 20,
                choice_set_size >= 10,
                choice_set_size >= 5,
            ], [3, 2, 1], default=0).astype('int8')
        
        # 算术组合特征（选择性创建）
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_pairs = [
            ('totalPrice_bin', 'total_duration'),
            ('price_rank', 'group_size'),
            ('days_ahead', 'totalPrice_bin')
        ]
        
        for col1, col2 in feature_pairs:
            if col1 in numeric_cols and col2 in numeric_cols:
                df[f'{col1}_{col2}_ratio'] = df[col1] / (df[col2] + 1)
                df[f'{col1}_{col2}_product'] = df[col1] * df[col2]
        
        return df


class FeatureSelector(BaseDataProcessor):
    """特征选择器"""
    
    def __init__(self, max_features: int = 200, method: str = 'variance', logger=None):
        self.max_features = max_features
        self.method = method
        self.logger = logger
        self.selected_features = None
    
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """特征选择主流程"""
        if self.method == 'none' or len(df.columns) <= self.max_features:
            return df
        
        if self.logger:
            self.logger.info(f"开始特征选择: {len(df.columns)} -> {self.max_features}")
        
        # 排除特征
        exclude_features = ['Id', 'selected', 'ranker_id', 'profileId', 'companyID']
        feature_cols = [col for col in df.columns if col not in exclude_features]
        
        # 特征选择
        if self.method == 'variance':
            selected = self._select_by_variance(df[feature_cols])
        elif self.method == 'correlation':
            selected = self._select_by_correlation(df[feature_cols])
        elif self.method == 'mutual_info':
            selected = self._select_by_mutual_info(df[feature_cols], df.get('selected'))
        else:
            selected = feature_cols[:self.max_features]
        
        # 保留必要列
        final_cols = exclude_features + selected
        final_cols = [col for col in final_cols if col in df.columns]
        
        self.selected_features = selected
        
        if self.logger:
            self.logger.info(f"特征选择完成: {len(selected)} 个特征被选中")
        
        return df[final_cols]
    
    def _select_by_variance(self, df: pd.DataFrame) -> List[str]:
        """基于方差的特征选择"""
        variances = df.var().sort_values(ascending=False)
        return variances.head(self.max_features).index.tolist()
    
    def _select_by_correlation(self, df: pd.DataFrame) -> List[str]:
        """基于相关性的特征选择"""
        # 先按方差过滤
        high_var_features = self._select_by_variance(df)
        df_filtered = df[high_var_features]
        
        # 移除高相关性特征
        corr_matrix = df_filtered.corr().abs()
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        to_drop = []
        for col in upper_triangle.columns:
            if any(upper_triangle[col] > 0.9):
                to_drop.append(col)
        
        remaining = [col for col in high_var_features if col not in to_drop]
        return remaining[:self.max_features]
    
    def _select_by_mutual_info(self, df: pd.DataFrame, target: pd.Series) -> List[str]:
        """基于互信息的特征选择"""
        try:
            from sklearn.feature_selection import mutual_info_regression
            from sklearn.preprocessing import LabelEncoder
            
            if target is None or target.isna().all():
                return self._select_by_variance(df)
            
            # 处理缺失值
            df_filled = df.fillna(df.median())
            target_filled = target.fillna(target.median())
            
            # 计算互信息
            mi_scores = mutual_info_regression(df_filled, target_filled)
            mi_df = pd.DataFrame({'feature': df.columns, 'mi_score': mi_scores})
            mi_df = mi_df.sort_values('mi_score', ascending=False)
            
            return mi_df.head(self.max_features)['feature'].tolist()
            
        except ImportError:
            print("sklearn不可用，使用方差选择")
            return self._select_by_variance(df)
        except Exception as e:
            print(f"互信息选择失败: {e}，使用方差选择")
            return self._select_by_variance(df)


class EnhancedDataProcessor:
    """增强的数据处理器"""
    
    def __init__(self, feature_level: str = 'enhanced', 
                 max_features: int = 200,
                 selection_mode: str = 'variance',
                 cache_dir: Optional[Path] = None,
                 enable_cache: bool = True,
                 logger=None):
        """
        初始化增强数据处理器
        
        Args:
            feature_level: 特征工程级别 ('none', 'basic', 'enhanced', 'advanced')
            max_features: 最大特征数
            selection_mode: 特征选择模式 ('none', 'variance', 'correlation', 'mutual_info')
            cache_dir: 缓存目录
            enable_cache: 是否启用缓存
            logger: 日志记录器
        """
        self.logger = logger
        self.enable_cache = enable_cache
        
        # 初始化处理组件
        self.encoder = DataEncoder(logger)
        self.engineer = FeatureEngineer(feature_level, logger)
        self.selector = FeatureSelector(max_features, selection_mode, logger) if selection_mode != 'none' else None
        
        # 初始化缓存管理器
        if enable_cache and cache_dir:
            self.cache_manager = CacheManager(cache_dir)
        else:
            self.cache_manager = None
        
        self.feature_columns = None
        self.processing_config = {
            'feature_level': feature_level,
            'max_features': max_features,
            'selection_mode': selection_mode
        }
    
    def load_and_process_data(self, file_path: Path, 
                             use_sampling: bool = False,
                             num_groups: int = 2000,
                             min_group_size: int = 20,
                             force_reprocess: bool = False) -> pd.DataFrame:
        """加载并处理数据"""
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        # 构建完整配置用于缓存键生成
        full_config = self.processing_config.copy()
        full_config.update({
            'use_sampling': use_sampling,
            'num_groups': num_groups,
            'min_group_size': min_group_size
        })
        
        # 尝试从缓存加载
        if not force_reprocess and self.cache_manager and self.enable_cache:
            cached_df = self.cache_manager.get_cached_data(file_path, full_config)
            if cached_df is not None:
                return cached_df
        
        # 加载原始数据
        df = pd.read_parquet(file_path)
        if self.logger:
            self.logger.info(f"加载数据: {df.shape}")
        
        # 数据抽样
        if use_sampling:
            df = self._sample_data(df, num_groups, min_group_size)
            if self.logger:
                self.logger.info(f"抽样后数据: {df.shape}")
        
        # 数据处理流水线
        df = self.encoder.process(df)
        df = self.engineer.process(df)
        
        if self.selector:
            df = self.selector.process(df)
            self.feature_columns = self.selector.selected_features
        
        # 保存到缓存
        if self.cache_manager and self.enable_cache:
            self.cache_manager.save_cached_data(df, file_path, full_config)
        
        return df
    
    def process_with_different_levels(self, file_path: Path, 
                                    levels: List[str] = None,
                                    **kwargs) -> Dict[str, pd.DataFrame]:
        """使用不同特征工程级别处理数据"""
        if levels is None:
            levels = ['none', 'basic', 'enhanced', 'advanced']
        
        results = {}
        
        for level in levels:
            print(f"🔄 处理特征级别: {level}")
            
            # 创建临时处理器
            temp_processor = EnhancedDataProcessor(
                feature_level=level,
                max_features=self.processing_config['max_features'],
                selection_mode=self.processing_config['selection_mode'],
                cache_dir=self.cache_manager.cache_dir if self.cache_manager else None,
                enable_cache=self.enable_cache,
                logger=self.logger
            )
            
            try:
                df = temp_processor.load_and_process_data(file_path, **kwargs)
                results[level] = df
                print(f"✅ {level}: {df.shape}")
            except Exception as e:
                print(f"❌ {level}: {e}")
                continue
        
        return results
    
    def compare_processing_modes(self, file_path: Path, **kwargs) -> Dict[str, Any]:
        """比较不同处理模式的效果"""
        comparison_results = {}
        
        # 比较不同特征工程级别
        level_results = self.process_with_different_levels(file_path, **kwargs)
        comparison_results['by_feature_level'] = level_results
        
        # 比较不同特征选择方法
        selection_results = {}
        for method in ['none', 'variance', 'correlation']:
            temp_processor = EnhancedDataProcessor(
                feature_level=self.processing_config['feature_level'],
                max_features=self.processing_config['max_features'],
                selection_mode=method,
                cache_dir=self.cache_manager.cache_dir if self.cache_manager else None,
                enable_cache=self.enable_cache,
                logger=self.logger
            )
            
            try:
                df = temp_processor.load_and_process_data(file_path, **kwargs)
                selection_results[method] = df
                print(f"✅ 特征选择 {method}: {df.shape}")
            except Exception as e:
                print(f"❌ 特征选择 {method}: {e}")
                continue
        
        comparison_results['by_selection_method'] = selection_results
        
        return comparison_results
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """准备特征矩阵"""
        exclude_features = ['Id', 'selected', 'ranker_id', 'profileId', 'companyID']
        
        if self.feature_columns:
            feature_cols = self.feature_columns
        else:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [col for col in numeric_cols if col not in exclude_features]
        
        # 处理缺失值
        df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())
        
        X = df[feature_cols].values.astype(np.float32)
        y = df['selected'].values.astype(np.float32) if 'selected' in df.columns else None
        
        return X, y, feature_cols
    
    def split_ranking_data(self, df: pd.DataFrame, 
                          test_size: float = 0.2,
                          random_state: int = 42) -> Tuple[np.ndarray, ...]:
        """分割排序数据"""
        X, y, feature_cols = self.prepare_features(df)
        groups = df['ranker_id'].values
        
        # 按组分割
        unique_groups = df['ranker_id'].unique()
        train_groups, test_groups = train_test_split(
            unique_groups, test_size=test_size, random_state=random_state
        )
        
        train_mask = df['ranker_id'].isin(train_groups)
        test_mask = df['ranker_id'].isin(test_groups)
        
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        
        # 计算组大小
        train_group_sizes = self._calculate_group_sizes(groups[train_mask])
        test_group_sizes = self._calculate_group_sizes(groups[test_mask])
        
        test_info = df[test_mask][['ranker_id', 'selected']].copy()
        
        return (X_train, X_test, y_train, y_test, 
                train_group_sizes, test_group_sizes, feature_cols, test_info)
    
    def prepare_test_features(self, test_df: pd.DataFrame, 
                            feature_names: List[str] = None) -> Tuple[np.ndarray, List[int]]:
        """准备测试特征"""
        # 应用相同的编码和特征工程
        test_df = self.encoder.process(test_df)
        test_df = self.engineer.process(test_df)
        
        # 使用指定的特征
        if feature_names is not None:
            feature_cols = feature_names
        elif self.feature_columns is not None:
            feature_cols = self.feature_columns
        else:
            exclude_features = ['Id', 'selected', 'ranker_id', 'profileId', 'companyID']
            numeric_cols = test_df.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [col for col in numeric_cols if col not in exclude_features]
        
        # 确保包含所需特征
        for feature in feature_cols:
            if feature not in test_df.columns:
                test_df[feature] = 0.0
        
        # 处理缺失值
        test_df[feature_cols] = test_df[feature_cols].fillna(test_df[feature_cols].median())
        
        X_test = test_df[feature_cols].values.astype(np.float32)
        group_sizes = self._calculate_group_sizes(test_df['ranker_id'].values)
        
        return X_test, group_sizes
    
    def assign_rankings(self, test_df: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
        """分配排名"""
        result_df = test_df[['Id', 'ranker_id']].copy()
        result_df['scores'] = scores
        
        result_df = result_df.sort_values(['ranker_id', 'scores', 'Id'], 
                                        ascending=[True, False, True])
        result_df['selected'] = result_df.groupby('ranker_id').cumcount() + 1
        
        return result_df[['Id', 'ranker_id', 'selected']]
    
    def _sample_data(self, df: pd.DataFrame, num_groups: int, min_group_size: int) -> pd.DataFrame:
        """数据抽样"""
        group_counts = df['ranker_id'].value_counts()
        valid_groups = group_counts[group_counts >= min_group_size].index
        
        if len(valid_groups) < num_groups:
            num_groups = len(valid_groups)
        
        np.random.seed(42)
        selected_groups = np.random.choice(valid_groups, size=num_groups, replace=False)
        
        return df[df['ranker_id'].isin(selected_groups)].copy()
    
    def _calculate_group_sizes(self, groups: np.ndarray) -> List[int]:
        """计算组大小"""
        group_sizes = []
        current_group = groups[0]
        current_size = 1
        
        for i in range(1, len(groups)):
            if groups[i] == current_group:
                current_size += 1
            else:
                group_sizes.append(current_size)
                current_group = groups[i]
                current_size = 1
        group_sizes.append(current_size)
        
        return group_sizes
    
    def get_cache_info(self) -> Dict[str, Any]:
        """获取缓存信息"""
        if self.cache_manager:
            return self.cache_manager.get_cache_info()
        else:
            return {'cache_enabled': False}
    
    def clear_cache(self):
        """清理缓存"""
        if self.cache_manager:
            self.cache_manager.clear_all_cache()
        else:
            print("缓存未启用")


# 向后兼容的DataProcessor类
class DataProcessor(EnhancedDataProcessor):
    """向后兼容的数据处理器"""
    
    def __init__(self, feature_level: str = 'enhanced', 
                 max_features: int = 200,
                 enable_selection: bool = True,
                 logger=None):
        selection_mode = 'variance' if enable_selection else 'none'
        super().__init__(
            feature_level=feature_level,
            max_features=max_features,
            selection_mode=selection_mode,
            logger=logger,
            enable_cache=False  # 默认关闭缓存以保持兼容性
        )