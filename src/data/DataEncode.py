import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import os
import gc
import hashlib
import tempfile
from multiprocessing import Pool, cpu_count
from typing import List, Tuple, Optional
import psutil

def process_chunk_worker(args: Tuple) -> Tuple:
    """多进程worker：处理单个chunk的编码"""
    chunk_data, chunk_id, temp_dir = args
    
    try:
        # 监控内存使用
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        optimized_df = optimize_data_types(chunk_data)
        temp_file = os.path.join(temp_dir, f'chunk_{chunk_id}.parquet')
        optimized_df.to_parquet(temp_file, index=False)
        
        # 清理内存
        del optimized_df, chunk_data
        gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_used = final_memory - initial_memory
        
        return temp_file, chunk_id, True, memory_used
    except Exception as e:
        return None, chunk_id, str(e), 0

def parse_duration(duration_str) -> int:
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

def optimize_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """优化数据类型 - 无torch依赖版本"""
    df = df.copy()
    
    # 整数类型列
    int_cols = [
        'legs1_segments3_baggageAllowance_weightMeasurementType', 'legs1_segments3_baggageAllowance_quantity',
        'legs1_segments3_cabinClass', 'legs1_segments3_seatsAvailable', 'legs0_segments3_baggageAllowance_quantity',
        'legs0_segments3_baggageAllowance_weightMeasurementType', 'legs0_segments3_cabinClass', 'legs0_segments3_seatsAvailable',
        'legs1_segments2_baggageAllowance_weightMeasurementType', 'legs1_segments2_baggageAllowance_quantity',
        'legs1_segments2_cabinClass', 'legs1_segments2_seatsAvailable', 'legs0_segments2_baggageAllowance_quantity',
        'legs0_segments2_baggageAllowance_weightMeasurementType', 'legs0_segments2_cabinClass', 'legs0_segments2_seatsAvailable',
        'miniRules1_percentage', 'miniRules0_percentage', 'legs1_segments1_seatsAvailable',
        'legs1_segments1_baggageAllowance_weightMeasurementType', 'legs1_segments1_baggageAllowance_quantity',
        'legs1_segments1_cabinClass', 'legs0_segments1_seatsAvailable', 'legs0_segments1_baggageAllowance_weightMeasurementType',
        'legs0_segments1_baggageAllowance_quantity', 'legs0_segments1_cabinClass', 'corporateTariffCode',
        'legs1_segments0_seatsAvailable', 'legs1_segments0_baggageAllowance_weightMeasurementType',
        'legs1_segments0_baggageAllowance_quantity', 'legs1_segments0_cabinClass', 'miniRules1_statusInfos',
        'miniRules0_statusInfos', 'miniRules1_monetaryAmount', 'miniRules0_monetaryAmount', 'pricingInfo_isAccessTP',
        'legs0_segments0_seatsAvailable', 'legs0_segments0_baggageAllowance_weightMeasurementType',
        'legs0_segments0_baggageAllowance_quantity', 'legs0_segments0_cabinClass', 'nationality',
        'Id', 'pricingInfo_passengerCount', 'profileId', 'companyID'
    ]
    
    # 时间类型列
    time_cols = ['legs1_departureAt', 'legs1_arrivalAt', 'legs0_departureAt', 'legs0_arrivalAt', 'requestDate']
    
    # 持续时间列
    duration_cols = [
        'legs1_duration', 'legs1_segments0_duration', 'legs0_segments0_duration', 'legs0_duration',
        'legs1_segments3_duration', 'legs0_segments3_duration', 'legs1_segments2_duration',
        'legs0_segments2_duration', 'legs1_segments1_duration', 'legs0_segments1_duration'
    ]
    
    # 布尔类型列
    bool_cols = ['isAccess3D', 'isVip', 'bySelf', 'sex']
    
    # 字符串分类变量
    str_cat_cols = [
        'frequentFlyer', 'legs1_segments3_marketingCarrier_code', 'legs1_segments3_operatingCarrier_code',
        'legs1_segments3_flightNumber', 'legs1_segments3_arrivalTo_airport_iata', 'legs1_segments3_departureFrom_airport_iata',
        'legs1_segments3_aircraft_code', 'legs1_segments3_arrivalTo_airport_city_iata', 'legs0_segments3_aircraft_code',
        'legs0_segments3_marketingCarrier_code', 'legs0_segments3_flightNumber', 'legs0_segments3_arrivalTo_airport_iata',
        'legs0_segments3_departureFrom_airport_iata', 'legs0_segments3_arrivalTo_airport_city_iata',
        'legs0_segments3_operatingCarrier_code', 'legs1_segments2_marketingCarrier_code', 'legs1_segments2_operatingCarrier_code',
        'legs1_segments2_flightNumber', 'legs1_segments2_arrivalTo_airport_iata', 'legs1_segments2_departureFrom_airport_iata',
        'legs1_segments2_aircraft_code', 'legs1_segments2_arrivalTo_airport_city_iata', 'legs0_segments2_aircraft_code',
        'legs0_segments2_marketingCarrier_code', 'legs0_segments2_flightNumber', 'legs0_segments2_arrivalTo_airport_iata',
        'legs0_segments2_departureFrom_airport_iata', 'legs0_segments2_arrivalTo_airport_city_iata',
        'legs0_segments2_operatingCarrier_code', 'legs1_segments1_arrivalTo_airport_iata',
        'legs1_segments1_arrivalTo_airport_city_iata', 'legs1_segments1_operatingCarrier_code',
        'legs1_segments1_departureFrom_airport_iata', 'legs1_segments1_marketingCarrier_code',
        'legs1_segments1_flightNumber', 'legs1_segments1_aircraft_code', 'legs0_segments1_aircraft_code',
        'legs0_segments1_arrivalTo_airport_city_iata', 'legs0_segments1_flightNumber', 'legs0_segments1_arrivalTo_airport_iata',
        'legs0_segments1_operatingCarrier_code', 'legs0_segments1_marketingCarrier_code',
        'legs0_segments1_departureFrom_airport_iata', 'legs1_segments0_arrivalTo_airport_city_iata',
        'legs1_segments0_departureFrom_airport_iata', 'legs1_segments0_flightNumber', 'legs1_segments0_operatingCarrier_code',
        'legs1_segments0_aircraft_code', 'legs1_segments0_marketingCarrier_code', 'legs1_segments0_arrivalTo_airport_iata',
        'legs0_segments0_arrivalTo_airport_city_iata', 'legs0_segments0_aircraft_code', 'legs0_segments0_arrivalTo_airport_iata',
        'legs0_segments0_departureFrom_airport_iata', 'legs0_segments0_marketingCarrier_code',
        'legs0_segments0_operatingCarrier_code', 'legs0_segments0_flightNumber'
    ]
    
    # 转换整数类型
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(-1).astype('int32')
    
    # 转换时间类型
    for col in time_cols:
        if col in df.columns:
            try:
                # 转换为datetime，无效值设为NaT
                df[col] = pd.to_datetime(df[col], errors='coerce')
                # 转换为Unix时间戳（秒）
                df[col] = (df[col] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
                # 用-1填充NaT转换后的NaN
                df[col] = df[col].fillna(-1).astype('int32')
            except Exception as e:
                df[col] = -1  # 使用-1统一表示缺失值，而非0
                df[col] = df[col].astype('int32')

    # 转换持续时间
    for col in duration_cols:
        if col in df.columns:
            # 确保parse_duration能处理NaN
            df[col] = df[col].apply(lambda x: parse_duration(x)).astype('int32')
        
    # 转换布尔类型
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype('bool')
    
    # 处理特殊列
    if 'searchRoute' in df.columns:
        df['is_round_trip'] = df['searchRoute'].str.contains('/', na=False).fillna(-1).astype('int8')
        df.drop(columns=['searchRoute'], inplace=True, errors='ignore')
    
    # 分箱编码
    for col in ['taxes', 'totalPrice']:
        if col in df.columns:
            try:
                df[f'{col}_bin'] = pd.qcut(df[col], q=10, labels=False, duplicates='drop').fillna(-1).astype('int8')
                df.drop(columns=[col], inplace=True, errors='ignore')
            except:
                df[f'{col}_bin'] = 0
    
    # 字符串哈希编码
    for col in str_cat_cols:
        if col in df.columns:
            # 向量化计算：非NA值哈希，NA值填-1，直接转int32
            df[col] = df[col].apply(
                lambda x: hash(x) & 0x7FFFFFFF if pd.notna(x) else -1
            ).astype('int32')
            
    # 目标变量
    if 'selected' in df.columns:
        df['selected'] = df['selected'].fillna(-1).astype('int8')
    
    return df

def get_system_memory_info():
    """获取系统内存信息"""
    mem = psutil.virtual_memory()
    return {
        'total': mem.total / 1024 / 1024 / 1024,  # GB
        'available': mem.available / 1024 / 1024 / 1024,  # GB
        'percent': mem.percent,
        'used': mem.used / 1024 / 1024 / 1024  # GB
    }

class DataEncode:
    """数据编码处理类 - 内存优化版本"""
    
    def __init__(self, logger=None):
        self.logger = logger
    
    def optimize_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """兼容性方法"""
        return optimize_data_types(df)
    
    def check_memory_requirements(self, input_file: str, chunk_size: int, n_processes: int) -> bool:
        """检查内存需求"""
        try:
            # 获取文件大小
            file_size_mb = os.path.getsize(input_file) / 1024 / 1024
            
            # 估算内存需求 (文件大小 * 进程数 * 安全系数)
            estimated_memory_gb = (file_size_mb * n_processes * 3) / 1024
            
            # 获取系统内存信息
            mem_info = get_system_memory_info()
            
            if self.logger:
                self.logger.info(f"文件大小: {file_size_mb:.1f} MB")
                self.logger.info(f"预计内存需求: {estimated_memory_gb:.1f} GB")
                self.logger.info(f"系统可用内存: {mem_info['available']:.1f} GB")
            
            if estimated_memory_gb > mem_info['available'] * 0.8:
                if self.logger:
                    self.logger.warning("内存可能不足，建议减少进程数或增加chunk_size")
                return False
                
            return True
        except Exception as e:
            if self.logger:
                self.logger.warning(f"内存检查失败: {e}")
            return True  # 如果检查失败，继续执行

    def process_file_sequential(self, input_file: str, output_file: str, 
                               chunk_size: int = 200000) -> Optional[str]:
        """单进程顺序处理"""
        if self.logger:
            self.logger.info(f"开始单进程编码 {input_file}")
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        try:
            pf = pq.ParquetFile(input_file)
            chunks = []
            
            for chunk_id, batch in enumerate(pf.iter_batches(batch_size=chunk_size)):
                df_chunk = batch.to_pandas()
                optimized_df = optimize_data_types(df_chunk)
                chunks.append(optimized_df)
                
                if self.logger and chunk_id % 10 == 0:
                    mem_info = get_system_memory_info()
                    self.logger.info(f"处理chunk {chunk_id}, 内存使用: {mem_info['percent']:.1f}%")
                
                # 定期清理内存
                if len(chunks) % 10 == 0:
                    gc.collect()
                    
                # 如果内存使用过高，提前合并
                if psutil.virtual_memory().percent > 85:
                    if self.logger:
                        self.logger.warning("内存使用过高，提前合并数据")
                    temp_df = pd.concat(chunks, ignore_index=True)
                    chunks = [temp_df]
                    gc.collect()
            
            final_df = pd.concat(chunks, ignore_index=True)
            final_df.to_parquet(output_file, index=False)
            
            if self.logger:
                self.logger.info(f"单进程编码完成: {len(final_df)} 行")
            
            return output_file
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"单进程编码失败: {str(e)}")
            return None