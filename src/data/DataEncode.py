import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import os
from datetime import datetime


CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
MAIN_PATH = os.path.abspath(os.path.join(CURRENT_PATH, "..", ".."))
os.chdir(MAIN_PATH)


class DataEncode:
    def __init__(self):
        pass


    @staticmethod
    def parse_duration(duration_str):
        """更健壮的持续时间解析函数，处理多种格式"""
        if pd.isna(duration_str):
            return 0
        
        try:
            # 尝试处理"HH:MM:SS"格式
            if ':' in duration_str:
                parts = duration_str.split(':')
                if len(parts) == 3:  # HH:MM:SS
                    return int(parts[0])*3600 + int(parts[1])*60 + int(parts[2])
                elif len(parts) == 2:  # MM:SS
                    return int(parts[0])*60 + int(parts[1])
            
            # 尝试处理小数格式（可能是小时）
            try:
                hours = float(duration_str)
                return int(hours * 3600)
            except ValueError:
                pass
            
            # 其他无法识别的格式返回0
            return 0
        except:
            return 0

    def optimize_data_types(self, df):
        """
        根据数据类型优化表转换数据类型
        """
        # 删除所有名称中含有segments3的列（不区分大小写）
        seg3_cols = [col for col in df.columns if 'segments3' in col.lower()]
        df.drop(columns=seg3_cols, inplace=True, errors='ignore')
        
        # 数据类型转换
        # 1. 将double类型转换为int32/int8
        double_to_int_cols = [
            'legs1_segments2_baggageAllowance_weightMeasurementType',
            'legs1_segments2_baggageAllowance_quantity',
            'legs1_segments2_cabinClass',
            'legs1_segments2_seatsAvailable',
            'legs0_segments2_baggageAllowance_quantity',
            'legs0_segments2_baggageAllowance_weightMeasurementType',
            'legs0_segments2_cabinClass',
            'legs0_segments2_seatsAvailable',
            'miniRules1_percentage',
            'miniRules0_percentage',
            'legs1_segments1_seatsAvailable',
            'legs1_segments1_baggageAllowance_weightMeasurementType',
            'legs1_segments1_baggageAllowance_quantity',
            'legs1_segments1_cabinClass',
            'legs0_segments1_seatsAvailable',
            'legs0_segments1_baggageAllowance_weightMeasurementType',
            'legs0_segments1_baggageAllowance_quantity',
            'legs0_segments1_cabinClass',
            'corporateTariffCode',
            'legs1_segments0_seatsAvailable',
            'legs1_segments0_baggageAllowance_weightMeasurementType',
            'legs1_segments0_baggageAllowance_quantity',
            'legs1_segments0_cabinClass',
            'miniRules1_statusInfos',
            'miniRules0_statusInfos',
            'miniRules1_monetaryAmount',
            'miniRules0_monetaryAmount',
            'pricingInfo_isAccessTP',
            'legs0_segments0_seatsAvailable',
            'legs0_segments0_baggageAllowance_weightMeasurementType',
            'legs0_segments0_baggageAllowance_quantity',
            'legs0_segments0_cabinClass',
            'nationality',
            'Id',
            'pricingInfo_passengerCount'
        ]
        
        for col in double_to_int_cols:
            if col in df.columns:
                # 先填充NaN为0（根据业务逻辑决定，可能需要其他填充方式）
                df[col] = df[col].fillna(0)
                # 转换为int32以节省内存
                try:
                    df[col] = pd.to_numeric(df[col], downcast='integer')
                except:
                    # 如果转换失败，尝试先转换为float再转换为int
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('int32')
        
        # 2. 时间类型转换
        time_cols = [
            'legs1_departureAt',
            'legs1_arrivalAt',
            'legs0_departureAt',
            'legs0_arrivalAt',
            'requestDate'
        ]
        
        duration_cols = [
            'legs1_duration',
            'legs1_segments0_duration',
            'legs0_segments0_duration',
            'legs0_duration',
            'legs1_segments2_duration',
            'legs0_segments2_duration',
            'legs1_segments1_duration',
            'legs0_segments1_duration'
        ]
        
        for col in time_cols:
            if col in df.columns:
                try:
                    # 处理时间戳
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    df[col] = (df[col] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
                    df[col] = pd.to_numeric(df[col], downcast='integer').fillna(0)
                except:
                    df[col] = 0
        
        for col in duration_cols:
            if col in df.columns:
                # 使用更健壮的持续时间解析函数
                df[col] = df[col].apply(self.parse_duration)
                df[col] = pd.to_numeric(df[col], downcast='integer')
        
        # 3. 布尔类型转换
        bool_cols = [
            'isAccess3D',
            'isVip',
            'bySelf',
            'sex'
        ]
        
        for col in bool_cols:
            if col in df.columns:
                df[col] = df[col].astype('bool')
        
        # 4. 分类变量编码
        # 对于searchRoute，检查是否包含"/"表示往返票
        if 'searchRoute' in df.columns:
            df['is_round_trip'] = df['searchRoute'].str.contains('/').fillna(False).astype('int8')
            df.drop(columns=['searchRoute'], inplace=True, errors='ignore')
        
        # 5. 分箱编码
        if 'taxes' in df.columns:
            try:
                df['taxes_bin'] = pd.qcut(df['taxes'], q=10, labels=False, duplicates='drop').astype('int8')
                df.drop(columns=['taxes'], inplace=True, errors='ignore')
            except:
                df['taxes_bin'] = 0
        
        if 'totalPrice' in df.columns:
            try:
                df['totalPrice_bin'] = pd.qcut(df['totalPrice'], q=10, labels=False, duplicates='drop').astype('int8')
                df.drop(columns=['totalPrice'], inplace=True, errors='ignore')
            except:
                df['totalPrice_bin'] = 0
        
        # 6. 字符串分类变量
        str_cat_cols = [
            'frequentFlyer',
            'legs1_segments2_marketingCarrier_code',
            'legs1_segments2_operatingCarrier_code',
            'legs1_segments2_flightNumber',
            'legs1_segments2_arrivalTo_airport_iata',
            'legs1_segments2_departureFrom_airport_iata',
            'legs1_segments2_aircraft_code',
            'legs1_segments2_arrivalTo_airport_city_iata',
            'legs0_segments2_aircraft_code',
            'legs0_segments2_marketingCarrier_code',
            'legs0_segments2_flightNumber',
            'legs0_segments2_arrivalTo_airport_iata',
            'legs0_segments2_departureFrom_airport_iata',
            'legs0_segments2_arrivalTo_airport_city_iata',
            'legs0_segments2_operatingCarrier_code',
            'legs1_segments1_arrivalTo_airport_iata',
            'legs1_segments1_arrivalTo_airport_city_iata',
            'legs1_segments1_operatingCarrier_code',
            'legs1_segments1_departureFrom_airport_iata',
            'legs1_segments1_marketingCarrier_code',
            'legs1_segments1_flightNumber',
            'legs1_segments1_aircraft_code',
            'legs0_segments1_aircraft_code',
            'legs0_segments1_arrivalTo_airport_city_iata',
            'legs0_segments1_flightNumber',
            'legs0_segments1_arrivalTo_airport_iata',
            'legs0_segments1_operatingCarrier_code',
            'legs0_segments1_marketingCarrier_code',
            'legs0_segments1_departureFrom_airport_iata',
            'legs1_segments0_arrivalTo_airport_city_iata',
            'legs1_segments0_departureFrom_airport_iata',
            'legs1_segments0_flightNumber',
            'legs1_segments0_operatingCarrier_code',
            'legs1_segments0_aircraft_code',
            'legs1_segments0_marketingCarrier_code',
            'legs1_segments0_arrivalTo_airport_iata',
            'legs0_segments0_arrivalTo_airport_city_iata',
            'legs0_segments0_aircraft_code',
            'legs0_segments0_arrivalTo_airport_iata',
            'legs0_segments0_departureFrom_airport_iata',
            'legs0_segments0_marketingCarrier_code',
            'legs0_segments0_operatingCarrier_code',
            'legs0_segments0_flightNumber'
        ]
        
        for col in str_cat_cols:
            if col in df.columns:
                # 对于高基数的分类变量，使用哈希编码减少内存
                df[col] = df[col].astype('str')
                df[col] = df[col].apply(lambda x: hash(x) % 65535 if pd.notna(x) else 0).astype('int32')
        
        # 7. 处理ranker_id和profileId
        if 'ranker_id' in df.columns:
            df['ranker_id'] = df['ranker_id'].apply(lambda x: hash(str(x)) % 65535).astype('int32')
        
        if 'profileId' in df.columns:
            df['profileId'] = df['profileId'].astype('int32')
        
        if 'companyID' in df.columns:
            df['companyID'] = df['companyID'].astype('int32')
        
        # 8. 目标变量
        if 'selected' in df.columns:
            df['selected'] = df['selected'].astype('int8')
        
        return df

    def process_large_parquet(self,final_output_path, file_path, chunk_size=100000):
        """
        分块处理大型Parquet文件
        """
        # 首先读取schema
        pf = pq.ParquetFile(file_path)
        
        # 获取所有列名
        all_columns = pf.schema.names
        
        # 预先删除segments3的列
        columns_to_read = [col for col in all_columns if 'segments3' not in col.lower()]
        
        # 创建输出目录
        output_dir = os.path.join(os.path.dirname(file_path), 'processed')
        os.makedirs(output_dir, exist_ok=True)
        
        # 分块处理
        for i, batch in enumerate(pf.iter_batches(batch_size=chunk_size, columns=columns_to_read)):
            print(f"Processing chunk {i+1}")
            df = batch.to_pandas()
            
            # 优化数据类型
            df = self.optimize_data_types(df)
            
            # 保存处理后的数据
            output_path = os.path.join(output_dir, f'chunk_{i}.parquet')
            df.to_parquet(output_path, index=False)
            
            # 手动清理内存
            del df
            import gc
            gc.collect()
        
        print("All chunks processed. Now merging...")
        
        # 合并所有分块
        chunk_files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.startswith('chunk_')]
        df_list = []
        
        for chunk_file in sorted(chunk_files, key=lambda x: int(x.split('_')[-1].split('.')[0])):
            df_chunk = pd.read_parquet(chunk_file)
            df_list.append(df_chunk)
        
        final_df = pd.concat(df_list, ignore_index=True)
        
        # 保存最终文件
        final_df.to_parquet(final_output_path, index=False)
        
        # 清理临时文件
        for chunk_file in chunk_files:
            os.remove(chunk_file)
        os.rmdir(output_dir)


# 使用示例
if __name__ == "__main__":
    encoder = DataEncode()
    train_file_path = os.path.join(MAIN_PATH, "data/aeroclub-recsys-2025/train.parquet")
    test_file_path = os.path.join(MAIN_PATH, "data/aeroclub-recsys-2025/test.parquet")
    output_train_path = os.path.join(MAIN_PATH, "data/aeroclub-recsys-2025/data_encoded/train_encoded.parquet")
    output_test_path = os.path.join(MAIN_PATH, "data/aeroclub-recsys-2025/data_encoded/test_encoded.parquet")
    encoder.process_large_parquet(output_train_path, train_file_path)
    encoder.process_large_parquet(output_test_path, test_file_path)
