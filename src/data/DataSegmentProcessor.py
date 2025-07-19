import pandas as pd
import numpy as np
import os
from typing import Dict, Set, Optional, List, Tuple
import logging
import pyarrow.parquet as pq
from datetime import datetime
from multiprocessing import Pool, Manager, cpu_count
from functools import partial
import pyarrow as pa
import gc
from concurrent.futures import ProcessPoolExecutor, as_completed

# 设置工作路径
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
MAIN_PATH = os.path.abspath(os.path.join(CURRENT_PATH, "..", ".."))
os.chdir(MAIN_PATH)

class FastDataSegmentProcessor:
    """高速数据分组处理类，支持多进程并行处理"""
    
    def __init__(self, chunk_size: int = 200000, output_dir: str = "data", n_processes: Optional[int] = None):
        self.chunk_size = chunk_size
        self.output_dir = output_dir
        self.n_processes = n_processes or min(cpu_count(), 8)  # 限制最大进程数
        
        # 使用os.path.join确保路径正确性
        self.train_output_dir = os.path.join(output_dir, "train")
        self.test_output_dir = os.path.join(output_dir, "test")
        
        os.makedirs(self.train_output_dir, exist_ok=True)
        os.makedirs(self.test_output_dir, exist_ok=True)
        
        # 配置logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"初始化处理器: {self.n_processes} 个进程, chunk_size={chunk_size}")
    
    def _setup_logging(self):
        """设置美观的logging配置"""
        # 清除已有的handlers
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        
        # 创建formatter
        formatter = logging.Formatter(
            fmt='%(asctime)s | %(levelname)8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 创建console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)
        
        # 创建file handler
        log_file = os.path.join(self.output_dir, f"fast_data_processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        
        # 配置root logger
        logging.basicConfig(
            level=logging.INFO,
            handlers=[console_handler, file_handler],
            force=True
        )
    
    def _get_segment_pattern(self, segment_level: int) -> str:
        """获取段级别对应的正则表达式模式"""
        return f'segments{segment_level}'
    
    def _detect_segment_ranker_ids(self, df: pd.DataFrame, segment_level: int) -> Set[str]:
        """检测包含指定段级别数据的ranker_id"""
        pattern = self._get_segment_pattern(segment_level)
        segment_mask = ~df.filter(regex=pattern).isnull().all(axis=1)
        return set(df.loc[segment_mask, 'ranker_id'].unique())
    
    def _get_parquet_row_groups_info(self, file_path: str) -> List[Tuple[int, int]]:
        """获取parquet文件的行组信息"""
        parquet_file = pq.ParquetFile(file_path)
        row_groups_info = []
        
        for i in range(parquet_file.num_row_groups):
            row_group = parquet_file.metadata.row_group(i)
            row_groups_info.append((i, row_group.num_rows))
        
        return row_groups_info
    
    def _read_row_group_batch(self, file_path: str, row_group_indices: List[int]) -> pd.DataFrame:
        """读取指定的行组"""
        parquet_file = pq.ParquetFile(file_path)
        
        tables = []
        for idx in row_group_indices:
            table = parquet_file.read_row_group(idx)
            tables.append(table)
        
        if tables:
            combined_table = pa.concat_tables(tables)
            return combined_table.to_pandas()
        else:
            return pd.DataFrame()
    
    def _process_batch_for_segment(self, batch_df: pd.DataFrame, segment_level: int, 
                                   data_type: str, batch_id: int) -> Dict:
        """处理单个批次的特定segment级别数据"""
        try:
            # 检测包含当前segment级别的ranker_id
            segment_ranker_ids = self._detect_segment_ranker_ids(batch_df, segment_level)
            
            if not segment_ranker_ids:
                return {
                    'batch_id': batch_id,
                    'segment_level': segment_level,
                    'processed_ranker_ids': set(),
                    'rows_processed': 0
                }
            
            # 提取对应的数据
            segment_data = batch_df[batch_df['ranker_id'].isin(segment_ranker_ids)].copy()
            
            # 保存数据到临时文件
            temp_filename = f"temp_{data_type}_segment_{segment_level}_batch_{batch_id}.parquet"
            temp_filepath = os.path.join(
                self.train_output_dir if data_type == 'train' else self.test_output_dir,
                temp_filename
            )
            
            segment_data.to_parquet(temp_filepath, index=False)
            
            # 清理内存
            del segment_data
            gc.collect()
            
            return {
                'batch_id': batch_id,
                'segment_level': segment_level,
                'processed_ranker_ids': segment_ranker_ids,
                'rows_processed': len(segment_ranker_ids),
                'temp_file': temp_filepath
            }
            
        except Exception as e:
            return {
                'batch_id': batch_id,
                'segment_level': segment_level,
                'error': str(e),
                'processed_ranker_ids': set(),
                'rows_processed': 0
            }
    
    def _merge_temp_files(self, data_type: str, segment_level: int, temp_files: List[str]):
        """合并临时文件"""
        if not temp_files:
            return
        
        output_dir = self.train_output_dir if data_type == 'train' else self.test_output_dir
        final_filename = f"{data_type}_segment_{segment_level}.parquet"
        final_filepath = os.path.join(output_dir, final_filename)
        
        # 读取所有临时文件并合并
        dfs = []
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                df = pd.read_parquet(temp_file)
                dfs.append(df)
                # 删除临时文件
                os.remove(temp_file)
        
        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
            
            # 如果最终文件已存在，合并
            if os.path.exists(final_filepath):
                existing_df = pd.read_parquet(final_filepath)
                combined_df = pd.concat([existing_df, combined_df], ignore_index=True)
            
            combined_df.to_parquet(final_filepath, index=False)
            self.logger.info(f"合并完成: {final_filepath}, 共 {len(combined_df)} 行")
        
        # 清理内存
        del dfs
        gc.collect()
    
    def _create_batches_from_row_groups(self, file_path: str, target_batch_size: int) -> List[List[int]]:
        """根据行组创建批次"""
        row_groups_info = self._get_parquet_row_groups_info(file_path)
        
        batches = []
        current_batch = []
        current_size = 0
        
        for row_group_idx, row_count in row_groups_info:
            # 如果当前批次加上这个行组会超过目标大小，并且当前批次不为空
            if current_size + row_count > target_batch_size and current_batch:
                batches.append(current_batch)
                current_batch = [row_group_idx]
                current_size = row_count
            else:
                current_batch.append(row_group_idx)
                current_size += row_count
        
        # 添加最后一个批次
        if current_batch:
            batches.append(current_batch)
        
        return batches
    
    def process_single_file(self, input_file: str, data_type: str):
        """使用多进程处理单个parquet数据文件"""
        if not input_file.endswith('.parquet'):
            raise ValueError(f"仅支持parquet格式文件，当前文件: {input_file}")
        
        self.logger.info(f"开始多进程处理 {input_file}")
        
        # 创建批次
        batches = self._create_batches_from_row_groups(input_file, self.chunk_size)
        self.logger.info(f"创建了 {len(batches)} 个批次")
        
        # 存储所有处理过的ranker_id
        all_processed_ranker_ids = set()
        
        # 按segment级别从高到低处理（3 -> 0）
        for segment_level in [3, 2, 1, 0]:
            self.logger.info(f"处理 segment_{segment_level} 级别数据...")
            
            temp_files = []
            
            # 使用进程池处理批次
            with ProcessPoolExecutor(max_workers=self.n_processes) as executor:
                # 准备任务
                futures = []
                
                for batch_id, row_group_indices in enumerate(batches):
                    # 读取批次数据
                    batch_df = self._read_row_group_batch(input_file, row_group_indices)
                    
                    # 过滤掉已处理的ranker_id
                    if all_processed_ranker_ids:
                        batch_df = batch_df[~batch_df['ranker_id'].isin(all_processed_ranker_ids)]
                    
                    if batch_df.empty:
                        continue
                    
                    # 提交任务
                    future = executor.submit(
                        self._process_batch_for_segment,
                        batch_df, segment_level, data_type, batch_id
                    )
                    futures.append(future)
                
                # 收集结果
                for future in as_completed(futures):
                    result = future.result()
                    
                    if 'error' in result:
                        self.logger.error(f"批次 {result['batch_id']} 处理失败: {result['error']}")
                        continue
                    
                    # 更新已处理的ranker_id
                    all_processed_ranker_ids.update(result['processed_ranker_ids'])
                    
                    # 收集临时文件
                    if 'temp_file' in result:
                        temp_files.append(result['temp_file'])
                    
                    self.logger.info(f"segment_{segment_level} 批次 {result['batch_id']}: "
                                   f"处理了 {result['rows_processed']} 个ranker_id")
            
            # 合并临时文件
            self._merge_temp_files(data_type, segment_level, temp_files)
            
            self.logger.info(f"segment_{segment_level} 处理完成, "
                           f"累计处理 {len(all_processed_ranker_ids)} 个ranker_id")
        
        self.logger.info(f"完成处理 {input_file}, 总计处理 {len(all_processed_ranker_ids)} 个ranker_id")
    
    def process_files(self, train_file: Optional[str] = None, test_file: Optional[str] = None):
        """处理训练和测试文件"""
        if train_file:
            self.logger.info("=" * 60)
            self.logger.info("开始处理训练数据")
            self.logger.info("=" * 60)
            self.process_single_file(train_file, 'train')
            
        if test_file:
            self.logger.info("=" * 60)
            self.logger.info("开始处理测试数据")
            self.logger.info("=" * 60)
            self.process_single_file(test_file, 'test')
    
    def get_statistics(self, data_type: str) -> Dict[int, Dict[str, int]]:
        """获取分组统计信息"""
        stats = {}
        
        for segment_count in [0, 1, 2, 3]:
            output_dir = self.train_output_dir if data_type == 'train' else self.test_output_dir
            filename = f"{data_type}_segment_{segment_count}.parquet"
            filepath = os.path.join(output_dir, filename)
            
            if os.path.exists(filepath):
                df = pd.read_parquet(filepath)
                stats[segment_count] = {
                    'total_rows': len(df),
                    'unique_ranker_ids': df['ranker_id'].nunique()
                }
            else:
                stats[segment_count] = {'total_rows': 0, 'unique_ranker_ids': 0}
        
        return stats
    
    def print_statistics(self):
        """打印统计信息"""
        self.logger.info("=" * 60)
        self.logger.info("数据处理统计结果")
        self.logger.info("=" * 60)
        
        for data_type in ['train', 'test']:
            stats = self.get_statistics(data_type)
            self.logger.info(f"\n{data_type.upper()} 数据统计:")
            self.logger.info("-" * 50)
            
            total_rows = 0
            total_ranker_ids = 0
            
            for segment_count, info in stats.items():
                total_rows += info['total_rows']
                total_ranker_ids += info['unique_ranker_ids']
                
                self.logger.info(f"航段数 {segment_count}: {info['total_rows']:,} 行, "
                               f"{info['unique_ranker_ids']:,} 个搜索会话")
            
            self.logger.info(f"总计: {total_rows:,} 行, {total_ranker_ids:,} 个搜索会话")


def process_batch_worker(args):
    """工作进程函数"""
    file_path, row_group_indices, segment_level, data_type, batch_id, chunk_size, output_dir = args
    
    try:
        # 读取数据
        parquet_file = pq.ParquetFile(file_path)
        tables = []
        for idx in row_group_indices:
            table = parquet_file.read_row_group(idx)
            tables.append(table)
        
        if tables:
            combined_table = pa.concat_tables(tables)
            batch_df = combined_table.to_pandas()
        else:
            return {
                'batch_id': batch_id,
                'segment_level': segment_level,
                'processed_ranker_ids': set(),
                'rows_processed': 0
            }
        
        # 处理segment数据
        pattern = f'segments{segment_level}'
        segment_mask = ~batch_df.filter(regex=pattern).isnull().all(axis=1)
        segment_ranker_ids = set(batch_df.loc[segment_mask, 'ranker_id'].unique())
        
        if not segment_ranker_ids:
            return {
                'batch_id': batch_id,
                'segment_level': segment_level,
                'processed_ranker_ids': set(),
                'rows_processed': 0
            }
        
        # 提取数据
        segment_data = batch_df[batch_df['ranker_id'].isin(segment_ranker_ids)].copy()
        
        # 保存到临时文件
        train_output_dir = os.path.join(output_dir, "train")
        test_output_dir = os.path.join(output_dir, "test")
        
        temp_filename = f"temp_{data_type}_segment_{segment_level}_batch_{batch_id}.parquet"
        temp_filepath = os.path.join(
            train_output_dir if data_type == 'train' else test_output_dir,
            temp_filename
        )
        
        segment_data.to_parquet(temp_filepath, index=False)
        
        return {
            'batch_id': batch_id,
            'segment_level': segment_level,
            'processed_ranker_ids': segment_ranker_ids,
            'rows_processed': len(segment_ranker_ids),
            'temp_file': temp_filepath
        }
        
    except Exception as e:
        return {
            'batch_id': batch_id,
            'segment_level': segment_level,
            'error': str(e),
            'processed_ranker_ids': set(),
            'rows_processed': 0
        }


def main():
    """主函数"""
    target_output_dir = os.path.join(MAIN_PATH, "data")
    
    # 目标文件列表（parquet格式）
    required_files = [
        "train/train_segment_0.parquet", "train/train_segment_1.parquet", 
        "train/train_segment_2.parquet", "train/train_segment_3.parquet",
        "test/test_segment_0.parquet", "test/test_segment_1.parquet",
        "test/test_segment_2.parquet", "test/test_segment_3.parquet"
    ]
    
    missing_files = [f for f in required_files 
                    if not os.path.exists(os.path.join(target_output_dir, f))]
    
    # 使用更多进程和更大的chunk_size来提高性能
    processor = FastDataSegmentProcessor(
        chunk_size=500000,  # 增大chunk_size
        output_dir=target_output_dir,
        n_processes=min(cpu_count(), 12)  # 使用更多进程
    )
    
    if not missing_files:
        processor.logger.info("所有目标文件已存在，跳过处理。")
        processor.logger.info(f"文件位置: {target_output_dir}")
        processor.print_statistics()
    else:
        processor.logger.info(f"检测到 {len(missing_files)} 个文件不存在，开始处理数据...")
        processor.logger.info(f"缺失文件: {missing_files}")
        
        # 检查输入文件是否存在
        train_file = os.path.join(MAIN_PATH, "data", "train.parquet")
        test_file = os.path.join(MAIN_PATH, "data", "test.parquet")
        
        if not os.path.exists(train_file):
            processor.logger.error(f"训练文件不存在: {train_file}")
            return
        
        if not os.path.exists(test_file):
            processor.logger.error(f"测试文件不存在: {test_file}")
            return
        
        start_time = datetime.now()
        processor.process_files(train_file=train_file, test_file=test_file)
        end_time = datetime.now()
        
        processor.logger.info(f"总处理时间: {end_time - start_time}")
        processor.logger.info(f"总处理时间: {end_time - start_time}")
        processor.print_statistics()


if __name__ == "__main__":
    main()