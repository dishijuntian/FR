import pandas as pd
import numpy as np
import os
from typing import Dict, Set, Optional, List, Tuple
import pyarrow.parquet as pq
from multiprocessing import cpu_count
import pyarrow as pa
import gc
from concurrent.futures import ProcessPoolExecutor, as_completed

class DataSegment:
    """数据分组处理类，支持多进程并行处理"""
    
    def __init__(self, chunk_size: int = 200000, n_processes: Optional[int] = None, logger=None):
        self.chunk_size = chunk_size
        self.n_processes = n_processes or min(cpu_count(), 8)
        self.logger = logger
        
        if self.logger:
            self.logger.info(f"初始化DataSegment: {self.n_processes} 个进程, chunk_size={chunk_size}")
    
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
                                   data_type: str, batch_id: int, output_dir: str) -> Dict:
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
            temp_filepath = os.path.join(output_dir, temp_filename)
            
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
    
    def _merge_temp_files(self, data_type: str, segment_level: int, temp_files: List[str], output_dir: str):
        """合并临时文件"""
        if not temp_files:
            return
        
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
            if self.logger:
                self.logger.info(f"分割合并完成: {final_filepath}, 共 {len(combined_df)} 行")
        
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
            if current_size + row_count > target_batch_size and current_batch:
                batches.append(current_batch)
                current_batch = [row_group_idx]
                current_size = row_count
            else:
                current_batch.append(row_group_idx)
                current_size += row_count
        
        if current_batch:
            batches.append(current_batch)
        
        return batches
    
    def process_file(self, input_file: str, data_type: str, output_dir: str):
        """处理单个文件进行分割"""
        if not input_file.endswith('.parquet'):
            raise ValueError(f"仅支持parquet格式文件，当前文件: {input_file}")
        
        if self.logger:
            self.logger.info(f"开始分割处理 {input_file}")
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建批次
        batches = self._create_batches_from_row_groups(input_file, self.chunk_size)
        if self.logger:
            self.logger.info(f"创建了 {len(batches)} 个批次")
        
        # 存储所有处理过的ranker_id
        all_processed_ranker_ids = set()
        
        # 按segment级别从高到低处理（3 -> 0）
        for segment_level in [3, 2, 1, 0]:
            if self.logger:
                self.logger.info(f"处理 segment_{segment_level} 级别数据...")
            
            temp_files = []
            
            # 使用进程池处理批次
            with ProcessPoolExecutor(max_workers=self.n_processes) as executor:
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
                        batch_df, segment_level, data_type, batch_id, output_dir
                    )
                    futures.append(future)
                
                # 收集结果
                for future in as_completed(futures):
                    result = future.result()
                    
                    if 'error' in result:
                        if self.logger:
                            self.logger.error(f"批次 {result['batch_id']} 处理失败: {result['error']}")
                        continue
                    
                    # 更新已处理的ranker_id
                    all_processed_ranker_ids.update(result['processed_ranker_ids'])
                    
                    # 收集临时文件
                    if 'temp_file' in result:
                        temp_files.append(result['temp_file'])
            
            # 合并临时文件
            self._merge_temp_files(data_type, segment_level, temp_files, output_dir)
            
            if self.logger:
                self.logger.info(f"segment_{segment_level} 处理完成, "
                               f"累计处理 {len(all_processed_ranker_ids)} 个ranker_id")
        
        if self.logger:
            self.logger.info(f"完成分割 {input_file}, 总计处理 {len(all_processed_ranker_ids)} 个ranker_id")
    
    def get_output_files(self, data_type: str, output_dir: str) -> List[str]:
        """获取输出文件列表"""
        files = []
        for segment_level in [0, 1, 2, 3]:
            filename = f"{data_type}_segment_{segment_level}.parquet"
            filepath = os.path.join(output_dir, filename)
            files.append(filepath)
        return files