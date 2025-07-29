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
    
    def _classify_ranker_ids_by_segments(self, df: pd.DataFrame) -> Dict[int, Set[str]]:
        """
        将ranker_id按照其包含的最高segment级别进行分类
        返回：{segment_level: set_of_ranker_ids}
        """
        ranker_segment_map = {}
        
        # 为每个ranker_id找到其最高的segment级别
        for ranker_id in df['ranker_id'].unique():
            ranker_data = df[df['ranker_id'] == ranker_id]
            max_segment = -1
            
            # 检查从高到低的segment级别
            for segment_level in [3, 2, 1, 0]:
                pattern = self._get_segment_pattern(segment_level)
                segment_cols = df.filter(regex=pattern).columns
                
                if len(segment_cols) > 0:
                    # 检查是否有有效数据（非-1且非null）
                    segment_data = ranker_data[segment_cols]
                    has_valid_data = ~((segment_data == -1) | segment_data.isnull()).all(axis=1).all()
                    
                    if has_valid_data:
                        max_segment = segment_level
                        break
            
            if max_segment >= 0:
                ranker_segment_map[ranker_id] = max_segment
        
        # 按segment级别分组
        segment_ranker_map = {0: set(), 1: set(), 2: set(), 3: set()}
        for ranker_id, segment_level in ranker_segment_map.items():
            segment_ranker_map[segment_level].add(ranker_id)
        
        return segment_ranker_map
    
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
    
    def _process_batch_for_segment(self, batch_df: pd.DataFrame, segment_ranker_ids: Set[str], 
                                   segment_level: int, data_type: str, batch_id: int, 
                                   output_dir: str) -> Dict:
        """处理单个批次的特定segment级别数据"""
        try:
            # 获取当前批次中属于该segment的ranker_id
            batch_ranker_ids = set(batch_df['ranker_id'].unique())
            target_ranker_ids = batch_ranker_ids.intersection(segment_ranker_ids)
            
            if not target_ranker_ids:
                return {
                    'batch_id': batch_id,
                    'segment_level': segment_level,
                    'processed_rows': 0,
                    'processed_ranker_ids': 0
                }
            
            # 提取对应的数据
            segment_data = batch_df[batch_df['ranker_id'].isin(target_ranker_ids)].copy()
            
            # 保存数据到临时文件
            temp_filename = f"temp_{data_type}_segment_{segment_level}_batch_{batch_id}.parquet"
            temp_filepath = os.path.join(output_dir, temp_filename)
            
            segment_data.to_parquet(temp_filepath, index=False)
            
            rows_count = len(segment_data)
            ranker_count = len(target_ranker_ids)
            
            # 清理内存
            del segment_data
            gc.collect()
            
            return {
                'batch_id': batch_id,
                'segment_level': segment_level,
                'processed_rows': rows_count,
                'processed_ranker_ids': ranker_count,
                'temp_file': temp_filepath
            }
            
        except Exception as e:
            return {
                'batch_id': batch_id,
                'segment_level': segment_level,
                'error': str(e),
                'processed_rows': 0,
                'processed_ranker_ids': 0
            }
    
    def _merge_temp_files(self, data_type: str, segment_level: int, temp_files: List[str], output_dir: str) -> int:
        """合并临时文件，返回总行数"""
        final_filename = f"{data_type}_segment_{segment_level}.parquet"
        final_filepath = os.path.join(output_dir, final_filename)
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 如果没有临时文件，创建空文件
        if not temp_files:
            empty_df = pd.DataFrame()
            empty_df.to_parquet(final_filepath, index=False)
            if self.logger:
                self.logger.info(f"创建空文件: {final_filepath}")
            return 0
        
        # 读取所有临时文件并合并
        dfs = []
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                df = pd.read_parquet(temp_file)
                dfs.append(df)
                # 删除临时文件
                os.remove(temp_file)
        
        total_rows = 0
        if dfs:
            # 1. 合并所有DataFrame
            combined_df = pd.concat(dfs, ignore_index=True)
            
            # 2. 剔除恒定值列（所有值相同或全为NaN的列）
            constant_columns = []
            for col in combined_df.columns:
                if combined_df[col].nunique(dropna=False) <= 1:  # 考虑全为NaN的情况
                    constant_columns.append(col)
            
            if constant_columns:
                self.logger.info(f"剔除 {len(constant_columns)} 个恒定值列: {constant_columns}")
                combined_df = combined_df.drop(columns=constant_columns)
            
            # 最终处理
            final_df = combined_df.sort_values(by='Id')
            final_df.to_parquet(final_filepath, index=False)
            total_rows = len(final_df)
            
            # 日志记录最终结果
            self.logger.info(f"处理后数据形状: {final_df.shape}")
            self.logger.info(f"分割合并完成: {final_filepath}, 共 {total_rows} 行")
            
        else:
            # 如果dfs为空，创建空文件
            empty_df = pd.DataFrame()
            empty_df.to_parquet(final_filepath, index=False)
            if self.logger:
                self.logger.info(f"创建空文件（无数据）: {final_filepath}")
        
        # 清理内存
        del dfs
        gc.collect()
        
        return total_rows
    
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
    
    def _get_ranker_segment_classification(self, input_file: str) -> Dict[int, Set[str]]:
        """
        预先扫描文件，对所有ranker_id进行segment分类
        这是一个关键优化：避免重复扫描和分类错误
        """
        if self.logger:
            self.logger.info("开始预扫描文件进行ranker_id分类...")
        
        parquet_file = pq.ParquetFile(input_file)
        all_ranker_segment_map = {}
        
        # 分批读取并分类
        for batch_idx, batch in enumerate(parquet_file.iter_batches(batch_size=self.chunk_size)):
            batch_df = batch.to_pandas()
            batch_classification = self._classify_ranker_ids_by_segments(batch_df)
            
            # 合并分类结果（取最高segment级别）
            for segment_level, ranker_ids in batch_classification.items():
                for ranker_id in ranker_ids:
                    if ranker_id not in all_ranker_segment_map:
                        all_ranker_segment_map[ranker_id] = segment_level
                    else:
                        # 如果已存在，取更高的segment级别
                        all_ranker_segment_map[ranker_id] = max(
                            all_ranker_segment_map[ranker_id], segment_level
                        )
            
            if self.logger and (batch_idx + 1) % 10 == 0:
                self.logger.info(f"已处理 {batch_idx + 1} 批次，发现 {len(all_ranker_segment_map)} 个ranker_id")
        
        # 按segment级别重新分组
        final_classification = {0: set(), 1: set(), 2: set(), 3: set()}
        for ranker_id, segment_level in all_ranker_segment_map.items():
            final_classification[segment_level].add(ranker_id)
        
        if self.logger:
            total_rankers = sum(len(ids) for ids in final_classification.values())
            self.logger.info(f"分类完成，总计 {total_rankers} 个ranker_id:")
            for level in [3, 2, 1, 0]:
                count = len(final_classification[level])
                if count > 0:
                    self.logger.info(f"  Segment {level}: {count} 个ranker_id")
        
        return final_classification
    
    def process_file(self, input_file: str, data_type: str, output_dir: str, pre_classification=False) -> Dict[int, int]:
        """
        处理单个文件进行分割
        返回：{segment_level: row_count}
        """
        if not input_file.endswith('.parquet'):
            raise ValueError(f"仅支持parquet格式文件，当前文件: {input_file}")
        
        if self.logger:
            self.logger.info(f"开始分割处理 {input_file}")
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # 步骤1：预先分类所有ranker_id
            if pre_classification:
                ranker_classification = self._get_ranker_segment_classification(input_file)
            
            # 步骤2：创建批次
            batches = self._create_batches_from_row_groups(input_file, self.chunk_size)
            if self.logger:
                self.logger.info(f"创建了 {len(batches)} 个批次")
            
            # 步骤3：按segment级别处理数据
            segment_results = {}
            
            for segment_level in [3, 2, 1, 0]:
                segment_ranker_ids = ranker_classification[segment_level]
                
                if not segment_ranker_ids:
                    # 创建空文件
                    final_filename = f"{data_type}_segment_{segment_level}.parquet"
                    final_filepath = os.path.join(output_dir, final_filename)
                    empty_df = pd.DataFrame()
                    empty_df.to_parquet(final_filepath, index=False)
                    segment_results[segment_level] = 0
                    if self.logger:
                        self.logger.info(f"Segment {segment_level}: 无数据，创建空文件")
                    continue
                
                if self.logger:
                    self.logger.info(f"处理 segment_{segment_level} 级别数据，包含 {len(segment_ranker_ids)} 个ranker_id...")
                
                temp_files = []
                
                # 使用进程池处理批次
                with ProcessPoolExecutor(max_workers=self.n_processes) as executor:
                    futures = []
                    
                    for batch_id, row_group_indices in enumerate(batches):
                        # 读取批次数据
                        batch_df = self._read_row_group_batch(input_file, row_group_indices)
                        
                        if batch_df.empty:
                            continue
                        
                        # 提交任务
                        future = executor.submit(
                            self._process_batch_for_segment,
                            batch_df, segment_ranker_ids, segment_level, 
                            data_type, batch_id, output_dir
                        )
                        futures.append(future)
                    
                    # 收集结果
                    total_processed_rows = 0
                    total_processed_rankers = 0
                    
                    for future in as_completed(futures):
                        result = future.result()
                        
                        if 'error' in result:
                            if self.logger:
                                self.logger.error(f"批次 {result['batch_id']} 处理失败: {result['error']}")
                            continue
                        
                        total_processed_rows += result['processed_rows']
                        total_processed_rankers += result['processed_ranker_ids']
                        
                        # 收集临时文件
                        if 'temp_file' in result and result['processed_rows'] > 0:
                            temp_files.append(result['temp_file'])
                
                # 合并临时文件
                segment_row_count = self._merge_temp_files(data_type, segment_level, temp_files, output_dir)
                segment_results[segment_level] = segment_row_count
                
                if self.logger:
                    self.logger.info(f"segment_{segment_level} 处理完成: {segment_row_count} 行")
            
            # 步骤4：验证结果
            total_output_rows = sum(segment_results.values())
            if self.logger:
                self.logger.info(f"完成分割 {input_file}")
                self.logger.info(f"分割结果汇总:")
                for level in [3, 2, 1, 0]:
                    if segment_results[level] > 0:
                        self.logger.info(f"  Segment {level}: {segment_results[level]:,} 行")
                self.logger.info(f"总计输出: {total_output_rows:,} 行")
            
            return segment_results
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"分割处理异常: {str(e)}")
                import traceback
                self.logger.error(f"详细错误:\n{traceback.format_exc()}")
            raise
    
    def get_output_files(self, data_type: str, output_dir: str) -> List[str]:
        """获取输出文件列表"""
        files = []
        for segment_level in [0, 1, 2, 3]:
            filename = f"{data_type}_segment_{segment_level}.parquet"
            filepath = os.path.join(output_dir, filename)
            files.append(filepath)
        return files
    
    def verify_segmentation(self, input_file: str, output_dir: str, data_type: str) -> Dict:
        """验证分割结果的完整性"""
        if self.logger:
            self.logger.info(f"开始验证 {data_type} 数据集的分割结果...")
        
        # 读取原始文件统计
        original_df = pd.read_parquet(input_file)
        original_rows = len(original_df)
        original_rankers = original_df['ranker_id'].nunique()
        
        # 读取分割文件统计
        total_segmented_rows = 0
        segmented_rankers = set()
        segment_stats = {}
        
        for level in [0, 1, 2, 3]:
            segment_file = os.path.join(output_dir, f"{data_type}_segment_{level}.parquet")
            if os.path.exists(segment_file):
                segment_df = pd.read_parquet(segment_file)
                rows = len(segment_df)
                rankers = set(segment_df['ranker_id'].unique()) if rows > 0 else set()
                
                total_segmented_rows += rows
                segmented_rankers.update(rankers)
                segment_stats[level] = {'rows': rows, 'rankers': len(rankers)}
        
        # 检查完整性
        row_match = original_rows == total_segmented_rows
        ranker_match = original_rankers == len(segmented_rankers)

        result = {
            'data_type': data_type,
            'original': {'rows': original_rows, 'rankers': original_rankers},
            'segmented': {'total_rows': total_segmented_rows, 'total_rankers': len(segmented_rankers)},
            'segments': segment_stats,
            'integrity': {'rows_match': row_match, 'rankers_match': ranker_match}
        }
        
        if self.logger:
            self.logger.info(f"验证结果:")
            self.logger.info(f"  原始数据: {original_rows:,} 行, {original_rankers:,} ranker_id")
            self.logger.info(f"  分割数据: {total_segmented_rows:,} 行, {len(segmented_rankers):,} ranker_id")
            self.logger.info(f"  行数匹配: {row_match}, ranker_id匹配: {ranker_match}")
            
            if not row_match:
                diff = original_rows - total_segmented_rows
                self.logger.warning(f"行数差异: {diff} 行")
            
            if not ranker_match:
                diff = original_rankers - len(segmented_rankers)
                self.logger.warning(f"ranker_id差异: {diff} 个")
        
        return result