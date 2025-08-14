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
    
    def _get_group_size_stats(self, df: pd.DataFrame) -> Dict:
        """计算组大小统计信息"""
        group_sizes = df.groupby('ranker_id').size()
        
        stats = {
            'min_size': group_sizes.min(),
            'max_size': group_sizes.max(),
            'q25': group_sizes.quantile(0.25),
            'q50': group_sizes.quantile(0.50),
            'q75': group_sizes.quantile(0.75),
            'mean': group_sizes.mean()
        }
        
        if self.logger:
            self.logger.info(f"组大小统计: min={stats['min_size']}, q25={stats['q25']:.1f}, "
                           f"median={stats['q50']:.1f}, q75={stats['q75']:.1f}, max={stats['max_size']}")
        
        return stats
    
    def _classify_ranker_ids_by_group_size(self, df: pd.DataFrame, size_stats: Dict) -> Dict[str, Set[str]]:
        """
        根据组大小将ranker_id分类为small, medium, big
        """
        group_sizes = df.groupby('ranker_id').size()
        
        # 定义分组阈值
        small_threshold = size_stats['q25']
        big_threshold = size_stats['q75']
        
        classification = {'small': set(), 'medium': set(), 'big': set()}
        
        for ranker_id, group_size in group_sizes.items():
            if group_size <= small_threshold:
                classification['small'].add(ranker_id)
            elif group_size <= big_threshold:
                classification['medium'].add(ranker_id)
            else:
                classification['big'].add(ranker_id)
        
        if self.logger:
            self.logger.info(f"按组大小分类结果:")
            self.logger.info(f"  Small (≤{small_threshold:.1f}): {len(classification['small'])} ranker_ids")
            self.logger.info(f"  Medium ({small_threshold:.1f}-{big_threshold:.1f}]: {len(classification['medium'])} ranker_ids")
            self.logger.info(f"  Big (>{big_threshold:.1f}): {len(classification['big'])} ranker_ids")
        
        return classification
    
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
    
    def _process_batch_for_segment_and_group(self, batch_df: pd.DataFrame, 
                                           segment_ranker_ids: Set[str], 
                                           group_ranker_ids: Dict[str, Set[str]],
                                           segment_level: int, data_type: str, 
                                           batch_id: int, output_dir: str) -> Dict:
        """处理单个批次的特定segment级别和组大小数据"""
        try:
            # 获取当前批次中属于该segment的ranker_id
            batch_ranker_ids = set(batch_df['ranker_id'].unique())
            target_ranker_ids = batch_ranker_ids.intersection(segment_ranker_ids)
            
            if not target_ranker_ids:
                return {
                    'batch_id': batch_id,
                    'segment_level': segment_level,
                    'processed_rows': 0,
                    'processed_ranker_ids': 0,
                    'group_results': {'small': 0, 'medium': 0, 'big': 0}
                }
            
            # 提取对应的数据
            segment_data = batch_df[batch_df['ranker_id'].isin(target_ranker_ids)].copy()
            
            # 按组大小进一步分类
            group_results = {}
            temp_files = {}
            
            for group_category in ['small', 'medium', 'big']:
                group_target_rankers = target_ranker_ids.intersection(group_ranker_ids[group_category])
                
                if group_target_rankers:
                    group_data = segment_data[segment_data['ranker_id'].isin(group_target_rankers)].copy()
                    
                    if not group_data.empty:
                        # 保存数据到临时文件
                        temp_filename = f"temp_{data_type}_segment_{segment_level}_{group_category}_batch_{batch_id}.parquet"
                        temp_filepath = os.path.join(output_dir, temp_filename)
                        
                        group_data.to_parquet(temp_filepath, index=False)
                        temp_files[group_category] = temp_filepath
                        group_results[group_category] = len(group_data)
                    else:
                        group_results[group_category] = 0
                else:
                    group_results[group_category] = 0
            
            total_rows = sum(group_results.values())
            
            # 清理内存
            del segment_data
            gc.collect()
            
            return {
                'batch_id': batch_id,
                'segment_level': segment_level,
                'processed_rows': total_rows,
                'processed_ranker_ids': len(target_ranker_ids),
                'group_results': group_results,
                'temp_files': temp_files
            }
            
        except Exception as e:
            return {
                'batch_id': batch_id,
                'segment_level': segment_level,
                'error': str(e),
                'processed_rows': 0,
                'processed_ranker_ids': 0,
                'group_results': {'small': 0, 'medium': 0, 'big': 0}
            }
    
    def _merge_temp_files_by_group(self, data_type: str, segment_level: int, 
                                  temp_files_by_group: Dict[str, List[str]], 
                                  output_dir: str) -> Dict[str, int]:
        """按组别合并临时文件，返回每个组的行数"""
        group_row_counts = {}
        
        for group_category in ['small', 'medium', 'big']:
            temp_files = temp_files_by_group.get(group_category, [])
            final_filename = f"{data_type}_segment_{segment_level}_{group_category}.parquet"
            final_filepath = os.path.join(output_dir, final_filename)
            
            # 确保输出目录存在
            os.makedirs(output_dir, exist_ok=True)
            
            # 如果没有临时文件，创建空文件
            if not temp_files:
                empty_df = pd.DataFrame()
                empty_df.to_parquet(final_filepath, index=False)
                group_row_counts[group_category] = 0
                if self.logger:
                    self.logger.debug(f"创建空文件: {final_filepath}")
                continue
            
            # 读取所有临时文件并合并
            dfs = []
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    df = pd.read_parquet(temp_file)
                    dfs.append(df)
                    # 删除临时文件
                    os.remove(temp_file)
            
            if dfs:
                # 合并所有DataFrame
                combined_df = pd.concat(dfs, ignore_index=True)
                
                # 剔除恒定值列（所有值相同或全为NaN的列）
                constant_columns = []
                for col in combined_df.columns:
                    if combined_df[col].nunique(dropna=False) <= 1:
                        constant_columns.append(col)
                
                if constant_columns and self.logger:
                    self.logger.debug(f"剔除 {len(constant_columns)} 个恒定值列")
                    combined_df = combined_df.drop(columns=constant_columns)
                
                # 最终处理
                final_df = combined_df.sort_values(by='Id')
                final_df.to_parquet(final_filepath, index=False)
                group_row_counts[group_category] = len(final_df)
                
                if self.logger:
                    self.logger.debug(f"分割合并完成: {final_filepath}, 共 {len(final_df)} 行")
                
                # 清理内存
                del combined_df, final_df
            else:
                # 如果dfs为空，创建空文件
                empty_df = pd.DataFrame()
                empty_df.to_parquet(final_filepath, index=False)
                group_row_counts[group_category] = 0
                if self.logger:
                    self.logger.debug(f"创建空文件（无数据）: {final_filepath}")
            
            # 清理内存
            del dfs
            gc.collect()
        
        return group_row_counts
    
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
    
    def _get_ranker_classifications(self, input_file: str) -> Tuple[Dict[int, Set[str]], Dict[str, Set[str]]]:
        """
        预先扫描文件，对所有ranker_id进行segment分类和组大小分类
        返回：(segment_classification, group_size_classification)
        """
        if self.logger:
            self.logger.info("开始预扫描文件进行ranker_id分类...")
        
        parquet_file = pq.ParquetFile(input_file)
        all_ranker_segment_map = {}
        all_data_for_group_size = []
        
        # 分批读取并分类
        for batch_idx, batch in enumerate(parquet_file.iter_batches(batch_size=self.chunk_size)):
            batch_df = batch.to_pandas()
            
            # Segment分类
            batch_classification = self._classify_ranker_ids_by_segments(batch_df)
            for segment_level, ranker_ids in batch_classification.items():
                for ranker_id in ranker_ids:
                    if ranker_id not in all_ranker_segment_map:
                        all_ranker_segment_map[ranker_id] = segment_level
                    else:
                        # 如果已存在，取更高的segment级别
                        all_ranker_segment_map[ranker_id] = max(
                            all_ranker_segment_map[ranker_id], segment_level
                        )
            
            # 收集数据用于组大小分类
            all_data_for_group_size.append(batch_df[['ranker_id']].copy())
            
            if self.logger and (batch_idx + 1) % 10 == 0:
                self.logger.info(f"已处理 {batch_idx + 1} 批次，发现 {len(all_ranker_segment_map)} 个ranker_id")
        
        # 按segment级别重新分组
        segment_classification = {0: set(), 1: set(), 2: set(), 3: set()}
        for ranker_id, segment_level in all_ranker_segment_map.items():
            segment_classification[segment_level].add(ranker_id)
        
        # 组大小分类
        all_ranker_data = pd.concat(all_data_for_group_size, ignore_index=True)
        size_stats = self._get_group_size_stats(all_ranker_data)
        group_size_classification = self._classify_ranker_ids_by_group_size(all_ranker_data, size_stats)
        
        if self.logger:
            total_rankers = sum(len(ids) for ids in segment_classification.values())
            self.logger.info(f"分类完成，总计 {total_rankers} 个ranker_id:")
            for level in [3, 2, 1, 0]:
                count = len(segment_classification[level])
                if count > 0:
                    self.logger.info(f"  Segment {level}: {count} 个ranker_id")
        
        # 清理内存
        del all_data_for_group_size, all_ranker_data
        gc.collect()
        
        return segment_classification, group_size_classification
    
    def process_file(self, input_file: str, data_type: str, output_dir: str, pre_classification=True) -> Dict[int, Dict[str, int]]:
        """
        处理单个文件进行分割（包含组大小分类）
        返回：{segment_level: {group_category: row_count}}
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
                segment_classification, group_size_classification = self._get_ranker_classifications(input_file)
            else:
                raise ValueError("必须启用pre_classification以支持组大小分类")
            
            # 步骤2：创建批次
            batches = self._create_batches_from_row_groups(input_file, self.chunk_size)
            if self.logger:
                self.logger.info(f"创建了 {len(batches)} 个批次")
            
            # 步骤3：按segment级别处理数据（同时进行组大小分类）
            segment_results = {}
            
            for segment_level in [3, 2, 1, 0]:
                segment_ranker_ids = segment_classification[segment_level]
                
                if not segment_ranker_ids:
                    # 创建空文件
                    group_results = {}
                    for group_category in ['small', 'medium', 'big']:
                        final_filename = f"{data_type}_segment_{segment_level}_{group_category}.parquet"
                        final_filepath = os.path.join(output_dir, final_filename)
                        empty_df = pd.DataFrame()
                        empty_df.to_parquet(final_filepath, index=False)
                        group_results[group_category] = 0
                    
                    segment_results[segment_level] = group_results
                    if self.logger:
                        self.logger.info(f"Segment {segment_level}: 无数据，创建空文件")
                    continue
                
                if self.logger:
                    self.logger.info(f"处理 segment_{segment_level} 级别数据，包含 {len(segment_ranker_ids)} 个ranker_id...")
                
                # 收集所有批次的临时文件
                temp_files_by_group = {'small': [], 'medium': [], 'big': []}
                
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
                            self._process_batch_for_segment_and_group,
                            batch_df, segment_ranker_ids, group_size_classification,
                            segment_level, data_type, batch_id, output_dir
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
                        if 'temp_files' in result:
                            for group_category, temp_file in result['temp_files'].items():
                                temp_files_by_group[group_category].append(temp_file)
                
                # 合并临时文件
                group_row_counts = self._merge_temp_files_by_group(
                    data_type, segment_level, temp_files_by_group, output_dir
                )
                segment_results[segment_level] = group_row_counts
                
                total_segment_rows = sum(group_row_counts.values())
                if self.logger:
                    self.logger.info(f"segment_{segment_level} 处理完成: {total_segment_rows} 行")
                    for group_cat, rows in group_row_counts.items():
                        if rows > 0:
                            self.logger.info(f"  {group_cat}: {rows} 行")
            
            # 步骤4：验证结果
            total_output_rows = sum(sum(groups.values()) for groups in segment_results.values())
            if self.logger:
                self.logger.info(f"完成分割 {input_file}")
                self.logger.info(f"分割结果汇总:")
                for level in [3, 2, 1, 0]:
                    level_total = sum(segment_results[level].values())
                    if level_total > 0:
                        self.logger.info(f"  Segment {level}: {level_total:,} 行")
                        for group_cat, rows in segment_results[level].items():
                            if rows > 0:
                                self.logger.info(f"    {group_cat}: {rows:,} 行")
                self.logger.info(f"总计输出: {total_output_rows:,} 行")
            
            return segment_results
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"分割处理异常: {str(e)}")
                import traceback
                self.logger.error(f"详细错误:\n{traceback.format_exc()}")
            raise
    
    def get_output_files(self, data_type: str, output_dir: str) -> List[str]:
        """获取按组大小划分的输出文件列表"""
        files = []
        
        for segment_level in [0, 1, 2, 3]:
            for group_category in ['small', 'medium', 'big']:
                filename = f"{data_type}_segment_{segment_level}_{group_category}.parquet"
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
            level_stats = {'total_rows': 0, 'total_rankers': 0, 'groups': {}}
            
            for group_category in ['small', 'medium', 'big']:
                segment_file = os.path.join(output_dir, f"{data_type}_segment_{level}_{group_category}.parquet")
                if os.path.exists(segment_file):
                    segment_df = pd.read_parquet(segment_file)
                    rows = len(segment_df)
                    rankers = set(segment_df['ranker_id'].unique()) if rows > 0 else set()
                    
                    level_stats['groups'][group_category] = {'rows': rows, 'rankers': len(rankers)}
                    level_stats['total_rows'] += rows
                    level_stats['total_rankers'] += len(rankers)
                    
                    total_segmented_rows += rows
                    segmented_rankers.update(rankers)
                else:
                    level_stats['groups'][group_category] = {'rows': 0, 'rankers': 0}
            
            segment_stats[level] = level_stats
        
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