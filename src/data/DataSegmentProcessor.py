import pandas as pd
import numpy as np
import os
from typing import Dict, Set, Optional
import logging
import pyarrow.parquet as pq
from datetime import datetime

# 设置工作路径
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
MAIN_PATH = os.path.abspath(os.path.join(CURRENT_PATH, "..", ".."))
os.chdir(MAIN_PATH)

class DataSegmentProcessor:
    """数据分组处理类，支持内存控制和批量处理，仅支持parquet格式"""
    
    def __init__(self, chunk_size: int = 100000, output_dir: str = "data"):
        self.chunk_size = chunk_size
        self.output_dir = output_dir
        self.processed_ranker_ids: Set[str] = set()
        
        # 使用os.path.join确保路径正确性
        self.train_output_dir = os.path.join(output_dir, "train")
        self.test_output_dir = os.path.join(output_dir, "test")
        
        os.makedirs(self.train_output_dir, exist_ok=True)
        os.makedirs(self.test_output_dir, exist_ok=True)
        
        # 配置更美观的logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
    
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
        log_file = os.path.join(self.output_dir, f"data_processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
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
    
    def _save_segment_data(self, df: pd.DataFrame, segment_count: int, data_type: str):
        """保存分组数据到parquet文件"""
        if df.empty:
            return
        
        # 确定输出目录
        output_dir = self.train_output_dir if data_type == 'train' else self.test_output_dir
        filename = f"{data_type}_segment_{segment_count}.parquet"
        filepath = os.path.join(output_dir, filename)
        
        if os.path.exists(filepath):
            # 读取现有数据并合并
            existing_df = pd.read_parquet(filepath)
            combined_df = pd.concat([existing_df, df], ignore_index=True)
            combined_df.to_parquet(filepath, index=False)
        else:
            df.to_parquet(filepath, index=False)
        
        self.logger.info(f"保存 {len(df)} 行数据到 {filepath}")
    
    def _read_parquet_in_batches(self, file_path: str):
        """使用pyarrow内存映射读取parquet文件并按ranker_id连续分组"""
        try:
            # 使用内存映射打开Parquet文件
            parquet_file = pq.ParquetFile(file_path)
            
            # 计算批次数
            row_group_count = parquet_file.num_row_groups
            self.logger.info(f"Parquet文件包含 {row_group_count} 个行组")
            
            # 存储上一个行组的尾部数据
            previous_tail = None
            
            # 逐行组处理
            for i in range(row_group_count):
                self.logger.info(f"处理行组 {i+1}/{row_group_count}")
                
                # 读取当前行组
                row_group = parquet_file.read_row_group(i)
                batch_df = row_group.to_pandas()
                
                # 如果有上一个行组的尾部数据，合并到当前批次
                if previous_tail is not None:
                    batch_df = pd.concat([previous_tail, batch_df], ignore_index=True)
                    previous_tail = None
                
                # 如果数据集为空，跳过处理
                if batch_df.empty:
                    continue
                
                # 确保按ranker_id排序（假设数据已按ranker_id排序）
                # 如果未排序，取消下面的注释（可能会增加内存消耗）
                # batch_df = batch_df.sort_values('ranker_id')
                
                # 找到每个ranker_id的起始行
                unique_ids = batch_df['ranker_id'].unique()
                self.logger.info(f"行组包含 {len(unique_ids)} 个唯一ranker_id")
                
                # 如果批次较小，直接返回
                if len(batch_df) <= self.chunk_size:
                    yield batch_df
                    continue
                
                # 按ranker_id分组，确保同一个id不被分割
                current_size = 0
                start_idx = 0
                
                for j, (id_start, id_end) in enumerate(zip(
                    batch_df.index[batch_df['ranker_id'] != batch_df['ranker_id'].shift(1)],
                    batch_df.index[batch_df['ranker_id'] != batch_df['ranker_id'].shift(-1)]
                )):
                    group_size = id_end - id_start + 1
                    current_size += group_size
                    
                    # 如果加入当前组后超过chunk_size，或者是最后一个组
                    if current_size > self.chunk_size or j == len(unique_ids) - 1:
                        # 如果是最后一个组，且当前批次不为空
                        if j == len(unique_ids) - 1 and start_idx <= id_end:
                            yield batch_df.loc[start_idx:id_end]
                            start_idx = id_end + 1
                            current_size = 0
                        # 如果不是最后一个组，但当前批次已超过chunk_size
                        elif start_idx <= id_end - group_size:
                            yield batch_df.loc[start_idx:id_end-group_size]
                            start_idx = id_end - group_size + 1
                            current_size = group_size
                
                # 保存剩余的数据作为下一个批次的头部
                if start_idx < len(batch_df):
                    previous_tail = batch_df.loc[start_idx:]
            
            # 处理最后一个行组的尾部数据
            if previous_tail is not None and not previous_tail.empty:
                yield previous_tail
                
        except Exception as e:
            self.logger.error(f"读取Parquet文件时出错: {e}")
            raise
    
    def _process_chunk_all_segments(self, chunk: pd.DataFrame, data_type: str):
        """在单个chunk中处理所有segment类型，避免重复读取"""
        # 过滤掉已处理的ranker_id
        remaining_chunk = chunk[~chunk['ranker_id'].isin(self.processed_ranker_ids)].copy()
        
        if remaining_chunk.empty:
            return
        
        self.logger.info(f"处理chunk: {len(remaining_chunk)} 行数据")
        
        # 从segment3到segment0依次处理，确保高等级segment不被低等级覆盖
        for segment_level in [3, 2, 1, 0]:
            if remaining_chunk.empty:
                break
                
            segment_count = segment_level
            self.logger.debug(f"处理 segment_{segment_count} 数据...")
            
            # 检测包含当前segment级别的ranker_id
            segment_ranker_ids = self._detect_segment_ranker_ids(remaining_chunk, segment_level)
            
            if segment_ranker_ids:
                # 提取对应的数据
                segment_data = remaining_chunk[remaining_chunk['ranker_id'].isin(segment_ranker_ids)]
                
                # 保存数据
                self._save_segment_data(segment_data, segment_count, data_type)
                
                # 更新已处理的ranker_id集合
                self.processed_ranker_ids.update(segment_ranker_ids)
                
                # 从remaining_chunk中删除已处理的行，避免重复处理
                remaining_chunk = remaining_chunk[~remaining_chunk['ranker_id'].isin(segment_ranker_ids)]
                
                self.logger.info(f"segment_{segment_count}: 处理了 {len(segment_ranker_ids)} 个ranker_id, "
                               f"剩余 {len(remaining_chunk)} 行待处理")
    
    def process_single_file(self, input_file: str, data_type: str):
        """处理单个parquet数据文件"""
        if not input_file.endswith('.parquet'):
            raise ValueError(f"仅支持parquet格式文件，当前文件: {input_file}")
        
        self.logger.info(f"开始处理 {input_file}")
        self.processed_ranker_ids.clear()
        
        # 使用parquet批量读取
        try:
            file_chunks = self._read_parquet_in_batches(input_file)
            
            chunk_count = 0
            for chunk in file_chunks:
                chunk_count += 1
                self.logger.info(f"处理第 {chunk_count} 个chunk")
                
                # 在单个chunk中处理所有segment类型
                self._process_chunk_all_segments(chunk, data_type)
                
                # 定期输出进度
                if chunk_count % 10 == 0:
                    self.logger.info(f"已处理 {chunk_count} 个chunk, "
                                   f"累计处理 {len(self.processed_ranker_ids)} 个ranker_id")
            
            self.logger.info(f"完成处理 {input_file}, 共处理 {chunk_count} 个chunk, "
                           f"总计 {len(self.processed_ranker_ids)} 个ranker_id")
            
        except Exception as e:
            self.logger.error(f"处理文件 {input_file} 时出错: {e}")
            raise
    
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
    
    processor = DataSegmentProcessor(chunk_size=200000, output_dir=target_output_dir)
    
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
        
        processor.process_files(train_file=train_file, test_file=test_file)
        processor.print_statistics()


if __name__ == "__main__":
    main()