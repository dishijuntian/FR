import os
import logging
import pandas as pd
import pyarrow.parquet as pq
from datetime import datetime
from multiprocessing import cpu_count
from typing import Dict, List, Optional

from src.data.DataEncode import DataEncode
from src.data.DataSegment import DataSegment


class DataProcessor:
    """数据处理流水线：编码 + 分割"""
    
    def __init__(self, base_dir: str = "data/aeroclub-recsys-2025", 
                 chunk_size: int = 200000, n_processes: Optional[int] = None,
                 logger: Optional[logging.Logger] = None):  # 添加logger参数
        self.base_dir = base_dir
        self.chunk_size = chunk_size
        self.n_processes = n_processes or min(cpu_count(), 8)
        
        # 设置路径
        self.encoded_dir = os.path.join(base_dir, "encoded")
        self.segment_dir = os.path.join(base_dir, "segmented")
        
        # 创建目录
        for dir_path in [self.encoded_dir, self.segment_dir]:
            os.makedirs(dir_path, exist_ok=True)
            for data_type in ['train', 'test']:
                os.makedirs(os.path.join(dir_path, data_type), exist_ok=True)
        
        # 使用传入的logger或创建新logger
        self.logger = logger or logging.getLogger(__name__)
        
        # 初始化处理器
        self.encoder = DataEncode(logger=self.logger)
        self.segmenter = DataSegment(
            chunk_size=self.chunk_size, 
            n_processes=self.n_processes, 
            logger=self.logger
        )
        
        self.logger.info(f"初始化DataProcessor: chunk_size={chunk_size}, n_processes={self.n_processes}")
    
    # 移除_setup_logging方法
    
    def _get_file_info(self, file_path: str) -> Dict:
        if not os.path.exists(file_path):
            return {'exists': False}
        
        try:
            pf = pq.ParquetFile(file_path)
            return {
                'exists': True,
                'rows': pf.metadata.num_rows,
                'size_mb': os.path.getsize(file_path) / 1024**2,
                'columns': len(pf.schema.names)
            }
        except Exception as e:
            return {'exists': True, 'error': str(e)}
    
    def _files_exist(self, file_paths: List[str]) -> bool:
        return all(os.path.exists(path) for path in file_paths)
    
    def encode_data(self, data_type: str, force: bool = False) -> bool:
        input_file = os.path.join(self.base_dir, f"{data_type}.parquet")
        output_file = os.path.join(self.encoded_dir, data_type, f"{data_type}_encoded.parquet")
        
        if not os.path.exists(input_file):
            self.logger.error(f"输入文件不存在: {input_file}")
            return False
        
        if os.path.exists(output_file) and not force:
            self.logger.info(f"编码文件已存在，跳过: {output_file}")
            return True
        
        try:
            self.logger.info(f"开始编码 {data_type} 数据")
            start_time = datetime.now()
            
            input_info = self._get_file_info(input_file)
            self.logger.info(f"输入: {input_info['rows']:,} 行, {input_info['size_mb']:.1f}MB")
            
            result = self.encoder.process_file_multiprocess(
                input_file=input_file,
                output_file=output_file,
                chunk_size=self.chunk_size,
                n_processes=self.n_processes
            )
            
            if result:
                output_info = self._get_file_info(output_file)
                duration = datetime.now() - start_time
                self.logger.info(f"编码完成: {output_info['rows']:,} 行, "
                               f"{output_info['size_mb']:.1f}MB, 耗时: {duration}")
                
                if input_info['rows'] != output_info['rows']:
                    self.logger.warning(f"行数不匹配! 输入: {input_info['rows']}, 输出: {output_info['rows']}")
                    return False
                
                return True
            else:
                self.logger.error("编码失败")
                return False
                
        except Exception as e:
            self.logger.error(f"编码异常: {str(e)}")
            return False
    
    def segment_data(self, data_type: str, force: bool = False, verify: bool = True) -> bool:
        input_file = os.path.join(self.encoded_dir, data_type, f"{data_type}_encoded.parquet")
        output_dir = os.path.join(self.segment_dir, data_type)
        
        if not os.path.exists(input_file):
            self.logger.error(f"编码文件不存在: {input_file}")
            return False
        
        output_files = self.segmenter.get_output_files(data_type, output_dir)
        if self._files_exist(output_files) and not force:
            self.logger.info(f"分割文件已存在，跳过")
            if verify:
                return self._verify_existing_segmentation(input_file, output_dir, data_type)
            return True
        
        try:
            self.logger.info(f"开始分割 {data_type} 数据")
            start_time = datetime.now()
            
            input_info = self._get_file_info(input_file)
            self.logger.info(f"输入: {input_info['rows']:,} 行, {input_info['size_mb']:.1f}MB")
            
            # 执行分割
            segment_results = self.segmenter.process_file(
                input_file=input_file,
                data_type=data_type,
                output_dir=output_dir
            )
            
            # 检查返回结果
            if segment_results is None:
                self.logger.error("DataSegment.process_file() 返回了 None")
                return False
            
            if not isinstance(segment_results, dict):
                self.logger.error(f"DataSegment.process_file() 返回了意外的类型: {type(segment_results)}")
                return False
            
            # 计算总数并验证
            total_segmented = sum(segment_results.values())
            duration = datetime.now() - start_time
            
            self.logger.info(f"分割完成，总计: {total_segmented:,} 行，耗时: {duration}")
            for level in [3, 2, 1, 0]:
                count = segment_results.get(level, 0)
                if count > 0:
                    self.logger.info(f"  Segment {level}: {count:,} 行")
            
            # 验证行数匹配
            if input_info['rows'] != total_segmented:
                self.logger.error(f"行数不匹配! 输入: {input_info['rows']}, 分割: {total_segmented}")
                
                # 尝试详细验证
                if verify:
                    verification_result = self.segmenter.verify_segmentation(input_file, output_dir, data_type)
                    if verification_result and not verification_result['integrity']['rows_match']:
                        self.logger.error("验证失败：数据完整性检查未通过")
                        return False
                else:
                    return False
            
            # 可选的完整性验证
            if verify:
                verification_result = self.segmenter.verify_segmentation(input_file, output_dir, data_type)
                if not verification_result['integrity']['rows_match']:
                    self.logger.error("数据完整性验证失败")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"分割异常: {str(e)}")
            import traceback
            self.logger.debug(f"详细错误信息:\n{traceback.format_exc()}")
            return False
    
    def _verify_existing_segmentation(self, input_file: str, output_dir: str, data_type: str) -> bool:
        """验证已存在的分割文件的完整性"""
        try:
            verification_result = self.segmenter.verify_segmentation(input_file, output_dir, data_type)
            
            if verification_result['integrity']['rows_match'] and verification_result['integrity']['rankers_match']:
                self.logger.info("现有分割文件验证通过")
                return True
            else:
                self.logger.warning("现有分割文件验证失败，建议重新分割")
                return False
                
        except Exception as e:
            self.logger.error(f"验证现有分割文件时出错: {str(e)}")
            return False
    
    def process_data_type(self, data_type: str, force_encode: bool = False, 
                         force_segment: bool = False, verify_segment: bool = True) -> bool:
        self.logger.info(f"开始处理 {data_type.upper()} 数据")
        
        if not self.encode_data(data_type, force=force_encode):
            return False
        
        if not self.segment_data(data_type, force=force_segment, verify=verify_segment):
            return False
        
        self.logger.info(f"{data_type.upper()} 数据处理完成")
        return True
    
    def process_pipeline(self, force: bool = False, verify: bool = True) -> bool:
        self.logger.info("开始数据处理流水线")
        
        train_file = os.path.join(self.base_dir, "train.parquet")
        test_file = os.path.join(self.base_dir, "test.parquet")
        
        for file_path in [train_file, test_file]:
            if not os.path.exists(file_path):
                self.logger.error(f"源文件不存在: {file_path}")
                return False
        
        if not force and self._all_outputs_exist():
            self.logger.info("所有输出文件已存在")
            if verify:
                self.logger.info("执行完整性验证...")
                if self._verify_all_outputs():
                    self.logger.info("验证通过，跳过处理")
                    return True
                else:
                    self.logger.warning("验证失败，重新处理")
            else:
                self.logger.info("跳过验证，直接完成")
                return True
        
        try:
            pipeline_start = datetime.now()
            
            # 阶段1：编码
            self.logger.info("=" * 50)
            self.logger.info("阶段1：编码数据")
            self.logger.info("=" * 50)
            
            encode_start = datetime.now()
            for data_type in ['train', 'test']:
                if not self.encode_data(data_type, force=force):
                    self.logger.error(f"{data_type} 编码失败")
                    return False
            encode_duration = datetime.now() - encode_start
            
            # 阶段2：分割
            self.logger.info("=" * 50)
            self.logger.info("阶段2：分割数据")  
            self.logger.info("=" * 50)
            
            segment_start = datetime.now()
            for data_type in ['train', 'test']:
                if not self.segment_data(data_type, force=force, verify=verify):
                    self.logger.error(f"{data_type} 分割失败")
                    return False
            segment_duration = datetime.now() - segment_start
            
            # 最终统计
            pipeline_duration = datetime.now() - pipeline_start
            self.logger.info("=" * 50)
            self.logger.info("处理完成")
            self.logger.info("=" * 50)
            
            self._log_final_stats()
            
            self.logger.info(f"编码耗时: {encode_duration}")
            self.logger.info(f"分割耗时: {segment_duration}")
            self.logger.info(f"总耗时: {pipeline_duration}")
            self.logger.info("数据处理流水线成功完成")
            
            return True
            
        except Exception as e:
            self.logger.error(f"流水线异常: {str(e)}")
            return False
    
    def _all_outputs_exist(self) -> bool:
        for data_type in ['train', 'test']:
            output_dir = os.path.join(self.segment_dir, data_type)
            output_files = self.segmenter.get_output_files(data_type, output_dir)
            if not self._files_exist(output_files):
                return False
        return True
    
    def _verify_all_outputs(self) -> bool:
        """验证所有输出文件的完整性"""
        for data_type in ['train', 'test']:
            input_file = os.path.join(self.encoded_dir, data_type, f"{data_type}_encoded.parquet")
            output_dir = os.path.join(self.segment_dir, data_type)
            
            if not self._verify_existing_segmentation(input_file, output_dir, data_type):
                return False
        return True
    
    def _log_final_stats(self):
        for data_type in ['train', 'test']:
            self.logger.info(f"{data_type.upper()} 数据统计:")
            
            encoded_file = os.path.join(self.encoded_dir, data_type, f"{data_type}_encoded.parquet")
            if os.path.exists(encoded_file):
                info = self._get_file_info(encoded_file)
                self.logger.info(f"  编码文件: {info['rows']:,} 行, {info['size_mb']:.1f}MB")
            
            output_dir = os.path.join(self.segment_dir, data_type)
            total_rows, total_size = 0, 0
            
            for level in [0, 1, 2, 3]:
                segment_file = os.path.join(output_dir, f"{data_type}_segment_{level}.parquet")
                if os.path.exists(segment_file):
                    info = self._get_file_info(segment_file)
                    if info['rows'] > 0:
                        self.logger.info(f"  Segment {level}: {info['rows']:,} 行, {info['size_mb']:.1f}MB")
                        total_rows += info['rows']
                        total_size += info['size_mb']
            
            self.logger.info(f"  分割总计: {total_rows:,} 行, {total_size:.1f}MB")
    
    def get_pipeline_status(self) -> Dict:
        status = {}
        for data_type in ['train', 'test']:
            encoded_file = os.path.join(self.encoded_dir, data_type, f"{data_type}_encoded.parquet")
            encoded = os.path.exists(encoded_file)
            
            output_dir = os.path.join(self.segment_dir, data_type)
            output_files = self.segmenter.get_output_files(data_type, output_dir)
            segmented = self._files_exist(output_files)
            
            # 添加验证状态
            verified = False
            if encoded and segmented:
                verified = self._verify_existing_segmentation(encoded_file, output_dir, data_type)
            
            status[data_type] = {
                'encoded': encoded, 
                'segmented': segmented,
                'verified': verified
            }
        
        return status
    
    def concatenate_segments(self, data_type: str, output_file: Optional[str] = None) -> str:
        """
        将分割的segment文件重新拼接成完整的数据文件
        
        Args:
            data_type: 'train' 或 'test'
            output_file: 输出文件路径，如果为None则自动生成
            
        Returns:
            输出文件路径
        """
        if output_file is None:
            output_file = os.path.join(self.base_dir, f"{data_type}_reconstructed.parquet")
        
        output_dir = os.path.join(self.segment_dir, data_type)
        
        self.logger.info(f"开始拼接 {data_type} 的segment文件...")
        
        # 收集所有segment文件
        segment_dfs = []
        total_rows = 0
        
        for level in [0, 1, 2, 3]:
            segment_file = os.path.join(output_dir, f"{data_type}_segment_{level}.parquet")
            if os.path.exists(segment_file):
                df = pd.read_parquet(segment_file)
                if len(df) > 0:
                    segment_dfs.append(df)
                    total_rows += len(df)
                    self.logger.info(f"  加载 segment_{level}: {len(df):,} 行")
        
        if not segment_dfs:
            self.logger.error("未找到任何segment文件")
            return None
        
        # 拼接所有数据
        self.logger.info("拼接数据...")
        combined_df = pd.concat(segment_dfs, ignore_index=True)
        
        # 按ranker_id和其他关键字段排序（保证一致性）
        if 'ranker_id' in combined_df.columns:
            sort_cols = ['ranker_id']
            if 'Id' in combined_df.columns:
                sort_cols.append('Id')
            combined_df = combined_df.sort_values(sort_cols).reset_index(drop=True)
        
        # 保存结果
        combined_df.to_parquet(output_file, index=False)
        
        self.logger.info(f"拼接完成: {output_file}")
        self.logger.info(f"  总行数: {len(combined_df):,}")
        self.logger.info(f"  文件大小: {os.path.getsize(output_file) / 1024**2:.1f}MB")
        
        # 验证拼接结果
        original_encoded = os.path.join(self.encoded_dir, data_type, f"{data_type}_encoded.parquet")
        if os.path.exists(original_encoded):
            original_df = pd.read_parquet(original_encoded)
            if len(original_df) == len(combined_df):
                self.logger.info("✓ 拼接验证通过：行数匹配")
            else:
                self.logger.warning(f"✗ 拼接验证失败：原始 {len(original_df)} 行，拼接 {len(combined_df)} 行")
        
        return output_file


def main():
    current_path = os.path.dirname(os.path.abspath(__file__))
    main_path = os.path.abspath(os.path.join(current_path, "..", ".."))
    os.chdir(main_path)
    print(f"工作目录: {main_path}")

    # 设置基本日志（仅用于独立运行）
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    processor = DataProcessor(
        base_dir="data/aeroclub-recsys-2025",
        chunk_size=200000,
    )
    
    status = processor.get_pipeline_status()
    processor.logger.info("当前状态:")
    for data_type, state in status.items():
        processor.logger.info(f"  {data_type}: 编码={state['encoded']}, 分割={state['segmented']}, 验证={state['verified']}")
    
    success = processor.process_pipeline(force=False, verify=True)
    
    if success:
        processor.logger.info("处理成功完成")
    else:
        processor.logger.error("处理失败")

if __name__ == "__main__":
    main()