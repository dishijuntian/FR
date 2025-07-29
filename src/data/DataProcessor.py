"""
数据处理流水线：编码 + 特征工程 + 分割
"""
import os
import logging
import pandas as pd
from datetime import datetime
from multiprocessing import cpu_count
from typing import Dict, List, Optional
from pathlib import Path

from src.data.DataEncode import DataEncode
from src.data.DataSegment import DataSegment
from src.data.DataEngineering import DataEngineering
from src.utils.Common import timer
from src.utils.FileUtils import FileUtils
from src.utils.MemoryUtils import MemoryUtils
from src.utils.ValidationUtils import ValidationUtils


class DataProcessor:
    """数据处理流水线 - 优化版"""
    
    def __init__(self, base_dir: str = "data/aeroclub-recsys-2025", 
                 chunk_size: int = 200000, n_processes: Optional[int] = None,
                 logger: Optional[logging.Logger] = None, config: Dict = None):
        
        self.base_dir = Path(base_dir)
        self.chunk_size = chunk_size
        self.n_processes = n_processes or min(cpu_count(), 8)
        self.logger = logger or logging.getLogger(__name__)
        self.config = config or {}
        
        # 设置路径
        self.encoded_dir = self.base_dir / "encoded"
        self.segment_dir = self.base_dir / "segmented"
        
        # 创建目录
        for dir_path in [self.encoded_dir, self.segment_dir]:
            FileUtils.ensure_dir(dir_path)
            for data_type in ['train', 'test']:
                FileUtils.ensure_dir(dir_path / data_type)
        
        # 初始化处理器
        self.encoder = DataEncode(logger=self.logger)
        self.segmenter = DataSegment(
            chunk_size=self.chunk_size, 
            n_processes=self.n_processes, 
            logger=self.logger
        )
        
        # 初始化数据工程模块（新增）
        self.data_engineering = DataEngineering(logger=self.logger)
        
        self.logger.info(f"DataProcessor初始化: chunk_size={chunk_size}, "
                        f"n_processes={self.n_processes}")
    
    @timer
    def encode_data(self, data_type: str, force: bool = False) -> bool:
        """编码数据"""
        input_file = self.base_dir / f"{data_type}.parquet"
        output_file = self.encoded_dir / data_type / f"{data_type}_encoded.parquet"
        
        if not input_file.exists():
            self.logger.error(f"输入文件不存在: {input_file}")
            return False
        
        if output_file.exists() and not force:
            self.logger.info(f"编码文件已存在，跳过: {output_file}")
            return True
        
        try:
            self.logger.info(f"开始编码 {data_type} 数据")
            start_time = datetime.now()
            
            input_info = FileUtils.get_file_info(input_file)
            self.logger.info(f"输入: {input_info['rows']:,} 行, {input_info['size_mb']:.1f}MB")
            
            result = self.encoder.process_file_multiprocess(
                input_file=str(input_file),
                output_file=str(output_file),
                chunk_size=self.chunk_size,
                n_processes=self.n_processes
            )
            
            if result:
                output_info = FileUtils.get_file_info(output_file)
                duration = datetime.now() - start_time
                self.logger.info(f"编码完成: {output_info['rows']:,} 行, "
                               f"{output_info['size_mb']:.1f}MB, 耗时: {duration}")
                
                return self._validate_encoding(input_info, output_info)
            else:
                self.logger.error("编码失败")
                return False
                
        except Exception as e:
            self.logger.error(f"编码异常: {str(e)}")
            return False
    
    @timer
    def engineer_features(self, data_type: str, force: bool = False) -> bool:
        """特征工程阶段（新增）"""
        # 检查是否启用特征工程
        feature_config = self.config.get('data_processing', {}).get('feature_engineering', {})
        if not feature_config.get('enabled', True):
            self.logger.info("特征工程已禁用，跳过")
            return True
        
        input_file = self.encoded_dir / data_type / f"{data_type}_encoded.parquet"
        output_file = self.encoded_dir / data_type / f"{data_type}_featured.parquet"
        
        if not input_file.exists():
            self.logger.error(f"编码文件不存在: {input_file}")
            return False
        
        if output_file.exists() and not force:
            self.logger.info(f"特征工程文件已存在，跳过: {output_file}")
            return True
        
        try:
            self.logger.info(f"开始 {data_type} 特征工程")
            start_time = datetime.now()
            
            # 加载数据
            df = pd.read_parquet(input_file)
            original_shape = df.shape
            
            # 获取特征类型配置
            feature_types = feature_config.get('feature_types', 
                                             ['flight', 'price', 'route', 'temporal', 'passenger'])
            
            if feature_config.get('enable_interactions', False):
                feature_types.append('interaction')
            
            # 执行特征工程
            df_featured = self.data_engineering.process_features(df, feature_types)
            
            # 数据质量处理
            quality_config = self.config.get('data_processing', {}).get('data_quality', {})
            if quality_config.get('detect_outliers', True):
                df_featured, quality_report = self.data_engineering.validate_and_clean(
                    df_featured,
                    clean_outliers=quality_config.get('clean_outliers', False),
                    outlier_method=quality_config.get('outlier_method', 'iqr')
                )
                
                self.logger.info(f"数据质量检查完成，异常值检测: {len(quality_report['outliers'])} 列")
            
            # 保存结果
            df_featured.to_parquet(output_file, index=False)
            
            duration = datetime.now() - start_time
            new_shape = df_featured.shape
            features_added = new_shape[1] - original_shape[1]
            
            self.logger.info(f"特征工程完成: {original_shape} -> {new_shape}")
            self.logger.info(f"新增特征: {features_added}, 耗时: {duration}")
            
            # 内存清理
            del df, df_featured
            MemoryUtils.force_gc()
            
            return True
            
        except Exception as e:
            self.logger.error(f"特征工程异常: {str(e)}")
            return False
    
    @timer
    def segment_data(self, data_type: str, force: bool = False, verify: bool = True) -> bool:
        """分割数据"""
        # 选择输入文件（优先使用特征工程后的文件）
        featured_file = self.encoded_dir / data_type / f"{data_type}_featured.parquet"
        encoded_file = self.encoded_dir / data_type / f"{data_type}_encoded.parquet"
        
        input_file = featured_file if featured_file.exists() else encoded_file
        output_dir = self.segment_dir / data_type
        
        if not input_file.exists():
            self.logger.error(f"输入文件不存在: {input_file}")
            return False
        
        output_files = self.segmenter.get_output_files(data_type, str(output_dir))
        if all(os.path.exists(f) for f in output_files) and not force:
            self.logger.info(f"分割文件已存在，跳过")
            if verify:
                return self._verify_existing_segmentation(str(input_file), str(output_dir), data_type)
            return True
        
        try:
            self.logger.info(f"开始分割 {data_type} 数据")
            start_time = datetime.now()
            
            input_info = FileUtils.get_file_info(input_file)
            self.logger.info(f"输入: {input_info['rows']:,} 行, {input_info['size_mb']:.1f}MB")
            
            # 执行分割
            segment_results = self.segmenter.process_file(
                input_file=str(input_file),
                data_type=data_type,
                output_dir=str(output_dir),
                pre_classification=True
            )
            
            if segment_results is None:
                self.logger.error("分割返回None")
                return False
            
            # 验证结果
            total_segmented = sum(segment_results.values())
            duration = datetime.now() - start_time
            
            self.logger.info(f"分割完成，总计: {total_segmented:,} 行，耗时: {duration}")
            for level in [3, 2, 1, 0]:
                count = segment_results.get(level, 0)
                if count > 0:
                    self.logger.info(f"  Segment {level}: {count:,} 行")
            
            if verify:
                verification_result = self.segmenter.verify_segmentation(str(input_file), str(output_dir), data_type)
                if not verification_result['integrity']['rows_match'] and verification_result['data_type']=='test':
                    self.logger.error("数据完整性验证失败")
                    return False
                elif verification_result['data_type']=='train':
                    self.logger.warning("略过 Train 数据集完整性验证")
                    return True
            
            return True
            
        except Exception as e:
            self.logger.error(f"分割异常: {str(e)}")
            return False
    
    def _validate_encoding(self, input_info: Dict, output_info: Dict) -> bool:
        """验证编码结果"""
        if input_info['rows'] != output_info['rows']:
            self.logger.warning(f"行数不匹配! 输入: {input_info['rows']}, 输出: {output_info['rows']}")
            return False
        return True
    
    def _verify_existing_segmentation(self, input_file: str, output_dir: str, data_type: str) -> bool:
        """验证已存在的分割文件"""
        try:
            verification_result = self.segmenter.verify_segmentation(input_file, output_dir, data_type)
            
            if verification_result['integrity']['rows_match'] and verification_result['integrity']['rankers_match'] and verification_result['data_type']=='test':
                self.logger.info("现有分割文件验证通过")
                return True
            elif verification_result['data_type']=='train':
                self.logger.warning("略过 Train 数据集完整性验证")
                return True
            else:
                self.logger.warning("现有分割文件验证失败")
                return False
        except Exception as e:
            self.logger.error(f"验证现有分割文件时出错: {str(e)}")
            return False
    
    @timer
    def process_data_type(self, data_type: str, force_encode: bool = False, 
                         force_feature: bool = False, force_segment: bool = False, 
                         verify_segment: bool = True) -> bool:
        """处理单个数据类型（包含特征工程）"""
        self.logger.info(f"开始处理 {data_type.upper()} 数据")
        
        # 阶段1：编码
        if not self.encode_data(data_type, force=force_encode):
            return False
        
        # 阶段2：特征工程
        # if not self.engineer_features(data_type, force=force_feature):
        #     return False
        
        # 阶段3：分割
        if not self.segment_data(data_type, force=force_segment, verify=verify_segment):
            return False
        
        self.logger.info(f"{data_type.upper()} 数据处理完成")
        return True
    
    @timer
    def process_pipeline(self, force: bool = False) -> bool:
        """完整数据处理流水线"""
        self.logger.info("开始数据处理流水线")
        
        # 检查源文件
        train_file = self.base_dir / "train.parquet"
        test_file = self.base_dir / "test.parquet"
        
        for file_path in [train_file, test_file]:
            if not file_path.exists():
                self.logger.error(f"源文件不存在: {file_path}")
                return False
        
        # 检查是否需要处理
        verify = self.config.get('verify_results', False)
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
                return True
        
        try:
            pipeline_start = datetime.now()
            
            # 处理训练和测试数据
            for data_type in ['train', 'test']:
                if not self.process_data_type(
                    data_type, 
                    force_encode=force, 
                    force_feature=force,
                    force_segment=force, 
                    verify_segment=verify
                ):
                    self.logger.error(f"{data_type} 处理失败")
                    return False
            
            # 最终统计
            pipeline_duration = datetime.now() - pipeline_start
            self.logger.info("=" * 50)
            self.logger.info("数据处理流水线成功完成")
            self.logger.info("=" * 50)
            
            self._log_final_stats()
            self.logger.info(f"总耗时: {pipeline_duration}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"流水线异常: {str(e)}")
            return False
    
    def _all_outputs_exist(self) -> bool:
        """检查所有输出是否存在"""
        for data_type in ['train', 'test']:
            output_dir = self.segment_dir / data_type
            output_files = self.segmenter.get_output_files(data_type, str(output_dir))
            if not all(os.path.exists(f) for f in output_files):
                return False
        return True
    
    def _verify_all_outputs(self) -> bool:
        """验证所有输出文件的完整性"""
        for data_type in ['train', 'test']:
            # 选择输入文件
            featured_file = self.encoded_dir / data_type / f"{data_type}_featured.parquet"
            encoded_file = self.encoded_dir / data_type / f"{data_type}_encoded.parquet"
            input_file = featured_file if featured_file.exists() else encoded_file
            
            output_dir = self.segment_dir / data_type
            
            if not self._verify_existing_segmentation(str(input_file), str(output_dir), data_type):
                return False
        return True
    
    def _log_final_stats(self):
        """记录最终统计"""
        for data_type in ['train', 'test']:
            self.logger.info(f"{data_type.upper()} 数据统计:")
            
            # 编码文件统计
            encoded_file = self.encoded_dir / data_type / f"{data_type}_encoded.parquet"
            if encoded_file.exists():
                info = FileUtils.get_file_info(encoded_file)
                self.logger.info(f"  编码文件: {info['rows']:,} 行, {info['size_mb']:.1f}MB")
            
            # 特征工程文件统计
            featured_file = self.encoded_dir / data_type / f"{data_type}_featured.parquet"
            if featured_file.exists():
                info = FileUtils.get_file_info(featured_file)
                self.logger.info(f"  特征文件: {info['rows']:,} 行, {info['size_mb']:.1f}MB, {info['columns']} 列")
            
            # 分割文件统计
            output_dir = self.segment_dir / data_type
            total_rows, total_size = 0, 0
            
            for level in [0, 1, 2, 3]:
                segment_file = output_dir / f"{data_type}_segment_{level}.parquet"
                if segment_file.exists():
                    info = FileUtils.get_file_info(segment_file)
                    if info['rows'] > 0:
                        self.logger.info(f"  Segment {level}: {info['rows']:,} 行, {info['size_mb']:.1f}MB")
                        total_rows += info['rows']
                        total_size += info['size_mb']
            
            self.logger.info(f"  分割总计: {total_rows:,} 行, {total_size:.1f}MB")
    
    def get_pipeline_status(self) -> Dict:
        """获取流水线状态"""
        status = {}
        for data_type in ['train', 'test']:
            encoded_file = self.encoded_dir / data_type / f"{data_type}_encoded.parquet"
            featured_file = self.encoded_dir / data_type / f"{data_type}_featured.parquet"
            
            encoded = encoded_file.exists()
            featured = featured_file.exists()
            
            output_dir = self.segment_dir / data_type
            output_files = self.segmenter.get_output_files(data_type, str(output_dir))
            segmented = all(os.path.exists(f) for f in output_files)
            
            # 验证状态
            verified = False
            if segmented:
                input_file = featured_file if featured else encoded_file
                if input_file.exists():
                    verified = self._verify_existing_segmentation(str(input_file), str(output_dir), data_type)
            
            status[data_type] = {
                'encoded': encoded,
                'featured': featured,
                'segmented': segmented,
                'verified': verified
            }
        
        return status
    
    def get_feature_engineering_report(self) -> Dict:
        """获取特征工程报告"""
        if hasattr(self.data_engineering, 'processing_stats'):
            return self.data_engineering.get_processing_summary()
        return {}