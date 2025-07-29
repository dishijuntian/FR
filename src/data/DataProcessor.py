<<<<<<< HEAD
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
=======
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
>>>>>>> e9addff7041905ba228e279124f71a4a54b1d4e3
        
        # 初始化处理器
        self.encoder = DataEncode(logger=self.logger)
        self.segmenter = DataSegment(
            chunk_size=self.chunk_size, 
            n_processes=self.n_processes, 
            logger=self.logger
        )
        
<<<<<<< HEAD
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
=======
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
>>>>>>> e9addff7041905ba228e279124f71a4a54b1d4e3
            self.logger.info(f"编码文件已存在，跳过: {output_file}")
            return True
        
        try:
            self.logger.info(f"开始编码 {data_type} 数据")
            start_time = datetime.now()
            
<<<<<<< HEAD
            input_info = FileUtils.get_file_info(input_file)
            self.logger.info(f"输入: {input_info['rows']:,} 行, {input_info['size_mb']:.1f}MB")
            
            result = self.encoder.process_file_multiprocess(
                input_file=str(input_file),
                output_file=str(output_file),
=======
            input_info = self._get_file_info(input_file)
            self.logger.info(f"输入: {input_info['rows']:,} 行, {input_info['size_mb']:.1f}MB")
            
            result = self.encoder.process_file_multiprocess(
                input_file=input_file,
                output_file=output_file,
>>>>>>> e9addff7041905ba228e279124f71a4a54b1d4e3
                chunk_size=self.chunk_size,
                n_processes=self.n_processes
            )
            
            if result:
<<<<<<< HEAD
                output_info = FileUtils.get_file_info(output_file)
=======
                output_info = self._get_file_info(output_file)
>>>>>>> e9addff7041905ba228e279124f71a4a54b1d4e3
                duration = datetime.now() - start_time
                self.logger.info(f"编码完成: {output_info['rows']:,} 行, "
                               f"{output_info['size_mb']:.1f}MB, 耗时: {duration}")
                
<<<<<<< HEAD
                return self._validate_encoding(input_info, output_info)
=======
                if input_info['rows'] != output_info['rows']:
                    self.logger.warning(f"行数不匹配! 输入: {input_info['rows']}, 输出: {output_info['rows']}")
                    return False
                
                return True
>>>>>>> e9addff7041905ba228e279124f71a4a54b1d4e3
            else:
                self.logger.error("编码失败")
                return False
                
        except Exception as e:
            self.logger.error(f"编码异常: {str(e)}")
            return False
    
<<<<<<< HEAD
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
=======
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
>>>>>>> e9addff7041905ba228e279124f71a4a54b1d4e3
            return True
        
        try:
            self.logger.info(f"开始分割 {data_type} 数据")
            start_time = datetime.now()
            
<<<<<<< HEAD
            input_info = FileUtils.get_file_info(input_file)
=======
            input_info = self._get_file_info(input_file)
>>>>>>> e9addff7041905ba228e279124f71a4a54b1d4e3
            self.logger.info(f"输入: {input_info['rows']:,} 行, {input_info['size_mb']:.1f}MB")
            
            # 执行分割
            segment_results = self.segmenter.process_file(
<<<<<<< HEAD
                input_file=str(input_file),
                data_type=data_type,
                output_dir=str(output_dir),
                pre_classification=True
            )
            
            if segment_results is None:
                self.logger.error("分割返回None")
                return False
            
            # 验证结果
=======
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
>>>>>>> e9addff7041905ba228e279124f71a4a54b1d4e3
            total_segmented = sum(segment_results.values())
            duration = datetime.now() - start_time
            
            self.logger.info(f"分割完成，总计: {total_segmented:,} 行，耗时: {duration}")
            for level in [3, 2, 1, 0]:
                count = segment_results.get(level, 0)
                if count > 0:
                    self.logger.info(f"  Segment {level}: {count:,} 行")
            
<<<<<<< HEAD
            if verify:
                verification_result = self.segmenter.verify_segmentation(str(input_file), str(output_dir), data_type)
                if not verification_result['integrity']['rows_match'] and verification_result['data_type']=='test':
                    self.logger.error("数据完整性验证失败")
                    return False
                elif verification_result['data_type']=='train':
                    self.logger.warning("略过 Train 数据集完整性验证")
                    return True
=======
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
>>>>>>> e9addff7041905ba228e279124f71a4a54b1d4e3
            
            return True
            
        except Exception as e:
            self.logger.error(f"分割异常: {str(e)}")
<<<<<<< HEAD
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
=======
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
                
>>>>>>> e9addff7041905ba228e279124f71a4a54b1d4e3
        except Exception as e:
            self.logger.error(f"验证现有分割文件时出错: {str(e)}")
            return False
    
<<<<<<< HEAD
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
=======
    def process_data_type(self, data_type: str, force_encode: bool = False, 
                         force_segment: bool = False, verify_segment: bool = True) -> bool:
        self.logger.info(f"开始处理 {data_type.upper()} 数据")
        
        if not self.encode_data(data_type, force=force_encode):
            return False
        
>>>>>>> e9addff7041905ba228e279124f71a4a54b1d4e3
        if not self.segment_data(data_type, force=force_segment, verify=verify_segment):
            return False
        
        self.logger.info(f"{data_type.upper()} 数据处理完成")
        return True
    
<<<<<<< HEAD
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
=======
    def process_pipeline(self, force: bool = False, verify: bool = True) -> bool:
        self.logger.info("开始数据处理流水线")
        
        train_file = os.path.join(self.base_dir, "train.parquet")
        test_file = os.path.join(self.base_dir, "test.parquet")
        
        for file_path in [train_file, test_file]:
            if not os.path.exists(file_path):
                self.logger.error(f"源文件不存在: {file_path}")
                return False
        
>>>>>>> e9addff7041905ba228e279124f71a4a54b1d4e3
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
<<<<<<< HEAD
=======
                self.logger.info("跳过验证，直接完成")
>>>>>>> e9addff7041905ba228e279124f71a4a54b1d4e3
                return True
        
        try:
            pipeline_start = datetime.now()
            
<<<<<<< HEAD
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
=======
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
>>>>>>> e9addff7041905ba228e279124f71a4a54b1d4e3
            
            # 最终统计
            pipeline_duration = datetime.now() - pipeline_start
            self.logger.info("=" * 50)
<<<<<<< HEAD
            self.logger.info("数据处理流水线成功完成")
            self.logger.info("=" * 50)
            
            self._log_final_stats()
            self.logger.info(f"总耗时: {pipeline_duration}")
=======
            self.logger.info("处理完成")
            self.logger.info("=" * 50)
            
            self._log_final_stats()
            
            self.logger.info(f"编码耗时: {encode_duration}")
            self.logger.info(f"分割耗时: {segment_duration}")
            self.logger.info(f"总耗时: {pipeline_duration}")
            self.logger.info("数据处理流水线成功完成")
>>>>>>> e9addff7041905ba228e279124f71a4a54b1d4e3
            
            return True
            
        except Exception as e:
            self.logger.error(f"流水线异常: {str(e)}")
            return False
    
    def _all_outputs_exist(self) -> bool:
<<<<<<< HEAD
        """检查所有输出是否存在"""
        for data_type in ['train', 'test']:
            output_dir = self.segment_dir / data_type
            output_files = self.segmenter.get_output_files(data_type, str(output_dir))
            if not all(os.path.exists(f) for f in output_files):
=======
        for data_type in ['train', 'test']:
            output_dir = os.path.join(self.segment_dir, data_type)
            output_files = self.segmenter.get_output_files(data_type, output_dir)
            if not self._files_exist(output_files):
>>>>>>> e9addff7041905ba228e279124f71a4a54b1d4e3
                return False
        return True
    
    def _verify_all_outputs(self) -> bool:
        """验证所有输出文件的完整性"""
        for data_type in ['train', 'test']:
<<<<<<< HEAD
            # 选择输入文件
            featured_file = self.encoded_dir / data_type / f"{data_type}_featured.parquet"
            encoded_file = self.encoded_dir / data_type / f"{data_type}_encoded.parquet"
            input_file = featured_file if featured_file.exists() else encoded_file
            
            output_dir = self.segment_dir / data_type
            
            if not self._verify_existing_segmentation(str(input_file), str(output_dir), data_type):
=======
            input_file = os.path.join(self.encoded_dir, data_type, f"{data_type}_encoded.parquet")
            output_dir = os.path.join(self.segment_dir, data_type)
            
            if not self._verify_existing_segmentation(input_file, output_dir, data_type):
>>>>>>> e9addff7041905ba228e279124f71a4a54b1d4e3
                return False
        return True
    
    def _log_final_stats(self):
<<<<<<< HEAD
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
=======
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
>>>>>>> e9addff7041905ba228e279124f71a4a54b1d4e3
                    if info['rows'] > 0:
                        self.logger.info(f"  Segment {level}: {info['rows']:,} 行, {info['size_mb']:.1f}MB")
                        total_rows += info['rows']
                        total_size += info['size_mb']
            
            self.logger.info(f"  分割总计: {total_rows:,} 行, {total_size:.1f}MB")
    
    def get_pipeline_status(self) -> Dict:
<<<<<<< HEAD
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
=======
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
>>>>>>> e9addff7041905ba228e279124f71a4a54b1d4e3
                'segmented': segmented,
                'verified': verified
            }
        
        return status
    
<<<<<<< HEAD
    def get_feature_engineering_report(self) -> Dict:
        """获取特征工程报告"""
        if hasattr(self.data_engineering, 'processing_stats'):
            return self.data_engineering.get_processing_summary()
        return {}
=======
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
>>>>>>> e9addff7041905ba228e279124f71a4a54b1d4e3
