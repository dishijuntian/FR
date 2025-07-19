import os
import logging
from datetime import datetime
from multiprocessing import cpu_count
from DataSegment import DataSegment
from DataEncode import DataEncode

# 设置工作路径
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
MAIN_PATH = os.path.abspath(os.path.join(CURRENT_PATH, "..", ".."))
os.chdir(MAIN_PATH)

class DataProcessor:
    """数据处理流水线主控制器"""
    
    def __init__(self, base_dir: str = "data/aeroclub-recsys-2025", 
                 chunk_size: int = 200000, n_processes: int = None):
        self.base_dir = base_dir
        self.chunk_size = chunk_size
        self.n_processes = n_processes or min(cpu_count(), 8)
        
        # 设置路径
        self.segment_dir = os.path.join(base_dir, "segment")
        self.encode_dir = os.path.join(base_dir, "encode")
        
        self.train_segment_dir = os.path.join(self.segment_dir, "train")
        self.test_segment_dir = os.path.join(self.segment_dir, "test")
        self.train_encode_dir = os.path.join(self.encode_dir, "train")
        self.test_encode_dir = os.path.join(self.encode_dir, "test")
        
        # 创建所有必要的目录
        for dir_path in [self.train_segment_dir, self.test_segment_dir, 
                        self.train_encode_dir, self.test_encode_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # 配置logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # 初始化处理器
        self.data_segment = DataSegment(
            chunk_size=chunk_size, 
            n_processes=self.n_processes,
            logger=self.logger
        )
        self.data_encode = DataEncode(logger=self.logger)
        
        self.logger.info(f"初始化DataProcessor: base_dir={base_dir}, "
                        f"chunk_size={chunk_size}, n_processes={self.n_processes}")
    
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
        log_file = os.path.join(self.base_dir, f"data_processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        
        # 配置root logger
        logging.basicConfig(
            level=logging.INFO,
            handlers=[console_handler, file_handler],
            force=True
        )
    
    def _check_source_files(self) -> tuple[str, str]:
        """检查原始数据文件是否存在"""
        train_file = os.path.join(self.base_dir, "train.parquet")
        test_file = os.path.join(self.base_dir, "test.parquet")
        
        if not os.path.exists(train_file):
            raise FileNotFoundError(f"训练文件不存在: {train_file}")
        
        if not os.path.exists(test_file):
            raise FileNotFoundError(f"测试文件不存在: {test_file}")
        
        return train_file, test_file
    
    def _get_segment_files(self) -> dict:
        """获取分割阶段的输出文件列表"""
        files = {
            'train': self.data_segment.get_output_files('train', self.train_segment_dir),
            'test': self.data_segment.get_output_files('test', self.test_segment_dir)
        }
        return files
    
    def _get_encode_files(self) -> dict:
        """获取编码阶段的输出文件列表"""
        files = {'train': [], 'test': []}
        
        for data_type in ['train', 'test']:
            encode_dir = self.train_encode_dir if data_type == 'train' else self.test_encode_dir
            for segment_level in [0, 1, 2, 3]:
                filename = f"{data_type}_segment_{segment_level}_encoded.parquet"
                filepath = os.path.join(encode_dir, filename)
                files[data_type].append(filepath)
        
        return files
    
    def _check_missing_files(self, files: dict) -> list:
        """检查缺失的文件"""
        missing = []
        for data_type in files:
            for file_path in files[data_type]:
                if not os.path.exists(file_path):
                    missing.append(file_path)
        return missing
    
    def process_segmentation(self, force: bool = False) -> bool:
        """执行数据分割阶段"""
        self.logger.info("=" * 60)
        self.logger.info("第一阶段：数据分割处理")
        self.logger.info("=" * 60)
        
        # 检查输出文件
        segment_files = self._get_segment_files()
        missing_segment_files = self._check_missing_files(segment_files)
        
        if not missing_segment_files and not force:
            self.logger.info("分割文件已存在，跳过分割阶段")
            return True
        
        if force:
            self.logger.info("强制重新执行分割处理...")
        else:
            self.logger.info(f"发现 {len(missing_segment_files)} 个分割文件缺失，开始分割处理...")
        
        try:
            # 获取源文件
            train_file, test_file = self._check_source_files()
            
            start_time = datetime.now()
            
            # 处理训练数据
            self.logger.info("处理训练数据分割...")
            self.data_segment.process_file(train_file, 'train', self.train_segment_dir)
            
            # 处理测试数据
            self.logger.info("处理测试数据分割...")
            self.data_segment.process_file(test_file, 'test', self.test_segment_dir)
            
            end_time = datetime.now()
            self.logger.info(f"分割阶段完成，耗时: {end_time - start_time}")
            return True
            
        except Exception as e:
            self.logger.error(f"分割阶段失败: {str(e)}")
            return False
    
    def process_encoding(self, force: bool = False) -> bool:
        """执行数据编码阶段"""
        self.logger.info("=" * 60)
        self.logger.info("第二阶段：数据编码处理")
        self.logger.info("=" * 60)
        
        # 检查分割文件是否存在
        segment_files = self._get_segment_files()
        missing_segment_files = self._check_missing_files(segment_files)
        
        if missing_segment_files:
            self.logger.error(f"分割文件缺失，无法进行编码: {missing_segment_files}")
            return False
        
        # 检查编码输出文件
        encode_files = self._get_encode_files()
        missing_encode_files = self._check_missing_files(encode_files)
        
        if not missing_encode_files and not force:
            self.logger.info("编码文件已存在，跳过编码阶段")
            return True
        
        if force:
            self.logger.info("强制重新执行编码处理...")
        else:
            self.logger.info(f"发现 {len(missing_encode_files)} 个编码文件缺失，开始编码处理...")
        
        try:
            start_time = datetime.now()
            
            # 对每个分割文件进行编码
            for data_type in ['train', 'test']:
                segment_dir = self.train_segment_dir if data_type == 'train' else self.test_segment_dir
                encode_dir = self.train_encode_dir if data_type == 'train' else self.test_encode_dir
                
                for segment_level in [0, 1, 2, 3]:
                    input_file = os.path.join(segment_dir, f"{data_type}_segment_{segment_level}.parquet")
                    output_file = os.path.join(encode_dir, f"{data_type}_segment_{segment_level}_encoded.parquet")
                    
                    if os.path.exists(input_file):
                        if not os.path.exists(output_file) or force:
                            self.logger.info(f"编码处理: {data_type}_segment_{segment_level}")
                            self.data_encode.process_file(input_file, output_file)
                        else:
                            self.logger.info(f"编码文件已存在，跳过: {output_file}")
                    else:
                        self.logger.warning(f"分割文件不存在，跳过编码: {input_file}")
            
            end_time = datetime.now()
            self.logger.info(f"编码阶段完成，耗时: {end_time - start_time}")
            return True
            
        except Exception as e:
            self.logger.error(f"编码阶段失败: {str(e)}")
            return False
    
    def process_pipeline(self, force_segment: bool = False, force_encode: bool = False) -> bool:
        """执行完整的数据处理流水线"""
        self.logger.info("开始数据处理流水线")
        self.logger.info(f"源数据目录: {self.base_dir}")
        self.logger.info(f"分割输出目录: {self.segment_dir}")
        self.logger.info(f"编码输出目录: {self.encode_dir}")
        
        pipeline_start_time = datetime.now()
        
        # 阶段1：数据分割
        if not self.process_segmentation(force=force_segment):
            self.logger.error("数据分割阶段失败，终止流水线")
            return False
        
        # 阶段2：数据编码
        if not self.process_encoding(force=force_encode):
            self.logger.error("数据编码阶段失败，终止流水线")
            return False
        
        pipeline_end_time = datetime.now()
        
        # 打印最终统计
        self._print_final_statistics()
        
        self.logger.info(f"数据处理流水线完成，总耗时: {pipeline_end_time - pipeline_start_time}")
        return True
    
    def _print_final_statistics(self):
        """打印最终统计信息"""
        self.logger.info("=" * 60)
        self.logger.info("数据处理流水线完成 - 最终统计")
        self.logger.info("=" * 60)
        
        for data_type in ['train', 'test']:
            self.logger.info(f"\n{data_type.upper()} 数据统计:")
            self.logger.info("-" * 50)
            
            segment_dir = self.train_segment_dir if data_type == 'train' else self.test_segment_dir
            encode_dir = self.train_encode_dir if data_type == 'train' else self.test_encode_dir
            
            total_segment_rows = 0
            total_encode_rows = 0
            
            for segment_level in [0, 1, 2, 3]:
                segment_file = os.path.join(segment_dir, f"{data_type}_segment_{segment_level}.parquet")
                encode_file = os.path.join(encode_dir, f"{data_type}_segment_{segment_level}_encoded.parquet")
                
                segment_rows = 0
                encode_rows = 0
                
                if os.path.exists(segment_file):
                    import pandas as pd
                    df = pd.read_parquet(segment_file)
                    segment_rows = len(df)
                    total_segment_rows += segment_rows
                
                if os.path.exists(encode_file):
                    import pandas as pd
                    df = pd.read_parquet(encode_file)
                    encode_rows = len(df)
                    total_encode_rows += encode_rows
                
                self.logger.info(f"Segment {segment_level}: {segment_rows:,} 行 -> {encode_rows:,} 行 (编码后)")
            
            self.logger.info(f"总计: {total_segment_rows:,} 行 -> {total_encode_rows:,} 行 (编码后)")


def main():
    """主函数"""
    # 初始化数据处理器
    processor = DataProcessor(
        base_dir="data/aeroclub-recsys-2025",
        chunk_size=500000,  # 可根据内存情况调整
        n_processes=min(cpu_count(), 12)
    )
    
    # 执行完整的处理流水线
    # force_segment=True 强制重新分割
    # force_encode=True 强制重新编码
    success = processor.process_pipeline(
        force_segment=False, 
        force_encode=False
    )
    
    if success:
        processor.logger.info("数据处理流水线成功完成！")
    else:
        processor.logger.error("数据处理流水线执行失败！")


if __name__ == "__main__":
    main()