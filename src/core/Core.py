"""
航班排名系统核心控制器
"""

import os
import sys
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path

current_path = Path(__file__).parent
project_root = current_path.parent.parent
sys.path.insert(0, str(project_root / "src"))

from src.data.DataProcessor import DataProcessor
from src.model.Trainer import FlightRankingTrainer
from src.model.Predictor import FlightRankingPredictor


class FlightRankingCore:
    """航班排名系统核心控制器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.project_root = project_root
        self._setup_environment()
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        self._init_components()
        
        self.logger.info("核心控制器初始化完成")
    
    def _setup_environment(self):
        """设置工作环境"""
        os.chdir(self.project_root)
        for dir_key in ['data_dir', 'model_input_dir', 'model_save_dir', 'output_dir', 'log_dir']:
            if dir_key in self.config['paths']:
                Path(self.config['paths'][dir_key]).mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self):
        """设置日志系统"""
        log_config = self.config.get('logging', {})
        log_file = Path(self.config['paths'].get('log_dir', 'logs')) / \
                  f"flight_ranking_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=getattr(logging, log_config.get('level', 'INFO').upper()),
            format=log_config.get('format', '%(asctime)s - %(levelname)s - %(message)s'),
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file, encoding='utf-8')
            ],
            force=True
        )
    
    def _init_components(self):
        """初始化组件"""
        data_config = self.config.get('data_processing', {})
        training_config = self.config.get('training', {})
        prediction_config = self.config.get('prediction', {})
        
        # 数据处理器
        self.data_processor = DataProcessor(
            base_dir=self.config['paths']['data_dir'],
            chunk_size=data_config.get('chunk_size', 200000),
            n_processes=data_config.get('n_processes'),
            logger=self.logger
        )
        
        # 模型训练器
        self.model_trainer = FlightRankingTrainer(
            config=self.config,
            logger=self.logger
        )
        
        # 显示训练模式
        if self.model_trainer.use_gpu:
            self.logger.info("训练模式: GPU加速 + 串行fold训练")
        else:
            self.logger.info("训练模式: CPU多进程 + 并行fold训练")
        
        # 模型预测器
        self.model_predictor = FlightRankingPredictor(
            config=self.config,
            logger=self.logger
        )
    
    def run_data_processing(self, force: bool = None) -> bool:
        """执行数据处理"""
        self.logger.info("开始数据处理")
        
        try:
            data_config = self.config.get('data_processing', {})
            force = force if force is not None else data_config.get('force_reprocess', False)
            
            success = self.data_processor.process_pipeline(force=force)
            
            if success:
                self.logger.info("✓ 数据处理完成")
            else:
                self.logger.error("✗ 数据处理失败")
            
            return success
        except Exception as e:
            self.logger.error(f"数据处理异常: {e}")
            return False
    
    def run_model_training(self, segments: List[int] = None) -> bool:
        """执行模型训练"""
        self.logger.info("开始模型训练")
        
        try:
            training_config = self.config.get('training', {})
            segments = segments or training_config.get('segments', [0, 1, 2])
            model_names = training_config.get('model_names', ['XGBRanker', 'LGBMRanker'])
            
            if not self._check_training_data(segments):
                return False
            
            results = self.model_trainer.train_all_segments()
            
            success = len(results) > 0
            if success:
                self.logger.info("✓ 模型训练完成")
            else:
                self.logger.error("✗ 模型训练失败")
            
            return success
        except Exception as e:
            self.logger.error(f"模型训练异常: {e}")
            return False
    
    def run_model_prediction(self, segments: List[int] = None, 
                           model_names: List[str] = None) -> bool:
        """执行模型预测"""
        self.logger.info("开始模型预测")
        
        try:
            prediction_config = self.config.get('prediction', {})
            segments = segments or prediction_config.get('segments', [0, 1, 2])
            model_names = model_names or prediction_config.get('model_names', ['XGBRanker', 'LGBMRanker'])
            
            if not self._check_prediction_data(segments):
                return False
            
            results = self.model_predictor.predict_all_segments()
            
            success = results is not None and len(results) > 0
            if success:
                self.logger.info(f"✓ 模型预测完成，记录数: {len(results)}")
            else:
                self.logger.error("✗ 模型预测失败")
            
            return success
        except Exception as e:
            self.logger.error(f"模型预测异常: {e}")
            return False
    
    def run_full_pipeline(self, skip_data: bool = False, 
                         skip_training: bool = False, 
                         skip_prediction: bool = False) -> bool:
        """执行完整流水线"""
        self.logger.info("开始完整流水线执行")
        
        start_time = datetime.now()
        pipeline_config = self.config.get('pipeline', {})
        
        try:
            # 数据处理
            if not skip_data and pipeline_config.get('run_data_processing', True):
                if not self.run_data_processing():
                    self.logger.error("数据处理失败")
                    return False
            else:
                self.logger.info("跳过数据处理")
            
            # 模型训练
            if not skip_training and pipeline_config.get('run_training', True):
                if not self.run_model_training():
                    self.logger.error("模型训练失败")
                    return False
            else:
                self.logger.info("跳过模型训练")
            
            # 模型预测
            if not skip_prediction and pipeline_config.get('run_prediction', True):
                if not self.run_model_prediction():
                    self.logger.error("模型预测失败")
                    return False
            else:
                self.logger.info("跳过模型预测")
            
            total_time = datetime.now() - start_time
            self.logger.info(f"✓ 完整流水线执行成功，耗时: {total_time}")
            return True
            
        except Exception as e:
            self.logger.error(f"流水线执行异常: {e}")
            return False
    
    def _check_training_data(self, segments: List[int]) -> bool:
        """检查训练数据"""
        train_path = Path(self.config['paths']['model_input_dir']) / "train"
        
        for segment in segments:
            file_path = train_path / f"train_segment_{segment}.parquet"
            if not file_path.exists():
                self.logger.error(f"训练数据不存在: {file_path}")
                return False
        
        self.logger.info(f"✓ 训练数据检查通过: {segments}")
        return True
    
    def _check_prediction_data(self, segments: List[int]) -> bool:
        """检查预测数据"""
        test_path = Path(self.config['paths']['model_input_dir']) / "test"
        model_path = Path(self.config['paths']['model_save_dir'])
        
        # 检查测试数据
        for segment in segments:
            file_path = test_path / f"test_segment_{segment}.parquet"
            if not file_path.exists():
                self.logger.error(f"测试数据不存在: {file_path}")
                return False
        
        # 检查模型文件
        model_files = list(model_path.glob("segment_*/*.pkl"))
        if not model_files:
            self.logger.error("没有找到训练好的模型")
            return False
        
        self.logger.info(f"✓ 预测数据检查通过: {segments}")
        return True
