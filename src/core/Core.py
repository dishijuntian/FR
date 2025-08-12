"""
航班排名系统核心控制器 - 模块化配置版本
"""

import os
import sys
import yaml
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
import psutil
import gc

current_path = Path(__file__).parent
project_root = current_path.parent.parent
sys.path.insert(0, str(project_root / "src"))

# 只导入数据处理相关模块，避免提前导入torch
from src.data.DataProcessor import DataProcessor

class FlightRankingCore:
    """航班排名系统核心控制器 - 模块化配置版本"""
    
    def __init__(self, core_config_path: str = "config/core.yaml"):
        # 加载核心配置
        self.core_config = self._load_config(core_config_path)
        self.project_root = project_root
        
        # 加载模块配置
        self.module_configs = self._load_module_configs()
        
        self._setup_environment()
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # 延迟初始化，避免提前加载torch
        self.data_processor = None
        self.model_trainer = None
        self.model_predictor = None
        
        self._init_data_processor()
        self.logger.info("核心控制器初始化完成")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _load_module_configs(self) -> Dict[str, Dict[str, Any]]:
        """加载所有模块配置"""
        module_configs = {}
        module_config_paths = self.core_config.get('module_configs', {})
        
        for module_name, config_path in module_config_paths.items():
            try:
                module_configs[module_name] = self._load_config(config_path)
                print(f"✓ 加载模块配置: {module_name}")
            except Exception as e:
                print(f"✗ 加载模块配置失败 {module_name}: {e}")
                module_configs[module_name] = {}
        
        return module_configs
    
    def _setup_environment(self):
        """设置工作环境"""
        os.chdir(self.project_root)
        for dir_key in ['data_dir', 'model_input_dir', 'model_save_dir', 'output_dir', 'log_dir']:
            if dir_key in self.core_config['paths']:
                Path(self.core_config['paths'][dir_key]).mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self):
        """设置日志系统"""
        log_config = self.core_config.get('logging', {})
        log_file = Path(self.core_config['paths'].get('log_dir', 'logs')) / \
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
    
    def _init_data_processor(self):
        """初始化数据处理器"""
        data_config = self.module_configs.get('data_processing', {})
        
        self.data_processor = DataProcessor(
            base_dir=self.core_config['paths']['data_dir'],
            chunk_size=data_config.get('chunk_size', 300000),
            n_processes=data_config.get('n_processes'),
            logger=self.logger,
            config=data_config
        )
        self.logger.info("数据处理器初始化完成")
    
    def _init_model_trainer(self):
        """延迟初始化模型训练器"""
        if self.model_trainer is not None:
            return
            
        try:
            self.logger.info("开始初始化模型训练器...")
            
            # 检查内存情况
            self._check_memory_before_torch()
            
            from src.model.Trainer import FlightRankingTrainer
            
            # 合并核心配置和训练配置
            trainer_config = {
                'paths': self.core_config['paths'],
                'training': self.module_configs.get('training', {})
            }
            
            self.model_trainer = FlightRankingTrainer(
                config=trainer_config,
                logger=self.logger
            )
            
            self.logger.info("模型训练器初始化完成")
            
        except Exception as e:
            self.logger.error(f"模型训练器初始化失败: {e}")
            raise
    
    def _init_model_predictor(self):
        """延迟初始化模型预测器"""
        if self.model_predictor is not None:
            return
            
        try:
            self.logger.info("开始初始化模型预测器...")
            
            from src.model.Predictor import FlightRankingPredictor
            
            # 合并核心配置和预测配置
            predictor_config = {
                'paths': self.core_config['paths'],
                'prediction': self.module_configs.get('prediction', {})
            }
            
            self.model_predictor = FlightRankingPredictor(
                config=predictor_config,
                logger=self.logger
            )
            
            self.logger.info("模型预测器初始化完成")
            
        except Exception as e:
            self.logger.error(f"模型预测器初始化失败: {e}")
            raise
    
    def _check_memory_before_torch(self):
        """在加载torch前检查内存"""
        mem_info = psutil.virtual_memory()
        available_gb = mem_info.available / 1024 / 1024 / 1024
        
        min_memory = self.core_config.get('memory_monitoring', {}).get('min_available_gb', 2.0)
        
        self.logger.info(f"当前可用内存: {available_gb:.1f} GB")
        
        if available_gb < min_memory:
            self.logger.warning(f"可用内存不足{min_memory}GB，torch加载可能失败")
            
        return available_gb
    
    def _monitor_memory_usage(self, stage: str):
        """监控内存使用"""
        if not self.core_config.get('memory_monitoring', {}).get('enabled', True):
            return
            
        try:
            mem_info = psutil.virtual_memory()
            process = psutil.Process()
            process_memory = process.memory_info().rss / 1024 / 1024 / 1024
            
            self.logger.info(f"[{stage}] 系统内存使用: {mem_info.percent:.1f}% "
                           f"({mem_info.used/1024/1024/1024:.1f}/{mem_info.total/1024/1024/1024:.1f} GB)")
            self.logger.info(f"[{stage}] 进程内存使用: {process_memory:.2f} GB")
            
            warning_threshold = self.core_config.get('memory_monitoring', {}).get('warning_threshold', 90)
            if mem_info.percent > warning_threshold:
                self.logger.warning(f"系统内存使用超过{warning_threshold}%，建议释放内存")
                
        except Exception as e:
            self.logger.debug(f"内存监控失败: {e}")
    
    def run_data_processing(self, force: bool = None) -> bool:
        """执行数据处理"""
        self.logger.info("=" * 50)
        self.logger.info("开始数据处理")
        self.logger.info("=" * 50)
        
        self._monitor_memory_usage("数据处理开始")
        
        try:
            data_config = self.module_configs.get('data_processing', {})
            force = force if force is not None else data_config.get('force_reprocess', False)
            
            success = self.data_processor.process_pipeline(force=force)
            
            # 数据处理完成后清理内存
            if self.core_config.get('memory_monitoring', {}).get('cleanup_between_stages', True):
                gc.collect()
            self._monitor_memory_usage("数据处理完成")
            
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
        self.logger.info("=" * 50)
        self.logger.info("开始模型训练")
        self.logger.info("=" * 50)
        
        try:
            # 延迟初始化模型训练器
            self._init_model_trainer()
            
            self._monitor_memory_usage("模型训练开始")
            
            training_config = self.module_configs.get('training', {})
            segments = segments or training_config.get('segments', [0, 1, 2])
            
            if not self._check_training_data(segments):
                return False
            
            results = self.model_trainer.train_all_segments()
            
            # 训练完成后清理内存
            if self.core_config.get('memory_monitoring', {}).get('cleanup_between_stages', True):
                gc.collect()
            self._monitor_memory_usage("模型训练完成")
            
            success = len(results) > 0
            if success:
                self.logger.info("✓ 模型训练完成")
            else:
                self.logger.error("✗ 模型训练失败")
            
            return success
        except MemoryError as e:
            self.logger.error(f"内存不足导致训练失败: {e}")
            return False
        except Exception as e:
            self.logger.error(f"模型训练异常: {e}")
            return False
    
    def run_model_prediction(self, segments: List[int] = None) -> bool:
        """执行模型预测"""
        self.logger.info("=" * 50)
        self.logger.info("开始模型预测")
        self.logger.info("=" * 50)
        
        try:
            # 延迟初始化模型预测器
            self._init_model_predictor()
            
            self._monitor_memory_usage("模型预测开始")
            
            prediction_config = self.module_configs.get('prediction', {})
            segments = segments or prediction_config.get('segments', [0, 1, 2])
            
            if not self._check_prediction_data(segments):
                return False
            
            results = self.model_predictor.predict_all_segments()
            
            # 预测完成后清理内存
            if self.core_config.get('memory_monitoring', {}).get('cleanup_between_stages', True):
                gc.collect()
            self._monitor_memory_usage("模型预测完成")
            
            success = results is not None and len(results) > 0
            if success:
                self.logger.info(f"✓ 模型预测完成，记录数: {len(results)}")
            else:
                self.logger.error("✗ 模型预测失败")
            
            return success
        except Exception as e:
            self.logger.error(f"模型预测异常: {e}")
            return False
    
    def run_full_pipeline(self) -> bool:
        """执行完整流水线"""
        self.logger.info("=" * 60)
        self.logger.info("开始完整流水线执行")
        self.logger.info("=" * 60)
        
        start_time = datetime.now()
        pipeline_config = self.core_config.get('pipeline', {})
        
        try:
            # 数据处理
            if pipeline_config.get('run_data_processing', True):
                if not self.run_data_processing():
                    self.logger.error("数据处理失败，停止流水线")
                    return False
            else:
                self.logger.info("跳过数据处理")
            
            # 模型训练
            if pipeline_config.get('run_training', True):
                if not self.run_model_training():
                    self.logger.error("模型训练失败，停止流水线")
                    return False
            else:
                self.logger.info("跳过模型训练")
            
            # 模型预测
            if pipeline_config.get('run_prediction', True):
                if not self.run_model_prediction():
                    self.logger.error("模型预测失败，停止流水线")
                    return False
            else:
                self.logger.info("跳过模型预测")
            
            total_time = datetime.now() - start_time
            self.logger.info("=" * 60)
            self.logger.info(f"✓ 完整流水线执行成功，总耗时: {total_time}")
            self.logger.info("=" * 60)
            
            # 最终内存状态
            self._monitor_memory_usage("流水线完成")
            
            return True
            
        except Exception as e:
            self.logger.error(f"流水线执行异常: {e}")
            return False
    
    def _check_training_data(self, segments: List[int]) -> bool:
        """检查训练数据"""
        train_path = Path(self.core_config['paths']['model_input_dir']) / "train"
        
        for segment in segments:
            file_path = train_path / f"train_segment_{segment}.parquet"
            if not file_path.exists():
                self.logger.error(f"训练数据不存在: {file_path}")
                return False
        
        self.logger.info(f"✓ 训练数据检查通过: {segments}")
        return True
    
    def _check_prediction_data(self, segments: List[int]) -> bool:
        """检查预测数据"""
        test_path = Path(self.core_config['paths']['model_input_dir']) / "test"
        model_path = Path(self.core_config['paths']['model_save_dir'])
        
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
    
    def get_config(self, module_name: str = None) -> Dict[str, Any]:
        """获取配置"""
        if module_name is None:
            return self.core_config
        return self.module_configs.get(module_name, {})