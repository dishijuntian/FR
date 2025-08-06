"""
航班排名系统核心控制器 - 重构版
专注于流水线协调，移除重复的配置和检查逻辑
"""

import os
import sys
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path

# 修复模块导入路径
current_path = Path(__file__).parent
project_root = current_path.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# 修复后的导入
try:
    from src.data.DataProcessor import DataProcessor
    from src.model.Trainer import FlightRankingTrainer
    from src.model.Predictor import FlightRankingPredictor
except ImportError:
    try:
        from data.DataProcessor import DataProcessor
        from model.Trainer import FlightRankingTrainer
        from model.Predictor import FlightRankingPredictor
    except ImportError:
        print("错误: 无法导入必要模块")
        print("请确保以下文件存在:")
        print("- src/data/DataProcessor.py")
        print("- src/model/Trainer.py") 
        print("- src/model/Predictor.py")
        raise

# 简单的工具函数
def setup_logger(name):
    import logging
    return logging.getLogger(name)


class FlightRankingCore:
    """航班排名系统核心控制器 - 重构版"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.project_root = project_root
        
        # 基础配置
        self.use_full_data = config.get('training', {}).get('use_full_data', False)
        
        self._setup_environment()
        self._setup_logging()
        self.logger = setup_logger(__name__)
        self._init_components()
        
        self.logger.info("核心控制器初始化完成")
    
    def _setup_environment(self):
        """设置工作环境"""
        os.chdir(self.project_root)
        # 确保必要目录存在
        for path_key in ['data_dir', 'model_save_dir', 'output_dir', 'log_dir']:
            path = self.config.get('paths', {}).get(path_key)
            if path:
                Path(path).mkdir(parents=True, exist_ok=True)
    
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
        # 数据处理器
        self.data_processor = DataProcessor(
            base_dir=self.config['paths']['data_dir'],
            chunk_size=self.config.get('data_processing', {}).get('chunk_size', 200000),
            n_processes=self.config.get('data_processing', {}).get('n_processes'),
            logger=self.logger,
            config=self.config.get('data_processing', {})
        )
        
        # 模型训练器和预测器
        self.model_trainer = FlightRankingTrainer(self.config, self.logger)
        self.model_predictor = FlightRankingPredictor(self.config, self.logger)
        
        # 显示模式信息
        mode = "全量数据" if self.use_full_data else "分段数据"
        gpu_mode = "GPU" if self.config.get('training', {}).get('use_gpu') else "CPU"
        self.logger.info(f"运行模式: {mode} + {gpu_mode}")
    
    def run_data_processing(self, force: bool = False) -> bool:
        """执行数据处理"""
        self.logger.info("开始数据处理")
        
        try:
            force = force or self.config.get('data_processing', {}).get('force_reprocess', False)
            success = self.data_processor.process_pipeline(force=force)
            
            if success:
                self.logger.info("✓ 数据处理完成")
            else:
                self.logger.error("✗ 数据处理失败")
            
            return success
        except Exception as e:
            self.logger.error(f"数据处理异常: {e}")
            return False
    
    def run_model_training(self) -> bool:
        """执行模型训练"""
        self.logger.info("开始模型训练")
        
        try:
            # 检查数据是否存在
            if not self._check_training_data():
                return False
            
            results = self.model_trainer.train_all_segments()
            success = len(results) > 0
            
            if success:
                mode = "全量数据" if self.use_full_data else "分段"
                self.logger.info(f"✓ {mode}模型训练完成")
                self._log_training_stats(results)
            else:
                self.logger.error("✗ 模型训练失败")
            
            return success
        except Exception as e:
            self.logger.error(f"模型训练异常: {e}")
            return False
    
    def run_model_prediction(self) -> bool:
        """执行模型预测"""
        self.logger.info("开始模型预测")
        
        try:
            if not self._check_prediction_data():
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
    
    def run_full_pipeline(self, **kwargs) -> bool:
        """执行完整流水线"""
        self.logger.info("开始完整流水线执行")
        start_time = datetime.now()
        
        try:
            pipeline_config = self.config.get('pipeline', {})
            
            # 执行各阶段
            if pipeline_config.get('run_data_processing', True):
                if not self.run_data_processing():
                    return False
            
            if pipeline_config.get('run_training', True):
                if not self.run_model_training():
                    return False
            
            if pipeline_config.get('run_prediction', True):
                if not self.run_model_prediction():
                    return False
            
            total_time = datetime.now() - start_time
            self.logger.info(f"✓ 完整流水线执行成功，耗时: {total_time}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"流水线执行异常: {e}")
            return False
    
    def _check_training_data(self) -> bool:
        """检查训练数据"""
        if self.use_full_data:
            # 检查全量数据
            data_path = Path(self.config['paths']['model_input_dir'])
            train_dir = data_path / "train"
            
            if train_dir.exists() and list(train_dir.glob("*.parquet")):
                return True
            
            # 检查其他可能的文件
            possible_files = ["train.parquet", "training_data.parquet"]
            if any((data_path / f).exists() for f in possible_files):
                return True
            
            self.logger.error("未找到全量训练数据")
            return False
        else:
            # 检查分段数据
            train_path = Path(self.config['paths']['model_input_dir']) / "train"
            segments = self.config.get('training', {}).get('segments', [0, 1, 2])
            
            for segment in segments:
                file_path = train_path / f"train_segment_{segment}.parquet"
                if not file_path.exists():
                    self.logger.error(f"训练数据不存在: {file_path}")
                    return False
            
            self.logger.info(f"✓ 分段训练数据检查通过: {segments}")
            return True
    
    def _check_prediction_data(self) -> bool:
        """检查预测数据"""
        test_path = Path(self.config['paths']['model_input_dir']) / "test"
        segments = self.config.get('prediction', {}).get('segments', [0, 1, 2])
        
        # 检查测试数据
        for segment in segments:
            file_path = test_path / f"test_segment_{segment}.parquet"
            if not file_path.exists():
                self.logger.error(f"测试数据不存在: {file_path}")
                return False
        
        # 检查模型文件
        model_path = Path(self.config['paths']['model_save_dir'])
        if self.use_full_data:
            model_dir = model_path / "full_data"
        else:
            model_dir = model_path
        
        if not any(model_dir.glob("**/*.pkl")):
            self.logger.error("没有找到训练好的模型")
            return False
        
        return True
    
    def _log_training_stats(self, results: Dict):
        """记录训练统计信息"""
        if self.use_full_data and 'full_data' in results:
            result = results['full_data']
            self.logger.info(f"训练统计: {result.get('n_rankers', 0)} rankers, "
                           f"{result.get('training_time', 0):.1f}s, "
                           f"{len(result.get('models', {}))} 模型")
        else:
            total_time = sum(r.get('training_time', 0) for r in results.values())
            total_models = sum(len(r.get('models', {})) for r in results.values())
            self.logger.info(f"训练统计: {len(results)} 段, "
                           f"{total_time:.1f}s, {total_models} 模型")
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        try:
            import torch
            gpu_available = torch.cuda.is_available()
        except ImportError:
            gpu_available = False
        
        return {
            'core_version': 'refactored_v1.0',
            'training_mode': 'full_data' if self.use_full_data else 'segments',
            'system_info': {
                'gpu_available': gpu_available,
            },
            'supported_models': self.config.get('training', {}).get('model_names', [])
        }