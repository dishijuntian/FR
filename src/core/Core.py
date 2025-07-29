<<<<<<< HEAD
"""
航班排名系统核心控制器
"""

=======
>>>>>>> e9addff7041905ba228e279124f71a4a54b1d4e3
import os
import sys
import logging
from datetime import datetime
<<<<<<< HEAD
from typing import Dict, Any, Optional, List
from pathlib import Path

=======
from typing import Dict, Any, Optional
from pathlib import Path

# 添加项目根目录到Python路径
>>>>>>> e9addff7041905ba228e279124f71a4a54b1d4e3
current_path = Path(__file__).parent
project_root = current_path.parent.parent
sys.path.insert(0, str(project_root / "src"))

from src.data.DataProcessor import DataProcessor
from src.model.Trainer import FlightRankingTrainer
from src.model.Predictor import FlightRankingPredictor


class FlightRankingCore:
<<<<<<< HEAD
    """航班排名系统核心控制器"""
=======
    """
    航班排名系统核心控制器
    整合数据处理、模型训练和预测功能
    """
>>>>>>> e9addff7041905ba228e279124f71a4a54b1d4e3
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.project_root = project_root
        self._setup_environment()
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
<<<<<<< HEAD
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
            data_path=self.config['paths']['model_input_dir'],
            model_save_path=self.config['paths']['model_save_dir'],
            use_gpu=training_config.get('use_gpu', True),
            enable_parallel=training_config.get('enable_parallel', True),
            enable_optimization=training_config.get('enable_optimization', True),
            n_folds=training_config.get('n_folds', 5),
            random_state=training_config.get('random_state', 42),
            logger=self.logger
        )
        
        # 显示训练模式
        if self.model_trainer.use_gpu:
            self.logger.info("训练模式: GPU加速 + 串行fold训练")
        else:
            self.logger.info("训练模式: CPU多进程 + 并行fold训练")
        
        # 模型预测器
        self.model_predictor = FlightRankingPredictor(
            data_path=self.config['paths']['model_input_dir'],
            model_save_path=self.config['paths']['model_save_dir'],
            output_path=self.config['paths']['output_dir'],
            use_gpu=prediction_config.get('use_gpu', True),
            enable_parallel=prediction_config.get('enable_parallel', True),
            enable_business_rules=prediction_config.get('enable_business_rules', False),
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
            
            results = self.model_trainer.train_all_segments(
                segments=segments,
                model_names=model_names,
                model_configs=self.config.get('training', {}).get('model_configs', {})
            )
            
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
            
            results = self.model_predictor.predict_all_segments(
                segments=segments,
                model_names=model_names
            )
            
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
    
    def get_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        status = {
            'config_loaded': True,
            'data_processor': hasattr(self, 'data_processor'),
            'model_trainer': hasattr(self, 'model_trainer'),
            'model_predictor': hasattr(self, 'model_predictor'),
            'gpu_available': self.model_trainer.use_gpu if hasattr(self, 'model_trainer') else False
        }
        
        # 检查数据文件
        data_path = Path(self.config['paths']['data_dir'])
        status['train_data'] = list((data_path / "train").glob("*.parquet")) if (data_path / "train").exists() else []
        status['test_data'] = list((data_path / "test").glob("*.parquet")) if (data_path / "test").exists() else []
        
        # 检查模型文件
        model_path = Path(self.config['paths']['model_save_dir'])
        status['models'] = list(model_path.glob("segment_*/*.pkl")) if model_path.exists() else []
        
        return status
=======
        
        # 初始化组件
        self._init_data_processor()
        self._init_model_trainer()
        self._init_model_predictor()
    
    def _setup_environment(self):
        """设置工作环境"""
        # 设置工作目录到项目根目录
        os.chdir(self.project_root)
        self.logger_setup = False
        
        # 确保必要目录存在
        required_dirs = [
            self.config['paths']['data_dir'],
            self.config['paths']['model_save_dir'],
            self.config['paths']['output_dir'],
            self.config['paths']['log_dir']
        ]
        
        for dir_path in required_dirs:
            os.makedirs(dir_path, exist_ok=True)
    
    def _setup_logging(self):
        """设置日志系统"""
        if self.logger_setup:
            return
            
        log_level = getattr(logging, self.config['logging']['level'].upper())
        log_format = self.config['logging']['format']
        
        # 清除现有处理器
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        
        # 创建格式化器
        formatter = logging.Formatter(log_format)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(log_level)
        
        # 文件处理器
        log_file = os.path.join(
            self.config['paths']['log_dir'],
            f"flight_ranking_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        
        # 配置根日志器
        logging.basicConfig(
            level=log_level,
            handlers=[console_handler, file_handler],
            force=True
        )
        
        self.logger_setup = True
    
    def _init_data_processor(self):
        """初始化数据处理器"""
        data_config = self.config['data_processing']
        
        self.data_processor = DataProcessor(
            base_dir=self.config['paths']['data_dir'],
            chunk_size=data_config['chunk_size'],
            n_processes=data_config.get('n_processes'),
            logger=self.logger  # 传递核心logger
        )
    
    def _init_model_trainer(self):
        """初始化模型训练器"""
        training_config = self.config['training']
        
        self.model_trainer = FlightRankingTrainer(
            data_path=os.path.join(self.project_root, self.config['paths']['data_dir'], "segmented"),
            model_save_path=self.config['paths']['model_save_dir'],
            use_gpu=training_config['use_gpu'],
            random_state=training_config['random_state'],
            logger=self.logger  # 传递核心logger
        )
    
    def _init_model_predictor(self):
        """初始化模型预测器"""
        prediction_config = self.config['prediction']
        
        self.model_predictor = FlightRankingPredictor(
            data_path=os.path.join(self.project_root, self.config['paths']['data_dir'], "segmented"),
            model_save_path=self.config['paths']['model_save_dir'],
            output_path=self.config['paths']['output_dir'],
            use_gpu=prediction_config['use_gpu'],
            random_state=prediction_config['random_state'],
            logger=self.logger  # 传递核心logger
        )
    
    def run_data_processing(self, force: bool = None, verify: bool = None) -> bool:
        """
        执行数据处理流水线
        
        Args:
            force: 是否强制重新处理，如果为None则使用配置文件设置
            verify: 是否验证处理结果，如果为None则使用配置文件设置
            
        Returns:
            bool: 处理是否成功
        """
        self.logger.info("=" * 60)
        self.logger.info("开始数据处理阶段")
        self.logger.info("=" * 60)
        
        # 使用参数或配置文件设置
        data_config = self.config['data_processing']
        force = force if force is not None else data_config.get('force_reprocess', False)
        verify = verify if verify is not None else data_config.get('verify_results', True)
        
        try:
            success = self.data_processor.process_pipeline(force=force, verify=verify)
            
            if success:
                self.logger.info("数据处理阶段完成")
                return True
            else:
                self.logger.error("数据处理阶段失败")
                return False
                
        except Exception as e:
            self.logger.error(f"数据处理阶段异常: {str(e)}")
            return False
    
    def run_model_training(self, segments: list = None) -> bool:
        """
        执行模型训练
        
        Args:
            segments: 要训练的数据段列表，如果为None则使用配置文件设置
            
        Returns:
            bool: 训练是否成功
        """
        self.logger.info("=" * 60)
        self.logger.info("开始模型训练阶段")
        self.logger.info("=" * 60)
        
        # 使用参数或配置文件设置
        training_config = self.config['training']
        segments = segments if segments is not None else training_config['segments']
        
        try:
            results = self.model_trainer.train_all(segments=segments)
            
            if results:
                self.logger.info("模型训练阶段完成")
                return True
            else:
                self.logger.error("模型训练阶段失败")
                return False
                
        except Exception as e:
            self.logger.error(f"模型训练阶段异常: {str(e)}")
            return False
    
    def run_model_prediction(self, segments: list = None, model_name: str = None) -> bool:
        """
        执行模型预测
        
        Args:
            segments: 要预测的数据段列表，如果为None则使用配置文件设置
            model_name: 模型名称，如果为None则使用配置文件设置
            
        Returns:
            bool: 预测是否成功
        """
        self.logger.info("=" * 60)
        self.logger.info("开始模型预测阶段")
        self.logger.info("=" * 60)
        
        # 使用参数或配置文件设置
        prediction_config = self.config['prediction']
        segments = segments if segments is not None else prediction_config['segments']
        model_name = model_name if model_name is not None else prediction_config['model_name']
        
        try:
            results = self.model_predictor.predict_all(segments=segments, model_name=model_name)
            
            if results is not None:
                self.logger.info("模型预测阶段完成")
                return True
            else:
                self.logger.error("模型预测阶段失败")
                return False
                
        except Exception as e:
            self.logger.error(f"模型预测阶段异常: {str(e)}")
            return False
    
    def run_full_pipeline(self) -> bool:
        """
        执行完整的流水线：数据处理 -> 模型训练 -> 模型预测
        
        Returns:
            bool: 整个流水线是否成功
        """
        self.logger.info("=" * 80)
        self.logger.info("开始完整流水线执行")
        self.logger.info("=" * 80)
        
        pipeline_config = self.config['pipeline']
        start_time = datetime.now()
        
        try:
            # 阶段1: 数据处理
            if pipeline_config['run_data_processing']:
                if not self.run_data_processing():
                    self.logger.error("数据处理失败，终止流水线")
                    return False
            else:
                self.logger.info("跳过数据处理阶段")
            
            # 阶段2: 模型训练
            if pipeline_config['run_training']:
                if not self.run_model_training():
                    self.logger.error("模型训练失败，终止流水线")
                    return False
            else:
                self.logger.info("跳过模型训练阶段")
            
            # 阶段3: 模型预测
            if pipeline_config['run_prediction']:
                if not self.run_model_prediction():
                    self.logger.error("模型预测失败，终止流水线")
                    return False
            else:
                self.logger.info("跳过模型预测阶段")
            
            # 完成统计
            total_time = datetime.now() - start_time
            self.logger.info("=" * 80)
            self.logger.info("完整流水线执行成功!")
            self.logger.info(f"总耗时: {total_time}")
            self.logger.info("=" * 80)
            
            return True
            
        except Exception as e:
            self.logger.error(f"流水线执行异常: {str(e)}")
            return False
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """
        获取流水线各阶段的状态
        
        Returns:
            Dict: 包含各阶段状态的字典
        """
        status = {
            'data_processing': self.data_processor.get_pipeline_status(),
            'model_files': self._get_model_files_status(),
            'prediction_files': self._get_prediction_files_status()
        }
        
        return status
    
    def _get_model_files_status(self) -> Dict[str, bool]:
        """检查模型文件状态"""
        model_dir = self.config['paths']['model_save_dir']
        segments = self.config['training']['segments']
        model_names = ['XGBRanker', 'LGBMRanker']
        
        status = {}
        for segment in segments:
            for model_name in model_names:
                model_file = os.path.join(model_dir, f"{model_name}_segment_{segment}.pkl")
                status[f"{model_name}_segment_{segment}"] = os.path.exists(model_file)
        
        return status
    
    def _get_prediction_files_status(self) -> Dict[str, bool]:
        """检查预测文件状态"""
        output_dir = self.config['paths']['output_dir']
        segments = self.config['prediction']['segments']
        model_name = self.config['prediction']['model_name']
        
        status = {}
        for segment in segments:
            pred_file = os.path.join(output_dir, f"{model_name}_segment_{segment}_prediction.csv")
            status[f"segment_{segment}_prediction"] = os.path.exists(pred_file)
        
        final_file = os.path.join(output_dir, f"{model_name}_final_submission.csv")
        status['final_submission'] = os.path.exists(final_file)
        
        return status
    
    def print_status_report(self):
        """打印详细的状态报告"""
        self.logger.info("=" * 60)
        self.logger.info("流水线状态报告")
        self.logger.info("=" * 60)
        
        status = self.get_pipeline_status()
        
        # 数据处理状态
        self.logger.info("数据处理状态:")
        for data_type, state in status['data_processing'].items():
            self.logger.info(f"  {data_type}:")
            self.logger.info(f"    编码: {'✓' if state['encoded'] else '✗'}")
            self.logger.info(f"    分割: {'✓' if state['segmented'] else '✗'}")
            self.logger.info(f"    验证: {'✓' if state['verified'] else '✗'}")
        
        # 模型文件状态
        self.logger.info("\n模型文件状态:")
        for model_file, exists in status['model_files'].items():
            self.logger.info(f"  {model_file}: {'✓' if exists else '✗'}")
        
        # 预测文件状态
        self.logger.info("\n预测文件状态:")
        for pred_file, exists in status['prediction_files'].items():
            self.logger.info(f"  {pred_file}: {'✓' if exists else '✗'}")
        
        self.logger.info("=" * 60)
>>>>>>> e9addff7041905ba228e279124f71a4a54b1d4e3
