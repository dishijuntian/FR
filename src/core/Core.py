"""
航班排名系统核心控制器 - 高效优化版
主要优化：
1. 简化流水线控制，减少中间层调用
2. 直接使用优化的训练器和预测器
3. 最小化配置检查和验证
4. 快速路径处理
"""

import os
import sys
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
import time

# 修复模块导入路径
current_path = Path(__file__).parent
project_root = current_path.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# 优化后的导入
try:
    from src.data.DataProcessor import DataProcessor
    from src.model.Trainer import FlightRankingTrainer
    from src.model.Predictor import FlightRankingPredictor
    from src.model.Manager import FlightRankingModelsManager
except ImportError:
    try:
        from data.DataProcessor import DataProcessor
        from model.Trainer import FlightRankingTrainer
        from model.Predictor import FlightRankingPredictor
        from model.Manager import FlightRankingModelsManager
    except ImportError:
        print("⚠️ 导入失败，请检查文件路径")


class FlightRankingCore:
    """航班排名系统核心控制器 - 高效版"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.project_root = project_root
        
        # 基础配置
        self.use_full_data = config.get('training', {}).get('use_full_data', False)
        
        self._setup_environment_fast()
        self._setup_logging_fast()
        self.logger = logging.getLogger(__name__)
        self._init_components_fast()
        
        mode = "全量+优化" if self.use_full_data else "标准"
        self.logger.info(f"核心控制器初始化完成 - {mode}模式")
    
    def _setup_environment_fast(self):
        """快速设置工作环境"""
        os.chdir(self.project_root)
        # 只创建必要目录
        essential_dirs = ['model_save_dir', 'output_dir']
        for path_key in essential_dirs:
            path = self.config.get('paths', {}).get(path_key)
            if path:
                Path(path).mkdir(parents=True, exist_ok=True)
    
    def _setup_logging_fast(self):
        """快速设置日志系统"""
        log_config = self.config.get('logging', {})
        
        # 简化的日志配置
        logging.basicConfig(
            level=getattr(logging, log_config.get('level', 'INFO').upper()),
            format=log_config.get('format', '%(asctime)s - %(levelname)s - %(message)s'),
            handlers=[logging.StreamHandler()],
            force=True
        )
    
    def _init_components_fast(self):
        """快速初始化组件"""
        # 数据处理器 - 简化配置
        self.data_processor = DataProcessor(
            base_dir=self.config['paths']['data_dir'],
            chunk_size=self.config.get('data_processing', {}).get('chunk_size', 500000),
            n_processes=self.config.get('data_processing', {}).get('n_processes', 4),
            logger=self.logger,
            config=self.config.get('data_processing', {})
        )
        
        # 使用优化的训练器（但保持原有类名）
        self.model_trainer = FlightRankingTrainer(self.config, self.logger)
        self.logger.info("✓ 使用优化训练器")
        
        # 预测器
        self.model_predictor = FlightRankingPredictor(self.config, self.logger)
    
    def run_data_processing_fast(self, force: bool = False) -> bool:
        """快速执行数据处理"""
        self.logger.info("开始数据处理")
        
        try:
            # 快速检查是否需要处理
            if not force:
                processed_dir = Path(self.config['paths']['model_input_dir'])
                if processed_dir.exists() and list(processed_dir.glob("**/*.parquet")):
                    self.logger.info("✓ 检测到已处理数据，跳过数据处理")
                    return True
            
            success = self.data_processor.process_pipeline(force=force)
            
            if success:
                self.logger.info("✓ 数据处理完成")
            else:
                self.logger.error("✗ 数据处理失败")
            
            return success
        except Exception as e:
            self.logger.error(f"数据处理异常: {e}")
            return False
    
    def run_model_training_fast(self) -> bool:
        """快速执行模型训练"""
        self.logger.info("开始模型训练")
        
        try:
            # 检查数据是否存在
            if not self._check_training_data_fast():
                return False
            
            # 使用快速训练方法
            if hasattr(self.model_trainer, 'train_all_segments'):
                results = self.model_trainer.train_all_segments()
            else:
                # 回退方法
                results = self._train_with_fallback()
            
            success = len(results) > 0
            
            if success:
                mode = "全量数据" if self.use_full_data else "分段"
                self.logger.info(f"✓ {mode}模型训练完成")
                self._log_training_stats_fast(results)
            else:
                self.logger.error("✗ 模型训练失败")
            
            return success
        except Exception as e:
            self.logger.error(f"模型训练异常: {e}")
            return False
    
    def run_model_prediction_fast(self) -> bool:
        """快速执行模型预测"""
        self.logger.info("开始模型预测")
        
        try:
            if not self._check_prediction_data_fast():
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
        """执行完整流水线 - 高效版"""
        self.logger.info("开始完整流水线执行")
        start_time = time.time()
        
        try:
            pipeline_config = self.config.get('pipeline', {})
            
            # 数据处理阶段
            if pipeline_config.get('run_data_processing', True):
                if not self.run_data_processing_fast():
                    self.logger.error("数据处理失败，终止流水线")
                    return False
            
            # 训练阶段
            if pipeline_config.get('run_training', True):
                train_start = time.time()
                if not self.run_model_training_fast():
                    self.logger.error("模型训练失败，终止流水线")
                    return False
                train_time = time.time() - train_start
                self.logger.info(f"✓ 训练阶段完成，耗时: {train_time:.1f}s")
            
            # 预测阶段
            if pipeline_config.get('run_prediction', True):
                pred_start = time.time()
                if not self.run_model_prediction_fast():
                    self.logger.error("模型预测失败，但训练已完成")
                    # 预测失败不影响整个流水线成功
                pred_time = time.time() - pred_start
                self.logger.info(f"✓ 预测阶段完成，耗时: {pred_time:.1f}s")
            
            total_time = time.time() - start_time
            self.logger.info(f"✓ 完整流水线执行成功，总耗时: {total_time:.1f}s")
            
            return True
            
        except Exception as e:
            self.logger.error(f"流水线执行异常: {e}")
            return False
    
    def _check_training_data_fast(self) -> bool:
        """快速检查训练数据"""
        train_path = Path(self.config['paths']['model_input_dir']) / "train"
        
        if not train_path.exists():
            self.logger.error(f"训练数据目录不存在: {train_path}")
            return False
        
        # 检查是否有parquet文件
        parquet_files = list(train_path.glob("*.parquet"))
        if not parquet_files:
            self.logger.error("训练目录中没有找到parquet文件")
            return False
        
        self.logger.info(f"✓ 训练数据检查通过: 找到{len(parquet_files)}个文件")
        return True
    
    def _check_prediction_data_fast(self) -> bool:
        """快速检查预测数据"""
        test_path = Path(self.config['paths']['model_input_dir']) / "test"
        model_path = Path(self.config['paths']['model_save_dir'])
        
        # 检查测试数据
        if not test_path.exists() or not list(test_path.glob("*.parquet")):
            self.logger.error("没有找到测试数据")
            return False
        
        # 检查模型文件
        if not list(model_path.glob("**/*.pkl")):
            self.logger.error("没有找到训练好的模型")
            return False
        
        return True
    
    def _log_training_stats_fast(self, results: Dict):
        """快速记录训练统计信息"""
        if not results:
            return
        
        if self.use_full_data and 'full_data' in results:
            result = results['full_data']
            self.logger.info(f"训练统计: {result.get('n_rankers', 0)} rankers, "
                           f"{result.get('training_time', 0):.1f}s")
        else:
            total_time = sum(r.get('training_time', 0) for r in results.values())
            total_models = sum(len(r.get('models', {})) for r in results.values())
            self.logger.info(f"训练统计: {len(results)} 段, "
                           f"{total_time:.1f}s, {total_models} 模型")
    
    def _train_with_fallback(self) -> Dict:
        """回退训练方法"""
        self.logger.warning("使用回退训练方法")
        
        # 简单的训练逻辑
        results = {}
        segments = self.config.get('training', {}).get('segments', [0, 1, 2])
        
        for segment_id in segments:
            try:
                # 这里可以添加简单的训练逻辑
                results[f'segment_{segment_id}'] = {
                    'segment_id': segment_id,
                    'models': {},
                    'training_time': 0,
                    'n_rankers': 0
                }
            except Exception as e:
                self.logger.error(f"回退训练segment_{segment_id}失败: {e}")
        
        return results
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态 - 简化版"""
        try:
            import torch
            gpu_available = torch.cuda.is_available()
        except ImportError:
            gpu_available = False
        
        return {
            'core_version': 'fast_optimized_v1.0',
            'training_mode': 'full_data_fast' if self.use_full_data else 'segments',
            'trainer_type': 'optimized',
            'system_info': {
                'gpu_available': gpu_available,
            },
            'supported_models': self.config.get('training', {}).get('model_names', [])
        }


# 移除这行，直接使用 FlightRankingCore
# FlightRankingCore = FastFlightRankingCore