"""
改进的航班排序预测器 - PyTorch版本

基于更优秀的预测代码结构重新设计，提供：
- 简化了排名处理逻辑
- 移除了复杂的分批验证处理
- 使用简单高效的排名分配方法
- 支持PyTorch模型的保存和加载

作者: Flight Ranking Team
版本: 3.3 (PyTorch版本)
"""

import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import os
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import warnings
import gc
import psutil
import pickle

# 导入原有模块
try:
    from .config import Config
    from .models import ModelFactory
    from .data_processor import DataProcessor
    from .progress_utils import progress_bar, create_data_loading_progress
except ImportError:
    from config import Config
    from models import ModelFactory
    from data_processor import DataProcessor
    from progress_utils import progress_bar, create_data_loading_progress

warnings.filterwarnings('ignore')

# PyTorch设备设置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FlightRankingPredictor:
    """改进的航班排序预测器 - PyTorch版本"""
    
    def __init__(self, 
                 data_path: str = None,
                 model_save_path: str = "models", 
                 output_path: str = "submissions",
                 use_gpu: bool = False, 
                 random_state: int = 42,
                 logger=None):
        """
        初始化预测器
        
        Args:
            data_path: 数据根目录路径
            model_save_path: 模型保存路径
            output_path: 输出路径
            use_gpu: 是否使用GPU
            random_state: 随机种子
            logger: 日志记录器
        """
        # 使用配置文件的路径或用户指定路径
        self.data_path = Path(data_path) if data_path else Path(Config.DATA_BASE_PATH)
        self.model_save_path = self.data_path / model_save_path
        self.output_path = self.data_path / output_path
        self.use_gpu = use_gpu
        self.random_state = random_state
        self.logger = logger
        
        # PyTorch设备
        self.device = DEVICE
        
        # 确保输出目录存在
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.model_save_path.mkdir(parents=True, exist_ok=True)
        
        # 初始化数据处理器
        self.data_processor = DataProcessor(logger=logger)
        
        # 缓存已加载的模型和特征
        self.loaded_models = {}
        self.loaded_features = {}
        
        self._log(f"预测器初始化完成 (PyTorch版本)")
        self._log(f"数据路径: {self.data_path}")
        self._log(f"模型路径: {self.model_save_path}")
        self._log(f"输出路径: {self.output_path}")
        self._log(f"使用设备: {self.device}")
    
    def _log(self, message: str):
        """记录日志"""
        if self.logger:
            self.logger.info(message)
        else:
            print(message)
    
    def _monitor_memory(self, stage: str = ""):
        """监控内存使用情况"""
        try:
            memory_info = psutil.Process().memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            
            # 添加GPU内存监控
            gpu_memory = ""
            if torch.cuda.is_available():
                gpu_allocated = torch.cuda.memory_allocated() / 1024 / 1024
                gpu_reserved = torch.cuda.memory_reserved() / 1024 / 1024
                gpu_memory = f", GPU已分配: {gpu_allocated:.1f}MB, GPU已保留: {gpu_reserved:.1f}MB"
            
            self._log(f"📊 {stage} 内存使用: {memory_mb:.1f}MB{gpu_memory}")
            return memory_mb
        except:
            return 0
    
    def _optimize_memory(self):
        """优化内存使用"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._monitor_memory("内存清理后")
    
    def save_model_and_features(self, model, model_name: str, segment_id: int, 
                               feature_names: List[str], performance: float = None):
        """
        保存模型和特征信息（支持PyTorch模型）
        
        Args:
            model: 训练好的模型
            model_name: 模型名称
            segment_id: 数据段ID
            feature_names: 特征名称列表
            performance: 模型性能指标
        """
        try:
            # 检查是否是PyTorch模型
            is_pytorch_model = isinstance(model, nn.Module)
            
            if is_pytorch_model:
                # 保存PyTorch模型
                model_path = self.model_save_path / f"{model_name}_segment_{segment_id}.pth"
                
                # 保存模型状态字典和结构信息
                model_info = {
                    'model_name': model_name,
                    'model_class': model.__class__.__name__,
                    'state_dict': model.state_dict(),
                    'model_params': getattr(model, 'params', {}),
                    'input_dim': getattr(model, 'input_dim', None),
                    'device': str(model.device) if hasattr(model, 'device') else str(self.device)
                }
                
                torch.save(model_info, model_path)
                self._log(f"已保存PyTorch模型: {model_path}")
                
                # 另外保存完整模型（备用）
                full_model_path = self.model_save_path / f"{model_name}_segment_{segment_id}_full.pkl"
                try:
                    # 将模型移到CPU再保存，避免设备相关问题
                    model_cpu = model.cpu()
                    joblib.dump(model_cpu, full_model_path)
                    # 恢复到原设备
                    model.to(self.device)
                    self._log(f"已保存完整PyTorch模型: {full_model_path}")
                except Exception as e:
                    self._log(f"保存完整PyTorch模型失败: {str(e)}")
            else:
                # 保存非PyTorch模型（XGBoost, LightGBM等）
                model_path = self.model_save_path / f"{model_name}_segment_{segment_id}.pkl"
                joblib.dump(model, model_path)
                self._log(f"已保存传统模型: {model_path}")
            
            # 保存特征名称
            feature_path = self.model_save_path / f"features_segment_{segment_id}.pkl"
            joblib.dump(feature_names, feature_path)
            
            # 保存模型信息
            info_path = self.model_save_path / f"info_{model_name}_segment_{segment_id}.json"
            import json
            model_info = {
                'model_name': model_name,
                'segment_id': segment_id,
                'feature_count': len(feature_names),
                'performance': performance,
                'is_pytorch_model': is_pytorch_model,
                'device_used': str(self.device),
                'saved_at': pd.Timestamp.now().isoformat()
            }
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(model_info, f, indent=2, ensure_ascii=False)
            
            self._log(f"已保存特征和信息: {feature_path}, {info_path}")
            
        except Exception as e:
            self._log(f"保存模型时出错: {str(e)}")
            raise
    
    def load_model_and_features(self, model_name: str, segment_id: int) -> Tuple[Any, List[str]]:
        """
        加载模型和特征信息（支持PyTorch模型）
        
        Args:
            model_name: 模型名称
            segment_id: 数据段ID
            
        Returns:
            Tuple: (模型对象, 特征名称列表)
        """
        cache_key = f"{model_name}_segment_{segment_id}"
        
        # 检查缓存
        if cache_key in self.loaded_models:
            return self.loaded_models[cache_key], self.loaded_features[cache_key]
        
        # 检查文件是否存在
        feature_path = self.model_save_path / f"features_segment_{segment_id}.pkl"
        info_path = self.model_save_path / f"info_{model_name}_segment_{segment_id}.json"
        
        if not feature_path.exists():
            raise FileNotFoundError(f"特征文件不存在: {feature_path}")
        
        try:
            # 加载特征名称
            feature_names = joblib.load(feature_path)
            
            # 读取模型信息
            is_pytorch_model = False
            if info_path.exists():
                import json
                with open(info_path, 'r', encoding='utf-8') as f:
                    info = json.load(f)
                is_pytorch_model = info.get('is_pytorch_model', False)
            else:
                # 通过模型名称判断
                is_pytorch_model = model_name in ['NeuralRanker', 'RankNet', 'TransformerRanker']
            
            # 加载模型
            if is_pytorch_model:
                model = self._load_pytorch_model(model_name, segment_id)
            else:
                model = self._load_traditional_model(model_name, segment_id)
            
            # 缓存加载的内容
            self.loaded_models[cache_key] = model
            self.loaded_features[cache_key] = feature_names
            
            self._log(f"已加载模型: {model_name}_segment_{segment_id}")
            return model, feature_names
            
        except Exception as e:
            self._log(f"加载模型时出错: {str(e)}")
            raise
    
    def _load_pytorch_model(self, model_name: str, segment_id: int):
        """加载PyTorch模型"""
        # 优先尝试加载.pth文件
        pth_path = self.model_save_path / f"{model_name}_segment_{segment_id}.pth"
        full_pkl_path = self.model_save_path / f"{model_name}_segment_{segment_id}_full.pkl"
        
        if pth_path.exists():
            try:
                # 加载模型信息
                model_info = torch.load(pth_path, map_location=self.device)
                
                # 重新创建模型
                input_dim = model_info.get('input_dim')
                model_params = model_info.get('model_params', {})
                
                if input_dim is None:
                    raise ValueError(f"无法获取模型输入维度")
                
                # 创建新的模型实例
                model = ModelFactory.create_model(
                    model_name,
                    use_gpu=self.use_gpu,
                    input_dim=input_dim,
                    **model_params
                )
                
                # 加载状态字典
                model.load_state_dict(model_info['state_dict'])
                model.eval()  # 设为评估模式
                
                self._log(f"已从.pth文件加载PyTorch模型: {pth_path}")
                return model
                
            except Exception as e:
                self._log(f"从.pth文件加载失败: {str(e)}，尝试备用方法")
        
        # 备用方法：加载完整模型
        if full_pkl_path.exists():
            try:
                model = joblib.load(full_pkl_path)
                model.to(self.device)
                model.eval()
                self._log(f"已从完整模型文件加载: {full_pkl_path}")
                return model
            except Exception as e:
                self._log(f"从完整模型文件加载失败: {str(e)}")
        
        raise FileNotFoundError(f"找不到PyTorch模型文件: {model_name}_segment_{segment_id}")
    
    def _load_traditional_model(self, model_name: str, segment_id: int):
        """加载传统模型（XGBoost, LightGBM等）"""
        model_path = self.model_save_path / f"{model_name}_segment_{segment_id}.pkl"
        
        if not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        model = joblib.load(model_path)
        self._log(f"已加载传统模型: {model_path}")
        return model
    
    def predict_segment(self, segment_id: int, model_name: str = 'XGBRanker') -> Optional[pd.DataFrame]:
        """
        预测单个数据段（PyTorch版本）
        
        Args:
            segment_id: 数据段ID
            model_name: 模型名称
            
        Returns:
            Optional[pd.DataFrame]: 预测结果
        """
        self._log(f"开始预测 segment_{segment_id}")
        self._monitor_memory("预测开始前")
        
        try:
            # 加载模型和特征
            self._log(f"正在加载模型和特征...")
            model, feature_names = self.load_model_and_features(model_name, segment_id)
            self._monitor_memory("模型加载后")
            
            # 查找测试数据文件
            self._log(f"正在查找测试数据文件...")
            possible_test_files = [
                self.data_path / "segmented" / "test" / f"test_segment_{segment_id}.parquet",
                self.data_path / "encode" / "test" / f"test_segment_{segment_id}_encoded.parquet",
                self.data_path / "test" / f"test_segment_{segment_id}.parquet"
            ]
            
            test_file = None
            for file_path in possible_test_files:
                if file_path.exists():
                    test_file = file_path
                    break
            
            if test_file is None:
                raise FileNotFoundError(f"找不到 segment_{segment_id} 的测试文件")
            
            # 加载测试数据
            self._log(f"正在加载测试数据: {test_file}")
            test_df = pd.read_parquet(test_file)
            self._log(f"测试数据形状: {test_df.shape}")
            self._monitor_memory("测试数据加载后")
            
            # 执行简化的预测流程
            return self._predict_segment_simplified(test_df, model, feature_names, segment_id)
                
        except Exception as e:
            self._log(f"预测 segment_{segment_id} 失败: {str(e)}")
            return None
        finally:
            # 清理内存
            self._optimize_memory()
    
    def _predict_segment_simplified(self, test_df: pd.DataFrame, model, 
                                  feature_names: List[str], segment_id: int) -> pd.DataFrame:
        """简化版预测流程（PyTorch版本）"""
        self._log(f"🚀 执行简化版预测流程...")
        
        # 准备特征数据
        self._log(f"准备特征数据...")
        X_test = self._prepare_test_features_simplified(test_df, feature_names)
        self._monitor_memory("特征准备后")
        
        # 执行预测
        self._log(f"执行模型预测...")
        if isinstance(model, nn.Module):
            # PyTorch模型预测
            pred_scores = self._predict_pytorch_model(model, X_test)
        else:
            # 传统模型预测
            pred_scores = model.predict(X_test)
        
        self._monitor_memory("模型预测后")
        
        # 创建结果DataFrame
        self._log(f"创建预测结果...")
        results = test_df[['Id', 'ranker_id']].copy()
        results['prediction_score'] = pred_scores
        
        # 使用简化的排名分配方法
        self._log(f"分配唯一排名...")
        results = self._assign_unique_rankings(results)
        
        self._log(f"✅ 简化预测完成，结果形状: {results.shape}")
        
        # 快速验证（仅抽样）
        self._quick_validation(results)
        
        return results[['Id', 'ranker_id', 'selected']]
    
    def _predict_pytorch_model(self, model: nn.Module, X_test: np.ndarray) -> np.ndarray:
        """
        使用PyTorch模型进行预测
        
        Args:
            model: PyTorch模型
            X_test: 测试特征
            
        Returns:
            np.ndarray: 预测分数
        """
        model.eval()
        predictions = []
        
        # 批量预测以避免内存问题
        batch_size = 1000
        
        with torch.no_grad():
            for i in range(0, len(X_test), batch_size):
                batch = X_test[i:i+batch_size]
                batch_tensor = torch.FloatTensor(batch).to(self.device)
                
                # 预测
                batch_pred = model(batch_tensor)
                
                # 转换为numpy
                if isinstance(batch_pred, torch.Tensor):
                    batch_pred = batch_pred.cpu().numpy().flatten()
                
                predictions.append(batch_pred)
        
        return np.concatenate(predictions)
    
    def _prepare_test_features_simplified(self, test_df: pd.DataFrame, 
                                        feature_names: List[str]) -> np.ndarray:
        """
        简化版特征准备
        
        Args:
            test_df: 测试数据
            feature_names: 特征名称列表
            
        Returns:
            np.ndarray: 特征矩阵
        """
        # 确保测试数据包含所需特征
        missing_features = set(feature_names) - set(test_df.columns)
        if missing_features:
            self._log(f"警告: 测试数据缺少特征: {missing_features}")
            # 为缺失特征添加0值
            for feature in missing_features:
                test_df[feature] = 0.0
        
        # 处理缺失值
        test_df[feature_names] = test_df[feature_names].fillna(
            test_df[feature_names].median()
        )
        
        return test_df[feature_names].values.astype(np.float32)
    
    def _assign_unique_rankings(self, results: pd.DataFrame) -> pd.DataFrame:
        """
        使用简化方法分配唯一排名
        
        Args:
            results: 包含Id, ranker_id, prediction_score的DataFrame
            
        Returns:
            pd.DataFrame: 添加了selected列的DataFrame
        """
        self._log("🎯 使用简化方法分配唯一排名...")
        
        # 确保唯一排名：使用Id作为tie-breaker
        results = results.sort_values(['ranker_id', 'prediction_score', 'Id'], 
                                    ascending=[True, False, True])
        results['selected'] = results.groupby('ranker_id').cumcount() + 1
        
        self._log("✅ 排名分配完成")
        return results
    
    def _quick_validation(self, results: pd.DataFrame, sample_size: int = 10):
        """
        快速验证排名的有效性
        
        Args:
            results: 预测结果
            sample_size: 验证样本数量
        """
        self._log("🔍 执行快速验证...")
        
        unique_rankers = results['ranker_id'].unique()
        sample_rankers = np.random.choice(
            unique_rankers, 
            min(sample_size, len(unique_rankers)), 
            replace=False
        )
        
        all_valid = True
        for ranker_id in sample_rankers:
            group_data = results[results['ranker_id'] == ranker_id]
            ranks = sorted(group_data['selected'].values)
            expected_ranks = list(range(1, len(group_data) + 1))
            
            if ranks != expected_ranks:
                self._log(f"❌ ranker_id {ranker_id} 排名无效: {ranks[:5]}...")
                all_valid = False
                break
        
        if all_valid:
            self._log(f"✅ 抽样验证通过 ({sample_size}个组)")
        else:
            self._log(f"⚠️ 抽样验证发现问题，但继续执行")
    
    def predict_all(self, 
                   segments: List[int] = None, 
                   model_names: List[str] = None,
                   ensemble_method: str = 'average') -> Optional[pd.DataFrame]:
        """
        预测所有指定数据段并生成最终提交文件（PyTorch版本）
        
        Args:
            segments: 要预测的数据段列表
            model_names: 要使用的模型名称列表
            ensemble_method: 集成方法
            
        Returns:
            Optional[pd.DataFrame]: 最终预测结果
        """
        if segments is None:
            segments = [0, 1, 2]  # 默认预测前3个段
        if model_names is None:
            model_names = ['XGBRanker']  # 默认使用XGBRanker
        
        self._log(f"开始预测所有数据段: {segments}")
        self._log(f"使用模型: {model_names}")
        self._monitor_memory("预测开始前")
        
        all_predictions = []
        
        # 对每个数据段进行预测
        for segment_id in progress_bar(segments, desc="预测数据段"):
            self._log(f"\n{'='*50}")
            segment_predictions = {}
            
            # 使用每个模型进行预测
            for model_name in model_names:
                try:
                    self._log(f"🔄 使用{model_name}预测segment_{segment_id}...")
                    prediction = self.predict_segment(segment_id, model_name)
                    if prediction is not None:
                        segment_predictions[model_name] = prediction
                        
                        # 保存单个段单个模型的预测结果
                        segment_output = self.output_path / f"{model_name}_segment_{segment_id}_prediction.csv"
                        prediction.to_csv(segment_output, index=False)
                        self._log(f"已保存预测结果: {segment_output}")
                    
                except Exception as e:
                    self._log(f"模型 {model_name} 预测 segment_{segment_id} 失败: {e}")
                    continue
                finally:
                    # 清理GPU内存
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            # 如果有多个模型，进行简化集成
            if len(segment_predictions) > 1:
                ensemble_prediction = self._ensemble_predictions_simplified(
                    segment_predictions, ensemble_method
                )
                all_predictions.append(ensemble_prediction)
            elif len(segment_predictions) == 1:
                all_predictions.append(list(segment_predictions.values())[0])
            else:
                self._log(f"segment_{segment_id} 没有成功的预测结果")
                continue
            
            # 清理当前段的内存
            self._optimize_memory()
        
        # 合并所有预测结果
        if not all_predictions:
            self._log("没有成功的预测结果")
            return None
        
        self._log(f"🔗 合并所有预测结果...")
        self._monitor_memory("合并前")
        
        final_submission = pd.concat(all_predictions, ignore_index=True)
        
        # 清理中间结果
        del all_predictions
        self._optimize_memory()
        
        # 按Id排序
        final_submission = final_submission.sort_values('Id').reset_index(drop=True)
        
        # 保存最终结果
        model_suffix = "_".join(model_names) if len(model_names) > 1 else model_names[0]
        final_output = self.output_path / f"{model_suffix}_final_submission.csv"
        final_submission.to_csv(final_output, index=False)
        
        # 结果总结
        self._log(f"\n{'='*50}")
        self._log(f"预测完成!")
        self._log(f"最终提交文件: {final_output}")
        self._log(f"总记录数: {len(final_submission)}")
        self._log(f"唯一ranker_id数量: {final_submission['ranker_id'].nunique()}")
        
        self._monitor_memory("最终完成")
        return final_submission
    
    def _ensemble_predictions_simplified(self, 
                                       predictions: Dict[str, pd.DataFrame],
                                       method: str = 'average') -> pd.DataFrame:
        """
        简化版集成多个模型预测
        
        Args:
            predictions: 模型预测结果字典
            method: 集成方法
            
        Returns:
            pd.DataFrame: 集成后的预测结果
        """
        self._log(f"集成 {len(predictions)} 个模型的预测结果，方法: {method}")
        
        if len(predictions) == 1:
            return list(predictions.values())[0]
        
        # 获取基础数据框架
        base_df = list(predictions.values())[0][['Id', 'ranker_id']].copy()
        
        # 简化的平均集成
        all_scores = []
        for model_name, pred_df in predictions.items():
            # 合并获取分数（如果有的话）
            temp_df = base_df.merge(pred_df, on=['Id', 'ranker_id'], how='left')
            if 'prediction_score' in temp_df.columns:
                all_scores.append(temp_df['prediction_score'].values)
            else:
                # 如果没有分数，用排名的倒数作为分数
                max_rank = temp_df.groupby('ranker_id')['selected'].transform('max')
                scores = max_rank - temp_df['selected'] + 1
                all_scores.append(scores.values)
        
        # 计算平均分数
        if all_scores:
            avg_scores = np.mean(all_scores, axis=0)
            base_df['prediction_score'] = avg_scores
            
            # 重新分配排名
            base_df = self._assign_unique_rankings(base_df)
        else:
            # 如果没有分数信息，使用第一个模型的结果
            first_result = list(predictions.values())[0]
            base_df = base_df.merge(first_result[['Id', 'ranker_id', 'selected']], 
                                  on=['Id', 'ranker_id'], how='left')
        
        return base_df[['Id', 'ranker_id', 'selected']]
    
    def validate_predictions(self, submission: pd.DataFrame, sample_size: int = 5):
        """
        验证预测结果的有效性
        
        Args:
            submission: 提交结果
            sample_size: 抽样验证的数量
        """
        self._log("\n验证预测结果:")
        
        unique_rankers = submission['ranker_id'].unique()
        total_groups = len(unique_rankers)
        valid_groups = 0
        
        # 检查所有组
        for ranker_id in unique_rankers:
            group_data = submission[submission['ranker_id'] == ranker_id]
            ranks = sorted(group_data['selected'].values)
            expected_ranks = list(range(1, len(group_data) + 1))
            if ranks == expected_ranks:
                valid_groups += 1
        
        self._log(f"总组数: {total_groups}")
        self._log(f"有效组数: {valid_groups}")
        self._log(f"有效率: {valid_groups/total_groups:.2%}")
        
        # 随机抽样详细检查
        if sample_size > 0:
            sample_rankers = np.random.choice(
                unique_rankers, 
                min(sample_size, len(unique_rankers)), 
                replace=False
            )
            
            self._log(f"\n抽样检查 {len(sample_rankers)} 个组:")
            for ranker_id in sample_rankers:
                group_data = submission[submission['ranker_id'] == ranker_id]
                ranks = sorted(group_data['selected'].values)
                expected_ranks = list(range(1, len(group_data) + 1))
                is_valid = ranks == expected_ranks
                self._log(f"  ranker_id {ranker_id}: 排名{'有效' if is_valid else '无效'} "
                         f"(大小: {len(group_data)})")
                if not is_valid:
                    self._log(f"    实际排名: {ranks[:10]}...")
                    self._log(f"    期望排名: {expected_ranks[:10]}...")
    
    def get_available_models(self) -> Dict[str, List[int]]:
        """
        获取可用的模型和对应的数据段
        
        Returns:
            Dict: {模型名称: [可用的段ID列表]}
        """
        available_models = {}
        
        if not self.model_save_path.exists():
            return available_models
        
        # 扫描模型文件（支持.pkl和.pth文件）
        for model_file in self.model_save_path.glob("*"):
            if model_file.name.startswith("features_") or model_file.name.startswith("info_"):
                continue
            
            # 解析文件名
            if model_file.suffix in ['.pkl', '.pth']:
                name_parts = model_file.stem.split("_")
                
                # 处理_full.pkl后缀
                if name_parts[-1] == "full":
                    name_parts = name_parts[:-1]
                
                if len(name_parts) >= 3 and name_parts[-2] == "segment":
                    model_name = "_".join(name_parts[:-2])
                    try:
                        segment_id = int(name_parts[-1])
                        if model_name not in available_models:
                            available_models[model_name] = []
                        if segment_id not in available_models[model_name]:
                            available_models[model_name].append(segment_id)
                    except ValueError:
                        continue
        
        # 排序段ID
        for model_name in available_models:
            available_models[model_name].sort()
        
        return available_models
    
    def print_model_summary(self):
        """打印模型摘要信息"""
        available_models = self.get_available_models()
        
        if not available_models:
            self._log("没有找到可用的模型")
            return
        
        self._log("\n可用模型摘要 (PyTorch版本):")
        self._log("="*50)
        
        for model_name, segments in available_models.items():
            is_pytorch = model_name in ['NeuralRanker', 'RankNet', 'TransformerRanker']
            model_type = "PyTorch" if is_pytorch else "传统"
            self._log(f"{model_name} ({model_type}): 段 {segments}")
            
            # 尝试读取模型信息
            for segment_id in segments[:3]:  # 只显示前3个段的详细信息
                info_path = self.model_save_path / f"info_{model_name}_segment_{segment_id}.json"
                if info_path.exists():
                    try:
                        import json
                        with open(info_path, 'r', encoding='utf-8') as f:
                            info = json.load(f)
                        performance = info.get('performance', 'N/A')
                        if isinstance(performance, (int, float)):
                            performance = f"{performance:.4f}"
                        self._log(f"  段{segment_id}: 特征数={info.get('feature_count', 'N/A')}, "
                                 f"性能={performance}, 设备={info.get('device_used', 'N/A')}")
                    except:
                        pass