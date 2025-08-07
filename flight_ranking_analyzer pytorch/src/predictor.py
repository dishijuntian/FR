"""
模型预测器 - 重构版

专注于：
- 模型保存和加载
- 预测流程管理
- 结果集成

作者: Flight Ranking Team
版本: 4.0 (重构版)
"""

import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import warnings
import gc

warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ModelManager:
    """模型管理器 - 负责模型的保存和加载"""
    
    def __init__(self, model_save_path: Path):
        self.model_save_path = model_save_path
        self.model_save_path.mkdir(parents=True, exist_ok=True)
        self.loaded_models = {}
        self.loaded_features = {}
    
    def save_model(self, model, model_name: str, segment_id: int, 
                   feature_names: List[str], performance: float = None):
        """
        保存模型和相关信息
        
        Args:
            model: 训练好的模型
            model_name: 模型名称
            segment_id: 数据段ID
            feature_names: 特征名称列表
            performance: 模型性能
        """
        try:
            # 保存模型
            if isinstance(model, nn.Module):
                self._save_pytorch_model(model, model_name, segment_id)
            else:
                self._save_traditional_model(model, model_name, segment_id)
            
            # 保存特征和信息
            self._save_model_info(model_name, segment_id, feature_names, performance)
            
            print(f"已保存模型: {model_name}_segment_{segment_id}")
            
        except Exception as e:
            print(f"保存模型失败: {e}")
    
    def load_model(self, model_name: str, segment_id: int) -> Tuple[Any, List[str]]:
        """
        加载模型和特征信息
        
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
        
        # 加载特征
        feature_path = self.model_save_path / f"features_segment_{segment_id}.pkl"
        if not feature_path.exists():
            raise FileNotFoundError(f"特征文件不存在: {feature_path}")
        
        feature_names = joblib.load(feature_path)
        
        # 加载模型
        if model_name in ['NeuralRanker', 'RankNet', 'TransformerRanker']:
            model = self._load_pytorch_model(model_name, segment_id)
        else:
            model = self._load_traditional_model(model_name, segment_id)
        
        # 缓存
        self.loaded_models[cache_key] = model
        self.loaded_features[cache_key] = feature_names
        
        return model, feature_names
    
    def get_available_models(self) -> Dict[str, List[int]]:
        """获取可用的模型和对应段"""
        available_models = {}
        
        for model_file in self.model_save_path.glob("*"):
            if model_file.suffix in ['.pkl', '.pth']:
                name_parts = model_file.stem.split("_")
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
    
    def _save_pytorch_model(self, model: nn.Module, model_name: str, segment_id: int):
        """保存PyTorch模型"""
        model_path = self.model_save_path / f"{model_name}_segment_{segment_id}.pth"
        
        model_info = {
            'model_name': model_name,
            'state_dict': model.state_dict(),
            'model_params': getattr(model, 'params', {}),
            'input_dim': getattr(model, 'input_dim', None)
        }
        
        torch.save(model_info, model_path)
    
    def _save_traditional_model(self, model, model_name: str, segment_id: int):
        """保存传统模型"""
        model_path = self.model_save_path / f"{model_name}_segment_{segment_id}.pkl"
        joblib.dump(model, model_path)
    
    def _save_model_info(self, model_name: str, segment_id: int, 
                        feature_names: List[str], performance: float):
        """保存模型信息"""
        # 保存特征
        feature_path = self.model_save_path / f"features_segment_{segment_id}.pkl"
        joblib.dump(feature_names, feature_path)
        
        # 保存信息
        info_path = self.model_save_path / f"info_{model_name}_segment_{segment_id}.json"
        import json
        info = {
            'model_name': model_name,
            'segment_id': segment_id,
            'feature_count': len(feature_names),
            'performance': performance,
            'is_pytorch_model': model_name in ['NeuralRanker', 'RankNet', 'TransformerRanker']
        }
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
    
    def _load_pytorch_model(self, model_name: str, segment_id: int):
        """加载PyTorch模型"""
        model_path = self.model_save_path / f"{model_name}_segment_{segment_id}.pth"
        
        if not model_path.exists():
            raise FileNotFoundError(f"PyTorch模型文件不存在: {model_path}")
        
        model_info = torch.load(model_path, map_location=DEVICE)
        
        # 修复：将相对导入改为绝对导入
        from models import ModelFactory
        
        # 重新创建模型
        model = ModelFactory.create_model(
            model_name,
            input_dim=model_info['input_dim'],
            **model_info['model_params']
        )
        
        model.load_state_dict(model_info['state_dict'])
        model.eval()
        
        return model
    
    def _load_traditional_model(self, model_name: str, segment_id: int):
        """加载传统模型"""
        model_path = self.model_save_path / f"{model_name}_segment_{segment_id}.pkl"
        
        if not model_path.exists():
            raise FileNotFoundError(f"传统模型文件不存在: {model_path}")
        
        return joblib.load(model_path)


class PredictionPipeline:
    """预测流水线"""
    
    def __init__(self, data_processor, model_manager: ModelManager):
        self.data_processor = data_processor
        self.model_manager = model_manager
    
    def predict_segment(self, test_file: Path, segment_id: int, 
                       model_name: str) -> Optional[pd.DataFrame]:
        """
        预测单个数据段
        
        Args:
            test_file: 测试文件路径
            segment_id: 段ID
            model_name: 模型名称
            
        Returns:
            Optional[pd.DataFrame]: 预测结果
        """
        try:
            # 加载模型和特征
            model, feature_names = self.model_manager.load_model(model_name, segment_id)
            
            # 加载测试数据
            test_df = pd.read_parquet(test_file)
            
            # 准备特征（传递特征名称）
            X_test, _ = self.data_processor.prepare_test_features(test_df, feature_names)
            
            # 预测
            if isinstance(model, nn.Module):
                scores = self._predict_pytorch_model(model, X_test)
            else:
                scores = model.predict(X_test)
            
            # 分配排名
            result_df = self.data_processor.assign_rankings(test_df, scores)
            
            return result_df
            
        except Exception as e:
            print(f"预测segment_{segment_id}失败: {e}")
            return None
        finally:
            # 清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def _predict_pytorch_model(self, model: nn.Module, X_test: np.ndarray) -> np.ndarray:
        """使用PyTorch模型预测"""
        model.eval()
        predictions = []
        batch_size = 1000
        
        with torch.no_grad():
            for i in range(0, len(X_test), batch_size):
                batch = X_test[i:i+batch_size]
                batch_tensor = torch.FloatTensor(batch).to(DEVICE)
                batch_pred = model(batch_tensor)
                predictions.append(batch_pred.cpu().numpy().flatten())
        
        return np.concatenate(predictions)


class FlightRankingPredictor:
    """航班排序预测器 - 主入口类"""
    
    def __init__(self, data_path: Path, use_gpu: bool = True, 
                 model_save_path: str = "models", output_path: str = "submissions"):
        """
        初始化预测器
        
        Args:
            data_path: 数据根路径
            use_gpu: 是否使用GPU
            model_save_path: 模型保存路径
            output_path: 输出路径
        """
        self.data_path = Path(data_path)
        self.output_path = self.data_path / output_path
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # 修复：将相对导入改为绝对导入
        from data_processor import DataProcessor
        
        # 初始化组件
        self.data_processor = DataProcessor()
        self.model_manager = ModelManager(self.data_path / model_save_path)
        self.prediction_pipeline = PredictionPipeline(
            self.data_processor, self.model_manager
        )
    
    def save_model_and_features(self, model, model_name: str, segment_id: int,
                               feature_names: List[str], performance: float = None):
        """保存模型和特征"""
        self.model_manager.save_model(model, model_name, segment_id, 
                                    feature_names, performance)
    
    def predict_all(self, segments: List[int] = None, 
                   model_names: List[str] = None,
                   ensemble_method: str = 'average') -> Optional[pd.DataFrame]:
        """
        预测所有指定数据段
        
        Args:
            segments: 要预测的数据段列表
            model_names: 要使用的模型名称列表
            ensemble_method: 集成方法
            
        Returns:
            Optional[pd.DataFrame]: 最终预测结果
        """
        if segments is None:
            segments = [0, 1, 2]
        if model_names is None:
            model_names = ['XGBRanker']
        
        print(f"开始预测数据段: {segments}")
        print(f"使用模型: {model_names}")
        
        all_predictions = []
        
        for segment_id in segments:
            segment_predictions = {}
            
            # 查找测试文件
            test_file = self._find_test_file(segment_id)
            if test_file is None:
                print(f"找不到segment_{segment_id}的测试文件")
                continue
            
            # 使用每个模型预测
            for model_name in model_names:
                try:
                    prediction = self.prediction_pipeline.predict_segment(
                        test_file, segment_id, model_name
                    )
                    if prediction is not None:
                        segment_predictions[model_name] = prediction
                except Exception as e:
                    print(f"模型{model_name}预测失败: {e}")
            
            # 集成预测结果
            if segment_predictions:
                if len(segment_predictions) == 1:
                    ensemble_result = list(segment_predictions.values())[0]
                else:
                    ensemble_result = self._ensemble_predictions(
                        segment_predictions, ensemble_method
                    )
                all_predictions.append(ensemble_result)
        
        if not all_predictions:
            print("没有成功的预测结果")
            return None
        
        # 合并所有预测
        final_submission = pd.concat(all_predictions, ignore_index=True)
        final_submission = final_submission.sort_values('Id').reset_index(drop=True)
        
        # 保存结果
        model_suffix = "_".join(model_names)
        output_file = self.output_path / f"{model_suffix}_final_submission.csv"
        final_submission.to_csv(output_file, index=False)
        
        print(f"预测完成，结果保存到: {output_file}")
        return final_submission
    
    def get_available_models(self) -> Dict[str, List[int]]:
        """获取可用模型"""
        return self.model_manager.get_available_models()
    
    def _find_test_file(self, segment_id: int) -> Optional[Path]:
        """查找测试文件"""
        possible_paths = [
            self.data_path / "segmented" / "test" / f"test_segment_{segment_id}.parquet",
            self.data_path / "test" / f"test_segment_{segment_id}.parquet"
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
        
        return None
    
    def _ensemble_predictions(self, predictions: Dict[str, pd.DataFrame], 
                            method: str = 'average') -> pd.DataFrame:
        """集成多个预测结果"""
        if method == 'average':
            # 简单平均集成
            base_df = list(predictions.values())[0][['Id', 'ranker_id']].copy()
            
            # 如果有分数列，进行分数平均
            all_scores = []
            for pred_df in predictions.values():
                if 'scores' in pred_df.columns:
                    all_scores.append(pred_df['scores'].values)
            
            if all_scores:
                avg_scores = np.mean(all_scores, axis=0)
                base_df['scores'] = avg_scores
                # 重新分配排名
                result_df = self.data_processor.assign_rankings(base_df, avg_scores)
            else:
                # 使用第一个模型的结果
                result_df = list(predictions.values())[0]
            
            return result_df[['Id', 'ranker_id', 'selected']]
        
        # 其他集成方法可以在这里扩展
        return list(predictions.values())[0]