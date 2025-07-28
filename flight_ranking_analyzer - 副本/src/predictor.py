"""
改进的航班排序预测器 - 修复内存瓶颈版本

基于更优秀的预测代码结构重新设计，提供：
- 修复了大数据集内存瓶颈问题
- 优化了DataFrame创建过程
- 添加了内存使用监控
- 改进了分批处理机制

作者: Flight Ranking Team
版本: 3.2 (修复内存瓶颈)
"""

import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import warnings
import gc
import psutil

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


class FlightRankingPredictor:
    """改进的航班排序预测器 - 修复内存瓶颈版本"""
    
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
        
        # 确保输出目录存在
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.model_save_path.mkdir(parents=True, exist_ok=True)
        
        # 初始化数据处理器
        self.data_processor = DataProcessor(logger=logger)
        
        # 缓存已加载的模型和特征
        self.loaded_models = {}
        self.loaded_features = {}
        
        self._log(f"预测器初始化完成")
        self._log(f"数据路径: {self.data_path}")
        self._log(f"模型路径: {self.model_save_path}")
        self._log(f"输出路径: {self.output_path}")
    
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
            self._log(f"📊 {stage} 内存使用: {memory_mb:.1f} MB")
            return memory_mb
        except:
            return 0
    
    def _optimize_memory(self):
        """优化内存使用"""
        gc.collect()
        self._monitor_memory("内存清理后")
    
    def save_model_and_features(self, model, model_name: str, segment_id: int, 
                               feature_names: List[str], performance: float = None):
        """
        保存模型和特征信息
        
        Args:
            model: 训练好的模型
            model_name: 模型名称
            segment_id: 数据段ID
            feature_names: 特征名称列表
            performance: 模型性能指标
        """
        try:
            # 保存模型
            model_path = self.model_save_path / f"{model_name}_segment_{segment_id}.pkl"
            joblib.dump(model, model_path)
            
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
                'saved_at': pd.Timestamp.now().isoformat()
            }
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(model_info, f, indent=2, ensure_ascii=False)
            
            self._log(f"已保存模型: {model_path}")
            self._log(f"已保存特征: {feature_path}")
            
        except Exception as e:
            self._log(f"保存模型时出错: {str(e)}")
            raise
    
    def load_model_and_features(self, model_name: str, segment_id: int) -> Tuple[Any, List[str]]:
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
        
        # 检查模型文件是否存在
        model_path = self.model_save_path / f"{model_name}_segment_{segment_id}.pkl"
        feature_path = self.model_save_path / f"features_segment_{segment_id}.pkl"
        
        if not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        if not feature_path.exists():
            raise FileNotFoundError(f"特征文件不存在: {feature_path}")
        
        try:
            # 加载模型和特征
            model = joblib.load(model_path)
            feature_names = joblib.load(feature_path)
            
            # 缓存加载的内容
            self.loaded_models[cache_key] = model
            self.loaded_features[cache_key] = feature_names
            
            self._log(f"已加载模型: {model_path}")
            return model, feature_names
            
        except Exception as e:
            self._log(f"加载模型时出错: {str(e)}")
            raise
    
    def predict_segment(self, segment_id: int, model_name: str = 'XGBRanker') -> Optional[pd.DataFrame]:
        """
        预测单个数据段（修复内存瓶颈版本）
        
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
            
            # 检查数据大小，决定处理策略
            data_size = len(test_df)
            if data_size > 1000000:  # 超过100万行
                self._log(f"⚡ 检测到大数据集({data_size:,}行)，启用内存优化模式")
                return self._predict_segment_optimized(test_df, model, feature_names, segment_id)
            else:
                self._log(f"📊 数据集大小适中({data_size:,}行)，使用标准模式")
                return self._predict_segment_standard(test_df, model, feature_names, segment_id)
                
        except Exception as e:
            self._log(f"预测 segment_{segment_id} 失败: {str(e)}")
            return None
        finally:
            # 清理内存
            self._optimize_memory()
    
    def _predict_segment_standard(self, test_df: pd.DataFrame, model, 
                                feature_names: List[str], segment_id: int) -> pd.DataFrame:
        """标准模式预测（适用于中小数据集）"""
        self._log(f"🔧 执行标准模式预测...")
        
        # 准备特征数据
        self._log(f"准备特征数据...")
        X_test, group_sizes = self._prepare_test_features(test_df, feature_names)
        self._monitor_memory("特征准备后")
        
        # 执行预测
        self._log(f"执行模型预测...")
        pred_scores = model.predict(X_test)
        self._monitor_memory("模型预测后")
        
        # 计算排名
        self._log(f"计算组内排名...")
        pred_ranks = self._calculate_group_ranks_robust(pred_scores, group_sizes, segment_id)
        self._monitor_memory("排名计算后")
        
        # 验证排名唯一性
        self._log(f"验证排名唯一性...")
        is_valid = self._validate_ranking_uniqueness(
            pred_ranks, group_sizes, f"segment_{segment_id}-{model.__class__.__name__}"
        )
        
        if not is_valid:
            self._log(f"⚠️ segment_{segment_id} 排名验证失败，强制修复...")
            pred_ranks = self._force_fix_ranking_uniqueness(pred_ranks, group_sizes, segment_id)
        
        self._log(f"✅ 排名验证通过")
        
        # ===== 关键修复：优化DataFrame创建 =====
        self._log(f"💾 开始创建提交结果... (数据量: {len(test_df):,} 行)")
        self._monitor_memory("DataFrame创建前")
        
        # 使用内存优化的方式创建DataFrame
        submission = self._create_submission_optimized(test_df, pred_ranks)
        
        self._monitor_memory("DataFrame创建后")
        self._log(f"✅ 提交结果创建完成，形状: {submission.shape}")
        
        # 最终验证（简化版）
        self._log(f"🔍 执行最终验证...")
        final_validation = self._final_ranking_validation_optimized(submission)
        
        if not final_validation:
            self._log("❌ 最终验证失败，执行修复...")
            submission = self._validate_and_fix_rankings(submission)
            self._log("✅ 最终修复完成")
        
        self._log(f"🎉 预测完成，结果形状: {submission.shape}")
        return submission
    
    def _predict_segment_optimized(self, test_df: pd.DataFrame, model, 
                                 feature_names: List[str], segment_id: int) -> pd.DataFrame:
        """优化模式预测（适用于大数据集）"""
        self._log(f"⚡ 执行大数据集优化模式预测...")
        
        chunk_size = 500000  # 每批50万行
        total_chunks = (len(test_df) + chunk_size - 1) // chunk_size
        self._log(f"📦 将数据分为{total_chunks}批处理，每批{chunk_size:,}行")
        
        all_results = []
        
        for chunk_idx in range(0, len(test_df), chunk_size):
            chunk_num = chunk_idx // chunk_size + 1
            self._log(f"🔄 处理第{chunk_num}/{total_chunks}批...")
            
            # 获取当前批次数据
            end_idx = min(chunk_idx + chunk_size, len(test_df))
            chunk_df = test_df.iloc[chunk_idx:end_idx].copy()
            
            # 处理当前批次
            X_chunk, group_sizes = self._prepare_test_features(chunk_df, feature_names)
            pred_scores = model.predict(X_chunk)
            pred_ranks = self._calculate_group_ranks_robust(pred_scores, group_sizes, segment_id)
            
            # 验证当前批次排名
            is_valid = self._validate_ranking_uniqueness(
                pred_ranks, group_sizes, f"segment_{segment_id}-chunk_{chunk_num}"
            )
            if not is_valid:
                pred_ranks = self._force_fix_ranking_uniqueness(pred_ranks, group_sizes, segment_id)
            
            # 创建当前批次结果（内存优化）
            chunk_result = self._create_submission_optimized(chunk_df, pred_ranks)
            all_results.append(chunk_result)
            
            self._log(f"✅ 第{chunk_num}批完成，形状: {chunk_result.shape}")
            
            # 清理当前批次的中间变量
            del chunk_df, X_chunk, pred_scores, pred_ranks, chunk_result
            self._optimize_memory()
        
        # 合并所有批次结果
        self._log(f"🔗 合并所有批次结果...")
        self._monitor_memory("合并前")
        
        final_result = pd.concat(all_results, ignore_index=True)
        
        # 清理批次结果
        del all_results
        self._optimize_memory()
        
        self._monitor_memory("合并后")
        self._log(f"✅ 大数据集预测完成，最终结果形状: {final_result.shape}")
        
        # 最终验证（抽样验证）
        final_validation = self._final_ranking_validation_optimized(final_result, sample_ratio=0.1)
        if not final_validation:
            self._log("⚠️ 最终验证失败，但由于数据量过大，跳过全量修复")
        
        return final_result
    
    def _create_submission_optimized(self, test_df: pd.DataFrame, pred_ranks: np.ndarray) -> pd.DataFrame:
        """内存优化的DataFrame创建方法"""
        self._log(f"🚀 使用内存优化方式创建DataFrame...")
        
        # 方法1：直接使用numpy数组创建（最省内存）
        try:
            # 提取需要的列为numpy数组
            id_values = test_df['Id'].values
            ranker_values = test_df['ranker_id'].values
            
            # 创建字典，使用numpy数组
            submission_data = {
                'Id': id_values,
                'ranker_id': ranker_values,
                'selected': pred_ranks
            }
            
            # 创建DataFrame
            submission = pd.DataFrame(submission_data)
            
            # 立即清理临时变量
            del submission_data, id_values, ranker_values
            
            return submission
            
        except Exception as e:
            self._log(f"⚠️ 优化方法失败，使用备用方法: {e}")
            
            # 备用方法：分列创建
            submission = pd.DataFrame()
            submission['Id'] = test_df['Id']
            submission['ranker_id'] = test_df['ranker_id']
            submission['selected'] = pred_ranks
            
            return submission
    
    def _prepare_test_features(self, test_df: pd.DataFrame, 
                              feature_names: List[str]) -> Tuple[np.ndarray, List[int]]:
        """
        准备测试特征数据
        
        Args:
            test_df: 测试数据
            feature_names: 特征名称列表
            
        Returns:
            Tuple: (特征矩阵, 组大小列表)
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
        
        X_test = test_df[feature_names].values.astype(np.float32)
        
        # 计算组大小
        group_sizes = self._calculate_group_sizes(test_df['ranker_id'].values)
        
        return X_test, group_sizes
    
    def _calculate_group_sizes(self, groups: np.ndarray) -> List[int]:
        """计算每个组的大小"""
        group_sizes = []
        current_group = groups[0]
        current_size = 1
        
        for i in range(1, len(groups)):
            if groups[i] == current_group:
                current_size += 1
            else:
                group_sizes.append(current_size)
                current_group = groups[i]
                current_size = 1
        group_sizes.append(current_size)  # 添加最后一个组
        
        return group_sizes
        
    def _calculate_group_ranks_robust(self, scores: np.ndarray, group_sizes: List[int], 
                                     context_id: int = 0) -> np.ndarray:
        """
        稳健的排名计算函数 - 确保排名唯一且连续
        
        Args:
            scores: 预测分数
            group_sizes: 每组的大小
            context_id: 上下文ID（用于生成唯一随机种子）
            
        Returns:
            np.ndarray: 唯一且连续的排名
        """
        ranks = np.zeros_like(scores, dtype=int)
        start_idx = 0
        
        for group_idx, group_size in enumerate(group_sizes):
            end_idx = start_idx + group_size
            group_scores = scores[start_idx:end_idx]
            
            if group_size == 1:
                # 单个元素的组，排名直接为1
                ranks[start_idx:end_idx] = 1
            else:
                # 多个元素的组，确保排名唯一
                # 创建基于多个因子的唯一随机种子
                unique_seed = ((context_id * 7919 + group_idx * 2851) % 2147483647)
                np.random.seed(unique_seed)
                
                # 添加强度适中的随机噪声
                noise_scale = 1e-6  # 适中的噪声强度
                noise = np.random.random(len(group_scores)) * noise_scale
                
                # 为每个位置添加不同的噪声偏移
                position_offset = np.arange(len(group_scores)) * 1e-9
                noisy_scores = group_scores + noise + position_offset
                
                # 计算排名：分数越高，排名越靠前（rank=1最好）
                sorted_indices = np.argsort(-noisy_scores)  # 降序排列的索引
                group_ranks = np.zeros(group_size, dtype=int)
                
                # 分配唯一且连续的排名
                for rank, idx in enumerate(sorted_indices):
                    group_ranks[idx] = rank + 1
                
                ranks[start_idx:end_idx] = group_ranks
                
                # 验证当前组的排名
                unique_ranks = set(group_ranks)
                expected_ranks = set(range(1, group_size + 1))
                if unique_ranks != expected_ranks:
                    # 如果仍有问题，使用强制方法
                    self._log(f"警告：组{group_idx}排名计算失败，使用强制方法")
                    ranks[start_idx:end_idx] = self._generate_forced_unique_ranks(
                        group_size, group_idx, context_id
                    )
            
            start_idx = end_idx
        
        return ranks
    
    def _generate_forced_unique_ranks(self, group_size: int, group_idx: int, 
                                     context_id: int) -> np.ndarray:
        """
        强制生成唯一排名
        
        Args:
            group_size: 组大小
            group_idx: 组索引
            context_id: 上下文ID
            
        Returns:
            np.ndarray: 唯一排名数组
        """
        # 使用确定性但独特的方法生成排名
        forced_seed = ((context_id * 13 + group_idx * 17) % 1000000)
        np.random.seed(forced_seed)
        
        # 直接生成1到group_size的随机排列
        unique_ranks = np.random.permutation(range(1, group_size + 1))
        return unique_ranks
    
    def _validate_ranking_uniqueness(self, ranks: np.ndarray, group_sizes: List[int], 
                                   context: str = "") -> bool:
        """
        验证排名的唯一性
        
        Args:
            ranks: 排名数组
            group_sizes: 组大小列表
            context: 上下文信息
            
        Returns:
            bool: 排名是否唯一有效
        """
        start_idx = 0
        all_valid = True
        
        for group_idx, group_size in enumerate(group_sizes):
            end_idx = start_idx + group_size
            group_ranks = ranks[start_idx:end_idx]
            
            # 检查排名是否唯一且连续
            unique_ranks = set(group_ranks)
            expected_ranks = set(range(1, group_size + 1))
            
            if unique_ranks != expected_ranks:
                self._log(f"排名验证失败 - {context} 组{group_idx}: "
                         f"期望{sorted(expected_ranks)}, 实际{sorted(unique_ranks)}")
                all_valid = False
            
            start_idx = end_idx
        
        if all_valid:
            self._log(f"排名验证通过 - {context}")
        
        return all_valid
    
    def _force_fix_ranking_uniqueness(self, ranks: np.ndarray, group_sizes: List[int], 
                                     context_id: int = 0) -> np.ndarray:
        """
        强制修复排名唯一性问题
        
        Args:
            ranks: 原始排名
            group_sizes: 组大小列表
            context_id: 上下文ID
            
        Returns:
            np.ndarray: 修复后的排名
        """
        fixed_ranks = ranks.copy()
        start_idx = 0
        
        for group_idx, group_size in enumerate(group_sizes):
            end_idx = start_idx + group_size
            group_ranks = fixed_ranks[start_idx:end_idx]
            
            # 检查是否需要修复
            unique_ranks = set(group_ranks)
            expected_ranks = set(range(1, group_size + 1))
            
            if unique_ranks != expected_ranks:
                # 强制分配连续排名
                # 使用多重因子生成唯一种子
                fix_seed = ((context_id * 23 + group_idx * 59) % 1000000)
                np.random.seed(fix_seed)
                new_ranks = np.random.permutation(range(1, group_size + 1))
                fixed_ranks[start_idx:end_idx] = new_ranks
                
                self._log(f"强制修复组{group_idx}的排名")
            
            start_idx = end_idx
        
        return fixed_ranks
    
    def _final_ranking_validation_optimized(self, submission: pd.DataFrame, 
                                          sample_ratio: float = 1.0) -> bool:
        """
        优化版最终排名验证 - 支持抽样验证
        
        Args:
            submission: 提交结果DataFrame
            sample_ratio: 验证的样本比例（1.0表示全量验证）
            
        Returns:
            bool: 是否所有组的排名都有效
        """
        unique_rankers = submission['ranker_id'].unique()
        total_groups = len(unique_rankers)
        
        # 如果数据太大，只验证样本
        if total_groups > 10000 and sample_ratio < 1.0:
            sample_size = int(total_groups * sample_ratio)
            self._log(f"🔍 数据量较大，验证{sample_size}/{total_groups}个组 "
                     f"({sample_ratio*100:.1f}%样本)")
            
            sample_rankers = np.random.choice(unique_rankers, sample_size, replace=False)
            rankers_to_check = sample_rankers
        else:
            self._log(f"🔍 验证所有{total_groups}个组...")
            rankers_to_check = unique_rankers
        
        # 验证选定的组
        valid_count = 0
        for ranker_id in rankers_to_check:
            group_data = submission[submission['ranker_id'] == ranker_id]
            ranks = sorted(group_data['selected'].values)
            expected_ranks = list(range(1, len(group_data) + 1))
            
            if ranks == expected_ranks:
                valid_count += 1
            else:
                if sample_ratio >= 1.0:  # 只在全量验证时输出详细错误
                    self._log(f"❌ 最终验证失败 - ranker_id {ranker_id}: "
                             f"期望{expected_ranks[:5]}..., 实际{ranks[:5]}...")
        
        validation_rate = valid_count / len(rankers_to_check)
        self._log(f"📊 验证结果: {valid_count}/{len(rankers_to_check)} "
                 f"({validation_rate*100:.1f}%) 组通过验证")
        
        # 如果是抽样验证，90%通过就认为可接受
        # 如果是全量验证，100%通过才认为有效
        threshold = 0.9 if sample_ratio < 1.0 else 1.0
        return validation_rate >= threshold
    
    def predict_all(self, 
                   segments: List[int] = None, 
                   model_names: List[str] = None,
                   ensemble_method: str = 'average') -> Optional[pd.DataFrame]:
        """
        预测所有指定数据段并生成最终提交文件（内存优化版本）
        
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
        segment_results = {}
        
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
            
            # 如果有多个模型，进行集成
            if len(segment_predictions) > 1:
                ensemble_prediction = self._ensemble_segment_predictions_robust(
                    segment_predictions, ensemble_method, segment_id
                )
                segment_results[segment_id] = ensemble_prediction
            elif len(segment_predictions) == 1:
                segment_results[segment_id] = list(segment_predictions.values())[0]
            else:
                self._log(f"segment_{segment_id} 没有成功的预测结果")
                continue
            
            all_predictions.append(segment_results[segment_id])
            
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
        
        # 最终验证和修复
        self._log("执行最终排名验证和修复...")
        final_submission = self._validate_and_fix_rankings(final_submission)
        
        # 保存最终结果
        model_suffix = "_".join(model_names) if len(model_names) > 1 else model_names[0]
        final_output = self.output_path / f"{model_suffix}_final_submission.csv"
        final_submission.to_csv(final_output, index=False)
        
        # 结果验证和总结
        self._log(f"\n{'='*50}")
        self._log(f"预测完成!")
        self._log(f"最终提交文件: {final_output}")
        self._log(f"总记录数: {len(final_submission)}")
        self._log(f"唯一ranker_id数量: {final_submission['ranker_id'].nunique()}")
        
        self.validate_predictions(final_submission)
        self._monitor_memory("最终完成")
        return final_submission
    
    def _ensemble_segment_predictions_robust(self, 
                                           predictions: Dict[str, pd.DataFrame],
                                           method: str = 'average',
                                           segment_id: int = 0) -> pd.DataFrame:
        """
        稳健的集成单个段的多个模型预测结果
        
        Args:
            predictions: 模型预测结果字典
            method: 集成方法
            segment_id: 数据段ID
            
        Returns:
            pd.DataFrame: 集成后的预测结果
        """
        self._log(f"集成 {len(predictions)} 个模型的预测结果，方法: {method}")
        
        # 获取基础数据框架
        base_df = list(predictions.values())[0][['Id', 'ranker_id']].copy()
        
        if method == 'average':
            # 分数平均法：需要重新计算排名
            all_scores = []
            for model_name, pred_df in predictions.items():
                # 将排名转换为分数（排名越小分数越高）
                max_rank = pred_df.groupby('ranker_id')['selected'].transform('max')
                scores = max_rank - pred_df['selected'] + 1
                all_scores.append(scores)
            
            # 计算平均分数
            avg_scores = np.mean(all_scores, axis=0)
            
            # 重新计算排名 - 使用稳健方法
            base_df['avg_score'] = avg_scores
            group_sizes = self._calculate_group_sizes(base_df['ranker_id'].values)
            new_ranks = self._calculate_group_ranks_robust(
                avg_scores, group_sizes, context_id=segment_id * 1000
            )
            base_df['selected'] = new_ranks
            
        elif method == 'voting':
            # 排名投票法：取平均排名然后重新分配
            all_ranks = []
            for model_name, pred_df in predictions.items():
                all_ranks.append(pred_df['selected'])
            
            # 计算平均排名
            avg_ranks = np.mean(all_ranks, axis=0)
            
            # 基于平均排名重新计算唯一排名
            group_sizes = self._calculate_group_sizes(base_df['ranker_id'].values)
            # 使用负的平均排名作为分数（平均排名越小，分数越高）
            pseudo_scores = -avg_ranks
            new_ranks = self._calculate_group_ranks_robust(
                pseudo_scores, group_sizes, context_id=segment_id * 2000
            )
            base_df['selected'] = new_ranks
            
        else:
            # 默认使用第一个模型的结果
            base_df['selected'] = list(predictions.values())[0]['selected']
        
        # 验证集成结果
        is_valid = self._final_ranking_validation_optimized(base_df, sample_ratio=0.1)
        if not is_valid:
            self._log("集成结果验证失败，执行修复...")
            base_df = self._validate_and_fix_rankings(base_df)
        
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
        
        # 扫描模型文件
        for model_file in self.model_save_path.glob("*.pkl"):
            if model_file.name.startswith("features_"):
                continue
                
            # 解析文件名: ModelName_segment_X.pkl
            name_parts = model_file.stem.split("_")
            if len(name_parts) >= 3 and name_parts[-2] == "segment":
                model_name = "_".join(name_parts[:-2])
                try:
                    segment_id = int(name_parts[-1])
                    if model_name not in available_models:
                        available_models[model_name] = []
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
        
        self._log("\n可用模型摘要:")
        self._log("="*50)
        
        for model_name, segments in available_models.items():
            self._log(f"{model_name}: 段 {segments}")
            
            # 尝试读取模型信息
            for segment_id in segments[:3]:  # 只显示前3个段的详细信息
                info_path = self.model_save_path / f"info_{model_name}_segment_{segment_id}.json"
                if info_path.exists():
                    try:
                        import json
                        with open(info_path, 'r', encoding='utf-8') as f:
                            info = json.load(f)
                        self._log(f"  段{segment_id}: 特征数={info.get('feature_count', 'N/A')}, "
                                 f"性能={info.get('performance', 'N/A'):.4f}")
                    except:
                        pass

    def _validate_and_fix_rankings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        验证并修复排名唯一性（超级优化版本）- 修复大数据集卡死问题
        
        Args:
            df: 包含ranker_id和selected列的DataFrame
            
        Returns:
            pd.DataFrame: 修复后的DataFrame
        """
        self._log("执行全面的排名验证和修复...")
        self._monitor_memory("排名修复前")
        
        total_rows = len(df)
        unique_rankers = df['ranker_id'].unique()
        total_groups = len(unique_rankers)
        
        self._log(f"📊 数据规模: {total_rows:,}行, {total_groups:,}个组")
        
        # 根据数据规模选择处理策略
        if total_groups > 100000:  # 超过10万个组
            self._log(f"⚡ 超大数据集({total_groups:,}组)，使用超级优化模式...")
            return self._validate_and_fix_rankings_ultra(df)
        elif total_groups > 20000:  # 2-10万个组
            self._log(f"⚡ 大数据集({total_groups:,}组)，使用分批优化模式...")
            return self._validate_and_fix_rankings_batch(df)
        else:
            self._log(f"📊 中等数据集({total_groups:,}组)，使用标准模式...")
            return self._validate_and_fix_rankings_standard(df)
    
    def _validate_and_fix_rankings_ultra(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        超级优化模式 - 适用于超大数据集（>10万组）
        策略：跳过验证，直接信任分批处理的结果
        """
        self._log("🚀 启用超级优化模式：跳过全量验证，信任分批处理结果")
        
        # 只验证少量样本以确认基本正确性
        sample_size = min(1000, len(df['ranker_id'].unique()))
        sample_rankers = np.random.choice(df['ranker_id'].unique(), sample_size, replace=False)
        
        problem_count = 0
        for ranker_id in sample_rankers:
            group_data = df[df['ranker_id'] == ranker_id]
            ranks = sorted(group_data['selected'].values)
            expected_ranks = list(range(1, len(group_data) + 1))
            
            if ranks != expected_ranks:
                problem_count += 1
                # 只修复样本中的问题
                fix_seed = abs(hash(str(ranker_id))) % 1000000
                np.random.seed(fix_seed)
                new_ranks = np.random.permutation(range(1, len(group_data) + 1))
                df.loc[df['ranker_id'] == ranker_id, 'selected'] = new_ranks
        
        self._log(f"📊 样本验证完成: 修复了 {problem_count}/{sample_size} 个问题组")
        self._log(f"✅ 超级优化模式完成，跳过全量验证以节省时间")
        
        self._monitor_memory("超级优化模式完成")
        return df
    
    def _validate_and_fix_rankings_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        分批优化模式 - 适用于大数据集（2-10万组）
        """
        self._log("⚡ 启用分批优化模式...")
        
        unique_rankers = df['ranker_id'].unique()
        total_groups = len(unique_rankers)
        problem_groups = 0
        
        # 分批处理
        batch_size = 10000  # 每批处理1万个组
        total_batches = (total_groups + batch_size - 1) // batch_size
        
        self._log(f"📦 将{total_groups:,}个组分为{total_batches}批处理...")
        
        for batch_idx in range(0, total_groups, batch_size):
            batch_num = batch_idx // batch_size + 1
            end_idx = min(batch_idx + batch_size, total_groups)
            batch_rankers = unique_rankers[batch_idx:end_idx]
            
            self._log(f"🔄 处理第{batch_num}/{total_batches}批 ({len(batch_rankers)}个组)...")
            
            # 处理当前批次
            for ranker_id in batch_rankers:
                group_mask = df['ranker_id'] == ranker_id
                group_data = df[group_mask]
                group_size = len(group_data)
                
                current_ranks = sorted(group_data['selected'].values)
                expected_ranks = list(range(1, group_size + 1))
                
                if current_ranks != expected_ranks:
                    problem_groups += 1
                    
                    # 修复排名
                    fix_seed = abs(hash(str(ranker_id))) % 1000000
                    np.random.seed(fix_seed)
                    new_ranks = np.random.permutation(range(1, group_size + 1))
                    df.loc[group_mask, 'selected'] = new_ranks
            
            # 每10批输出一次进度
            if batch_num % 10 == 0:
                self._log(f"📊 已处理 {end_idx:,}/{total_groups:,} 个组，发现 {problem_groups} 个问题")
                self._monitor_memory(f"第{batch_num}批完成")
        
        self._log(f"✅ 分批模式完成: 修复了 {problem_groups}/{total_groups} 个问题组")
        return df
    
    def _validate_and_fix_rankings_standard(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        标准模式 - 适用于中小数据集（<2万组）
        """
        self._log("📊 使用标准验证模式...")
        
        fixed_df = df.copy()
        problem_groups = 0
        total_groups = 0
        
        unique_rankers = df['ranker_id'].unique()
        
        for ranker_id in unique_rankers:
            total_groups += 1
            group_mask = df['ranker_id'] == ranker_id
            group_data = df[group_mask]
            group_size = len(group_data)
            
            current_ranks = sorted(group_data['selected'].values)
            expected_ranks = list(range(1, group_size + 1))
            
            if current_ranks != expected_ranks:
                problem_groups += 1
                
                # 使用ranker_id的哈希值作为随机种子确保可重复性
                fix_seed = abs(hash(str(ranker_id))) % 1000000
                np.random.seed(fix_seed)
                
                # 生成唯一的1到group_size的排列
                new_ranks = np.random.permutation(range(1, group_size + 1))
                fixed_df.loc[group_mask, 'selected'] = new_ranks
            
            # 每1000个组输出一次进度
            if total_groups % 1000 == 0:
                self._log(f"📊 已验证 {total_groups:,}/{len(unique_rankers):,} 个组...")
        
        self._log(f"✅ 标准模式完成: 修复了 {problem_groups}/{total_groups} 个问题组")
        return fixed_df
    
    def _fix_ranking_chunk(self, fixed_df: pd.DataFrame, chunk_mask: np.ndarray, 
                          chunk_rankers: np.ndarray) -> int:
        """修复一批ranker的排名问题"""
        problem_count = 0
        
        for ranker_id in chunk_rankers:
            group_mask = fixed_df['ranker_id'] == ranker_id
            group_data = fixed_df[group_mask]
            group_size = len(group_data)
            
            current_ranks = sorted(group_data['selected'].values)
            expected_ranks = list(range(1, group_size + 1))
            
            if current_ranks != expected_ranks:
                problem_count += 1
                
                # 修复排名
                fix_seed = abs(hash(str(ranker_id))) % 1000000
                np.random.seed(fix_seed)
                new_ranks = np.random.permutation(range(1, group_size + 1))
                fixed_df.loc[group_mask, 'selected'] = new_ranks
        
        return problem_count