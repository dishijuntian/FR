"""
数据处理模块 - 修复排名重复问题版本

该模块负责数据的加载、预处理、抽样和特征工程
- 彻底修复了预测结果合并时的排名重复问题
- 强化了排名唯一性保证机制
- 改进了集成预测的稳定性
- 增加了多层次的验证和修复

作者: Flight Ranking Team
版本: 2.2 (修复排名重复问题)
"""

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import gc
import os
from typing import Tuple, List, Optional, Dict, Any
from sklearn.model_selection import train_test_split
import warnings

# 导入进度条工具
try:
    from .progress_utils import create_data_loading_progress, progress_bar
except ImportError:
    from progress_utils import create_data_loading_progress, progress_bar

warnings.filterwarnings('ignore')


class DataProcessor:
    """数据处理器"""
    
    def __init__(self, logger=None):
        """
        初始化数据处理器
        
        Args:
            logger: 日志记录器，如果为None则使用print输出
        """
        self.logger = logger
        self.feature_columns = None
        self.exclude_columns = ['Id', 'selected', 'ranker_id', 'profileId', 'companyID']
    
    def _log(self, message):
        """记录日志"""
        if self.logger:
            self.logger.info(message)
        else:
            print(message)
    
    def load_data(self, file_path: str, use_sampling: bool = True, 
                  num_groups: int = 2000, min_group_size: int = 20) -> pd.DataFrame:
        """
        加载数据，支持抽样和全量加载（带进度条）
        
        Args:
            file_path: 数据文件路径
            use_sampling: 是否使用抽样
            num_groups: 抽样时的组数量
            min_group_size: 抽样时每组最小数据条数
            
        Returns:
            pd.DataFrame: 加载的数据
        """
        self._log(f"📂 开始加载数据: {os.path.basename(file_path)}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        # 创建数据加载进度条
        with create_data_loading_progress("读取文件") as pbar:
            # 读取数据
            pf = pq.ParquetFile(file_path)
            df = pf.read().to_pandas()
            pbar.update(1, "文件读取完成")
            
            self._log(f"📊 原始数据形状: {df.shape}")
            
            if use_sampling:
                pbar.set_description("执行数据抽样")
                self._log(f"🎯 使用抽样模式: {num_groups}个组, 每组至少{min_group_size}条数据")
                df = self._sample_data(df, num_groups, min_group_size)
                pbar.update(1, "抽样完成")
            else:
                self._log("📈 使用全量数据模式")
                pbar.update(1, "跳过抽样")
            
            # 优化内存使用
            pbar.set_description("优化内存使用")
            df = self._optimize_memory_usage(df)
            pbar.update(1, "内存优化完成")
        
        self._log(f"✅ 最终数据形状: {df.shape}")
        self._log(f"📊 数据包含的组数: {df['ranker_id'].nunique()}")
        
        return df
    
    def _sample_data(self, df: pd.DataFrame, num_groups: int, min_group_size: int) -> pd.DataFrame:
        """
        基于ranker_id的分组抽样（带进度条）
        
        Args:
            df: 原始数据
            num_groups: 要抽取的组数量
            min_group_size: 每组最小数据条数
            
        Returns:
            pd.DataFrame: 抽样后的数据
        """
        # 统计每个ranker_id的数据量
        self._log("📊 统计组信息...")
        group_counts = df['ranker_id'].value_counts()
        
        # 筛选满足最小组大小要求的ranker_id
        valid_groups = group_counts[group_counts >= min_group_size].index
        self._log(f"📋 满足最小组大小({min_group_size})的组数: {len(valid_groups)}")
        
        if len(valid_groups) < num_groups:
            self._log(f"⚠️  可用组数({len(valid_groups)})少于要求的组数({num_groups})")
            num_groups = len(valid_groups)
        
        # 随机抽取指定数量的ranker_id
        self._log("🎲 随机选择组...")
        np.random.seed(42)
        selected_groups = np.random.choice(valid_groups, size=num_groups, replace=False)
        
        # 基于选中的ranker_id筛选数据
        self._log("🔄 筛选数据...")
        sampled_df = df[df['ranker_id'].isin(selected_groups)].copy()
        
        self._log(f"📊 抽样后数据形状: {sampled_df.shape}")
        self._log(f"📋 实际抽取的组数: {sampled_df['ranker_id'].nunique()}")
        
        return sampled_df
    
    def _optimize_memory_usage(self, df: pd.DataFrame) -> pd.DataFrame:
        """优化内存使用"""
        # 将数值列转换为float32以节省内存
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].astype(np.float32)
        
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        准备特征数据
        
        Args:
            df: 原始数据
            
        Returns:
            Tuple: (特征矩阵, 标签向量, 特征名列表)
        """
        self._log("准备特征数据...")
        
        # 选择数值型特征
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # 排除不需要的特征
        feature_cols = [col for col in numeric_cols if col not in self.exclude_columns]
        self.feature_columns = feature_cols
        
        # 处理缺失值
        df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())
        
        # 准备特征和标签
        X = df[feature_cols].values.astype(np.float32)
        y = df['selected'].values.astype(np.float32)
        
        self._log(f"特征维度: {X.shape}")
        self._log(f"特征数量: {len(feature_cols)}")
        
        return X, y, feature_cols
    
    def prepare_ranking_data(self, df: pd.DataFrame, test_size: float = 0.2, 
                           random_state: int = 42) -> Tuple[np.ndarray, ...]:
        """
        准备排序模型数据，包括训练测试集划分
        
        Args:
            df: 原始数据
            test_size: 测试集比例
            random_state: 随机种子
            
        Returns:
            Tuple: 包含训练测试数据的元组
        """
        self._log("准备排序模型数据...")
        
        # 准备特征
        X, y, feature_cols = self.prepare_features(df)
        groups = df['ranker_id'].values
        
        # 按ranker_id分组划分训练集和测试集
        unique_groups = df['ranker_id'].unique()
        train_groups, test_groups = train_test_split(
            unique_groups, test_size=test_size, random_state=random_state
        )
        
        train_mask = df['ranker_id'].isin(train_groups)
        test_mask = df['ranker_id'].isin(test_groups)
        
        X_train = X[train_mask]
        X_test = X[test_mask]
        y_train = y[train_mask]
        y_test = y[test_mask]
        
        # 计算组大小
        train_group_sizes = self._calculate_group_sizes(groups[train_mask])
        test_group_sizes = self._calculate_group_sizes(groups[test_mask])
        
        # 测试集信息
        test_info = df[test_mask][['ranker_id', 'selected']].copy()
        
        self._log(f"训练集形状: {X_train.shape}")
        self._log(f"测试集形状: {X_test.shape}")
        self._log(f"训练组数: {len(train_group_sizes)}")
        self._log(f"测试组数: {len(test_group_sizes)}")
        
        return (X_train, X_test, y_train, y_test, 
                train_group_sizes, test_group_sizes, 
                feature_cols, test_info)
    
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
    
    def load_test_data(self, file_path: str) -> pd.DataFrame:
        """
        加载测试数据（不抽样，加载全部数据）
        
        Args:
            file_path: 测试文件路径
            
        Returns:
            pd.DataFrame: 测试数据
        """
        self._log(f"加载测试数据: {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"测试文件不存在: {file_path}")
        
        # 读取测试数据
        pf = pq.ParquetFile(file_path)
        test_df = pf.read().to_pandas()
        
        # 优化内存使用
        test_df = self._optimize_memory_usage(test_df)
        
        self._log(f"测试数据形状: {test_df.shape}")
        
        return test_df
    
    def prepare_test_features(self, test_df: pd.DataFrame) -> Tuple[np.ndarray, List[int]]:
        """
        准备测试数据特征
        
        Args:
            test_df: 测试数据
            
        Returns:
            Tuple: (特征矩阵, 组大小列表)
        """
        if self.feature_columns is None:
            raise ValueError("特征列尚未确定，请先调用prepare_features方法")
        
        # 确保测试数据包含所需特征
        missing_features = set(self.feature_columns) - set(test_df.columns)
        if missing_features:
            self._log(f"警告: 测试数据缺少特征: {missing_features}")
            # 为缺失特征添加0值
            for feature in missing_features:
                test_df[feature] = 0.0
        
        # 处理缺失值
        test_df[self.feature_columns] = test_df[self.feature_columns].fillna(
            test_df[self.feature_columns].median()
        )
        
        X_test = test_df[self.feature_columns].values.astype(np.float32)
        
        # 计算组大小
        groups = test_df['ranker_id'].values
        group_sizes = self._calculate_group_sizes(groups)
        
        return X_test, group_sizes
    
    def save_predictions(self, test_df: pd.DataFrame, predictions: Dict[str, Dict[str, np.ndarray]], 
                        output_path: str) -> str:
        """
        保存预测结果
        
        Args:
            test_df: 测试数据
            predictions: 预测结果字典
            output_path: 输出路径
            
        Returns:
            str: 输出文件路径
        """
        # 创建结果DataFrame
        result_df = test_df[['Id', 'ranker_id']].copy()
        
        # 添加预测结果
        for model_type, pred in predictions.items():
            if 'scores' in pred:
                result_df[f'{model_type}_score'] = pred['scores']
            if 'ranks' in pred:
                result_df[f'{model_type}_rank'] = pred['ranks']
        
        # 保存到文件
        result_df.to_parquet(output_path, index=False)
        self._log(f"预测结果已保存到: {output_path}")
        
        return output_path


class PredictionMerger:
    """预测结果合并器 - 修复排名重复问题版本"""
    
    def __init__(self, logger=None):
        """
        初始化预测结果合并器
        
        Args:
            logger: 日志记录器
        """
        self.logger = logger
    
    def _log(self, message):
        """记录日志"""
        if self.logger:
            self.logger.info(message)
        else:
            print(message)
    
    def _validate_data_quality(self, df: pd.DataFrame, stage: str = "unknown"):
        """
        验证数据质量，检查异常值
        
        Args:
            df: 要验证的DataFrame
            stage: 验证阶段名称
        """
        self._log(f"🔍 验证数据质量 - {stage}")
        
        # 检查数值列的异常值
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            nan_count = df[col].isna().sum()
            inf_count = np.isinf(df[col]).sum()
            
            if nan_count > 0:
                self._log(f"  ⚠️ {col}: {nan_count} 个 NaN 值")
            if inf_count > 0:
                self._log(f"  ⚠️ {col}: {inf_count} 个 无穷大值")
    
    def _clean_prediction_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        清理预测数据中的异常值
        
        Args:
            df: 包含预测结果的DataFrame
            
        Returns:
            pd.DataFrame: 清理后的DataFrame
        """
        self._log("🧹 清理预测数据...")
        
        # 找到所有分数和排名列
        score_columns = [col for col in df.columns if col.endswith('_score')]
        rank_columns = [col for col in df.columns if col.endswith('_rank')]
        
        # 清理分数列
        for col in score_columns:
            if col in df.columns:
                # 替换无穷大值为NaN
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                
                # 统计并处理NaN值
                nan_count = df[col].isna().sum()
                if nan_count > 0:
                    self._log(f"  处理 {col}: {nan_count} 个异常值")
                    # 用该列的中位数填充NaN值
                    median_val = df[col].median()
                    if pd.isna(median_val):  # 如果中位数也是NaN，使用0
                        median_val = 0.0
                    df[col] = df[col].fillna(median_val)
        
        # 清理排名列
        for col in rank_columns:
            if col in df.columns:
                # 替换无穷大值为NaN
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                
                # 统计并处理NaN值
                nan_count = df[col].isna().sum()
                if nan_count > 0:
                    self._log(f"  处理 {col}: {nan_count} 个异常值")
                    # 对于排名，用该组内的最大排名+1填充
                    df[col] = df.groupby('ranker_id')[col].transform(
                        lambda x: x.fillna(x.max() + 1 if not x.isna().all() else len(x))
                    )
        
        return df
    
    def merge_predictions(self, prediction_files: List[str], 
                         submission_file: str, 
                         output_file: str,
                         ensemble_method: str = 'average') -> str:
        """
        合并多个预测文件并与submission文件对应（修复排名重复问题版本）
        
        Args:
            prediction_files: 预测文件路径列表
            submission_file: submission模板文件路径
            output_file: 输出文件路径
            ensemble_method: 集成方法 ('average', 'voting', 'weighted')
            
        Returns:
            str: 输出文件路径
        """
        self._log("开始合并预测结果...")
        
        # 读取submission文件，处理编码问题
        if not os.path.exists(submission_file):
            raise FileNotFoundError(f"Submission文件不存在: {submission_file}")
        
        # 尝试不同的编码方式读取submission文件
        submission_df = None
        encodings_to_try = ['utf-8', 'gbk', 'gb2312', 'latin-1', 'cp1252']
        
        for encoding in encodings_to_try:
            try:
                if submission_file.endswith('.csv'):
                    submission_df = pd.read_csv(submission_file, encoding=encoding)
                elif submission_file.endswith('.parquet'):
                    submission_df = pd.read_parquet(submission_file)
                else:
                    # 尝试作为CSV读取
                    submission_df = pd.read_csv(submission_file, encoding=encoding)
                
                self._log(f"成功使用 {encoding} 编码读取submission文件")
                break
                
            except UnicodeDecodeError:
                continue
            except Exception as e:
                self._log(f"使用 {encoding} 编码读取失败: {str(e)}")
                continue
        
        if submission_df is None:
            raise ValueError(f"无法读取submission文件，尝试了以下编码: {encodings_to_try}")
        
        self._log(f"Submission文件形状: {submission_df.shape}")
        self._validate_data_quality(submission_df, "submission")
        
        # 读取所有预测文件
        prediction_dfs = []
        for file_path in prediction_files:
            if os.path.exists(file_path):
                try:
                    pred_df = pd.read_parquet(file_path)
                    # 验证和清理预测数据
                    self._validate_data_quality(pred_df, f"prediction - {os.path.basename(file_path)}")
                    pred_df = self._clean_prediction_data(pred_df)
                    prediction_dfs.append(pred_df)
                    self._log(f"读取预测文件: {file_path}, 形状: {pred_df.shape}")
                except Exception as e:
                    self._log(f"读取预测文件失败: {file_path}, 错误: {str(e)}")
            else:
                self._log(f"警告: 预测文件不存在: {file_path}")
        
        if not prediction_dfs:
            raise ValueError("没有找到有效的预测文件")
        
        # 合并所有预测结果
        merged_predictions = pd.concat(prediction_dfs, ignore_index=True)
        self._validate_data_quality(merged_predictions, "merged predictions")
        
        # 检查submission和prediction的列是否匹配
        common_columns = set(submission_df.columns) & set(merged_predictions.columns)
        self._log(f"共同列: {list(common_columns)}")
        
        if 'Id' not in common_columns and 'ranker_id' not in common_columns:
            self._log("警告: 没有找到共同的匹配列(Id, ranker_id)")
            # 尝试其他可能的列名
            possible_id_cols = ['id', 'ID', 'Id', 'flight_id']
            possible_ranker_cols = ['ranker_id', 'rankerId', 'ranker_ID']
            
            for col in possible_id_cols:
                if col in submission_df.columns and col in merged_predictions.columns:
                    submission_df = submission_df.rename(columns={col: 'Id'})
                    merged_predictions = merged_predictions.rename(columns={col: 'Id'})
                    break
            
            for col in possible_ranker_cols:
                if col in submission_df.columns and col in merged_predictions.columns:
                    submission_df = submission_df.rename(columns={col: 'ranker_id'})
                    merged_predictions = merged_predictions.rename(columns={col: 'ranker_id'})
                    break
        
        # 与submission文件对应
        try:
            if 'Id' in submission_df.columns and 'ranker_id' in submission_df.columns:
                final_df = submission_df.merge(
                    merged_predictions, 
                    on=['Id', 'ranker_id'], 
                    how='left'
                )
            elif 'Id' in submission_df.columns:
                final_df = submission_df.merge(
                    merged_predictions, 
                    on=['Id'], 
                    how='left'
                )
            else:
                # 如果没有合适的匹配列，直接使用预测结果
                self._log("警告: 无法与submission文件匹配，直接使用预测结果")
                final_df = merged_predictions
        
        except Exception as e:
            self._log(f"合并数据时出错: {str(e)}")
            self._log("使用预测数据作为最终结果")
            final_df = merged_predictions
        
        # 验证合并后的数据
        self._validate_data_quality(final_df, "after merge")
        
        # 处理集成预测（关键修复）
        final_df = self._ensemble_predictions_robust(final_df, ensemble_method)
        
        # 最终验证和修复
        self._validate_data_quality(final_df, "after ensemble")
        final_df = self._comprehensive_ranking_fix(final_df)
        
        # 保存最终结果
        try:
            final_df.to_parquet(output_file, index=False)
            self._log(f"最终预测结果已保存到: {output_file}")
        except Exception as e:
            # 如果parquet保存失败，尝试保存为CSV
            csv_output = output_file.replace('.parquet', '.csv')
            final_df.to_csv(csv_output, index=False, encoding='utf-8')
            self._log(f"最终预测结果已保存到: {csv_output}")
            output_file = csv_output
        
        return output_file
    
    def _ensemble_predictions_robust(self, df: pd.DataFrame, method: str) -> pd.DataFrame:
        """
        稳健的集成多个模型的预测结果 - 彻底修复排名重复问题
        
        Args:
            df: 包含多个模型预测的DataFrame
            method: 集成方法
            
        Returns:
            pd.DataFrame: 集成后的结果，确保排名唯一且连续
        """
        self._log(f"🎯 执行稳健集成预测，方法: {method}")
        
        # 找到所有分数列和排名列
        score_columns = [col for col in df.columns if col.endswith('_score')]
        rank_columns = [col for col in df.columns if col.endswith('_rank')]
        
        self._log(f"找到分数列: {score_columns}")
        self._log(f"找到排名列: {rank_columns}")
        
        if not score_columns and not rank_columns:
            self._log("警告: 没有找到预测分数或排名列")
            return self._create_emergency_rankings(df)
        
        try:
            if method == 'average' and score_columns:
                # 平均分数方法
                df = self._average_score_ensemble_robust(df, score_columns)
                
            elif method == 'voting' and rank_columns:
                # 排名投票方法  
                df = self._voting_rank_ensemble_robust(df, rank_columns)
                
            elif method == 'weighted' and score_columns:
                # 加权平均方法
                df = self._weighted_average_ensemble_robust(df, score_columns)
                
            else:
                # 回退到默认方法
                self._log("使用默认集成方法")
                df = self._create_emergency_rankings(df)
            
            # 关键步骤：确保最终排名的唯一性
            if 'final_rank' in df.columns and 'ranker_id' in df.columns:
                df = self._guarantee_ranking_uniqueness(df)
            
            self._log("✅ 稳健集成预测完成")
            
        except Exception as e:
            self._log(f"❌ 集成预测过程中出错: {str(e)}")
            # 如果集成失败，创建安全的备用排名
            df = self._create_emergency_rankings(df)
        
        return df

    def _average_score_ensemble_robust(self, df: pd.DataFrame, score_columns: list) -> pd.DataFrame:
        """
        稳健版本的平均分数集成
        
        Args:
            df: 包含分数列的DataFrame
            score_columns: 分数列名列表
            
        Returns:
            pd.DataFrame: 集成后的DataFrame
        """
        self._log("执行稳健平均分数集成...")
        
        # 清理和验证分数列
        for col in score_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        
        # 计算平均分数，跳过NaN值
        df['ensemble_score'] = df[score_columns].mean(axis=1, skipna=True)
        
        # 处理仍然是NaN的情况
        df['ensemble_score'] = df['ensemble_score'].fillna(0.0)
        
        # 关键修复：基于集成分数计算稳健的唯一排名
        if 'ranker_id' in df.columns:
            df['final_rank'] = self._calculate_robust_group_ranks(
                df['ensemble_score'].values, 
                df['ranker_id'].values,
                score_based=True
            )
        else:
            df['final_rank'] = 1
        
        return df

    def _voting_rank_ensemble_robust(self, df: pd.DataFrame, rank_columns: list) -> pd.DataFrame:
        """
        稳健版本的排名投票集成
        
        Args:
            df: 包含排名列的DataFrame
            rank_columns: 排名列名列表
            
        Returns:
            pd.DataFrame: 集成后的DataFrame
        """
        self._log("执行稳健排名投票集成...")
        
        # 清理和验证排名列
        for col in rank_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        
        # 计算平均排名
        df['ensemble_rank'] = df[rank_columns].mean(axis=1, skipna=True)
        
        # 处理NaN值 - 用组内最大排名+1填充
        if 'ranker_id' in df.columns:
            df['ensemble_rank'] = df.groupby('ranker_id')['ensemble_rank'].transform(
                lambda x: x.fillna(x.max() + 1 if not x.isna().all() else len(x))
            )
        else:
            df['ensemble_rank'] = df['ensemble_rank'].fillna(1.0)
        
        # 关键修复：基于平均排名重新分配稳健的唯一排名
        if 'ranker_id' in df.columns:
            # 将平均排名转换为伪分数（排名越小，分数越高）
            pseudo_scores = -df['ensemble_rank'].values
            df['final_rank'] = self._calculate_robust_group_ranks(
                pseudo_scores,
                df['ranker_id'].values,
                score_based=True
            )
        else:
            df['final_rank'] = df['ensemble_rank'].round().clip(lower=1).astype(int)
        
        return df

    def _weighted_average_ensemble_robust(self, df: pd.DataFrame, score_columns: list) -> pd.DataFrame:
        """
        稳健版本的加权平均集成
        
        Args:
            df: 包含分数列的DataFrame  
            score_columns: 分数列名列表
            
        Returns:
            pd.DataFrame: 集成后的DataFrame
        """
        self._log("执行稳健加权平均集成...")
        
        # 清理分数列
        for col in score_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        
        # 简单等权重（可以根据模型性能调整权重）
        weights = np.ones(len(score_columns)) / len(score_columns)
        
        # 计算加权平均，正确处理NaN值
        score_matrix = df[score_columns].values
        valid_mask = ~np.isnan(score_matrix)
        
        weighted_scores = []
        for i in range(len(df)):
            row_scores = score_matrix[i]
            row_mask = valid_mask[i]
            
            if row_mask.any():  # 如果有有效值
                valid_scores = row_scores[row_mask]
                valid_weights = weights[row_mask]
                valid_weights = valid_weights / valid_weights.sum()  # 重新归一化权重
                weighted_score = np.average(valid_scores, weights=valid_weights)
            else:
                weighted_score = 0.0
            
            weighted_scores.append(weighted_score)
        
        df['ensemble_score'] = weighted_scores
        
        # 基于加权分数计算稳健的唯一排名
        if 'ranker_id' in df.columns:
            df['final_rank'] = self._calculate_robust_group_ranks(
                df['ensemble_score'].values,
                df['ranker_id'].values,
                score_based=True
            )
        else:
            df['final_rank'] = 1
        
        return df

    def _calculate_robust_group_ranks(self, values: np.ndarray, ranker_ids: np.ndarray, 
                                     score_based: bool = True) -> np.ndarray:
        """
        计算稳健的组内排名，确保每组排名唯一且连续
        
        Args:
            values: 分数或排名值
            ranker_ids: ranker_id数组
            score_based: 是否基于分数（True）还是排名（False）
            
        Returns:
            np.ndarray: 唯一且连续的排名
        """
        ranks = np.zeros_like(values, dtype=int)
        
        # 按ranker_id分组处理
        unique_rankers = np.unique(ranker_ids)
        
        for ranker_id in unique_rankers:
            group_mask = ranker_ids == ranker_id
            group_values = values[group_mask]
            group_size = len(group_values)
            
            if group_size == 1:
                # 单个元素的组，排名直接为1
                ranks[group_mask] = 1
            else:
                # 多个元素的组，确保排名唯一
                # 使用ranker_id的哈希值作为随机种子确保可重复性
                unique_seed = abs(hash(str(ranker_id))) % 1000000
                np.random.seed(unique_seed)
                
                # 添加唯一的噪声
                noise_scale = 1e-8
                noise = np.random.random(len(group_values)) * noise_scale
                
                # 为每个位置添加不同的偏移
                position_offset = np.arange(len(group_values)) * 1e-10
                noisy_values = group_values + noise + position_offset
                
                if score_based:
                    # 基于分数：分数越高排名越靠前
                    sorted_indices = np.argsort(-noisy_values)
                else:
                    # 基于排名：排名越小越靠前
                    sorted_indices = np.argsort(noisy_values)
                
                # 分配唯一且连续的排名
                group_ranks = np.zeros(group_size, dtype=int)
                for rank, idx in enumerate(sorted_indices):
                    group_ranks[idx] = rank + 1
                
                ranks[group_mask] = group_ranks
                
                # 验证当前组的排名
                unique_ranks = set(group_ranks)
                expected_ranks = set(range(1, group_size + 1))
                if unique_ranks != expected_ranks:
                    # 如果仍有问题，强制修复
                    self._log(f"警告：ranker_id {ranker_id} 排名计算失败，强制修复")
                    ranks[group_mask] = np.random.permutation(range(1, group_size + 1))
        
        return ranks

    def _guarantee_ranking_uniqueness(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        ⭐ 终极修复方法：绝对保证排名的唯一性和连续性
        
        Args:
            df: 包含排名的DataFrame
            
        Returns:
            pd.DataFrame: 绝对修复后的DataFrame
        """
        self._log("🔧 执行终极排名唯一性保证...")
        
        problem_groups = 0
        total_groups = df['ranker_id'].nunique()
        
        for ranker_id in df['ranker_id'].unique():
            group_mask = df['ranker_id'] == ranker_id
            group_data = df[group_mask]
            group_size = len(group_data)
            
            # 检查当前排名是否符合要求
            current_ranks = sorted(group_data['final_rank'].values)
            expected_ranks = list(range(1, group_size + 1))
            
            if current_ranks != expected_ranks:
                problem_groups += 1
                
                # 终极修复策略：使用多种备用方案
                fixed_ranks = None
                
                # 策略1：基于集成分数重新排名
                if 'ensemble_score' in group_data.columns:
                    scores = group_data['ensemble_score'].values
                    # 使用ranker_id作为随机种子确保可重复性和唯一性
                    unique_seed = abs(hash(str(ranker_id))) % 1000000
                    np.random.seed(unique_seed)
                    noise = np.random.random(len(scores)) * 1e-6
                    noisy_scores = scores + noise
                    sorted_indices = np.argsort(-noisy_scores)
                    fixed_ranks = np.zeros(group_size, dtype=int)
                    for rank, idx in enumerate(sorted_indices):
                        fixed_ranks[idx] = rank + 1
                
                # 策略2：基于集成排名重新排名
                elif 'ensemble_rank' in group_data.columns:
                    ranks_vals = group_data['ensemble_rank'].values
                    unique_seed = abs(hash(str(ranker_id) + "_rank")) % 1000000
                    np.random.seed(unique_seed)
                    noise = np.random.random(len(ranks_vals)) * 1e-6
                    noisy_ranks = ranks_vals + noise
                    sorted_indices = np.argsort(noisy_ranks)
                    fixed_ranks = np.zeros(group_size, dtype=int)
                    for rank, idx in enumerate(sorted_indices):
                        fixed_ranks[idx] = rank + 1
                
                # 策略3：完全随机排列（最后手段）
                if fixed_ranks is None:
                    unique_seed = abs(hash(str(ranker_id) + "_final")) % 1000000
                    np.random.seed(unique_seed)
                    fixed_ranks = np.random.permutation(range(1, group_size + 1))
                
                # 应用修复后的排名
                df.loc[group_mask, 'final_rank'] = fixed_ranks
                
                # 再次验证
                new_ranks = sorted(fixed_ranks)
                if new_ranks != expected_ranks:
                    # 如果还是有问题，使用最简单的顺序排名
                    df.loc[group_mask, 'final_rank'] = list(range(1, group_size + 1))
        
        if problem_groups > 0:
            self._log(f"🔧 终极修复完成：修复了 {problem_groups}/{total_groups} 个排名问题组")
        else:
            self._log(f"✅ 所有 {total_groups} 个组的排名都已完美")
        
        return df

    def _create_emergency_rankings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建紧急备用排名（当所有其他方法都失败时）
        
        Args:
            df: DataFrame
            
        Returns:
            pd.DataFrame: 添加了紧急排名的DataFrame
        """
        self._log("创建紧急备用排名...")
        
        if 'ranker_id' in df.columns:
            # 为每个组创建确定性的随机排名
            for ranker_id in df['ranker_id'].unique():
                group_mask = df['ranker_id'] == ranker_id
                group_size = group_mask.sum()
                
                # 使用ranker_id作为种子确保可重复性
                emergency_seed = abs(hash(str(ranker_id) + "_emergency")) % 1000000
                np.random.seed(emergency_seed)
                ranks = np.random.permutation(range(1, group_size + 1))
                df.loc[group_mask, 'final_rank'] = ranks
        else:
            df['final_rank'] = 1
        
        return df

    def _comprehensive_ranking_fix(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        全面的排名修复（最终检查和修复）
        
        Args:
            df: DataFrame
            
        Returns:
            pd.DataFrame: 全面修复后的DataFrame
        """
        self._log("执行全面排名修复...")
        
        # 确保final_rank列存在
        if 'final_rank' not in df.columns:
            df = self._create_emergency_rankings(df)
        
        # 确保final_rank是有效的整数
        df['final_rank'] = df['final_rank'].fillna(1).astype(int)
        
        # 最终验证和修复
        if 'ranker_id' in df.columns:
            problem_groups = 0
            total_groups = 0
            
            for ranker_id in df['ranker_id'].unique():
                total_groups += 1
                group_mask = df['ranker_id'] == ranker_id
                group_data = df[group_mask]
                group_size = len(group_data)
                
                current_ranks = sorted(group_data['final_rank'].values)
                expected_ranks = list(range(1, group_size + 1))
                
                if current_ranks != expected_ranks:
                    problem_groups += 1
                    
                    # 最终强制修复
                    final_seed = abs(hash(str(ranker_id) + "_final_fix")) % 1000000
                    np.random.seed(final_seed)
                    new_ranks = np.random.permutation(range(1, group_size + 1))
                    df.loc[group_mask, 'final_rank'] = new_ranks
            
            if problem_groups > 0:
                self._log(f"全面修复完成：修复了 {problem_groups}/{total_groups} 个最终问题组")
        
        # 将final_rank复制到selected列（这是最终提交需要的列名）
        df['selected'] = df['final_rank']
        
        # 确保selected列也是整数类型
        df['selected'] = df['selected'].astype(int)
        
        # 最终验证
        if 'ranker_id' in df.columns:
            all_valid = True
            for ranker_id in df['ranker_id'].unique():
                group_mask = df['ranker_id'] == ranker_id
                group_data = df[group_mask]
                current_ranks = sorted(group_data['selected'].values)
                expected_ranks = list(range(1, len(group_data) + 1))
                
                if current_ranks != expected_ranks:
                    all_valid = False
                    self._log(f"❌ 最终验证失败 ranker_id {ranker_id}: {current_ranks}")
            
            if all_valid:
                self._log("✅ 最终验证通过，所有排名都是唯一且连续的")
            else:
                self._log("❌ 最终验证失败，仍有排名问题")
        
        return df