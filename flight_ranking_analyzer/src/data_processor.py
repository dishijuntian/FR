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
    """预测结果合并器 - 简化排名处理版本"""
    
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
    
    def merge_predictions(self, prediction_files: List[str], 
                         submission_file: str, 
                         output_file: str,
                         ensemble_method: str = 'average') -> str:
        """
        合并多个预测文件并与submission文件对应（简化版本）
        
        Args:
            prediction_files: 预测文件路径列表
            submission_file: submission模板文件路径
            output_file: 输出文件路径
            ensemble_method: 集成方法 ('average', 'voting', 'weighted')
            
        Returns:
            str: 输出文件路径
        """
        self._log("开始合并预测结果...")
        
        # 读取submission文件
        if not os.path.exists(submission_file):
            raise FileNotFoundError(f"Submission文件不存在: {submission_file}")
        
        try:
            if submission_file.endswith('.parquet'):
                submission_df = pd.read_parquet(submission_file)
            else:
                submission_df = pd.read_csv(submission_file)
            
            self._log(f"Submission文件形状: {submission_df.shape}")
            
        except Exception as e:
            raise ValueError(f"无法读取submission文件: {str(e)}")
        
        # 读取所有预测文件
        prediction_dfs = []
        for file_path in prediction_files:
            if os.path.exists(file_path):
                try:
                    if file_path.endswith('.parquet'):
                        pred_df = pd.read_parquet(file_path)
                    else:
                        pred_df = pd.read_csv(file_path)
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
        self._log(f"合并预测结果形状: {merged_predictions.shape}")
        
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
        
        # 处理集成预测（简化版本）
        final_df = self._ensemble_predictions_simplified(final_df, ensemble_method)
        
        # 保存最终结果
        try:
            if output_file.endswith('.parquet'):
                final_df.to_parquet(output_file, index=False)
            else:
                final_df.to_csv(output_file, index=False, encoding='utf-8')
            self._log(f"最终预测结果已保存到: {output_file}")
        except Exception as e:
            # 如果保存失败，尝试保存为CSV
            csv_output = output_file.replace('.parquet', '.csv')
            final_df.to_csv(csv_output, index=False, encoding='utf-8')
            self._log(f"最终预测结果已保存到: {csv_output}")
            output_file = csv_output
        
        return output_file
    
    def _ensemble_predictions_simplified(self, df: pd.DataFrame, method: str) -> pd.DataFrame:
        """
        简化版集成多个模型的预测结果
        
        Args:
            df: 包含多个模型预测的DataFrame
            method: 集成方法
            
        Returns:
            pd.DataFrame: 集成后的结果
        """
        self._log(f"🎯 执行简化集成预测，方法: {method}")
        
        # 找到所有分数列和排名列
        score_columns = [col for col in df.columns if col.endswith('_score')]
        rank_columns = [col for col in df.columns if col.endswith('_rank')]
        
        self._log(f"找到分数列: {score_columns}")
        self._log(f"找到排名列: {rank_columns}")
        
        if not score_columns and not rank_columns:
            self._log("警告: 没有找到预测分数或排名列，使用第一列作为分数")
            # 尝试找到数值列作为分数
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                score_columns = [numeric_cols[0]]
            else:
                # 创建默认排名
                return self._create_default_rankings(df)
        
        try:
            if method == 'average' and score_columns:
                # 平均分数方法
                df = self._average_score_ensemble_simplified(df, score_columns)
                
            elif method == 'voting' and rank_columns:
                # 排名投票方法  
                df = self._voting_rank_ensemble_simplified(df, rank_columns)
                
            elif method == 'weighted' and score_columns:
                # 加权平均方法
                df = self._weighted_average_ensemble_simplified(df, score_columns)
                
            else:
                # 回退到默认方法
                self._log("使用默认集成方法")
                df = self._create_default_rankings(df)
            
            self._log("✅ 简化集成预测完成")
            
        except Exception as e:
            self._log(f"❌ 集成预测过程中出错: {str(e)}")
            # 如果集成失败，创建安全的备用排名
            df = self._create_default_rankings(df)
        
        return df

    def _average_score_ensemble_simplified(self, df: pd.DataFrame, score_columns: list) -> pd.DataFrame:
        """简化版平均分数集成"""
        self._log("执行简化平均分数集成...")
        
        # 清理分数列
        for col in score_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        
        # 计算平均分数
        df['ensemble_score'] = df[score_columns].mean(axis=1, skipna=True)
        
        # 使用简化的排名分配方法
        if 'ranker_id' in df.columns:
            df = self._assign_unique_rankings_simplified(df)
        else:
            df['selected'] = 1
        
        return df

    def _voting_rank_ensemble_simplified(self, df: pd.DataFrame, rank_columns: list) -> pd.DataFrame:
        """简化版排名投票集成"""
        self._log("执行简化排名投票集成...")
        
        # 清理排名列
        for col in rank_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(1.0)
        
        # 计算平均排名
        df['ensemble_rank'] = df[rank_columns].mean(axis=1, skipna=True)
        
        # 将平均排名转换为分数（排名越小，分数越高）
        df['ensemble_score'] = -df['ensemble_rank']
        
        # 重新分配排名
        if 'ranker_id' in df.columns:
            df = self._assign_unique_rankings_simplified(df)
        else:
            df['selected'] = df['ensemble_rank'].round().clip(lower=1).astype(int)
        
        return df

    def _weighted_average_ensemble_simplified(self, df: pd.DataFrame, score_columns: list) -> pd.DataFrame:
        """简化版加权平均集成"""
        self._log("执行简化加权平均集成...")
        
        # 清理分数列
        for col in score_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        
        # 简单等权重
        weights = np.ones(len(score_columns)) / len(score_columns)
        
        # 计算加权平均
        df['ensemble_score'] = df[score_columns].multiply(weights).sum(axis=1)
        
        # 分配排名
        if 'ranker_id' in df.columns:
            df = self._assign_unique_rankings_simplified(df)
        else:
            df['selected'] = 1
        
        return df

    def _assign_unique_rankings_simplified(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        使用简化方法分配唯一排名
        
        Args:
            df: 包含ranker_id和ensemble_score的DataFrame
            
        Returns:
            pd.DataFrame: 添加了selected列的DataFrame
        """
        self._log("🎯 使用简化方法分配唯一排名...")
        
        # 确保Id列存在，如果不存在则创建
        if 'Id' not in df.columns:
            df['Id'] = range(len(df))
        
        # 确保唯一排名：使用Id作为tie-breaker
        df = df.sort_values(['ranker_id', 'ensemble_score', 'Id'], 
                          ascending=[True, False, True])
        df['selected'] = df.groupby('ranker_id').cumcount() + 1
        
        self._log("✅ 排名分配完成")
        return df

    def _create_default_rankings(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建默认排名"""
        self._log("创建默认排名...")
        
        if 'ranker_id' in df.columns:
            # 确保Id列存在
            if 'Id' not in df.columns:
                df['Id'] = range(len(df))
            
            # 按ranker_id分组，按Id排序分配排名
            df = df.sort_values(['ranker_id', 'Id'])
            df['selected'] = df.groupby('ranker_id').cumcount() + 1
        else:
            df['selected'] = 1
        
        return df