import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.model.FlightRankingAnalyzer import FlightRankingAnalyzer
import joblib
from pathlib import Path

class FlightRankingTrainer:
    def __init__(self, data_path="data/aeroclub-recsys-2025/segmented", 
                 model_save_path="models", use_gpu=False, random_state=42):
        self.data_path = Path(data_path)
        self.model_save_path = self.data_path / model_save_path
        self.use_gpu = use_gpu
        self.random_state = random_state
        
        # 确保模型保存目录存在
        self.model_save_path.mkdir(parents=True, exist_ok=True)
    
    def train_segment(self, segment_id):
        """训练单个数据段"""
        print(f"开始训练 segment_{segment_id}")
        
        # 加载数据
        train_file = self.data_path / "train" / f"train_segment_{segment_id}.parquet"
        df = pd.read_parquet(train_file)
        print(f"数据形状: {df.shape}")
        
        # 初始化分析器
        analyzer = FlightRankingAnalyzer(use_gpu=self.use_gpu, random_state=self.random_state)
        
        # 准备数据
        X, y, groups, feature_cols, df_processed = analyzer.prepare_data(df)
        print(f"特征数量: {len(feature_cols)}")
        
        # 按ranker_id进行训练集验证集划分
        unique_rankers = np.unique(groups)
        train_rankers, test_rankers = train_test_split(
            unique_rankers, test_size=0.2, random_state=self.random_state
        )
        
        train_mask = np.isin(groups, train_rankers)
        test_mask = np.isin(groups, test_rankers)
        
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        groups_train, groups_test = groups[train_mask], groups[test_mask]
        
        print(f"训练集: {X_train.shape}, 验证集: {X_test.shape}")
        
        # 训练模型
        results = analyzer.train_models(X_train, X_test, y_train, y_test, groups_train, groups_test)
        
        # 保存模型
        for model_name in analyzer.trained_models:
            model_path = self.model_save_path / f"{model_name}_segment_{segment_id}.pkl"
            analyzer.save_model(str(model_path), model_name)
            print(f"已保存模型: {model_path}")
        
        # 保存特征名称
        feature_path = self.model_save_path / f"features_segment_{segment_id}.pkl"
        joblib.dump(feature_cols, feature_path)
        
        # 输出结果
        for result in results:
            print(f"段{segment_id} - {result['Model']}: HitRate@3 = {result['HitRate@3']}")
        
        return results
    
    def train_all(self, segments=[0, 1, 2]):
        """训练所有指定数据段"""
        all_results = {}
        
        for segment_id in segments:
            try:
                print(f"\n{'='*50}")
                results = self.train_segment(segment_id)
                all_results[f"segment_{segment_id}"] = results
                print(f"完成训练 segment_{segment_id}\n")
            except Exception as e:
                print(f"训练 segment_{segment_id} 失败: {e}")
                continue
        
        # 汇总结果
        print("\n" + "="*50)
        print("训练结果汇总:")
        for segment, results in all_results.items():
            print(f"\n{segment}:")
            for result in results:
                print(f"  {result['Model']}: HitRate@3 = {result['HitRate@3']}")
        
        return all_results