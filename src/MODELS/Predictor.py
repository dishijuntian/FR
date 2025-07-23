import pandas as pd
import numpy as np
from FlightRankingAnalyzer import FlightRankingAnalyzer
import joblib
import os

class FlightRankingPredictor:
    def __init__(self, data_path="data/aeroclub-recsys-2025/encode", 
                 model_save_path="models", output_path="submissions",
                 use_gpu=False, random_state=42):
        self.data_path = data_path
        self.model_save_path = model_save_path
        self.output_path = output_path
        self.use_gpu = use_gpu
        self.random_state = random_state
        os.makedirs(output_path, exist_ok=True)
    
    def predict_segment(self, segment_id, model_name='XGBRanker'):
        """预测单个数据段"""
        print(f"开始预测 segment_{segment_id}")
        
        # 检查模型文件是否存在
        model_path = f"{self.model_save_path}/{model_name}_segment_{segment_id}.pkl"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # 加载测试数据
        test_file = f"{self.data_path}/test/test_segment_{segment_id}_encoded.parquet"
        df = pd.read_parquet(test_file)
        print(f"测试数据形状: {df.shape}")
        
        # 初始化分析器
        analyzer = FlightRankingAnalyzer(use_gpu=self.use_gpu, random_state=self.random_state)
        
        # 加载特征名称
        feature_path = f"{self.model_save_path}/features_segment_{segment_id}.pkl"
        feature_cols = joblib.load(feature_path)
        analyzer.feature_names = feature_cols
        
        # 准备测试数据
        X, _, groups, _, df_processed = analyzer.prepare_data(df, target_col='selected')
        
        # 加载模型
        analyzer.load_model(model_path, model_name)
        print(f"已加载模型: {model_path}")
        
        # 预测排名
        ranks = analyzer.predict_ranks(X, groups, model_name)
        
        # 准备提交结果
        submission = pd.DataFrame({
            'Id': df['Id'],
            'ranker_id': df['ranker_id'],
            'selected': ranks
        })
        
        print(f"预测完成，结果形状: {submission.shape}")
        return submission
    
    def predict_all(self, segments=[0, 1, 2], model_name='XGBRanker'):
        """预测所有指定数据段并生成最终提交文件"""
        all_predictions = []
        
        for segment_id in segments:
            try:
                print(f"\n{'='*50}")
                prediction = self.predict_segment(segment_id, model_name)
                all_predictions.append(prediction)
                
                # 保存单个段的预测结果
                segment_output = f"{self.output_path}/{model_name}_segment_{segment_id}_prediction.csv"
                prediction.to_csv(segment_output, index=False)
                print(f"已保存预测结果: {segment_output}")
                
            except Exception as e:
                print(f"预测 segment_{segment_id} 失败: {e}")
                continue
        
        # 合并所有预测结果
        if not all_predictions:
            print("没有成功的预测结果")
            return None
        
        final_submission = pd.concat(all_predictions, ignore_index=True)
        
        # 按Id排序
        final_submission = final_submission.sort_values('Id').reset_index(drop=True)
        
        # 保存最终结果
        final_output = f"{self.output_path}/{model_name}_final_submission.csv"
        final_submission.to_csv(final_output, index=False)
        
        # 结果验证
        print(f"\n{'='*50}")
        print(f"预测完成!")
        print(f"最终提交文件: {final_output}")
        print(f"总记录数: {len(final_submission)}")
        print(f"唯一ranker_id数量: {final_submission['ranker_id'].nunique()}")
        
        self.validate_predictions(final_submission)
        return final_submission
    
    def validate_predictions(self, submission, sample_size=5):
        """验证预测结果的有效性"""
        print("\n验证预测结果:")
        unique_rankers = submission['ranker_id'].unique()
        
        # 随机抽样检查
        sample_rankers = np.random.choice(unique_rankers, min(sample_size, len(unique_rankers)), replace=False)
        
        for ranker_id in sample_rankers:
            group_data = submission[submission['ranker_id'] == ranker_id]
            ranks = sorted(group_data['selected'].values)
            expected_ranks = list(range(1, len(group_data) + 1))
            is_valid = ranks == expected_ranks
            print(f"ranker_id {ranker_id}: 排名{'有效' if is_valid else '无效'}")