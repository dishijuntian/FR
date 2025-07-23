import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from memory_profiler import profile
import matplotlib.pyplot as plt
import seaborn as sns

@profile
def analyze_feature_distributions(df, features_to_analyze):
    """
    分析关键特征的分布，内存优化版本
    """
    results = {}
    
    for feature in features_to_analyze:
        try:
            if feature in df.columns:
                # 对于分类特征
                if df[feature].dtype.name == 'category' or df[feature].dtype == 'object':
                    # 只计算top N值避免内存问题
                    value_counts = df[feature].value_counts().head(20)
                    results[feature] = {
                        'type': 'categorical',
                        'top_values': value_counts.to_dict(),
                        'unique_count': df[feature].nunique()
                    }
                    
                    # 可视化
                    plt.figure(figsize=(10, 6))
                    value_counts.plot(kind='bar')
                    plt.title(f'Top 20 values for {feature}')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    plt.savefig(f'{feature}_distribution.png', dpi=300)
                    plt.close()
                
                # 对于数值特征
                elif np.issubdtype(df[feature].dtype, np.number):
                    desc = df[feature].describe()
                    results[feature] = {
                        'type': 'numeric',
                        'stats': desc.to_dict(),
                        'missing': df[feature].isna().sum()
                    }
                    
                    # 可视化 - 使用直方图
                    plt.figure(figsize=(10, 6))
                    plt.hist(df[feature].dropna(), bins=50)
                    plt.title(f'Distribution of {feature}')
                    plt.savefig(f'{feature}_distribution.png', dpi=300)
                    plt.close()
            else:
                print(f"特征 {feature} 不存在于数据中")
        except Exception as e:
            print(f"分析特征 {feature} 时出错: {e}")
    
    return results

# 选择关键特征进行分析
key_features = [
    'selected', 'Total Price', 'Taxes', 'Legs0_duration', 
    'Legs0_segments0_cabin Class', 'Pricing info_is Access TP',
    'Mini Rules0_status Infos', 'Company ID', 'Frequent Flyer'
]

train_sample = pq.read_table('E:/GIT PROJECT/FR/kaggle/input/aeroclub-recsys-2025/train.parquet', columns=['ranker_id', 'selected']).to_pandas().sample(frac=0.01)
feature_distributions = analyze_feature_distributions(train_sample, key_features)