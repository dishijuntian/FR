import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from memory_profiler import profile
import matplotlib.pyplot as plt
import seaborn as sns

# 内存优化设置
plt.switch_backend('Agg')  # 减少内存使用将Matplotlib的后端设置为'Agg'，
#这是一个非交互式的后端，可以减少内存使用，特别适合在服务器或无GUI环境下运行。
sns.set(style='whitegrid')
#设置Seaborn的绘图风格为白色网格背景。

# 使用PyArrow加载parquet文件（内存高效）
@profile
#这是一个装饰器，通常用于内存分析工具（如memory_profiler）来跟踪函数的内存使用情况。
def load_data_with_memory_optimization(file_path):
    """
    内存优化的数据加载函数
    使用PyArrow的迭代式加载和Pandas的块处理
    """
    # 首先只读取元数据查看结构
    parquet_file = pq.ParquetFile(file_path)
    print(f"文件包含 {parquet_file.num_row_groups} 个行组")
    #使用PyArrow的ParquetFile类打开文件，但不立即加载所有数,打印文件包含的行组数量（Parquet文件通常会被分成多个行组存储）
    
    # 分块读取策略
    chunk_size = 1000000  # 每次处理1M行
    chunks = []
    
    for i in range(parquet_file.num_row_groups):
       #逐个读取行组并将其转换为Pandas DataFrame。
        chunk = parquet_file.read_row_group(i).to_pandas()
        
        # 优化内存 - 转换数据类型对于object类型（通常是字符串），如果唯一值比例小于50%，转换为更节省内存的category类型。
        # 对于float64和int64类型，尝试向下转换到更小的数据类型（如float32、int32等）。    
        for col in chunk.columns:
            if chunk[col].dtype == 'object':
                # 尝试转换为分类或更小的类型
                unique_ratio = len(chunk[col].unique()) / len(chunk[col])
                if unique_ratio < 0.5:
                    chunk[col] = chunk[col].astype('category')
            elif chunk[col].dtype == 'float64':
                chunk[col] = pd.to_numeric(chunk[col], downcast='float')
            elif chunk[col].dtype == 'int64':
                chunk[col] = pd.to_numeric(chunk[col], downcast='integer')
        #将处理后的数据块添加到列表中。
        #当累计行数超过500万时，合并现有数据块并重置列表，防止内存占用过高。
        chunks.append(chunk)
        if len(chunks) * chunk_size > 5000000:  # 每500万行处理一次
            partial_df = pd.concat(chunks, ignore_index=True)
            chunks = [partial_df]
    
    full_df = pd.concat(chunks, ignore_index=True)
    return full_df

# 加载数据
try:
    # 尝试加载小样本进行初步探索
    train_sample = pq.read_table('E:/GIT PROJECT/FR/kaggle/input/aeroclub-recsys-2025/train.parquet', columns=['ranker_id', 'selected']).to_pandas().sample(frac=0.01)
    print("样本数据加载成功")
except Exception as e:
    print(f"数据加载错误: {e}")



@profile
def analyze_group_distribution(df, group_col='ranker_id'):
    """
    分析组大小分布，内存优化版本
    """
    # 使用PySpark或Dask处理大数据集会更合适
    # 这里使用Pandas的优化方法
    
    # 方法1: 使用value_counts() + 抽样
    try:
        #计算 group_col列中每个唯一值的出现次数（即每个组的大小）。
        #返回一个 Pandas Series，索引是组名，值是对应的计数。
        group_sizes = df[group_col].value_counts()
        print(f"总组数: {len(group_sizes)}")
        
        # 描述统计
        print("\n组大小描述统计:")
        print(group_sizes.describe())
    #group_sizes.describe()​​：
    # 计算并返回 group_sizes的描述统计信息，包括：
    # count：非空值的数量。
    # mean：平均值。
    # std：标准差。
    # min：最小值。
    # 25%、50%（中位数）、75%：四分位数。
    # max：最大值。
        # 可视化 - 使用对数尺度
        plt.figure(figsize=(10, 6))
        plt.hist(group_sizes, bins=50, log=True)
        plt.title('Group Size Distribution (log scale)')
        plt.xlabel('Number of options per search session')
        plt.ylabel('Frequency (log scale)')
        plt.savefig('group_size_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 分析超过10个选项的组（用于评估的组）
        large_groups = group_sizes[group_sizes > 10]
        print(f"\n超过10个选项的组占比: {len(large_groups)/len(group_sizes):.2%}")
        
        return group_sizes
    except MemoryError:
        print("内存不足，采用抽样方法")
        sample_size = min(1000000, len(df))
        sample = df.sample(n=sample_size, random_state=42)
        return analyze_group_distribution(sample, group_col)

# 执行分析
group_sizes = analyze_group_distribution(train_sample)