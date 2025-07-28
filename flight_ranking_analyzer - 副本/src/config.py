"""
配置文件 - 定义全局配置参数

作者: Flight Ranking Team
版本: 2.1
"""

import os
from typing import List, Dict, Any

class Config:
    """全局配置类"""
    
    # 数据路径配置
    DATA_BASE_PATH = "E:/GIT PROJECT/FR/data/aeroclub-recsys-2025"
    TRAIN_DATA_PATH = os.path.join(DATA_BASE_PATH, "segmented/train")
    TEST_DATA_PATH = os.path.join(DATA_BASE_PATH, "segmented/test")
    SUBMISSION_FILE_PATH = os.path.join(DATA_BASE_PATH, "sample_submission.parquet")
    OUTPUT_PATH = os.path.join(DATA_BASE_PATH, "results")
    
# 抽样配置
    DEFAULT_NUM_GROUPS = 2000
    DEFAULT_MIN_GROUP_SIZE = 20
    USE_SAMPLING = True  # 默认使用抽样
    
    # GPU配置
    AUTO_DETECT_GPU = True
    FORCE_USE_GPU = False    # 强制使用GPU（跳过检测）
    FORCE_USE_CPU = False    # 强制使用CPU
    
    # 自动调参配置
    ENABLE_AUTO_TUNING = False  # 默认关闭自动调参
    AUTO_TUNING_TRIALS = 50  # 自动调参试验次数
    AUTO_TUNING_TIMEOUT = 3600  # 自动调参超时时间(秒)
    
    # 模型配置
    AVAILABLE_MODELS = [
        'XGBRanker', 'LGBMRanker', 'LambdaMART', 
        'ListNet', 'NeuralRanker', 'BM25Ranker'
    ]
    
    # 默认模型参数
    DEFAULT_MODEL_PARAMS = {
        'XGBRanker': {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'eval_metric': 'ndcg'
        },
        'LGBMRanker': {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'objective': 'lambdarank',
            'metric': 'ndcg'
        },
        'LambdaMART': {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        },
        'ListNet': {
            'n_estimators': 100,
            'learning_rate': 0.05,
            'max_depth': 7,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        },
        'NeuralRanker': {
            'hidden_units': [256, 128, 64],
            'learning_rate': 0.001,
            'epochs': 10,
            'batch_size': 32
        },
        'BM25Ranker': {}
    }
    
    # 自动调参搜索空间
    TUNING_SEARCH_SPACES = {
        'XGBRanker': {
            'n_estimators': [50, 100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 4, 5, 6, 7, 8],
            'subsample': [0.6, 0.7, 0.8, 0.9],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9]
        },
        'LGBMRanker': {
            'n_estimators': [50, 100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 4, 5, 6, 7, 8],
            'subsample': [0.6, 0.7, 0.8, 0.9],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9]
        },
        'NeuralRanker': {
            'hidden_units': [
                [128, 64], [256, 128], [256, 128, 64], 
                [512, 256, 128], [128, 64, 32]
            ],
            'learning_rate': [0.0001, 0.0005, 0.001, 0.005],
            'epochs': [5, 10, 15, 20],
            'batch_size': [16, 32, 64, 128]
        }
    }
    
    # 特征配置
    EXCLUDE_FEATURES = ['Id', 'selected', 'ranker_id', 'profileId', 'companyID']
    
    # 评估配置
    HITRATE_K = 3
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    
    # 日志配置
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # 内存和性能配置
    MAX_SHAP_SAMPLES = 2000
    MAX_IMPORTANCE_SAMPLES = 1000
    ENABLE_MEMORY_OPTIMIZATION = True
    
    # 输出配置
    SAVE_FEATURE_IMPORTANCE = True
    SAVE_SHAP_VALUES = True
    SAVE_MODEL_PREDICTIONS = True
    FINAL_PREDICTION_FILENAME = "final_predictions.parquet"
    
    @classmethod
    def get_train_files(cls) -> List[str]:
        """获取训练文件路径列表，自动检测目录结构"""
        train_files = []
        
        # 可能的训练文件路径
        possible_train_paths = [
            os.path.join(cls.DATA_BASE_PATH, "encode/train"),
            os.path.join(cls.DATA_BASE_PATH, "segmented/train"),
            os.path.join(cls.DATA_BASE_PATH, "train"),
            cls.DATA_BASE_PATH  # 直接在根目录
        ]
        
        # 尝试每个可能的路径
        for train_path in possible_train_paths:
            if os.path.exists(train_path):
                # 查找训练文件
                for i in range(10):  # 最多检查10个segment
                    possible_files = [
                        os.path.join(train_path, f"train_segment_{i}_encoded.parquet"),
                        os.path.join(train_path, f"train_segment_{i}.parquet"),
                        os.path.join(train_path, f"train_{i}.parquet")
                    ]
                    
                    for file_path in possible_files:
                        if os.path.exists(file_path):
                            train_files.append(file_path)
                            break  # 找到一个就跳出内层循环
                
                if train_files:  # 如果找到文件，就使用这个路径
                    break
        
        return sorted(train_files)  # 排序确保顺序一致
    
    @classmethod
    def get_test_files(cls) -> List[str]:
        """获取测试文件路径列表，自动检测目录结构"""
        test_files = []
        
        # 可能的测试文件路径
        possible_test_paths = [
            os.path.join(cls.DATA_BASE_PATH, "encode/test"),
            os.path.join(cls.DATA_BASE_PATH, "segmented/test"),
            os.path.join(cls.DATA_BASE_PATH, "test"),
            cls.DATA_BASE_PATH  # 直接在根目录
        ]
        
        # 尝试每个可能的路径
        for test_path in possible_test_paths:
            if os.path.exists(test_path):
                # 查找测试文件
                for i in range(10):  # 最多检查10个segment
                    possible_files = [
                        os.path.join(test_path, f"test_segment_{i}_encoded.parquet"),
                        os.path.join(test_path, f"test_segment_{i}.parquet"),
                        os.path.join(test_path, f"test_{i}.parquet")
                    ]
                    
                    for file_path in possible_files:
                        if os.path.exists(file_path):
                            test_files.append(file_path)
                            break  # 找到一个就跳出内层循环
                
                if test_files:  # 如果找到文件，就使用这个路径
                    break
        
        return sorted(test_files)  # 排序确保顺序一致
    
    @classmethod
    def ensure_output_dir(cls):
        """确保输出目录存在"""
        try:
            os.makedirs(cls.OUTPUT_PATH, exist_ok=True)
            # 测试写入权限
            test_file = os.path.join(cls.OUTPUT_PATH, 'test_write.tmp')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
        except Exception as e:
            raise ValueError(f"无法创建或写入输出目录 {cls.OUTPUT_PATH}: {str(e)}")
        
    @classmethod
    def validate_config(cls):
        """验证配置的有效性"""
        # 检查数据基础路径
        if not os.path.exists(cls.DATA_BASE_PATH):
            print(f"警告: 数据基础路径不存在: {cls.DATA_BASE_PATH}")
            print("请在 src/config.py 中修改 DATA_BASE_PATH 为正确的路径")
        
        # 检查训练数据路径
        if not os.path.exists(cls.TRAIN_DATA_PATH):
            print(f"警告: 训练数据路径不存在: {cls.TRAIN_DATA_PATH}")
        
        # 检查测试数据路径
        if not os.path.exists(cls.TEST_DATA_PATH):
            print(f"警告: 测试数据路径不存在: {cls.TEST_DATA_PATH}")
        
        # 确保输出目录存在（这个是必需的）
        cls.ensure_output_dir()