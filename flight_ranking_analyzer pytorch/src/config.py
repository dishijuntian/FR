"""
配置文件 - PyTorch版本

定义全局配置参数，包括PyTorch特定设置

作者: Flight Ranking Team
版本: 3.0 (PyTorch版本)
"""

import os
import torch
from typing import List, Dict, Any

class Config:
    """全局配置类（PyTorch版本）"""
    
    # 数据路径配置 - 修改为segmented目录
    DATA_BASE_PATH = "E:/GIT PROJECT/FR/data/aeroclub-recsys-2025"
    TRAIN_DATA_PATH = os.path.join(DATA_BASE_PATH, "segmented/train")
    TEST_DATA_PATH = os.path.join(DATA_BASE_PATH, "segmented/test")
    SUBMISSION_FILE_PATH = os.path.join(DATA_BASE_PATH, "sample_submission.parquet")
    OUTPUT_PATH = os.path.join(DATA_BASE_PATH, "results")
    
    # 抽样配置
    DEFAULT_NUM_GROUPS = 2000
    DEFAULT_MIN_GROUP_SIZE = 20
    USE_SAMPLING = True  # 默认使用抽样
    
    # PyTorch/GPU配置
    AUTO_DETECT_GPU = True
    FORCE_USE_GPU = False    # 强制使用GPU（跳过检测）
    FORCE_USE_CPU = False    # 强制使用CPU
    GPU_MEMORY_FRACTION = 0.8  # GPU内存使用比例
    MIXED_PRECISION = False    # 混合精度训练
    
    # PyTorch优化设置
    TORCH_NUM_THREADS = None   # PyTorch线程数，None为自动
    TORCH_DETERMINISTIC = False  # 确定性计算（影响性能）
    TORCH_BENCHMARK = True     # 自动选择最优算法
    
    # 自动调参配置
    ENABLE_AUTO_TUNING = False  # 默认关闭自动调参
    AUTO_TUNING_TRIALS = 50  # 自动调参试验次数
    AUTO_TUNING_TIMEOUT = 3600  # 自动调参超时时间(秒)
    PYTORCH_TUNING_TRIALS = 30  # PyTorch模型专用调参次数（较少以节省时间）
    
    # 模型配置
    AVAILABLE_MODELS = [
        'XGBRanker', 'LGBMRanker', 'LambdaMART', 
        'ListNet', 'NeuralRanker', 'RankNet', 
        'TransformerRanker', 'BM25Ranker'
    ]
    
    # PyTorch模型列表
    PYTORCH_MODELS = ['NeuralRanker', 'RankNet', 'TransformerRanker']
    
    # 传统模型列表
    TRADITIONAL_MODELS = ['XGBRanker', 'LGBMRanker', 'LambdaMART', 'ListNet', 'BM25Ranker']
    
    # 默认模型参数（针对PyTorch优化）
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
            'epochs': 15,                    # 增加epochs
            'batch_size': 64,                # 优化batch_size
            'dropout_rate': 0.2,
            'weight_decay': 1e-5,            # L2正则化
            'early_stopping_patience': 5     # 早停耐心
        },
        'RankNet': {
            'hidden_units': [128, 64, 32],
            'learning_rate': 0.001,
            'epochs': 20,                    # 增加epochs
            'batch_size': 128,               # 更大的batch_size
            'dropout_rate': 0.3,
            'weight_decay': 1e-4,            # L2正则化
            'early_stopping_patience': 7     # 早停耐心
        },
        'TransformerRanker': {
            'num_heads': 4,                  # 保守的头数
            'num_layers': 2,                 # 保守的层数
            'd_model': 64,                   # 适中的模型维度
            'dff': 128,                      # 前馈网络维度
            'learning_rate': 0.001,
            'epochs': 15,                    # 适中的训练轮数
            'batch_size': 64,
            'dropout_rate': 0.1,
            'max_seq_length': 16,            # 序列长度
            'weight_decay': 1e-5,            # L2正则化
            'early_stopping_patience': 5,    # 早停耐心
            'warmup_steps': 1000             # 学习率预热
        },
        'BM25Ranker': {}
    }
    
    # 自动调参搜索空间（针对PyTorch优化）
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
            'epochs': [10, 15, 20, 25],              # 适当的epochs范围
            'batch_size': [32, 64, 128, 256],
            'dropout_rate': [0.1, 0.2, 0.3, 0.4],
            'weight_decay': [1e-6, 1e-5, 1e-4, 1e-3]
        },
        'RankNet': {
            'hidden_units': [
                [64, 32], [128, 64], [128, 64, 32], 
                [256, 128, 64], [64, 32, 16]
            ],
            'learning_rate': [0.0001, 0.0005, 0.001, 0.002],
            'epochs': [15, 20, 25, 30],              # 适当的epochs范围
            'batch_size': [64, 128, 256, 512],
            'dropout_rate': [0.2, 0.3, 0.4, 0.5],
            'weight_decay': [1e-6, 1e-5, 1e-4, 1e-3]
        },
        'TransformerRanker': {
            'num_heads': [2, 4, 8],                  # 保守的选择
            'num_layers': [1, 2, 3],                 # 减少层数选择
            'd_model': [32, 64, 128],                # 合适的模型维度
            'dff': [64, 128, 256],                   # 前馈网络维度
            'learning_rate': [0.0005, 0.001, 0.002], # 稳定的学习率范围
            'epochs': [10, 15, 20],                  # 适当的训练轮数
            'batch_size': [32, 64, 128],             # 合适的批次大小
            'dropout_rate': [0.1, 0.2, 0.3],        # 适中的dropout
            'max_seq_length': [8, 16, 32],           # 不同的序列长度
            'weight_decay': [1e-6, 1e-5, 1e-4]      # L2正则化
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
    
    # PyTorch特定内存配置
    PYTORCH_MAX_MEMORY_MB = None    # 最大内存使用（MB），None为不限制
    PYTORCH_CACHE_SIZE = 1000       # 模型缓存大小
    ENABLE_GPU_MEMORY_MONITOR = True # 启用GPU内存监控
    
    # 训练配置
    DEFAULT_EARLY_STOPPING = True   # 默认启用早停
    DEFAULT_GRADIENT_CLIPPING = 1.0 # 梯度裁剪阈值
    DEFAULT_LR_SCHEDULER = 'cosine' # 学习率调度器
    
    # 输出配置
    SAVE_FEATURE_IMPORTANCE = True
    SAVE_SHAP_VALUES = True
    SAVE_MODEL_PREDICTIONS = True
    FINAL_PREDICTION_FILENAME = "final_predictions.parquet"
    
    # PyTorch模型保存配置
    SAVE_MODEL_STATE_DICT = True    # 保存状态字典
    SAVE_FULL_MODEL = True          # 保存完整模型（备用）
    MODEL_SAVE_FORMAT = 'pth'       # 保存格式: 'pth' 或 'pkl'
    
    @classmethod
    def setup_pytorch_environment(cls):
        """设置PyTorch运行环境"""
        # 设置线程数
        if cls.TORCH_NUM_THREADS is not None:
            torch.set_num_threads(cls.TORCH_NUM_THREADS)
        
        # 设置确定性计算
        if cls.TORCH_DETERMINISTIC:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = cls.TORCH_BENCHMARK
        
        # 设置随机种子
        torch.manual_seed(cls.RANDOM_STATE)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cls.RANDOM_STATE)
        
        print(f"PyTorch环境设置完成:")
        print(f"  版本: {torch.__version__}")
        print(f"  设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        print(f"  线程数: {torch.get_num_threads()}")
        print(f"  确定性: {cls.TORCH_DETERMINISTIC}")
        print(f"  基准测试: {torch.backends.cudnn.benchmark}")
    
    @classmethod
    def get_device_info(cls) -> Dict[str, Any]:
        """获取设备信息"""
        info = {
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cpu_count': torch.get_num_threads(),
        }
        
        if torch.cuda.is_available():
            info.update({
                'cuda_version': torch.version.cuda,
                'gpu_count': torch.cuda.device_count(),
                'gpu_name': torch.cuda.get_device_name(0),
                'gpu_memory': torch.cuda.get_device_properties(0).total_memory / 1024**3,
                'current_device': torch.cuda.current_device()
            })
        
        return info
    
    @classmethod
    def optimize_for_device(cls, device: str = 'auto'):
        """根据设备优化配置"""
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if device == 'cuda' and torch.cuda.is_available():
            # GPU优化配置
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            if gpu_memory < 4:  # 小于4GB显存
                cls.DEFAULT_MODEL_PARAMS['NeuralRanker']['batch_size'] = 32
                cls.DEFAULT_MODEL_PARAMS['RankNet']['batch_size'] = 64
                cls.DEFAULT_MODEL_PARAMS['TransformerRanker']['batch_size'] = 32
                cls.DEFAULT_MODEL_PARAMS['TransformerRanker']['d_model'] = 32
            elif gpu_memory < 8:  # 4-8GB显存
                cls.DEFAULT_MODEL_PARAMS['NeuralRanker']['batch_size'] = 64
                cls.DEFAULT_MODEL_PARAMS['RankNet']['batch_size'] = 128
                cls.DEFAULT_MODEL_PARAMS['TransformerRanker']['batch_size'] = 64
            else:  # 8GB+显存
                cls.DEFAULT_MODEL_PARAMS['NeuralRanker']['batch_size'] = 128
                cls.DEFAULT_MODEL_PARAMS['RankNet']['batch_size'] = 256
                cls.DEFAULT_MODEL_PARAMS['TransformerRanker']['batch_size'] = 128
            
            print(f"已针对GPU优化配置 (显存: {gpu_memory:.1f}GB)")
        else:
            # CPU优化配置
            cls.DEFAULT_MODEL_PARAMS['NeuralRanker']['batch_size'] = 32
            cls.DEFAULT_MODEL_PARAMS['RankNet']['batch_size'] = 64
            cls.DEFAULT_MODEL_PARAMS['TransformerRanker']['batch_size'] = 32
            
            # 减少模型复杂度
            cls.DEFAULT_MODEL_PARAMS['NeuralRanker']['hidden_units'] = [128, 64]
            cls.DEFAULT_MODEL_PARAMS['RankNet']['hidden_units'] = [64, 32]
            cls.DEFAULT_MODEL_PARAMS['TransformerRanker']['d_model'] = 32
            cls.DEFAULT_MODEL_PARAMS['TransformerRanker']['num_layers'] = 1
            
            print("已针对CPU优化配置")
    
    @classmethod
    def get_train_files(cls) -> List[str]:
        """获取训练文件路径列表，自动检测目录结构"""
        train_files = []
        
        # 可能的训练文件路径
        possible_train_paths = [
            os.path.join(cls.DATA_BASE_PATH, "segmented/train"),
        ]
        
        # 尝试每个可能的路径
        for train_path in possible_train_paths:
            if os.path.exists(train_path):
                # 查找训练文件
                for i in range(10):  # 最多检查10个segment
                    possible_files = [
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
            os.path.join(cls.DATA_BASE_PATH, "segmented/test"),
        ]
        
        # 尝试每个可能的路径
        for test_path in possible_test_paths:
            if os.path.exists(test_path):
                # 查找测试文件
                for i in range(10):  # 最多检查10个segment
                    possible_files = [
                        os.path.join(test_path, f"test_segment_{i}.parquet"),
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
        
        # 设置PyTorch环境
        cls.setup_pytorch_environment()
        
        # 根据设备优化配置
        cls.optimize_for_device()
    
    @classmethod
    def get_model_config(cls, model_name: str, use_tuning: bool = False) -> Dict[str, Any]:
        """获取模型配置"""
        if use_tuning and model_name in cls.TUNING_SEARCH_SPACES:
            # 返回搜索空间中的默认值
            search_space = cls.TUNING_SEARCH_SPACES[model_name]
            config = {}
            for param, values in search_space.items():
                if isinstance(values, list):
                    config[param] = values[0]  # 使用第一个值作为默认
                elif isinstance(values, dict) and 'default' in values:
                    config[param] = values['default']
            return config
        else:
            return cls.DEFAULT_MODEL_PARAMS.get(model_name, {}).copy()
    
    @classmethod
    def is_pytorch_model(cls, model_name: str) -> bool:
        """判断是否为PyTorch模型"""
        return model_name in cls.PYTORCH_MODELS
    
    @classmethod
    def get_pytorch_models(cls) -> List[str]:
        """获取所有PyTorch模型名称"""
        return cls.PYTORCH_MODELS.copy()
    
    @classmethod
    def get_traditional_models(cls) -> List[str]:
        """获取所有传统模型名称"""
        return cls.TRADITIONAL_MODELS.copy()

# 自动初始化
if __name__ != "__main__":
    try:
        Config.setup_pytorch_environment()
    except:
        pass  # 静默失败，避免导入时出错