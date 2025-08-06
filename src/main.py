#!/usr/bin/env python3

"""
航班排名系统主入口 - 性能优化版
解决启动延迟和警告问题
"""

import os
import sys
import time
import warnings
from pathlib import Path

# ==================== 性能优化设置 ====================
def setup_environment():
    """优化环境设置"""
    # 抑制各种警告和不必要的输出
    warnings.filterwarnings('ignore')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # TensorFlow静默
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # CUDA异步模式
    os.environ['PYTHONWARNINGS'] = 'ignore'   # Python警告静默
    os.environ['NUMBA_DISABLE_JIT'] = '0'     # 启用Numba加速
    
    # XGBoost优化
    os.environ['OMP_NUM_THREADS'] = str(min(os.cpu_count(), 8))
    os.environ['MKL_NUM_THREADS'] = str(min(os.cpu_count(), 8))
    
    # 设置CUDA缓存
    os.environ['CUDA_CACHE_DISABLE'] = '0'
    os.environ['CUDA_CACHE_PATH'] = os.path.expanduser('~/.nv/ComputeCache')

def preload_gpu():
    """预加载GPU，减少后续初始化时间"""
    try:
        import torch
        if torch.cuda.is_available():
            print("🚀 正在预热GPU...")
            start_time = time.time()
            
            # 清理GPU内存
            torch.cuda.empty_cache()
            
            # 简单的GPU预热操作
            device = torch.device('cuda:0')
            x = torch.randn(1000, 1000, device=device)
            y = torch.matmul(x, x.t())
            del x, y
            torch.cuda.empty_cache()
            
            warm_time = time.time() - start_time
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✓ GPU预热完成 ({gpu_name}, 耗时: {warm_time:.2f}s)")
            return True
    except Exception as e:
        print(f"⚠️ GPU预热失败: {e}")
        return False
    return False

def setup_python_path():
    """设置Python路径 - 优化版"""
    current_file = Path(__file__).resolve()
    
    if current_file.parent.name == 'src':
        project_root = current_file.parent.parent
    else:
        project_root = current_file.parent
    
    paths_to_add = [
        str(project_root),
        str(project_root / "src"),
    ]
    
    for path in paths_to_add:
        if path not in sys.path:
            sys.path.insert(0, path)
    
    os.chdir(project_root)
    return project_root

def import_flight_ranking_core():
    """导入核心模块 - 优化版"""
    print("📦 正在导入核心模块...")
    import_start = time.time()
    
    try:
        from src.core.Core import FlightRankingCore
        import_time = time.time() - import_start
        print(f"✓ 核心模块导入成功 (耗时: {import_time:.2f}s)")
        return FlightRankingCore
    except ImportError as e1:
        try:
            from core.Core import FlightRankingCore
            import_time = time.time() - import_start
            print(f"✓ 核心模块导入成功 (耗时: {import_time:.2f}s)")
            return FlightRankingCore
        except ImportError as e2:
            print(f"❌ 导入错误1: {e1}")
            print(f"❌ 导入错误2: {e2}")
            raise ImportError("无法导入FlightRankingCore模块")

def create_optimized_config():
    """创建优化配置"""
    return {
        'paths': {
            'data_dir': "data/aeroclub-recsys-2025",
            'model_input_dir': "data/aeroclub-recsys-2025/processed",
            'model_save_dir': "data/aeroclub-recsys-2025/models",
            'output_dir': "data/aeroclub-recsys-2025/submissions",
            'log_dir': "logs"
        },
        'data_processing': {
            'chunk_size': 500000,  # 增大chunk提升效率
            'n_processes': min(os.cpu_count(), 6),  # 限制进程数
            'force_reprocess': False
        },
        'training': {
            'segments': [0, 1, 2],
            'model_names': ['XGBRanker', 'LGBMRanker', 'LambdaMART'],
            'use_gpu': True,
            'random_state': 42,
            'use_full_data': False,
            'model_configs': {
                'XGBRanker': {
                    'n_estimators': 100,  # 减少树数量加快训练
                    'max_depth': 6,       # 减少深度
                    'learning_rate': 0.1, # 提高学习率
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'max_bin': 256        # GPU优化
                },
                'LGBMRanker': {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'max_bin': 255,       # GPU优化
                    'force_col_wise': True
                },
                'LambdaMART': {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'max_bin': 256
                }
            }
        },
        'prediction': {
            'segments': [0, 1, 2],
            'model_names': ['XGBRanker', 'LGBMRanker', 'LambdaMART'],
            'use_gpu': True,
        },
        'pipeline': {
            'run_data_processing': False,  # 跳过数据处理加快测试
            'run_training': True,
            'run_prediction': True
        },
        'logging': {
            'level': "INFO",
            'format': "%(asctime)s | %(levelname)8s | %(name)s | %(message)s"
        }
    }

def load_config():
    """加载配置文件 - 优化版"""
    try:
        import yaml
        config_path = Path("config/conf.yaml")
        
        if config_path.exists():
            print("📋 正在加载配置文件...")
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 应用性能优化
            if 'training' in config and 'model_configs' in config['training']:
                # 优化XGBoost参数
                if 'XGBRanker' in config['training']['model_configs']:
                    config['training']['model_configs']['XGBRanker'].update({
                        'max_bin': 256,
                        'verbosity': 0
                    })
                
                # 优化LightGBM参数
                if 'LGBMRanker' in config['training']['model_configs']:
                    config['training']['model_configs']['LGBMRanker'].update({
                        'max_bin': 255,
                        'verbose': -1,
                        'force_col_wise': True
                    })
            
            print("✓ 配置文件加载成功")
            return config
        else:
            print("⚠️ 配置文件不存在，使用优化默认配置")
            return create_optimized_config()
    except Exception as e:
        print(f"⚠️ 加载配置文件失败: {e}")
        print("使用优化默认配置")
        return create_optimized_config()

def check_dependencies():
    """检查依赖 - 优化版"""
    print("🔍 正在检查依赖...")
    
    required = {
        'pandas': 'pandas',
        'numpy': 'numpy', 
        'sklearn': 'scikit-learn',
        'xgboost': 'xgboost',
        'lightgbm': 'lightgbm'
    }
    
    missing = []
    for pkg_name, install_name in required.items():
        try:
            __import__(pkg_name)
        except ImportError:
            missing.append(install_name)
    
    if missing:
        print(f"❌ 缺少依赖包: {missing}")
        print(f"请安装: pip install {' '.join(missing)}")
        return False
    
    # 检查GPU库
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ GPU可用: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️ GPU不可用，将使用CPU模式")
    except ImportError:
        print("⚠️ PyTorch未安装，部分功能可能受限")
    
    print("✓ 依赖检查完成")
    return True

def show_system_info():
    """显示系统信息"""
    import psutil
    
    print("\n" + "="*50)
    print("🖥️  系统信息")
    print("="*50)
    print(f"CPU核心数: {os.cpu_count()}")
    print(f"内存: {psutil.virtual_memory().total // (1024**3)}GB")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory // (1024**3)}GB")
        else:
            print("GPU: 不可用")
    except:
        print("GPU: 检查失败")
    print("="*50 + "\n")

def main():
    """主函数 - 优化版"""
    total_start = time.time()
    
    print("="*60)
    print("🚀 航班排名系统启动 - 性能优化版")
    print("="*60)
    
    try:
        # 1. 环境优化
        setup_environment()
        
        # 2. 显示系统信息
        show_system_info()
        
        # 3. 路径设置
        project_root = setup_python_path()
        print(f"📁 项目根目录: {project_root}")
        
        # 4. 预热GPU（并行进行）
        gpu_ready = preload_gpu()
        
        # 5. 检查依赖
        if not check_dependencies():
            return 1
        
        # 6. 导入核心模块
        FlightRankingCore = import_flight_ranking_core()
        
        # 7. 加载配置
        config = load_config()
        
        # 8. 系统初始化
        print("⚙️ 正在初始化系统...")
        init_start = time.time()
        core = FlightRankingCore(config)
        init_time = time.time() - init_start
        print(f"✓ 系统初始化完成 (耗时: {init_time:.2f}s)")
        
        # 9. 显示启动统计
        startup_time = time.time() - total_start
        print(f"\n🎯 系统启动完成! 总耗时: {startup_time:.2f}s")
        print(f"💡 优化效果: GPU预热 {'✓' if gpu_ready else '✗'}")
        
        # 10. 运行流水线
        print("\n" + "="*60)
        print("🏃 开始执行流水线...")
        print("="*60)
        
        pipeline_start = time.time()
        success = core.run_full_pipeline()
        pipeline_time = time.time() - pipeline_start
        
        if success:
            print("\n" + "="*60)
            print("✅ 航班排名系统执行成功!")
            print(f"📊 流水线耗时: {pipeline_time:.2f}s")
            print(f"🎯 总耗时: {time.time() - total_start:.2f}s")
            print("="*60)
        else:
            print("\n" + "="*60)
            print("❌ 航班排名系统执行失败")
            print("="*60)
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n⏸️ 用户中断执行")
        return 1
    except Exception as e:
        print(f"\n💥 系统运行错误: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)