#!/usr/bin/env python3

"""
航班排名系统主入口 - 高效优化版
主要优化：
1. 简化启动流程，去除不必要的检查
2. 直接GPU初始化，减少预热时间
3. 最小化环境设置
4. 快速路径配置
"""

import os
import sys
import time
import warnings
from pathlib import Path

# ==================== 简化环境设置 ====================
def setup_environment_fast():
    """快速环境设置"""
    warnings.filterwarnings('ignore')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['PYTHONWARNINGS'] = 'ignore'
    
    # GPU设置
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    os.environ['OMP_NUM_THREADS'] = str(min(os.cpu_count(), 8))

def setup_python_path_fast():
    """快速设置Python路径"""
    current_file = Path(__file__).resolve()
    
    if current_file.parent.name == 'src':
        project_root = current_file.parent.parent
    else:
        project_root = current_file.parent
    
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / "src"))
    
    os.chdir(project_root)
    return project_root

def check_gpu_fast():
    """快速GPU检查"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✓ GPU可用: {gpu_name}")
            
            # 简单预热，避免复杂的预热操作
            device = torch.device('cuda:0')
            x = torch.randn(100, 100, device=device)
            y = torch.matmul(x, x.t())
            del x, y
            torch.cuda.empty_cache()
            
            return True
    except Exception:
        pass
    print("⚠️ GPU不可用，使用CPU模式")
    return False

def import_core_fast():
    """快速导入核心模块"""
    try:
        from src.core.Core import FlightRankingCore
        return FlightRankingCore
    except ImportError:
        try:
            from core.Core import FlightRankingCore
            return FlightRankingCore
        except ImportError:
            raise ImportError("无法导入FlightRankingCore模块")

def create_fast_config():
    """创建快速配置 - 针对高效训练优化"""
    return {
        'paths': {
            'data_dir': "data/aeroclub-recsys-2025",
            'model_input_dir': "data/aeroclub-recsys-2025/processed",
            'model_save_dir': "data/aeroclub-recsys-2025/models",
            'output_dir': "data/aeroclub-recsys-2025/submissions",
            'log_dir': "logs"
        },
        'data_processing': {
            'chunk_size': 500000,  # 大块处理提升效率
            'n_processes': min(os.cpu_count(), 4),  # 适中的进程数
            'force_reprocess': False
        },
        'training': {
            'segments': [0, 1, 2],
            'model_names': ['XGBRanker', 'LGBMRanker', 'LambdaMART'],
            'use_gpu': True,
            'random_state': 42,
            'use_full_data': True,  # 启用全量数据模式
            'model_configs': {
                'XGBRanker': {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'verbosity': 0
                },
                'LGBMRanker': {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'verbose': -1
                },
                'LambdaMART': {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1
                }
            }
        },
        'prediction': {
            'segments': [0, 1, 2],
            'model_names': ['XGBRanker', 'LGBMRanker', 'LambdaMART'],
            'use_gpu': True,
        },
        'pipeline': {
            'run_data_processing': False,  # 跳过数据处理加快启动
            'run_training': True,
            'run_prediction': True
        },
        'logging': {
            'level': "INFO",
            'format': "%(asctime)s | %(levelname)8s | %(message)s"
        }
    }

def load_config_fast():
    """快速加载配置"""
    try:
        import yaml
        config_path = Path("config/conf.yaml")
        
        if config_path.exists():
            print("📋 加载配置文件...")
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 应用高效优化
            if 'training' in config:
                # 确保使用全量数据模式以获得最佳性能
                config['training']['use_full_data'] = True
                
                # 优化模型参数以提升训练速度
                if 'model_configs' in config['training']:
                    for model_name, model_config in config['training']['model_configs'].items():
                        if model_name in ['XGBRanker', 'LGBMRanker', 'LambdaMART']:
                            model_config.update({
                                'verbosity': 0,
                                'verbose': -1,
                                'n_jobs': -1
                            })
            
            print("✓ 配置文件加载成功")
            return config
        else:
            print("⚠️ 配置文件不存在，使用快速默认配置")
            return create_fast_config()
    except Exception as e:
        print(f"⚠️ 加载配置文件失败: {e}")
        return create_fast_config()

def show_system_info_fast():
    """显示系统信息 - 简化版"""
    import psutil
    
    print("\n" + "="*50)
    print("🖥️  系统信息")
    print("="*50)
    print(f"CPU核心数: {os.cpu_count()}")
    print(f"可用内存: {psutil.virtual_memory().available // (1024**3)}GB")
    
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
    """主函数 - 高效优化版"""
    total_start = time.time()
    
    print("="*60)
    print("🚀 航班排名系统启动 - 高效优化版")
    print("="*60)
    
    try:
        # 1. 快速环境设置
        setup_environment_fast()
        
        # 2. 路径设置
        project_root = setup_python_path_fast()
        print(f"📁 项目根目录: {project_root}")
        
        # 3. 显示系统信息
        show_system_info_fast()
        
        # 4. GPU检查和预热
        gpu_ready = check_gpu_fast()
        
        # 5. 导入核心模块
        print("📦 导入核心模块...")
        import_start = time.time()
        FlightRankingCore = import_core_fast()
        import_time = time.time() - import_start
        print(f"✓ 核心模块导入完成 (耗时: {import_time:.2f}s)")
        
        # 6. 加载配置
        config = load_config_fast()
        
        # 7. 系统初始化
        print("⚙️ 初始化系统...")
        init_start = time.time()
        
        core = FlightRankingCore(config)
        init_time = time.time() - init_start
        print(f"✓ 系统初始化完成 (耗时: {init_time:.2f}s)")
        
        # 8. 显示启动统计
        startup_time = time.time() - total_start
        print(f"\n🎯 系统启动完成! 总耗时: {startup_time:.2f}s")
        print(f"💡 优化状态: GPU {'✓' if gpu_ready else '✗'}, 全量数据模式 {'✓' if config.get('training', {}).get('use_full_data') else '✗'}")
        
        # 9. 运行流水线
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