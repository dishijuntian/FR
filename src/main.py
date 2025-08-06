#!/usr/bin/env python3

"""
航班排名系统主入口 - 完全修复版本
"""

import os
import sys
from pathlib import Path

def setup_python_path():
    """设置Python路径"""
    # 获取正确的项目根目录
    current_file = Path(__file__).resolve()
    
    # 如果当前文件在src目录下，项目根目录是其父目录
    if current_file.parent.name == 'src':
        project_root = current_file.parent.parent
    else:
        project_root = current_file.parent
    
    # 添加必要的路径
    paths_to_add = [
        str(project_root),
        str(project_root / "src"),
    ]
    
    for path in paths_to_add:
        if path not in sys.path:
            sys.path.insert(0, path)
    
    # 切换到项目根目录
    os.chdir(project_root)
    
    print(f"项目根目录: {project_root}")
    print(f"当前工作目录: {os.getcwd()}")
    
    return project_root

def import_flight_ranking_core():
    """导入核心模块"""
    try:
        # 直接导入，因为路径已经设置好了
        from src.core.Core import FlightRankingCore
        return FlightRankingCore
    except ImportError as e1:
        try:
            from core.Core import FlightRankingCore
            return FlightRankingCore
        except ImportError as e2:
            print(f"导入错误1: {e1}")
            print(f"导入错误2: {e2}")
            raise ImportError("无法导入FlightRankingCore模块")

def create_basic_config():
    """创建基础配置"""
    return {
        'paths': {
            'data_dir': "data/aeroclub-recsys-2025",
            'model_input_dir': "data/aeroclub-recsys-2025/processed",
            'model_save_dir': "data/aeroclub-recsys-2025/models",
            'output_dir': "data/aeroclub-recsys-2025/submissions",
            'log_dir': "logs"
        },
        'data_processing': {
            'chunk_size': 200000,
            'n_processes': None,
            'force_reprocess': False
        },
        'training': {
            'segments': [0, 1, 2],
            'model_names': ['XGBRanker', 'LGBMRanker'],
            'use_gpu': True,
            'random_state': 42,
            'use_full_data': False,
            'model_configs': {}
        },
        'prediction': {
            'segments': [0, 1, 2],
            'model_names': ['XGBRanker', 'LGBMRanker'],
            'use_gpu': True,
            'ensemble_weights': {
                'XGBRanker': 0.5,
                'LGBMRanker': 0.5
            }
        },
        'pipeline': {
            'run_data_processing': True,
            'run_training': True,
            'run_prediction': True
        },
        'logging': {
            'level': "INFO",
            'format': "%(asctime)s | %(levelname)8s | %(name)s | %(message)s"
        }
    }

def load_config():
    """加载配置文件"""
    try:
        import yaml
        config_path = Path("config/conf.yaml")
        
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        else:
            print("使用默认配置")
            return create_basic_config()
    except Exception as e:
        print(f"加载配置文件失败: {e}")
        print("使用默认配置")
        return create_basic_config()

def check_basic_dependencies():
    """检查基础依赖"""
    required = ['pandas', 'numpy', 'sklearn']
    missing = []
    
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    
    if missing:
        print(f"缺少依赖包: {missing}")
        print(f"请安装: pip install {' '.join(missing)}")
        return False
    
    return True

def main():
    """主函数"""
    print("=" * 60)
    print("航班排名系统启动")
    print("=" * 60)
    
    try:
        # 设置路径
        project_root = setup_python_path()
        
        # 检查基础依赖
        if not check_basic_dependencies():
            print("依赖检查失败")
            return 1
        
        print("✓ 依赖检查通过")
        
        # 导入核心模块
        FlightRankingCore = import_flight_ranking_core()
        print("✓ 核心模块导入成功")
        
        # 加载配置
        config = load_config()
        print("✓ 配置加载成功")
        
        # 运行系统
        core = FlightRankingCore(config)
        print("✓ 系统初始化成功")
        
        success = core.run_full_pipeline()
        
        if success:
            print("\n" + "=" * 60)
            print("✓ 航班排名系统执行成功")
            print("=" * 60)
        else:
            print("\n" + "=" * 60)
            print("✗ 航班排名系统执行失败")
            print("=" * 60)
        
        return 0 if success else 1
        
    except Exception as e:
        print(f"\n系统运行错误: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)