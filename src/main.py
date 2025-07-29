#!/usr/bin/env python3

"""
航班排名系统主入口
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any

current_path = Path(__file__).parent
project_root = current_path.parent if (current_path / "src").exists() else current_path
sys.path.insert(0, str(project_root))

from core.Core import FlightRankingCore


def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"配置文件格式错误: {e}")


def setup_argument_parser() -> argparse.ArgumentParser:
    """设置命令行参数"""
    parser = argparse.ArgumentParser(
        description="航班排名系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python main.py                          # 运行完整流水线
  python main.py --config conf.yaml       # 使用指定配置
  python main.py --mode data              # 只执行数据处理
  python main.py --mode training          # 只执行模型训练
  python main.py --mode prediction        # 只执行模型预测
  python main.py --segments 0 1 2         # 指定数据段
  python main.py --models XGBRanker RankNet # 指定模型
  python main.py --force                  # 强制重新处理
  python main.py --gpu off                # 禁用GPU
        """
    )
    
    parser.add_argument('--config', '-c', default='config/conf.yaml',
                       help='配置文件路径')
    
    parser.add_argument('--mode', '-m', 
                       choices=['full', 'data', 'training', 'prediction'],
                       default='full', help='运行模式')
    
    parser.add_argument('--segments', type=int, nargs='+',
                       help='指定数据段 (例: --segments 0 1 2)')
    
    parser.add_argument('--models', nargs='+',
                       choices=['XGBRanker', 'LGBMRanker', 'RankNet', 'TransformerRanker'],
                       help='指定模型')
    
    parser.add_argument('--gpu', choices=['on', 'off'],
                       help='GPU设置')
    
    parser.add_argument('--force', '-f', action='store_true',
                       help='强制重新处理')
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='详细输出')
    
    parser.add_argument('--status', action='store_true',
                       help='显示状态信息')
    
    return parser


def apply_cli_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """应用命令行参数覆盖"""
    # GPU设置
    if args.gpu:
        gpu_setting = args.gpu == 'on'
        config['training']['use_gpu'] = gpu_setting
        config['prediction']['use_gpu'] = gpu_setting
    
    # 强制重新处理
    if args.force:
        config['data_processing']['force_reprocess'] = True
    
    # 数据段设置
    if args.segments:
        config['training']['segments'] = args.segments
        config['prediction']['segments'] = args.segments
    
    # 模型设置
    if args.models:
        config['training']['model_names'] = args.models
        config['prediction']['model_names'] = args.models
    
    # 详细输出
    if args.verbose:
        config['logging']['level'] = 'DEBUG'
    
    return config


def print_status(core: FlightRankingCore):
    """打印状态信息"""
    status = core.get_status()
    
    print("=" * 50)
    print("系统状态")
    print("=" * 50)
    print(f"GPU可用: {'是' if status['gpu_available'] else '否'}")
    print(f"训练数据: {len(status['train_data'])} 文件")
    print(f"测试数据: {len(status['test_data'])} 文件")
    print(f"模型文件: {len(status['models'])} 个")
    print("=" * 50)


def main():
    """主函数"""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    try:
        # 加载配置
        print(f"加载配置: {args.config}")
        config = load_config(args.config)
        
        # 应用命令行覆盖
        config = apply_cli_overrides(config, args)
        
        # 初始化核心控制器
        print("初始化系统...")
        core = FlightRankingCore(config)
        
        # 显示状态
        if args.status:
            print_status(core)
            return
        
        # 执行操作
        success = True
        
        if args.mode == 'full':
            success = core.run_full_pipeline()
        elif args.mode == 'data':
            success = core.run_data_processing(force=args.force)
        elif args.mode == 'training':
            success = core.run_model_training(segments=args.segments)
        elif args.mode == 'prediction':
            success = core.run_model_prediction(
                segments=args.segments,
                model_names=args.models
            )
        
        # 输出结果
        if success:
            print("\n✓ 执行成功")
        else:
            print("\n✗ 执行失败")
        
        sys.exit(0 if success else 1)
        
    except FileNotFoundError as e:
        print(f"文件错误: {e}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"配置错误: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n用户中断")
        sys.exit(130)
    except Exception as e:
        print(f"运行错误: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()