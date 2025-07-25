#!/usr/bin/env python3
"""
航班排名系统主入口
读取配置文件并启动核心控制器
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any

# 添加项目路径
current_path = Path(__file__).parent
project_root = current_path.parent
sys.path.insert(0, str(current_path))

from core.Core import FlightRankingCore


def load_config(config_path: str) -> Dict[str, Any]:
    """
    加载YAML配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        Dict: 配置字典
        
    Raises:
        FileNotFoundError: 配置文件不存在
        yaml.YAMLError: YAML格式错误
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 验证必要的配置项
        required_sections = ['paths', 'data_processing', 'training', 'prediction', 'pipeline', 'logging']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"配置文件缺少必要部分: {section}")
        
        return config
        
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"YAML配置文件格式错误: {e}")


def validate_config(config: Dict[str, Any]) -> bool:
    """
    验证配置文件的有效性
    
    Args:
        config: 配置字典
        
    Returns:
        bool: 配置是否有效
    """
    try:
        # 验证路径配置
        paths = config['paths']
        required_paths = ['data_dir', 'model_save_dir', 'output_dir', 'log_dir']
        for path_key in required_paths:
            if path_key not in paths:
                print(f"错误: 路径配置缺少 {path_key}")
                return False
        
        # 验证数据处理配置
        data_config = config['data_processing']
        if 'chunk_size' not in data_config or data_config['chunk_size'] <= 0:
            print("错误: data_processing.chunk_size 必须是正整数")
            return False
        
        # 验证训练配置
        training_config = config['training']
        required_training = ['segments', 'use_gpu', 'random_state']
        for key in required_training:
            if key not in training_config:
                print(f"错误: 训练配置缺少 {key}")
                return False
        
        # 验证预测配置
        prediction_config = config['prediction']
        required_prediction = ['segments', 'model_name', 'use_gpu', 'random_state']
        for key in required_prediction:
            if key not in prediction_config:
                print(f"错误: 预测配置缺少 {key}")
                return False
        
        # 验证流水线配置
        pipeline_config = config['pipeline']
        required_pipeline = ['run_data_processing', 'run_training', 'run_prediction']
        for key in required_pipeline:
            if key not in pipeline_config:
                print(f"错误: 流水线配置缺少 {key}")
                return False
        
        return True
        
    except Exception as e:
        print(f"配置验证出错: {e}")
        return False


def setup_argument_parser() -> argparse.ArgumentParser:
    """
    设置命令行参数解析器
    
    Returns:
        ArgumentParser: 配置好的参数解析器
    """
    parser = argparse.ArgumentParser(
        description="航班排名系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
                使用示例:
                python main.py                          # 使用默认配置运行完整流水线
                python main.py --config custom.yaml     # 使用自定义配置文件
                python main.py --mode data              # 只执行数据处理
                python main.py --mode training          # 只执行模型训练
                python main.py --mode prediction        # 只执行模型预测
                python main.py --status                 # 查看状态报告
                python main.py --force                  # 强制重新处理所有步骤
                """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config/conf.yaml',
        help='配置文件路径 (默认: config/conf.yaml)'
    )
    
    parser.add_argument(
        '--mode', '-m',
        choices=['full', 'data', 'training', 'prediction'],
        default='full',
        help='运行模式 (默认: full)'
    )
    
    parser.add_argument(
        '--status', '-s',
        action='store_true',
        help='显示流水线状态报告并退出'
    )
    
    parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='强制重新处理（忽略现有文件）'
    )
    
    parser.add_argument(
        '--no-verify',
        action='store_true',
        help='跳过数据验证步骤'
    )
    
    parser.add_argument(
        '--segments',
        type=int,
        nargs='+',
        help='指定要处理的数据段 (例: --segments 0 1 2)'
    )
    
    parser.add_argument(
        '--model',
        choices=['XGBRanker', 'LGBMRanker'],
        help='指定预测使用的模型'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='详细输出模式'
    )
    
    return parser


def main():
    """主函数"""
    # 解析命令行参数
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    try:
        # 加载配置文件
        config_path = os.path.join(project_root, args.config)
        print(f"加载配置文件: {config_path}")
        config = load_config(config_path)
        
        # 验证配置
        if not validate_config(config):
            print("配置文件验证失败，退出")
            sys.exit(1)
        
        # 根据命令行参数调整配置
        if args.force:
            config['data_processing']['force_reprocess'] = True
        
        if args.no_verify:
            config['data_processing']['verify_results'] = False
        
        if args.segments:
            config['training']['segments'] = args.segments
            config['prediction']['segments'] = args.segments
        
        if args.model:
            config['prediction']['model_name'] = args.model
        
        if args.verbose:
            config['logging']['level'] = 'DEBUG'
        
        # 初始化核心控制器
        print("初始化航班排名系统核心控制器...")
        core = FlightRankingCore(config)
        
        # 根据模式执行相应操作
        if args.status:
            # 显示状态报告
            core.print_status_report()
            
        elif args.mode == 'full':
            # 运行完整流水线
            success = core.run_full_pipeline()
            sys.exit(0 if success else 1)
            
        elif args.mode == 'data':
            # 只执行数据处理
            success = core.run_data_processing(
                force=args.force,
                verify=not args.no_verify
            )
            sys.exit(0 if success else 1)
            
        elif args.mode == 'training':
            # 只执行模型训练
            segments = args.segments if args.segments else None
            success = core.run_model_training(segments=segments)
            sys.exit(0 if success else 1)
            
        elif args.mode == 'prediction':
            # 只执行模型预测
            segments = args.segments if args.segments else None
            model_name = args.model if args.model else None
            success = core.run_model_prediction(segments=segments, model_name=model_name)
            sys.exit(0 if success else 1)
        
    except FileNotFoundError as e:
        print(f"文件错误: {e}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"配置文件错误: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"运行错误: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()