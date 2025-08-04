#!/usr/bin/env python3

"""
航班排名系统主入口
"""

import os
import sys
import yaml
from pathlib import Path
from typing import Dict, Any

# 确定项目根目录
current_path = Path(__file__).parent
project_root = current_path.parent if (current_path / "src").exists() else current_path
sys.path.insert(0, str(project_root))

from core.Core import FlightRankingCore

# 默认配置文件路径
DEFAULT_CONFIG_PATH = "config/conf.yaml"

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

def main():
    """主函数"""
    try:
        # 加载配置文件
        print(f"加载配置文件: {DEFAULT_CONFIG_PATH}")
        config = load_config(DEFAULT_CONFIG_PATH)
        
        # 初始化核心控制器
        print("初始化系统...")
        core = FlightRankingCore(config)

        # 执行完整流水线
        print("=" * 50)
        print("开始执行完整流水线")
        print("=" * 50)
        
        success = core.run_full_pipeline()
        
        # 输出结果
        if success:
            print("\n✓ 航班排名系统执行成功")
        else:
            print("\n✗ 航班排名系统执行失败")
        
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
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()