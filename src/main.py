#!/usr/bin/env python3

"""
航班排名系统主入口 - 模块化配置版本
"""

import os
import sys
from pathlib import Path

# 确定项目根目录
current_path = Path(__file__).parent
project_root = current_path.parent if (current_path / "src").exists() else current_path
sys.path.insert(0, str(project_root))

from core.Core import FlightRankingCore

# 默认核心配置文件路径
DEFAULT_CORE_CONFIG = "config/core.yaml"

def main():
    """主函数"""
    try:
        # 检查核心配置文件
        if not Path(DEFAULT_CORE_CONFIG).exists():
            print(f"错误: 核心配置文件不存在 - {DEFAULT_CORE_CONFIG}")
            print("请确保以下配置文件存在:")
            print("  - config/core.yaml")
            print("  - config/data_processing.yaml")
            print("  - config/training.yaml")
            print("  - config/prediction.yaml")
            sys.exit(1)
        
        # 初始化核心控制器
        print(f"加载核心配置: {DEFAULT_CORE_CONFIG}")
        core = FlightRankingCore(DEFAULT_CORE_CONFIG)

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