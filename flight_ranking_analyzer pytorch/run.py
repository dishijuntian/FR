#!/usr/bin/env python3
"""
简化启动脚本

该脚本解决模块导入路径问题，并提供简化的启动方式

使用方法:
    python run.py
"""

import os
import sys

# 确保能找到src目录
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')

if not os.path.exists(src_dir):
    print("错误: 找不到src目录")
    print(f"当前目录: {current_dir}")
    print("请确保在正确的项目根目录下运行此脚本")
    sys.exit(1)

# 添加src目录到Python路径
sys.path.insert(0, src_dir)

try:
    # 导入并运行主程序
    from main import main
    
    if __name__ == "__main__":
        main()
        
except ImportError as e:
    print(f"导入模块失败: {e}")
    print("请检查依赖是否正确安装:")
    print("pip install -r requirements.txt")
    sys.exit(1)
except Exception as e:
    print(f"程序运行失败: {e}")
    sys.exit(1)