#!/usr/bin/env python3
"""
快速启动脚本

该脚本提供简化的配置和启动方式，自动处理常见问题

使用方法:
    python quick_start.py
"""

import os
import sys
import traceback

def check_and_setup_environment():
    """检查并设置运行环境"""
    print("检查运行环境...")
    
    # 检查Python版本
    if sys.version_info < (3, 8):
        print("错误: 需要Python 3.8或更高版本")
        return False
    
    # 检查必需的包
    required_packages = [
        'pandas', 'numpy', 'sklearn', 'xgboost', 'lightgbm', 
        'tensorflow', 'matplotlib', 'seaborn'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"错误: 缺少必需的包: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements.txt")
        return False
    
    print("✓ Python版本和依赖包检查通过")
    return True

def setup_data_paths():
    """设置数据路径"""
    print("\n配置数据路径...")
    
    # 检查是否存在src目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(current_dir, 'src')
    
    if not os.path.exists(src_dir):
        print("错误: 找不到src目录")
        return False
    
    config_file = os.path.join(src_dir, 'config.py')
    if not os.path.exists(config_file):
        print("错误: 找不到配置文件 src/config.py")
        return False
    
    # 读取当前配置
    sys.path.insert(0, src_dir)
    try:
        from config import Config
        
        print(f"当前数据基础路径: {Config.DATA_BASE_PATH}")
        
        # 检查数据路径是否存在
        if not os.path.exists(Config.DATA_BASE_PATH):
            print(f"警告: 数据路径不存在: {Config.DATA_BASE_PATH}")
            
            # 提供修改选项
            while True:
                choice = input("是否修改数据路径? (y/n): ").strip().lower()
                if choice == 'y':
                    new_path = input("请输入新的数据基础路径: ").strip()
                    if os.path.exists(new_path):
                        # 更新配置文件
                        update_config_path(config_file, new_path)
                        print(f"✓ 数据路径已更新为: {new_path}")
                        break
                    else:
                        print(f"路径不存在: {new_path}")
                elif choice == 'n':
                    print("继续使用当前路径（可能导致运行错误）")
                    break
                else:
                    print("请输入 y 或 n")
        else:
            print("✓ 数据路径配置正确")
        
        return True
        
    except Exception as e:
        print(f"配置检查失败: {e}")
        return False

def update_config_path(config_file, new_path):
    """更新配置文件中的数据路径"""
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 替换数据基础路径
        old_line = 'DATA_BASE_PATH = "E:/GIT PROJECT/FR/data/aeroclub-recsys-2025"'
        new_line = f'DATA_BASE_PATH = "{new_path.replace(os.sep, "/")}"'
        
        updated_content = content.replace(old_line, new_line)
        
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(updated_content)
            
    except Exception as e:
        print(f"更新配置文件失败: {e}")

def run_main_program():
    """运行主程序"""
    print("\n启动主程序...")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(current_dir, 'src')
    sys.path.insert(0, src_dir)
    
    try:
        from main import main
        main()
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"\n程序运行失败: {e}")
        print("\n详细错误信息:")
        traceback.print_exc()
        
        print("\n可能的解决方案:")
        print("1. 检查数据文件路径是否正确")
        print("2. 确保所有依赖包已安装")
        print("3. 检查Python版本是否为3.8+")
        return False
    
    return True

def main():
    """快速启动主函数"""
    print("="*60)
    print("航班排序分析器 - 快速启动")
    print("="*60)
    
    # 1. 检查环境
    if not check_and_setup_environment():
        input("\n按Enter键退出...")
        return
    
    # 2. 设置数据路径
    if not setup_data_paths():
        input("\n按Enter键退出...")
        return
    
    # 3. 运行主程序
    if run_main_program():
        print("\n程序执行完成!")
    
    input("\n按Enter键退出...")

if __name__ == "__main__":
    main()