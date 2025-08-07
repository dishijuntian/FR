#!/usr/bin/env python3
"""
最终启动脚本

该脚本解决所有导入和路径问题，提供最可靠的启动方式

使用方法:
    python start.py
"""

import os
import sys
import subprocess
import traceback

def check_dependencies():
    """检查必要的依赖"""
    required_packages = [
        'pandas', 'numpy', 'sklearn', 'xgboost', 'lightgbm', 
        'tensorflow', 'matplotlib', 'seaborn', 'shap', 'optuna'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ 缺少依赖包: {', '.join(missing)}")
        print("请运行: pip install -r requirements.txt")
        return False
    
    print("✅ 所有依赖包检查通过")
    return True

def setup_paths():
    """设置Python路径"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(current_dir, 'src')
    
    if not os.path.exists(src_dir):
        print(f"❌ 找不到src目录: {src_dir}")
        return False
    
    # 确保src目录在Python路径中
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    
    print(f"✅ Python路径已设置: {src_dir}")
    return True

def check_config():
    """检查配置"""
    try:
        import config
        data_path = config.Config.DATA_BASE_PATH
        
        if not os.path.exists(data_path):
            print(f"⚠️  数据路径不存在: {data_path}")
            print("请修改 src/config.py 中的 DATA_BASE_PATH")
            
            choice = input("是否继续运行? (y/n): ").strip().lower()
            if choice != 'y':
                return False
        else:
            print(f"✅ 数据路径配置正确: {data_path}")
        
        # 确保输出目录存在
        config.Config.ensure_output_dir()
        print(f"✅ 输出目录已创建: {config.Config.OUTPUT_PATH}")
        
        return True
        
    except Exception as e:
        print(f"❌ 配置检查失败: {e}")
        return False

def run_analysis():
    """运行分析程序"""
    print("\n" + "="*60)
    print("启动航班排序分析器")
    print("="*60)
    
    try:
        # 导入主程序
        import main
        
        # 运行主程序
        main.main()
        
        print("\n✅ 程序执行完成!")
        return True
        
    except KeyboardInterrupt:
        print("\n⚠️  程序被用户中断")
        return False
        
    except Exception as e:
        print(f"\n❌ 程序执行失败: {e}")
        print("\n详细错误信息:")
        traceback.print_exc()
        
        print("\n🔧 可能的解决方案:")
        print("1. 检查 src/config.py 中的数据路径配置")
        print("2. 确保数据文件存在于正确位置")
        print("3. 检查Python版本是否为3.8+")
        print("4. 重新安装依赖: pip install -r requirements.txt")
        return False

def main():
    """主函数"""
    print("🚀 航班排序分析器启动脚本")
    print("="*40)
    
    # 1. 检查依赖
    print("\n📦 检查依赖包...")
    if not check_dependencies():
        input("\n按Enter键退出...")
        return
    
    # 2. 设置路径
    print("\n📁 设置Python路径...")
    if not setup_paths():
        input("\n按Enter键退出...")
        return
    
    # 3. 检查配置
    print("\n⚙️  检查配置...")
    if not check_config():
        input("\n按Enter键退出...")
        return
    
    # 4. 运行分析
    success = run_analysis()
    
    if success:
        print("\n🎉 所有任务完成!")
    else:
        print("\n❌ 执行过程中出现问题")
    
    input("\n按Enter键退出...")

if __name__ == "__main__":
    main()  