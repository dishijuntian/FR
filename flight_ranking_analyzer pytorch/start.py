#!/usr/bin/env python3
"""
启动脚本 - 重构版

专注于：
- 依赖检查
- 环境设置
- 程序启动

作者: Flight Ranking Team
版本: 4.0 (重构版)
"""

import os
import sys
from pathlib import Path
import subprocess


class DependencyChecker:
    """依赖检查器"""
    
    REQUIRED_PACKAGES = [
        'pandas', 'numpy', 'sklearn', 'xgboost', 'lightgbm',
        'torch', 'matplotlib', 'seaborn', 'optuna'
    ]
    
    @classmethod
    def check_all(cls) -> bool:
        """检查所有依赖"""
        print("📦 检查依赖包...")
        
        missing = []
        for package in cls.REQUIRED_PACKAGES:
            if not cls._check_package(package):
                missing.append(package)
        
        if missing:
            print(f"❌ 缺少依赖包: {', '.join(missing)}")
            cls._show_install_instructions(missing)
            return False
        
        print("✅ 所有依赖包检查通过")
        cls._show_pytorch_info()
        return True
    
    @staticmethod
    def _check_package(package: str) -> bool:
        """检查单个包"""
        try:
            if package == 'sklearn':
                import sklearn
            else:
                __import__(package)
            return True
        except ImportError:
            return False
    
    @staticmethod
    def _show_install_instructions(missing: list):
        """显示安装说明"""
        print("\n💡 安装说明:")
        print("pip install -r requirements.txt")
        
        if 'torch' in missing:
            print("\n🔥 PyTorch安装:")
            print("访问 https://pytorch.org/ 选择适合的版本")
            print("CPU版本: pip install torch torchvision")
            print("GPU版本: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    
    @staticmethod
    def _show_pytorch_info():
        """显示PyTorch信息"""
        try:
            import torch
            print(f"PyTorch版本: {torch.__version__}")
            if torch.cuda.is_available():
                print(f"✅ CUDA可用: {torch.cuda.get_device_name(0)}")
            else:
                print("⚠️ CUDA不可用，将使用CPU")
        except:
            pass


class PathManager:
    """路径管理器"""
    
    @staticmethod
    def setup_python_path() -> bool:
        """设置Python路径"""
        print("📁 设置Python路径...")
        
        current_dir = Path(__file__).parent
        src_dir = current_dir / 'src'
        
        if not src_dir.exists():
            print(f"❌ 找不到src目录: {src_dir}")
            return False
        
        # 添加到Python路径
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))
        
        print(f"✅ Python路径已设置: {src_dir}")
        return True


class ConfigValidator:
    """配置验证器"""
    
    @staticmethod
    def validate() -> bool:
        """验证配置"""
        print("⚙️ 检查配置...")
        
        try:
            from config import Config
            
            # 检查数据路径
            if not Config.DATA_BASE_PATH.exists():
                print(f"⚠️ 数据路径不存在: {Config.DATA_BASE_PATH}")
                print("请修改 src/config.py 中的 DATA_BASE_PATH")
                
                choice = input("是否继续运行? (y/n): ").strip().lower()
                if choice != 'y':
                    return False
            else:
                print(f"✅ 数据路径正确: {Config.DATA_BASE_PATH}")
            
            # 确保输出目录
            Config.ensure_paths()
            print(f"✅ 输出目录已创建: {Config.OUTPUT_PATH}")
            
            return True
            
        except Exception as e:
            print(f"❌ 配置检查失败: {e}")
            return False


class ApplicationLauncher:
    """应用启动器"""
    
    def __init__(self):
        self.dependency_checker = DependencyChecker()
        self.path_manager = PathManager()
        self.config_validator = ConfigValidator()
    
    def launch(self):
        """启动应用"""
        print("🚀 航班排序分析器启动脚本 v4.0 (重构版)")
        print("="*60)
        
        # 1. 检查依赖
        if not self.dependency_checker.check_all():
            self._exit_with_message("依赖检查失败")
            return
        
        # 2. 设置路径
        if not self.path_manager.setup_python_path():
            self._exit_with_message("路径设置失败")
            return
        
        # 3. 验证配置
        if not self.config_validator.validate():
            self._exit_with_message("配置验证失败")
            return
        
        # 4. 启动主程序
        self._run_main_program()
    
    def _run_main_program(self):
        """运行主程序"""
        print(f"\n{'='*60}")
        print("启动航班排序分析器")
        print('='*60)
        
        try:
            from main import main
            main()
            print("\n✅ 程序执行完成!")
        except KeyboardInterrupt:
            print("\n⚠️ 程序被用户中断")
        except Exception as e:
            print(f"\n❌ 程序执行失败: {e}")
            self._show_troubleshooting()
    
    def _show_troubleshooting(self):
        """显示故障排除信息"""
        print("\n🔧 故障排除:")
        print("1. 检查 src/config.py 中的数据路径配置")
        print("2. 确保数据文件存在于正确位置")
        print("3. 检查Python版本是否为3.8+")
        print("4. 重新安装依赖: pip install -r requirements.txt")
        print("5. 如果使用GPU，确保CUDA和PyTorch版本兼容")
    
    def _exit_with_message(self, message: str):
        """带消息退出"""
        print(f"\n❌ {message}")
        input("\n按Enter键退出...")


def main():
    """主函数"""
    launcher = ApplicationLauncher()
    launcher.launch()


if __name__ == "__main__":
    main()