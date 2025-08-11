"""
工具函数模块 - 重构版
统一管理辅助功能：进度跟踪、系统检查、文件管理等

作者: Flight Ranking Team
版本: 5.0 (重构版)
"""

import os
import sys
import psutil
import torch
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator
from contextlib import contextmanager
import time
import warnings
from dataclasses import dataclass

warnings.filterwarnings('ignore')

# 尝试导入tqdm用于进度条
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    
# 尝试导入Rich用于美化输出
try:
    from rich.console import Console
    from rich.progress import Progress, TaskID, BarColumn, TextColumn, TimeRemainingColumn
    from rich.table import Table
    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False
    console = None


@dataclass
class SystemInfo:
    """系统信息数据类"""
    python_version: str
    platform: str
    cpu_count: int
    memory_total: float
    memory_available: float
    gpu_available: bool
    gpu_name: Optional[str] = None
    gpu_memory: Optional[float] = None
    torch_version: str = None
    cuda_version: Optional[str] = None


class ProgressTracker:
    """进度跟踪器"""
    
    def __init__(self, use_rich: bool = None):
        """
        初始化进度跟踪器
        
        Args:
            use_rich: 是否使用Rich库（None表示自动检测）
        """
        self.use_rich = use_rich if use_rich is not None else HAS_RICH
        self.current_progress = None
        self.current_task = None
    
    @contextmanager
    def create_progress_bar(self, total: int, description: str = "处理中"):
        """创建进度条上下文管理器"""
        if self.use_rich and HAS_RICH:
            with Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
                console=console
            ) as progress:
                task = progress.add_task(description, total=total)
                yield lambda advance=1: progress.update(task, advance=advance)
        elif HAS_TQDM:
            with tqdm(total=total, desc=description, 
                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
                yield lambda advance=1: pbar.update(advance)
        else:
            # 简单的文本进度显示
            current = [0]
            
            def update(advance=1):
                current[0] += advance
                progress_pct = (current[0] / total) * 100
                print(f"\r{description}: {current[0]}/{total} ({progress_pct:.1f}%)", end="")
                if current[0] >= total:
                    print()
            
            yield update
    
    @contextmanager
    def create_training_progress(self, num_segments: int):
        """创建训练专用进度跟踪"""
        if self.use_rich and HAS_RICH:
            yield RichTrainingProgress(num_segments)
        else:
            yield SimpleTrainingProgress(num_segments)
    
    def show_spinner(self, message: str = "处理中..."):
        """显示旋转器（仅Rich支持）"""
        if self.use_rich and HAS_RICH:
            return console.status(message)
        else:
            print(f"{message}")
            return SimpleSpinner()


class SimpleTrainingProgress:
    """简单训练进度跟踪"""
    
    def __init__(self, num_segments: int):
        self.num_segments = num_segments
        self.current_segment = 0
        self.start_time = time.time()
    
    def update_current_stage(self, stage_description: str):
        """更新当前阶段"""
        elapsed = time.time() - self.start_time
        print(f"[{elapsed:.1f}s] {stage_description}")
    
    def complete_stage(self, success: bool = True):
        """完成当前阶段"""
        self.current_segment += 1
        status = "✅" if success else "❌"
        progress_pct = (self.current_segment / self.num_segments) * 100
        print(f"{status} 进度: {self.current_segment}/{self.num_segments} ({progress_pct:.1f}%)")


class RichTrainingProgress:
    """Rich训练进度跟踪"""
    
    def __init__(self, num_segments: int):
        self.num_segments = num_segments
        self.current_segment = 0
        self.start_time = time.time()
        self.console = Console()
    
    def update_current_stage(self, stage_description: str):
        """更新当前阶段"""
        elapsed = time.time() - self.start_time
        self.console.print(f"[bold blue][{elapsed:.1f}s][/bold blue] {stage_description}")
    
    def complete_stage(self, success: bool = True):
        """完成当前阶段"""
        self.current_segment += 1
        status = "[green]✅[/green]" if success else "[red]❌[/red]"
        progress_pct = (self.current_segment / self.num_segments) * 100
        self.console.print(f"{status} 进度: {self.current_segment}/{self.num_segments} ({progress_pct:.1f}%)")


class SimpleSpinner:
    """简单旋转器（无操作）"""
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class SystemChecker:
    """系统检查器"""
    
    def __init__(self):
        self.info = None
    
    def check_system(self) -> Dict[str, Any]:
        """检查系统信息"""
        # 基础系统信息
        system_info = {
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            'platform': sys.platform,
            'cpu_count': os.cpu_count(),
        }
        
        # 内存信息
        try:
            memory = psutil.virtual_memory()
            system_info.update({
                'memory_total_gb': memory.total / (1024**3),
                'memory_available_gb': memory.available / (1024**3),
                'memory_usage_percent': memory.percent
            })
        except:
            system_info.update({
                'memory_total_gb': 'Unknown',
                'memory_available_gb': 'Unknown',
                'memory_usage_percent': 'Unknown'
            })
        
        # GPU信息
        gpu_info = self._check_gpu()
        system_info.update(gpu_info)
        
        # 依赖库检查
        dependencies = self._check_dependencies()
        system_info.update(dependencies)
        
        self.info = system_info
        return system_info
    
    def _check_gpu(self) -> Dict[str, Any]:
        """检查GPU信息"""
        gpu_info = {
            'gpu_available': torch.cuda.is_available(),
            'torch_version': torch.__version__
        }
        
        if torch.cuda.is_available():
            try:
                gpu_info.update({
                    'gpu_name': torch.cuda.get_device_name(0),
                    'gpu_memory_gb': torch.cuda.get_device_properties(0).total_memory / (1024**3),
                    'cuda_version': torch.version.cuda,
                    'gpu_count': torch.cuda.device_count()
                })
            except Exception as e:
                gpu_info['gpu_error'] = str(e)
        
        return gpu_info
    
    def _check_dependencies(self) -> Dict[str, Any]:
        """检查依赖库"""
        dependencies = {}
        
        # 检查核心依赖
        core_deps = ['pandas', 'numpy', 'matplotlib', 'seaborn', 'sklearn']
        for dep in core_deps:
            try:
                __import__(dep)
                dependencies[f'{dep}_available'] = True
            except ImportError:
                dependencies[f'{dep}_available'] = False
        
        # 检查可选依赖
        optional_deps = ['xgboost', 'lightgbm', 'optuna', 'tqdm', 'rich']
        for dep in optional_deps:
            try:
                __import__(dep)
                dependencies[f'{dep}_available'] = True
            except ImportError:
                dependencies[f'{dep}_available'] = False
        
        return dependencies
    
    def print_system_summary(self):
        """打印系统信息摘要"""
        if self.info is None:
            self.check_system()
        
        if HAS_RICH and console:
            self._print_rich_summary()
        else:
            self._print_simple_summary()
    
    def _print_rich_summary(self):
        """使用Rich打印美化的系统信息"""
        table = Table(title="系统信息摘要")
        table.add_column("项目", style="cyan", no_wrap=True)
        table.add_column("值", style="magenta")
        table.add_column("状态", justify="center")
        
        # 基础信息
        table.add_row("Python版本", self.info['python_version'], "✅")
        table.add_row("平台", self.info['platform'], "✅")
        table.add_row("CPU核心数", str(self.info['cpu_count']), "✅")
        
        # 内存信息
        if isinstance(self.info['memory_total_gb'], (int, float)):
            table.add_row("总内存", f"{self.info['memory_total_gb']:.1f} GB", "✅")
            table.add_row("可用内存", f"{self.info['memory_available_gb']:.1f} GB", "✅")
        
        # GPU信息
        gpu_status = "✅" if self.info['gpu_available'] else "❌"
        table.add_row("GPU可用", str(self.info['gpu_available']), gpu_status)
        
        if self.info['gpu_available']:
            table.add_row("GPU名称", self.info.get('gpu_name', 'Unknown'), "✅")
            if 'gpu_memory_gb' in self.info:
                table.add_row("GPU内存", f"{self.info['gpu_memory_gb']:.1f} GB", "✅")
        
        # 关键依赖
        key_deps = ['pandas', 'torch', 'xgboost', 'lightgbm']
        for dep in key_deps:
            available = self.info.get(f'{dep}_available', False)
            status = "✅" if available else "❌"
            table.add_row(f"{dep}可用", str(available), status)
        
        console.print(table)
    
    def _print_simple_summary(self):
        """打印简单的系统信息"""
        print("\n系统信息摘要:")
        print("=" * 40)
        print(f"Python版本: {self.info['python_version']}")
        print(f"平台: {self.info['platform']}")
        print(f"CPU核心数: {self.info['cpu_count']}")
        
        if isinstance(self.info['memory_total_gb'], (int, float)):
            print(f"总内存: {self.info['memory_total_gb']:.1f} GB")
            print(f"可用内存: {self.info['memory_available_gb']:.1f} GB")
        
        gpu_status = "✅" if self.info['gpu_available'] else "❌"
        print(f"GPU可用: {self.info['gpu_available']} {gpu_status}")
        
        if self.info['gpu_available']:
            print(f"GPU名称: {self.info.get('gpu_name', 'Unknown')}")
            if 'gpu_memory_gb' in self.info:
                print(f"GPU内存: {self.info['gpu_memory_gb']:.1f} GB")
        
        print("\n关键依赖:")
        key_deps = ['pandas', 'torch', 'xgboost', 'lightgbm', 'optuna']
        for dep in key_deps:
            available = self.info.get(f'{dep}_available', False)
            status = "✅" if available else "❌"
            print(f"  {dep}: {available} {status}")


class FileManager:
    """文件管理器"""
    
    def __init__(self, paths_config):
        """
        初始化文件管理器
        
        Args:
            paths_config: 路径配置对象
        """
        self.paths = paths_config
    
    def find_data_files(self) -> tuple[List[Path], List[Path]]:
        """查找训练和测试文件"""
        train_files = []
        test_files = []
        
        # 查找训练文件
        if self.paths.train_data.exists():
            train_files = sorted(self.paths.train_data.glob("*.parquet"))
            if not train_files:
                train_files = sorted(self.paths.train_data.glob("*.csv"))
        
        # 查找测试文件
        if self.paths.test_data.exists():
            test_files = sorted(self.paths.test_data.glob("*.parquet"))
            if not test_files:
                test_files = sorted(self.paths.test_data.glob("*.csv"))
        
        # 打印发现的文件
        print(f"\n文件发现:")
        print(f"训练文件: {len(train_files)} 个")
        for i, file_path in enumerate(train_files[:5]):  # 只显示前5个
            print(f"  {i}: {file_path.name}")
        if len(train_files) > 5:
            print(f"  ... 还有 {len(train_files) - 5} 个文件")
        
        print(f"测试文件: {len(test_files)} 个")
        for i, file_path in enumerate(test_files[:5]):  # 只显示前5个
            print(f"  {i}: {file_path.name}")
        if len(test_files) > 5:
            print(f"  ... 还有 {len(test_files) - 5} 个文件")
        
        if not train_files:
            raise FileNotFoundError(f"在 {self.paths.train_data} 中未找到训练文件")
        
        return train_files, test_files
    
    def validate_file(self, file_path: Path) -> Dict[str, Any]:
        """验证文件"""
        validation_result = {
            'exists': file_path.exists(),
            'readable': False,
            'size_mb': 0,
            'error': None
        }
        
        if validation_result['exists']:
            try:
                validation_result['size_mb'] = file_path.stat().st_size / (1024 * 1024)
                # 尝试读取文件头部
                if file_path.suffix == '.parquet':
                    import pandas as pd
                    df = pd.read_parquet(file_path, nrows=5)
                    validation_result['readable'] = True
                    validation_result['columns'] = list(df.columns)
                    validation_result['sample_shape'] = df.shape
                elif file_path.suffix == '.csv':
                    import pandas as pd
                    df = pd.read_csv(file_path, nrows=5)
                    validation_result['readable'] = True
                    validation_result['columns'] = list(df.columns)
                    validation_result['sample_shape'] = df.shape
                else:
                    validation_result['readable'] = True
                    
            except Exception as e:
                validation_result['error'] = str(e)
        
        return validation_result
    
    def create_backup(self, file_path: Path, backup_dir: Optional[Path] = None) -> Path:
        """创建文件备份"""
        if backup_dir is None:
            backup_dir = file_path.parent / "backup"
        
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成备份文件名（添加时间戳）
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_name = f"{file_path.stem}_{timestamp}{file_path.suffix}"
        backup_path = backup_dir / backup_name
        
        # 复制文件
        import shutil
        shutil.copy2(file_path, backup_path)
        
        return backup_path
    
    def clean_temp_files(self, temp_dir: Optional[Path] = None):
        """清理临时文件"""
        if temp_dir is None:
            temp_dir = Path.cwd() / "temp"
        
        if temp_dir.exists():
            import shutil
            try:
                shutil.rmtree(temp_dir)
                print(f"已清理临时目录: {temp_dir}")
            except Exception as e:
                print(f"清理临时目录失败: {e}")


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.start_time = None
        self.checkpoints = {}
    
    def start(self):
        """开始监控"""
        self.start_time = time.time()
        self.checkpoints = {}
    
    def checkpoint(self, name: str) -> float:
        """添加检查点"""
        if self.start_time is None:
            self.start()
        
        current_time = time.time()
        elapsed = current_time - self.start_time
        self.checkpoints[name] = elapsed
        return elapsed
    
    def get_summary(self) -> Dict[str, float]:
        """获取性能摘要"""
        summary = {}
        
        if self.start_time:
            summary['total_time'] = time.time() - self.start_time
        
        # 计算各阶段耗时
        checkpoint_names = list(self.checkpoints.keys())
        for i, name in enumerate(checkpoint_names):
            if i == 0:
                summary[f'{name}_duration'] = self.checkpoints[name]
            else:
                prev_time = self.checkpoints[checkpoint_names[i-1]]
                summary[f'{name}_duration'] = self.checkpoints[name] - prev_time
        
        return summary
    
    def print_summary(self):
        """打印性能摘要"""
        summary = self.get_summary()
        
        print("\n性能监控摘要:")
        print("=" * 40)
        
        for key, value in summary.items():
            if key == 'total_time':
                print(f"总用时: {value:.2f}秒")
            else:
                stage_name = key.replace('_duration', '')
                print(f"{stage_name}: {value:.2f}秒")


class Logger:
    """简单日志记录器"""
    
    def __init__(self, log_file: Optional[Path] = None, console_output: bool = True):
        """
        初始化日志记录器
        
        Args:
            log_file: 日志文件路径
            console_output: 是否输出到控制台
        """
        self.log_file = log_file
        self.console_output = console_output
        
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
    
    def log(self, message: str, level: str = "INFO"):
        """记录日志"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] [{level}] {message}"
        
        if self.console_output:
            print(log_message)
        
        if self.log_file:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(log_message + '\n')
    
    def info(self, message: str):
        """记录信息日志"""
        self.log(message, "INFO")
    
    def warning(self, message: str):
        """记录警告日志"""
        self.log(message, "WARNING")
    
    def error(self, message: str):
        """记录错误日志"""
        self.log(message, "ERROR")


# 便捷函数
def create_logger(output_path: Path, name: str = "flight_ranking") -> Logger:
    """创建日志记录器的便捷函数"""
    log_file = output_path / f"{name}.log"
    return Logger(log_file, console_output=True)


def format_time(seconds: float) -> str:
    """格式化时间显示"""
    if seconds < 60:
        return f"{seconds:.1f}秒"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{int(minutes)}分{secs:.1f}秒"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{int(hours)}小时{int(minutes)}分"


def format_memory(bytes_size: float) -> str:
    """格式化内存大小显示"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f}{unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f}TB"


@contextmanager
def timer(description: str = "操作"):
    """计时器上下文管理器"""
    start_time = time.time()
    print(f"开始{description}...")
    try:
        yield
    finally:
        elapsed = time.time() - start_time
        print(f"{description}完成，用时: {format_time(elapsed)}")


def safe_import(module_name: str, package: Optional[str] = None):
    """安全导入模块"""
    try:
        if package:
            return __import__(f"{package}.{module_name}", fromlist=[module_name])
        else:
            return __import__(module_name)
    except ImportError as e:
        print(f"警告: 无法导入 {module_name}: {e}")
        return None


def ensure_dir(path: Path):
    """确保目录存在"""
    path.mkdir(parents=True, exist_ok=True)


def get_file_size(file_path: Path) -> float:
    """获取文件大小（MB）"""
    if file_path.exists():
        return file_path.stat().st_size / (1024 * 1024)
    return 0.0