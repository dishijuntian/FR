"""
进度显示工具 - 重构版

专注于：
- 统一的进度条显示
- 简化的接口
- 可选的Rich支持

作者: Flight Ranking Team
版本: 4.0 (重构版)
"""

from tqdm import tqdm
from typing import Optional, Any, Iterator
from contextlib import contextmanager

# 可选的Rich支持
try:
    from rich.console import Console
    from rich.progress import Progress, TaskID, BarColumn, TextColumn, TimeRemainingColumn
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None


class ProgressBar:
    """统一的进度条接口"""
    
    def __init__(self, total: Optional[int] = None, description: str = "Processing", 
                 use_rich: bool = True):
        """
        初始化进度条
        
        Args:
            total: 总步数
            description: 描述文字
            use_rich: 是否使用Rich（如果可用）
        """
        self.total = total
        self.description = description
        self.use_rich = use_rich and RICH_AVAILABLE
        
        self.progress = None
        self.task_id = None
        self.tqdm_bar = None
    
    def __enter__(self):
        if self.use_rich:
            self.progress = Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
                console=console
            )
            self.progress.start()
            self.task_id = self.progress.add_task(self.description, total=self.total)
        else:
            self.tqdm_bar = tqdm(
                total=self.total,
                desc=self.description,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
            )
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.progress:
            self.progress.stop()
        if self.tqdm_bar:
            self.tqdm_bar.close()
    
    def update(self, advance: int = 1, description: Optional[str] = None):
        """更新进度"""
        if self.progress and self.task_id is not None:
            if description:
                self.progress.update(self.task_id, description=description, advance=advance)
            else:
                self.progress.update(self.task_id, advance=advance)
        elif self.tqdm_bar:
            self.tqdm_bar.update(advance)
            if description:
                self.tqdm_bar.set_description(description)
    
    def set_total(self, total: int):
        """设置总数"""
        if self.progress and self.task_id is not None:
            self.progress.update(self.task_id, total=total)
        elif self.tqdm_bar:
            self.tqdm_bar.total = total


@contextmanager
def progress_bar(iterable=None, total: Optional[int] = None, 
                description: str = "Processing", use_rich: bool = True):
    """
    进度条上下文管理器
    
    Args:
        iterable: 可迭代对象（可选）
        total: 总步数
        description: 描述文字
        use_rich: 是否使用Rich
        
    Yields:
        ProgressBar或迭代器
    """
    if iterable is not None:
        # 直接包装迭代器
        if use_rich and RICH_AVAILABLE:
            yield tqdm(iterable, desc=description, leave=False)
        else:
            yield tqdm(iterable, desc=description)
    else:
        # 返回进度条对象
        with ProgressBar(total=total, description=description, use_rich=use_rich) as pbar:
            yield pbar


def simple_progress(iterable, description: str = "Processing") -> Iterator:
    """
    简单进度条包装器
    
    Args:
        iterable: 可迭代对象
        description: 描述文字
        
    Returns:
        Iterator: 带进度条的迭代器
    """
    return tqdm(iterable, desc=description, leave=False)


class TrainingProgress:
    """训练进度显示器"""
    
    def __init__(self, model_names: list):
        """
        初始化训练进度
        
        Args:
            model_names: 模型名称列表
        """
        self.model_names = model_names
        self.current_model = 0
        self.overall_bar = None
        self.model_bar = None
    
    def __enter__(self):
        self.overall_bar = tqdm(
            total=len(self.model_names),
            desc="总体进度",
            position=0,
            leave=True
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.overall_bar:
            self.overall_bar.close()
        if self.model_bar:
            self.model_bar.close()
    
    def start_model(self, model_name: str):
        """开始训练新模型"""
        if self.model_bar:
            self.model_bar.close()
        
        self.model_bar = tqdm(
            desc=f"训练 {model_name}",
            position=1,
            leave=False
        )
    
    def update_model(self, description: str = None):
        """更新模型进度"""
        if self.model_bar and description:
            self.model_bar.set_description(description)
    
    def finish_model(self, model_name: str, performance: float = None):
        """完成模型训练"""
        if self.model_bar:
            self.model_bar.close()
            self.model_bar = None
        
        self.overall_bar.update(1)
        if performance is not None:
            desc = f"完成: {model_name} (HitRate@3: {performance:.4f})"
        else:
            desc = f"完成: {model_name}"
        self.overall_bar.set_description(desc)


def show_summary(title: str, results: dict):
    """
    显示结果总结
    
    Args:
        title: 标题
        results: 结果字典
    """
    if RICH_AVAILABLE:
        console.print(f"\n🎉 {title}", style="bold green")
        console.print("="*50, style="green")
        
        for key, value in results.items():
            if isinstance(value, float):
                console.print(f"{key}: {value:.4f}", style="cyan")
            else:
                console.print(f"{key}: {value}", style="cyan")
    else:
        print(f"\n🎉 {title}")
        print("="*50)
        for key, value in results.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")


# 向后兼容的简化函数
def create_data_loading_progress(description: str = "加载数据"):
    """创建数据加载进度条"""
    return ProgressBar(description=description)


def create_file_progress(files: list, description: str = "处理文件"):
    """创建文件处理进度条"""
    return simple_progress(files, description)