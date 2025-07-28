"""
进度条工具模块

该模块提供统一的进度条显示功能
- 支持数据加载进度
- 支持模型训练进度
- 支持预测进度
- 支持文件处理进度

作者: Flight Ranking Team
版本: 2.1
"""

from tqdm import tqdm
import time
from typing import Optional, Any, Iterator
from contextlib import contextmanager
import sys

try:
    from rich.console import Console
    from rich.progress import (
        Progress, TaskID, BarColumn, TextColumn, 
        TimeRemainingColumn, TimeElapsedColumn,
        MofNCompleteColumn, SpinnerColumn
    )
    from rich.text import Text
    from rich.live import Live
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class ProgressTracker:
    """进度跟踪器"""
    
    def __init__(self, use_rich: bool = True):
        """
        初始化进度跟踪器
        
        Args:
            use_rich: 是否使用rich库的高级进度条
        """
        self.use_rich = use_rich and RICH_AVAILABLE
        self.console = Console() if self.use_rich else None
        self.current_progress = None
        
    def create_progress(self, description: str, total: Optional[int] = None, 
                       show_speed: bool = True) -> 'ProgressContext':
        """
        创建进度条上下文
        
        Args:
            description: 进度描述
            total: 总步数
            show_speed: 是否显示速度
            
        Returns:
            ProgressContext: 进度条上下文管理器
        """
        return ProgressContext(
            description=description,
            total=total,
            show_speed=show_speed,
            use_rich=self.use_rich,
            console=self.console
        )
    
    def simple_progress(self, iterable, description: str = "Processing") -> Iterator:
        """
        简单进度条包装器
        
        Args:
            iterable: 可迭代对象
            description: 描述文字
            
        Returns:
            Iterator: 带进度条的迭代器
        """
        if self.use_rich:
            return tqdm(iterable, desc=description, 
                       bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        else:
            return tqdm(iterable, desc=description)


class ProgressContext:
    """进度条上下文管理器"""
    
    def __init__(self, description: str, total: Optional[int] = None,
                 show_speed: bool = True, use_rich: bool = True, 
                 console: Optional[Any] = None):
        self.description = description
        self.total = total
        self.show_speed = show_speed
        self.use_rich = use_rich
        self.console = console
        self.progress = None
        self.task_id = None
        self.pbar = None
        
    def __enter__(self):
        if self.use_rich and self.console:
            # 使用rich进度条
            self.progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TextColumn("•"),
                TimeElapsedColumn(),
                TextColumn("•"),
                TimeRemainingColumn(),
                console=self.console
            )
            self.progress.start()
            self.task_id = self.progress.add_task(
                self.description, 
                total=self.total
            )
            return self
        else:
            # 使用tqdm进度条
            self.pbar = tqdm(
                total=self.total,
                desc=self.description,
                unit="items" if self.total else "it",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            )
            return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.progress:
            self.progress.stop()
        if self.pbar:
            self.pbar.close()
    
    def update(self, advance: int = 1, description: Optional[str] = None):
        """更新进度"""
        if self.progress and self.task_id is not None:
            self.progress.update(
                self.task_id, 
                advance=advance,
                description=description or self.description
            )
        elif self.pbar:
            self.pbar.update(advance)
            if description:
                self.pbar.set_description(description)
    
    def set_total(self, total: int):
        """设置总数"""
        self.total = total
        if self.progress and self.task_id is not None:
            self.progress.update(self.task_id, total=total)
        elif self.pbar:
            self.pbar.total = total
    
    def set_description(self, description: str):
        """设置描述"""
        if self.progress and self.task_id is not None:
            self.progress.update(self.task_id, description=description)
        elif self.pbar:
            self.pbar.set_description(description)


class ModelTrainingProgress:
    """模型训练进度显示"""
    
    def __init__(self, model_names: list, use_rich: bool = True):
        self.model_names = model_names
        self.use_rich = use_rich and RICH_AVAILABLE
        self.console = Console() if self.use_rich else None
        self.current_model_idx = 0
        
    @contextmanager
    def training_session(self):
        """训练会话上下文"""
        if self.use_rich:
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold green]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TextColumn("•"),
                TimeElapsedColumn(),
                console=self.console
            ) as progress:
                overall_task = progress.add_task(
                    "总体训练进度", 
                    total=len(self.model_names)
                )
                model_task = progress.add_task(
                    "当前模型", 
                    total=None
                )
                
                yield ModelTrainingContext(progress, overall_task, model_task, self.model_names)
        else:
            yield SimpleTrainingContext(self.model_names)


class ModelTrainingContext:
    """Rich进度条训练上下文"""
    
    def __init__(self, progress, overall_task, model_task, model_names):
        self.progress = progress
        self.overall_task = overall_task
        self.model_task = model_task
        self.model_names = model_names
        self.current_model_idx = 0
    
    def start_model(self, model_name: str, steps: Optional[int] = None):
        """开始训练新模型"""
        self.progress.update(
            self.model_task,
            description=f"训练 {model_name}",
            completed=0,
            total=steps
        )
    
    def update_model_progress(self, advance: int = 1, description: Optional[str] = None):
        """更新当前模型进度"""
        self.progress.update(self.model_task, advance=advance)
        if description:
            self.progress.update(self.model_task, description=description)
    
    def finish_model(self, model_name: str, performance: float):
        """完成当前模型训练"""
        self.progress.update(
            self.overall_task, 
            advance=1,
            description=f"已完成: {model_name} (HitRate@3: {performance:.4f})"
        )
        self.current_model_idx += 1


class SimpleTrainingContext:
    """简单训练上下文（无rich时使用）"""
    
    def __init__(self, model_names):
        self.model_names = model_names
        self.current_pbar = None
        self.overall_pbar = tqdm(total=len(model_names), desc="总体进度", position=0)
        
    def start_model(self, model_name: str, steps: Optional[int] = None):
        """开始训练新模型"""
        if self.current_pbar:
            self.current_pbar.close()
        
        self.current_pbar = tqdm(
            total=steps,
            desc=f"训练 {model_name}",
            position=1,
            leave=False
        )
    
    def update_model_progress(self, advance: int = 1, description: Optional[str] = None):
        """更新当前模型进度"""
        if self.current_pbar:
            self.current_pbar.update(advance)
            if description:
                self.current_pbar.set_description(description)
    
    def finish_model(self, model_name: str, performance: float):
        """完成当前模型训练"""
        if self.current_pbar:
            self.current_pbar.close()
            self.current_pbar = None
        
        self.overall_pbar.update(1)
        self.overall_pbar.set_description(f"已完成: {model_name} (HitRate@3: {performance:.4f})")
    
    def __del__(self):
        """清理资源"""
        if self.current_pbar:
            self.current_pbar.close()
        if self.overall_pbar:
            self.overall_pbar.close()


def create_file_progress(files: list, description: str = "处理文件") -> Iterator:
    """
    创建文件处理进度条
    
    Args:
        files: 文件列表
        description: 描述文字
        
    Returns:
        Iterator: 带进度条的文件迭代器
    """
    return tqdm(
        files, 
        desc=description,
        unit="文件",
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
    )


def create_data_loading_progress(description: str = "加载数据") -> ProgressContext:
    """
    创建数据加载进度条
    
    Args:
        description: 描述文字
        
    Returns:
        ProgressContext: 进度条上下文
    """
    tracker = ProgressTracker()
    return tracker.create_progress(description, show_speed=True)


def show_completion_summary(results: dict, title: str = "完成总结"):
    """
    显示完成总结
    
    Args:
        results: 结果字典
        title: 标题
    """
    if RICH_AVAILABLE:
        console = Console()
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


# 简化的全局函数
def progress_bar(iterable, desc: str = "Processing", **kwargs):
    """全局进度条函数"""
    return tqdm(iterable, desc=desc, **kwargs)


def progress_range(n: int, desc: str = "Processing", **kwargs):
    """进度条范围函数"""
    return tqdm(range(n), desc=desc, **kwargs)