"""
可视化模块 - 重构版
统一管理所有结果可视化功能

作者: Flight Ranking Team
版本: 5.0 (重构版)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Any, Optional
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8')

# 颜色配置
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72', 
    'accent': '#F18F01',
    'success': '#C73E1D',
    'pytorch': '#EE4C2C',
    'traditional': '#306998',
    'background': '#F8F9FA'
}


class BaseVisualizer:
    """可视化基类"""
    
    def __init__(self, output_path: Path, figsize: tuple = (12, 8)):
        """
        初始化可视化器
        
        Args:
            output_path: 输出路径
            figsize: 图形大小
        """
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.figsize = figsize
        
    def _setup_plot(self, title: str, figsize: Optional[tuple] = None):
        """设置图形"""
        figsize = figsize or self.figsize
        fig, ax = plt.subplots(figsize=figsize)
        fig.suptitle(title, fontsize=16, fontweight='bold')
        return fig, ax
    
    def _save_plot(self, filename: str, dpi: int = 300, bbox_inches: str = 'tight'):
        """保存图形"""
        filepath = self.output_path / filename
        plt.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches, 
                   facecolor='white', edgecolor='none')
        print(f"图表已保存: {filepath}")
    
    def _get_model_color(self, model_name: str) -> str:
        """根据模型类型获取颜色"""
        pytorch_models = ['NeuralRanker', 'RankNet', 'TransformerRanker']
        if model_name in pytorch_models:
            return COLORS['pytorch']
        else:
            return COLORS['traditional']


class ModelPerformanceVisualizer(BaseVisualizer):
    """模型性能可视化器"""
    
    def plot_model_comparison(self, results_df: pd.DataFrame, segment_id: int,
                            metrics: List[str] = None, show_plot: bool = True):
        """绘制模型性能对比图"""
        if results_df.empty:
            print("⚠️ 结果数据为空，跳过可视化")
            return
        
        if metrics is None:
            metrics = [col for col in results_df.columns 
                      if col not in ['Model', 'Segment', 'Parameters']]
        
        n_metrics = len(metrics)
        if n_metrics == 0:
            print("⚠️ 没有找到可用的性能指标")
            return
        
        # 创建子图
        fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 6))
        if n_metrics == 1:
            axes = [axes]
        
        fig.suptitle(f'模型性能对比 - 数据段 {segment_id}', fontsize=16, fontweight='bold')
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # 获取颜色
            colors = [self._get_model_color(model) for model in results_df['Model']]
            
            # 绘制柱状图
            bars = ax.bar(results_df['Model'], results_df[metric], color=colors, alpha=0.8)
            
            # 添加数值标签
            for bar, value in zip(bars, results_df[metric]):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
            
            ax.set_title(f'{metric}', fontweight='bold')
            ax.set_ylabel(metric)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
            
            # 设置y轴范围
            if results_df[metric].max() > 0:
                ax.set_ylim(0, results_df[metric].max() * 1.15)
        
        plt.tight_layout()
        
        # 保存图片
        self._save_plot(f'model_performance_segment_{segment_id}.png')
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def plot_training_progress(self, training_results: List[Dict[str, Any]], 
                             metric: str = 'HitRate@3', show_plot: bool = True):
        """绘制训练进度图"""
        if not training_results:
            return
        
        # 准备数据
        segments = []
        models_data = {}
        
        for result in training_results:
            segment_id = result['segment_id']
            segments.append(segment_id)
            
            for _, row in result['results'].iterrows():
                model_name = row['Model']
                if model_name not in models_data:
                    models_data[model_name] = []
                models_data[model_name].append(row[metric])
        
        # 创建图形
        fig, ax = self._setup_plot(f'训练进度 - {metric}')
        
        # 绘制每个模型的性能曲线
        for model_name, scores in models_data.items():
            color = self._get_model_color(model_name)
            ax.plot(segments[:len(scores)], scores, 'o-', 
                   label=model_name, color=color, linewidth=2, markersize=8)
        
        ax.set_xlabel('数据段', fontweight='bold')
        ax.set_ylabel(metric, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._save_plot(f'training_progress_{metric.lower()}.png')
        
        if show_plot:
            plt.show()
        else:
            plt.close()


class FeatureAnalysisVisualizer(BaseVisualizer):
    """特征分析可视化器"""
    
    def plot_feature_importance(self, feature_importance: pd.DataFrame, 
                               segment_id: int, top_k: int = 20, show_plot: bool = True):
        """绘制特征重要性图"""
        if feature_importance.empty:
            print("⚠️ 特征重要性数据为空")
            return
        
        # 获取前K个最重要的特征
        top_features = feature_importance.head(top_k)
        
        fig, ax = self._setup_plot(f'特征重要性分析 - 数据段 {segment_id}')
        
        # 绘制水平柱状图
        y_pos = np.arange(len(top_features))
        bars = ax.barh(y_pos, top_features['Average'], color=COLORS['primary'], alpha=0.8)
        
        # 设置标签
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features.index, fontsize=10)
        ax.set_xlabel('重要性分数', fontweight='bold')
        ax.set_title(f'前{top_k}个最重要特征', fontweight='bold')
        
        # 添加数值标签
        for i, (bar, value) in enumerate(zip(bars, top_features['Average'])):
            width = bar.get_width()
            ax.text(width + width*0.01, bar.get_y() + bar.get_height()/2,
                   f'{value:.4f}', ha='left', va='center', fontweight='bold')
        
        ax.grid(True, alpha=0.3, axis='x')
        ax.invert_yaxis()  # 最重要的特征在顶部
        
        plt.tight_layout()
        self._save_plot(f'feature_importance_segment_{segment_id}.png')
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def plot_feature_correlation_heatmap(self, correlation_matrix: pd.DataFrame,
                                       show_plot: bool = True):
        """绘制特征相关性热力图"""
        if correlation_matrix.empty:
            return
        
        # 只显示相关性较高的特征对
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        fig, ax = self._setup_plot('特征相关性热力图', figsize=(12, 10))
        
        # 绘制热力图
        sns.heatmap(correlation_matrix, mask=mask, annot=False, cmap='coolwarm',
                   center=0, square=True, ax=ax, cbar_kws={'shrink': 0.8})
        
        ax.set_title('特征相关性矩阵', fontweight='bold')
        
        plt.tight_layout()
        self._save_plot('feature_correlation_heatmap.png')
        
        if show_plot:
            plt.show()
        else:
            plt.close()


class ComparisonVisualizer(BaseVisualizer):
    """比较分析可视化器"""
    
    def create_model_comparison_dashboard(self, combined_results: pd.DataFrame,
                                        show_plot: bool = True):
        """创建模型比较仪表板"""
        if combined_results.empty:
            return
        
        # 创建2x2的子图布局
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('模型比较分析仪表板', fontsize=20, fontweight='bold')
        
        # 1. 平均性能对比
        ax1 = axes[0, 0]
        avg_performance = combined_results.groupby('Model')['HitRate@3'].mean().sort_values(ascending=False)
        colors = [self._get_model_color(model) for model in avg_performance.index]
        bars1 = ax1.bar(avg_performance.index, avg_performance.values, color=colors, alpha=0.8)
        ax1.set_title('平均HitRate@3性能', fontweight='bold')
        ax1.set_ylabel('HitRate@3')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, value in zip(bars1, avg_performance.values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. 性能稳定性(标准差)
        ax2 = axes[0, 1]
        std_performance = combined_results.groupby('Model')['HitRate@3'].std().sort_values()
        colors = [self._get_model_color(model) for model in std_performance.index]
        bars2 = ax2.bar(std_performance.index, std_performance.values, color=colors, alpha=0.8)
        ax2.set_title('性能稳定性 (标准差越小越稳定)', fontweight='bold')
        ax2.set_ylabel('HitRate@3 标准差')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # 3. 各段性能分布箱线图
        ax3 = axes[1, 0]
        models_for_boxplot = combined_results['Model'].unique()
        data_for_boxplot = [combined_results[combined_results['Model'] == model]['HitRate@3'].values 
                           for model in models_for_boxplot]
        
        box_plot = ax3.boxplot(data_for_boxplot, labels=models_for_boxplot, patch_artist=True)
        
        # 设置箱线图颜色
        for patch, model in zip(box_plot['boxes'], models_for_boxplot):
            patch.set_facecolor(self._get_model_color(model))
            patch.set_alpha(0.7)
        
        ax3.set_title('各数据段性能分布', fontweight='bold')
        ax3.set_ylabel('HitRate@3')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # 4. 最佳模型统计
        ax4 = axes[1, 1]
        best_models = combined_results.loc[combined_results.groupby('Segment')['HitRate@3'].idxmax()]
        best_model_counts = best_models['Model'].value_counts()
        
        # 饼图
        colors_pie = [self._get_model_color(model) for model in best_model_counts.index]
        wedges, texts, autotexts = ax4.pie(best_model_counts.values, labels=best_model_counts.index,
                                          autopct='%1.1f%%', colors=colors_pie, startangle=90)
        
        # 设置文本样式
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax4.set_title('各段最佳模型分布', fontweight='bold')
        
        plt.tight_layout()
        self._save_plot('model_comparison_dashboard.png')
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def create_performance_trend_analysis(self, combined_results: pd.DataFrame,
                                        show_plot: bool = True):
        """创建性能趋势分析"""
        if combined_results.empty or 'Segment' not in combined_results.columns:
            return
        
        fig, ax = self._setup_plot('模型性能趋势分析', figsize=(14, 8))
        
        # 为每个模型绘制趋势线
        for model in combined_results['Model'].unique():
            model_data = combined_results[combined_results['Model'] == model].sort_values('Segment')
            color = self._get_model_color(model)
            
            ax.plot(model_data['Segment'], model_data['HitRate@3'], 
                   'o-', label=model, color=color, linewidth=2, markersize=8)
            
            # 添加趋势线
            z = np.polyfit(model_data['Segment'], model_data['HitRate@3'], 1)
            p = np.poly1d(z)
            ax.plot(model_data['Segment'], p(model_data['Segment']), 
                   '--', color=color, alpha=0.7, linewidth=1)
        
        ax.set_xlabel('数据段', fontweight='bold')
        ax.set_ylabel('HitRate@3', fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._save_plot('performance_trend_analysis.png')
        
        if show_plot:
            plt.show()
        else:
            plt.close()


class ResultsVisualizer:
    """结果可视化主控制器"""
    
    def __init__(self, output_path: Path):
        """初始化结果可视化器"""
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # 初始化各个可视化器
        self.model_performance = ModelPerformanceVisualizer(output_path)
        self.feature_analysis = FeatureAnalysisVisualizer(output_path)
        self.comparison = ComparisonVisualizer(output_path)
    
    def plot_model_performance(self, results_df: pd.DataFrame, segment_id: int,
                             show_plot: bool = True):
        """绘制模型性能对比"""
        self.model_performance.plot_model_comparison(results_df, segment_id, show_plot=show_plot)
    
    def plot_feature_importance(self, feature_importance: pd.DataFrame, 
                              segment_id: int, top_k: int = 20, show_plot: bool = True):
        """绘制特征重要性"""
        self.feature_analysis.plot_feature_importance(
            feature_importance, segment_id, top_k, show_plot=show_plot
        )
    
    def create_training_summary(self, training_results: List[Dict[str, Any]],
                              show_plot: bool = True):
        """创建训练总结可视化"""
        # 训练进度图
        self.model_performance.plot_training_progress(
            training_results, show_plot=show_plot
        )
        
        # 合并所有结果创建比较分析
        all_results = []
        for result in training_results:
            df = result['results'].copy()
            df['Segment'] = result['segment_id']
            all_results.append(df)
        
        if all_results:
            combined_results = pd.concat(all_results, ignore_index=True)
            self.create_model_comparison_dashboard(combined_results, show_plot=show_plot)
    
    def create_model_comparison_dashboard(self, combined_results: pd.DataFrame,
                                        show_plot: bool = True):
        """创建模型比较仪表板"""
        self.comparison.create_model_comparison_dashboard(combined_results, show_plot=show_plot)
        self.comparison.create_performance_trend_analysis(combined_results, show_plot=show_plot)
    
    def generate_summary_report(self, training_results: List[Dict[str, Any]]) -> str:
        """生成文本总结报告"""
        if not training_results:
            return "无训练结果可供分析"
        
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("训练结果总结报告")
        report_lines.append("=" * 60)
        
        # 总体统计
        total_segments = len(training_results)
        all_models = set()
        
        for result in training_results:
            all_models.update(result['results']['Model'].tolist())
        
        report_lines.append(f"训练数据段数: {total_segments}")
        report_lines.append(f"测试模型数: {len(all_models)}")
        report_lines.append(f"使用模型: {', '.join(sorted(all_models))}")
        report_lines.append("")
        
        # 各段最佳模型
        report_lines.append("各段最佳模型:")
        for result in training_results:
            best_model = result['best_model']
            report_lines.append(f"  段 {result['segment_id']}: {best_model['Model']} "
                              f"(HitRate@3: {best_model['HitRate@3']:.4f})")
        
        report_lines.append("")
        
        # 模型平均性能
        all_results = []
        for result in training_results:
            df = result['results'].copy()
            all_results.append(df)
        
        if all_results:
            combined_df = pd.concat(all_results, ignore_index=True)
            avg_performance = combined_df.groupby('Model')['HitRate@3'].agg(['mean', 'std'])
            
            report_lines.append("模型平均性能 (HitRate@3):")
            for model in avg_performance.index:
                mean_val = avg_performance.loc[model, 'mean']
                std_val = avg_performance.loc[model, 'std']
                report_lines.append(f"  {model}: {mean_val:.4f} ± {std_val:.4f}")
        
        report_lines.append("")
        report_lines.append("=" * 60)
        
        # 保存报告
        report_text = "\n".join(report_lines)
        report_file = self.output_path / "training_summary_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"总结报告已保存: {report_file}")
        return report_text


# 便捷函数
def quick_plot_model_comparison(results_df: pd.DataFrame, output_path: Path = None,
                              segment_id: int = 0):
    """快速绘制模型比较图的便捷函数"""
    if output_path is None:
        output_path = Path("./results")
    
    visualizer = ModelPerformanceVisualizer(output_path)
    visualizer.plot_model_comparison(results_df, segment_id)


def quick_plot_feature_importance(feature_importance: pd.DataFrame, 
                                 output_path: Path = None, segment_id: int = 0):
    """快速绘制特征重要性图的便捷函数"""
    if output_path is None:
        output_path = Path("./results")
    
    visualizer = FeatureAnalysisVisualizer(output_path)
    visualizer.plot_feature_importance(feature_importance, segment_id)
    