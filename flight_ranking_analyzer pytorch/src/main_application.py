"""
主程序模块 - 重构版 v5.1
统一的程序入口和工作流管理，提供完善的流水线选择

作者: Flight Ranking Team
版本: 5.1 (改进版)
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# 导入重构后的模块
from core_config import ConfigManager, FeatureLevel, DataProcessMode, FeatureSelectionMode
from data_processor import DataProcessor
from models_module import ModelFactory
from training_prediction import ModelManager, ModelTrainer, ModelPredictor
from evaluation_metrics import ModelEvaluator
from visualization import ResultsVisualizer
from utils import ProgressTracker, SystemChecker, FileManager

# 全局配置
config = ConfigManager()


class EnhancedUserInterface:
    """增强的用户交互界面"""
    
    @staticmethod
    def show_welcome():
        """显示欢迎界面"""
        print(" 航班排序分析系统 v5.1 (改进版)")
        print("新增功能:")
        print("  • 灵活的数据处理流水线选择")
        print("  •  智能缓存管理")
        print("  •  多种特征工程模式比较")
        print("  • ⚡ 快速重用已处理数据")
    
    @staticmethod
    def get_run_mode() -> str:
        """获取运行模式"""
        print("\n🎯 运行模式选择:")
        print("1. 💻 完整流程 (数据处理 + 训练 + 预测)")
        print("2. 🔧 仅数据处理与特征工程")
        print("3. 🎯 仅模型训练")
        print("4. 📈 仅预测 (使用已保存的模型)")
        print("5. 📊 模型比较分析")
        print("6. 🔄 缓存管理")
        
        while True:
            choice = input("\n请选择运行模式 (1-6): ").strip()
            if choice in ['1', '2', '3', '4', '5', '6']:
                return choice
            print("❌ 请输入有效选择 (1-6)")
    
    @staticmethod
    def get_data_processing_config() -> Dict[str, Any]:
        """获取数据处理配置"""
        print("\n" + "="*60)
        print("🔄 数据处理流水线配置")
        print("="*60)
        
        config_dict = {}
        
        # 检查缓存情况
        cache_info = config.get_cache_info()
        has_cache = cache_info['cache_count'] > 0 if 'cache_count' in cache_info else False
        
        if has_cache:
            print(f"\n💾 发现缓存数据: {cache_info['cache_count']} 个文件")
            print("缓存文件:", ", ".join(cache_info['cached_files'][:3]))
            if len(cache_info['cached_files']) > 3:
                print(f"... 还有 {len(cache_info['cached_files']) - 3} 个文件")
        
        # 数据处理模式选择
        print("\n🔄 数据处理模式:")
        modes = [
            ("完整处理", "full_process", "编码 → 特征工程 → 特征选择"),
            ("仅数据编码", "encoding_only", "仅对原始数据进行编码处理"),
            ("仅特征工程", "feature_only", "假设数据已编码，仅做特征工程"),
            ("使用原始数据", "raw_data", "直接使用原始数据训练"),
        ]
        
        if has_cache:
            modes.insert(1, ("加载缓存数据", "load_cached", "使用已保存的处理结果"))
            modes.append(("比较处理模式", "compare_modes", "比较不同处理方式的效果"))
        
        for i, (name, _, desc) in enumerate(modes, 1):
            print(f"{i}. {name} - {desc}")
        
        while True:
            try:
                choice = int(input(f"\n请选择数据处理模式 (1-{len(modes)}, 推荐1): ") or "1")
                if 1 <= choice <= len(modes):
                    config_dict['data_process_mode'] = modes[choice-1][1]
                    break
                else:
                    print(f"❌ 请输入 1-{len(modes)} 之间的数字")
            except ValueError:
                print("❌ 请输入有效数字")
        
        # 根据模式获取详细配置
        if config_dict['data_process_mode'] == 'load_cached':
            return config_dict  # 加载缓存模式不需要其他配置
        
        elif config_dict['data_process_mode'] == 'raw_data':
            config_dict['feature_level'] = 'none'
            config_dict['selection_mode'] = 'none'
            return config_dict
        
        elif config_dict['data_process_mode'] == 'encoding_only':
            config_dict['feature_level'] = 'none'
            config_dict.update(EnhancedUserInterface._get_feature_selection_config())
            return config_dict
        
        elif config_dict['data_process_mode'] == 'compare_modes':
            return EnhancedUserInterface._get_comparison_config()
        
        # 完整处理或仅特征工程模式
        config_dict.update(EnhancedUserInterface._get_feature_engineering_config())
        config_dict.update(EnhancedUserInterface._get_feature_selection_config())
        config_dict.update(EnhancedUserInterface._get_cache_config())
        
        return config_dict
    
    @staticmethod
    def _get_feature_engineering_config() -> Dict[str, Any]:
        """获取特征工程配置"""
        print("\n🛠️ 特征工程配置:")
        
        levels = [
            ("跳过特征工程", "none", "直接使用编码后的原始特征"),
            ("基础特征工程", "basic", "价格、时间、持续时间等核心特征"),
            ("增强特征工程", "enhanced", "基础 + 预订时机、舱位等级、用户类型特征"),
            ("高级特征工程", "advanced", "增强 + 经济学特征、选择复杂度、组合特征"),
        ]
        
        for i, (name, _, desc) in enumerate(levels, 1):
            emoji = "⭐" if i == 3 else ""  # 推荐增强级别
            print(f"{i}. {name} {emoji}")
            print(f"   └─ {desc}")
        
        while True:
            try:
                choice = int(input(f"\n请选择特征工程级别 (1-{len(levels)}, 推荐3): ") or "3")
                if 1 <= choice <= len(levels):
                    return {'feature_level': levels[choice-1][1]}
                else:
                    print(f"❌ 请输入 1-{len(levels)} 之间的数字")
            except ValueError:
                print("❌ 请输入有效数字")
    
    @staticmethod
    def _get_feature_selection_config() -> Dict[str, Any]:
        """获取特征选择配置"""
        print("\n🎯 特征选择配置:")
        
        selections = [
            ("跳过特征选择", "none", "使用所有生成的特征"),
            ("方差选择", "variance", "基于特征方差进行选择 (推荐)"),
            ("相关性选择", "correlation", "移除高相关性特征"),
            ("互信息选择", "mutual_info", "基于互信息进行选择"),
        ]
        
        for i, (name, _, desc) in enumerate(selections, 1):
            emoji = "⭐" if i == 2 else ""  # 推荐方差选择
            print(f"{i}. {name} {emoji}")
            print(f"   └─ {desc}")
        
        while True:
            try:
                choice = int(input(f"\n请选择特征选择方法 (1-{len(selections)}, 推荐2): ") or "2")
                if 1 <= choice <= len(selections):
                    selection_mode = selections[choice-1][1]
                    config_dict = {'selection_mode': selection_mode}
                    
                    # 如果选择了特征选择，询问特征数量
                    if selection_mode != 'none':
                        try:
                            max_features = int(input("最大特征数 (默认200): ") or "200")
                            config_dict['max_features'] = max_features
                        except ValueError:
                            config_dict['max_features'] = 200
                    
                    return config_dict
                else:
                    print(f"❌ 请输入 1-{len(selections)} 之间的数字")
            except ValueError:
                print("❌ 请输入有效数字")
    
    @staticmethod
    def _get_cache_config() -> Dict[str, Any]:
        """获取缓存配置"""
        print("\n💾 缓存配置:")
        cache_data = input("保存处理后的数据到缓存? (y/n, 默认y): ").strip().lower() != 'n'
        auto_load = input("下次自动加载缓存? (y/n, 默认y): ").strip().lower() != 'n'
        
        return {
            'cache_processed_data': cache_data,
            'auto_load_cache': auto_load
        }
    
    @staticmethod
    def _get_comparison_config() -> Dict[str, Any]:
        """获取比较模式配置"""
        print("\n📊 比较模式配置:")
        print("将比较以下处理方式的效果:")
        print("  • 原始数据")
        print("  • 仅编码")
        print("  • 基础特征工程")
        print("  • 增强特征工程")
        print("  • 高级特征工程")
        
        return {
            'data_process_mode': 'compare_modes',
            'comparison_modes': ['raw_data', 'encoding_only', 'basic', 'enhanced', 'advanced']
        }
    
    @staticmethod
    def get_training_config() -> Dict[str, Any]:
        """获取训练配置"""
        print("\n" + "="*60)
        print("🎯 模型训练配置")
        print("="*60)
        
        config_dict = {}
        
        # 数据模式
        print("\n📊 数据加载模式:")
        print("1. 🔬 抽样模式 (推荐，快速验证)")
        print("2. 🏢 全量模式 (完整数据，耗时较长)")
        
        data_mode = input("请选择 (1-2, 默认1): ").strip() or '1'
        config_dict['use_sampling'] = data_mode == '1'
        
        if config_dict['use_sampling']:
            try:
                config_dict['num_groups'] = int(input(f"抽样组数 (默认{config.training.num_groups}): ") 
                                               or config.training.num_groups)
                config_dict['min_group_size'] = int(input(f"最小组大小 (默认{config.training.min_group_size}): ") 
                                                   or config.training.min_group_size)
            except ValueError:
                config_dict['num_groups'] = config.training.num_groups
                config_dict['min_group_size'] = config.training.min_group_size
        
        # 模型选择
        available_models = ModelFactory.get_available_models()
        print(f"\n🤖 可用模型:")
        for i, model in enumerate(available_models, 1):
            model_type = "🔥 PyTorch" if ModelFactory.is_pytorch_model(model) else "📊 传统"
            print(f"{i}. {model} ({model_type})")
        print(f"{len(available_models) + 1}. 所有模型")
        print(f"{len(available_models) + 2}. 快速模式 (仅XGBRanker + NeuralRanker)")
        
        model_input = input("\n选择模型 (用逗号分隔, 默认快速模式): ").strip()
        if not model_input:
            config_dict['selected_models'] = ['XGBRanker', 'NeuralRanker']
        else:
            try:
                choices = [int(x.strip()) for x in model_input.split(',')]
                if len(available_models) + 1 in choices:
                    config_dict['selected_models'] = available_models
                elif len(available_models) + 2 in choices:
                    config_dict['selected_models'] = ['XGBRanker', 'NeuralRanker']
                else:
                    config_dict['selected_models'] = [
                        available_models[i-1] 
                        for i in choices 
                        if 1 <= i <= len(available_models)
                    ]
            except:
                config_dict['selected_models'] = ['XGBRanker', 'NeuralRanker']
        
        # 自动调参
        config_dict['enable_auto_tuning'] = input("\n🔧 启用自动调参? (y/n, 默认n): ").strip().lower() == 'y'
        if config_dict['enable_auto_tuning']:
            try:
                config_dict['auto_tuning_trials'] = int(input("调参试验次数 (默认20): ") or 20)
            except:
                config_dict['auto_tuning_trials'] = 20
        
        # 模型保存
        config_dict['save_models'] = input("💾 保存训练的模型? (y/n, 默认y): ").strip().lower() != 'n'
        
        return config_dict
    
    @staticmethod
    def get_prediction_config(model_manager: ModelManager) -> Dict[str, Any]:
        """获取预测配置"""
        print("\n" + "="*60)
        print("📈 预测配置")
        print("="*60)
        
        config_dict = {}
        
        # 检查可用模型
        available_models = model_manager.get_available_models()
        
        if not available_models:
            print("❌ 没有找到保存的模型，请先训练模型")
            config_dict['prediction_possible'] = False
            return config_dict
        
        print("\n🤖 可用的已保存模型:")
        for model_name, segments in available_models.items():
            model_type = "🔥 PyTorch" if ModelFactory.is_pytorch_model(model_name) else "📊 传统"
            print(f"  {model_name} ({model_type}): 段 {segments}")
        
        # 选择模型
        model_names = list(available_models.keys())
        print(f"\n模型选择:")
        for i, model_name in enumerate(model_names, 1):
            print(f"{i}. {model_name}")
        print(f"{len(model_names) + 1}. 所有模型")
        print(f"{len(model_names) + 2}. 最佳组合 (推荐)")
        
        model_input = input("选择模型 (用逗号分隔, 默认最佳组合): ").strip()
        if not model_input or model_input == str(len(model_names) + 2):
            # 选择最佳组合：传统模型 + PyTorch模型各一个
            traditional = [m for m in model_names if not ModelFactory.is_pytorch_model(m)]
            pytorch = [m for m in model_names if ModelFactory.is_pytorch_model(m)]
            config_dict['prediction_models'] = (traditional[:1] + pytorch[:1]) or model_names[:2]
        elif model_input == str(len(model_names) + 1):
            config_dict['prediction_models'] = model_names
        else:
            try:
                choices = [int(x.strip()) for x in model_input.split(',')]
                config_dict['prediction_models'] = [
                    model_names[i-1] for i in choices if 1 <= i <= len(model_names)
                ]
            except:
                config_dict['prediction_models'] = model_names[:2]
        
        # 选择数据段
        all_segments = set()
        for segments in available_models.values():
            all_segments.update(segments)
        all_segments = sorted(list(all_segments))
        
        print(f"\n📊 可预测数据段: {all_segments}")
        segment_input = input("选择数据段 (用逗号分隔，默认全部): ").strip()
        
        if segment_input:
            try:
                config_dict['prediction_segments'] = [int(s.strip()) for s in segment_input.split(',')]
                config_dict['prediction_segments'] = [s for s in config_dict['prediction_segments'] if s in all_segments]
            except:
                config_dict['prediction_segments'] = all_segments
        else:
            config_dict['prediction_segments'] = all_segments
        
        # 集成方法
        print("\n🔗 集成方法:")
        print("1. 简单平均")
        print("2. 加权平均")
        print("3. 投票机制")
        
        ensemble_choice = input("选择集成方法 (1-3, 默认1): ").strip() or '1'
        ensemble_map = {'1': 'average', '2': 'weighted_average', '3': 'voting'}
        config_dict['ensemble_method'] = ensemble_map.get(ensemble_choice, 'average')
        
        config_dict['prediction_possible'] = True
        return config_dict
    
    @staticmethod
    def show_cache_management() -> str:
        """显示缓存管理界面"""
        print("\n" + "="*60)
        print("💾 缓存管理")
        print("="*60)
        
        cache_info = config.get_cache_info()
        
        if cache_info['cache_count'] > 0:
            print(f"\n当前缓存状态:")
            print(f"  缓存目录: {cache_info['cache_dir']}")
            print(f"  缓存文件数: {cache_info['cache_count']}")
            print(f"  缓存文件:")
            for file_name in cache_info['cached_files']:
                print(f"    • {file_name}")
        else:
            print("\n📭 暂无缓存数据")
        
        print(f"\n缓存操作:")
        print("1. 📋 查看缓存详情")
        print("2. 🗑️ 清理所有缓存")
        print("3. 🔄 重新生成缓存")
        print("4. ↩️ 返回主菜单")
        
        return input("请选择操作 (1-4): ").strip()
    
    @staticmethod
    def show_config_summary(data_config: Dict[str, Any], training_config: Dict[str, Any] = None):
        """显示配置总结"""
        print(f"\n{'='*60}")
        print("📋 配置总结")
        print('='*60)
        
        # 系统信息
        device_info = config.get_device_info()
        print(f"💻 设备: {device_info['device']}")
        print(f"🐍 PyTorch版本: {device_info['pytorch_version']}")
        
        # 数据处理配置
        print(f"\n🔄 数据处理:")
        mode_desc = {
            'full_process': '完整处理 (编码→特征工程→选择)',
            'encoding_only': '仅数据编码',
            'feature_only': '仅特征工程', 
            'load_cached': '加载缓存数据',
            'raw_data': '使用原始数据',
            'compare_modes': '比较多种处理模式'
        }
        print(f"  模式: {mode_desc.get(data_config.get('data_process_mode'), '未知')}")
        
        if 'feature_level' in data_config:
            level_desc = {
                'none': '跳过',
                'basic': '基础',
                'enhanced': '增强',
                'advanced': '高级'
            }
            print(f"  特征工程: {level_desc.get(data_config['feature_level'], '未知')}")
        
        if 'selection_mode' in data_config:
            selection_desc = {
                'none': '跳过',
                'variance': '方差选择',
                'correlation': '相关性选择',
                'mutual_info': '互信息选择'
            }
            print(f"  特征选择: {selection_desc.get(data_config['selection_mode'], '未知')}")
            
            if data_config.get('max_features'):
                print(f"  最大特征数: {data_config['max_features']}")
        
        # 训练配置
        if training_config:
            print(f"\n🎯 模型训练:")
            print(f"  数据模式: {'抽样' if training_config.get('use_sampling', True) else '全量'}")
            if training_config.get('selected_models'):
                models_str = ', '.join(training_config['selected_models'])
                print(f"  选择模型: {models_str}")
            print(f"  自动调参: {'启用' if training_config.get('enable_auto_tuning', False) else '关闭'}")


class EnhancedWorkflowManager:
    """增强的工作流管理器"""
    
    def __init__(self):
        self.ui = EnhancedUserInterface()
        self.system_checker = SystemChecker()
        self.file_manager = FileManager(config.paths)
        self.progress_tracker = ProgressTracker()
        
        # 初始化组件
        self.model_manager = ModelManager(config.paths.models)
        self.visualizer = ResultsVisualizer(config.paths.output)
        
        # 确保路径存在
        config.ensure_paths()
    
    def run(self):
        """运行主工作流"""
        try:
            # 显示欢迎界面
            self.ui.show_welcome()
            
            # 系统检查
            print("🔍 系统检查中...")
            system_info = self.system_checker.check_system()
            self._display_system_info(system_info)
            
            # 文件检查
            train_files, test_files = self.file_manager.find_data_files()
            
            # 获取运行模式
            run_mode = self.ui.get_run_mode()
            
            # 根据模式执行不同工作流
            if run_mode == '1':
                self._run_full_workflow(train_files, test_files)
            elif run_mode == '2':
                self._run_data_processing_only(train_files)
            elif run_mode == '3':
                self._run_training_only(train_files)
            elif run_mode == '4':
                self._run_prediction_only(test_files)
            elif run_mode == '5':
                self._run_model_comparison()
            elif run_mode == '6':
                self._run_cache_management()
            
            print("\n🎉 程序执行完成!")
            
        except KeyboardInterrupt:
            print("\n⚠️ 程序被用户中断")
        except Exception as e:
            print(f"\n❌ 程序执行失败: {e}")
            import traceback
            traceback.print_exc()
        finally:
            input("\n按Enter键退出...")
    
    def _run_full_workflow(self, train_files: List[Path], test_files: List[Path]):
        """完整工作流"""
        print("\n🚀 完整工作流模式")
        
        # 获取配置
        data_config = self.ui.get_data_processing_config()
        training_config = self.ui.get_training_config()
        
        self.ui.show_config_summary(data_config, training_config)
        input("\n按Enter键开始...")
        
        # 数据处理阶段
        processed_data = self._execute_data_processing(train_files, data_config)
        if not processed_data:
            print("❌ 数据处理失败，无法继续")
            return
        
        # 训练阶段
        training_results = self._execute_training_with_processed_data(processed_data, training_config)
        
        # 预测阶段
        if test_files and training_results:
            print(f"\n{'='*60}")
            print("📈 预测阶段")
            print('='*60)
            
            prediction_config = {
                'prediction_models': training_config['selected_models'],
                'prediction_segments': list(range(len(test_files))),
                'ensemble_method': 'average'
            }
            
            self._execute_prediction(test_files, prediction_config)
        
        # 结果可视化
        if training_results:
            self._generate_visualizations(training_results)
    
    def _run_data_processing_only(self, train_files: List[Path]):
        """仅数据处理模式"""
        print("\n🔄 仅数据处理模式")
        
        data_config = self.ui.get_data_processing_config()
        self.ui.show_config_summary(data_config)
        
        input("\n按Enter键开始...")
        self._execute_data_processing(train_files, data_config)
    
    def _run_training_only(self, train_files: List[Path]):
        """仅训练模式"""
        print("\n🎯 仅训练模式")
        
        # 检查是否有可用的处理数据
        cache_info = config.get_cache_info()
        if cache_info['cache_count'] > 0:
            use_cache = input(f"发现 {cache_info['cache_count']} 个缓存文件，是否使用? (y/n, 默认y): ").strip().lower() != 'n'
            if use_cache:
                data_config = {'data_process_mode': 'load_cached'}
            else:
                data_config = self.ui.get_data_processing_config()
        else:
            data_config = self.ui.get_data_processing_config()
        
        # 训练配置
        training_config = self.ui.get_training_config()
        self.ui.show_config_summary(data_config, training_config)
        
        input("\n按Enter键开始...")
        
        # 数据处理
        processed_data = self._execute_data_processing(train_files, data_config)
        if not processed_data:
            return
        
        # 训练
        training_results = self._execute_training_with_processed_data(processed_data, training_config)
        
        if training_results:
            self._generate_visualizations(training_results)
    
    def _run_prediction_only(self, test_files: List[Path]):
        """仅预测模式"""
        print("\n📈 仅预测模式")
        
        prediction_config = self.ui.get_prediction_config(self.model_manager)
        if not prediction_config.get('prediction_possible', False):
            return
        
        self._execute_prediction(test_files, prediction_config)
    
    def _run_model_comparison(self):
        """模型比较分析"""
        print("\n📊 模型比较分析模式")
        
        # 读取已保存的结果
        result_files = list(config.paths.output.glob("model_results_segment_*.csv"))
        
        if not result_files:
            print("❌ 没有找到已保存的模型结果")
            return
        
        # 合并所有结果
        all_results = []
        for file_path in result_files:
            df = pd.read_csv(file_path)
            segment_id = int(file_path.stem.split('_')[-1])
            df['Segment'] = segment_id
            all_results.append(df)
        
        if all_results:
            combined_results = pd.concat(all_results, ignore_index=True)
            self.visualizer.create_model_comparison_dashboard(combined_results)
            print("✅ 模型比较分析完成")
    
    def _run_cache_management(self):
        """缓存管理"""
        while True:
            choice = self.ui.show_cache_management()
            
            if choice == '1':
                self._show_cache_details()
            elif choice == '2':
                self._clear_cache()
            elif choice == '3':
                print("🔄 重新生成缓存需要重新处理数据，请选择模式2进行数据处理")
            elif choice == '4':
                break
            else:
                print("❌ 无效选择")
    
    def _execute_data_processing(self, train_files: List[Path], data_config: Dict[str, Any]) -> Optional[List[Dict]]:
        """执行数据处理"""
        mode = data_config.get('data_process_mode', 'full_process')
        
        if mode == 'compare_modes':
            return self._execute_comparison_processing(train_files, data_config)
        elif mode == 'load_cached':
            return self._load_cached_data(train_files)
        else:
            return self._execute_single_processing(train_files, data_config)
    
    def _execute_single_processing(self, train_files: List[Path], data_config: Dict[str, Any]) -> Optional[List[Dict]]:
        """执行单一数据处理模式"""
        # 创建数据处理器
        data_processor = DataProcessor(
            feature_level=data_config.get('feature_level', 'enhanced'),
            max_features=data_config.get('max_features', 200),
            enable_selection=data_config.get('selection_mode', 'variance') != 'none'
        )
        
        processed_data = []
        
        with self.progress_tracker.create_training_progress(len(train_files)) as progress:
            for i, train_file in enumerate(train_files):
                progress.update_current_stage(f"处理段 {i}: {train_file.name}")
                
                try:
                    # 加载和处理数据
                    df = data_processor.load_and_process_data(
                        train_file,
                        use_sampling=data_config.get('use_sampling', True),
                        num_groups=data_config.get('num_groups', 2000),
                        min_group_size=data_config.get('min_group_size', 20)
                    )
                    
                    # 保存到缓存
                    if data_config.get('cache_processed_data', True):
                        cache_file = config.paths.cache_data / f"processed_segment_{i}.pkl"
                        df.to_pickle(cache_file)
                        print(f"💾 已缓存: {cache_file.name}")
                    
                    processed_data.append({
                        'segment_id': i,
                        'data': df,
                        'processor': data_processor
                    })
                    
                    progress.complete_stage()
                    
                except Exception as e:
                    print(f"❌ 处理段 {i} 失败: {e}")
                    progress.complete_stage(success=False)
                    continue
        
        return processed_data
    
    def _load_cached_data(self, train_files: List[Path]) -> Optional[List[Dict]]:
        """加载缓存数据"""
        processed_data = []
        
        for i in range(len(train_files)):
            cache_file = config.paths.cache_data / f"processed_segment_{i}.pkl"
            if cache_file.exists():
                try:
                    df = pd.read_pickle(cache_file)
                    # 创建一个默认的数据处理器
                    data_processor = DataProcessor()
                    
                    processed_data.append({
                        'segment_id': i,
                        'data': df,
                        'processor': data_processor
                    })
                    print(f"✅ 加载缓存段 {i}: {len(df)} 条记录")
                except Exception as e:
                    print(f"❌ 加载缓存段 {i} 失败: {e}")
        
        if not processed_data:
            print("❌ 没有找到可用的缓存数据")
            return None
        
        return processed_data
    
    def _execute_training_with_processed_data(self, processed_data: List[Dict], training_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """使用处理后的数据执行训练"""
        # 创建训练器
        trainer = ModelTrainer(
            model_manager=self.model_manager,
            enable_auto_tuning=training_config.get('enable_auto_tuning', False),
            auto_tuning_trials=training_config.get('auto_tuning_trials', 30)
        )
        
        training_results = []
        
        with self.progress_tracker.create_training_progress(len(processed_data)) as progress:
            for processed_item in processed_data:
                segment_id = processed_item['segment_id']
                df = processed_item['data']
                data_processor = processed_item['processor']
                
                progress.update_current_stage(f"训练段 {segment_id}")
                
                try:
                    # 分割数据
                    data_split = data_processor.split_ranking_data(df)
                    (X_train, X_test, y_train, y_test, 
                     train_group_sizes, test_group_sizes, feature_cols, test_info) = data_split
                    
                    # 训练模型
                    results_df = trainer.train_models(
                        model_names=training_config['selected_models'],
                        X_train=X_train, y_train=y_train, train_groups=train_group_sizes,
                        X_val=X_test, y_val=y_test, val_groups=test_group_sizes,
                        segment_id=segment_id, feature_names=feature_cols,
                        save_models=training_config.get('save_models', True)
                    )
                    
                    # 保存结果
                    if not results_df.empty:
                        results_file = config.paths.output / f"model_results_segment_{segment_id}.csv"
                        results_df.to_csv(results_file, index=False)
                        
                        best_model = results_df.loc[results_df['HitRate@3'].idxmax()]
                        training_results.append({
                            'segment_id': segment_id,
                            'results': results_df,
                            'best_model': best_model,
                            'feature_count': len(feature_cols)
                        })
                        
                        print(f"✅ 段 {segment_id} 最佳: {best_model['Model']} "
                              f"(HitRate@3: {best_model['HitRate@3']:.4f})")
                    
                    progress.complete_stage()
                    
                except Exception as e:
                    print(f"❌ 训练段 {segment_id} 失败: {e}")
                    progress.complete_stage(success=False)
                    continue
        
        return training_results
    
    def _execute_prediction(self, test_files: List[Path], prediction_config: Dict[str, Any]):
        """执行预测"""
        # 创建数据处理器（用于测试数据处理）
        from data_processor import EnhancedDataProcessor
        data_processor = EnhancedDataProcessor()
        
        # 创建预测器
        predictor = ModelPredictor(self.model_manager, data_processor)
        
        # 执行预测
        final_result = predictor.predict_all_segments(
            segments=prediction_config['prediction_segments'],
            model_names=prediction_config['prediction_models'],
            test_data_path=config.paths.test_data,
            ensemble_method=prediction_config['ensemble_method'],
            output_path=config.paths.output
        )
        
        if final_result is not None:
            print(f"✅ 预测完成，总记录数: {len(final_result)}")
        else:
            print("❌ 预测失败")
    
    def _execute_comparison_processing(self, train_files: List[Path], data_config: Dict[str, Any]) -> Optional[List[Dict]]:
        """执行比较处理模式"""
        print("🔄 比较不同处理模式...")
        
        comparison_modes = data_config.get('comparison_modes', ['raw_data', 'encoding_only', 'basic', 'enhanced', 'advanced'])
        comparison_results = []
        
        for mode in comparison_modes:
            print(f"\n处理模式: {mode}")
            
            # 配置该模式
            mode_config = data_config.copy()
            if mode == 'raw_data':
                mode_config.update({
                    'data_process_mode': 'raw_data',
                    'feature_level': 'none',
                    'selection_mode': 'none'
                })
            elif mode == 'encoding_only':
                mode_config.update({
                    'data_process_mode': 'encoding_only',
                    'feature_level': 'none'
                })
            else:
                mode_config.update({
                    'data_process_mode': 'full_process',
                    'feature_level': mode
                })
            
            # 处理数据
            try:
                processed_data = self._execute_single_processing(train_files, mode_config)
                if processed_data:
                    comparison_results.extend(processed_data)
                    for item in processed_data:
                        item['processing_mode'] = mode
                    print(f"✅ {mode}: 处理完成")
            except Exception as e:
                print(f"❌ {mode}: 处理失败 - {e}")
        
        return comparison_results if comparison_results else None
    
    def _load_cached_data(self, train_files: List[Path]) -> Optional[List[Dict]]:
        """加载缓存数据"""
        print("💾 加载缓存数据...")
        
        processed_data = []
        
        for i in range(len(train_files)):
            cache_file = config.paths.cache_data / f"processed_segment_{i}.pkl"
            if cache_file.exists():
                try:
                    df = pd.read_pickle(cache_file)
                    # 创建一个默认的数据处理器
                    from data_processor import EnhancedDataProcessor
                    data_processor = EnhancedDataProcessor()
                    
                    processed_data.append({
                        'segment_id': i,
                        'data': df,
                        'processor': data_processor
                    })
                    print(f"✅ 加载缓存段 {i}: {len(df)} 条记录")
                except Exception as e:
                    print(f"❌ 加载缓存段 {i} 失败: {e}")
        
        if not processed_data:
            print("❌ 没有找到可用的缓存数据")
            return None
        
        return processed_data
    
    def _generate_visualizations(self, training_results: List[Dict[str, Any]]):
        """生成可视化结果"""
        try:
            # 创建模型性能对比图
            for result in training_results:
                self.visualizer.plot_model_performance(
                    result['results'], 
                    result['segment_id']
                )
            
            # 创建综合分析报告
            if len(training_results) > 1:
                self.visualizer.create_training_summary(training_results)
            
            print("✅ 可视化结果已生成")
            
        except Exception as e:
            print(f"⚠️ 生成可视化失败: {e}")
    
    def _show_cache_details(self):
        """显示缓存详细信息"""
        cache_info = config.get_cache_info()
        
        print("\n📋 缓存详细信息:")
        print(f"缓存目录: {cache_info['cache_dir']}")
        print(f"缓存文件数: {cache_info.get('cache_count', 0)}")
        
        if cache_info.get('cache_count', 0) > 0:
            print("缓存文件列表:")
            for file_name in cache_info.get('cached_files', []):
                cache_file = Path(cache_info['cache_dir']) / file_name
                if cache_file.exists():
                    size_mb = cache_file.stat().st_size / (1024 * 1024)
                    print(f"  • {file_name} ({size_mb:.1f}MB)")
        
        input("\n按Enter键继续...")
    
    def _clear_cache(self):
        """清理缓存"""
        cache_info = config.get_cache_info()
        
        if cache_info.get('cache_count', 0) == 0:
            print("📭 当前没有缓存文件")
            return
        
        confirm = input(f"确认清理 {cache_info['cache_count']} 个缓存文件? (y/N): ").strip().lower()
        
        if confirm == 'y':
            try:
                # 清理缓存目录中的所有pkl文件
                cache_dir = Path(cache_info['cache_dir'])
                for cache_file in cache_dir.glob("*.pkl"):
                    cache_file.unlink()
                
                print("✅ 缓存已清理")
            except Exception as e:
                print(f"❌ 清理缓存失败: {e}")
        else:
            print("取消清理操作")
    
    def _display_system_info(self, system_info: Dict[str, Any]):
        """显示系统信息"""
        print("💻 系统信息:")
        key_items = ['python_version', 'gpu_available', 'memory_total_gb']
        for key in key_items:
            if key in system_info:
                value = system_info[key]
                if isinstance(value, bool):
                    status = "✅" if value else "❌"
                    print(f"  {status} {key.replace('_', ' ').title()}: {value}")
                else:
                    print(f"  • {key.replace('_', ' ').title()}: {value}")


class ApplicationManager:
    """应用程序管理器"""
    
    def __init__(self):
        self.workflow_manager = EnhancedWorkflowManager()
    
    def start(self):
        """启动应用程序"""
        try:
            # 检查Python版本
            if sys.version_info < (3, 8):
                print("❌ 需要Python 3.8或更高版本")
                return
            
            # 运行工作流
            self.workflow_manager.run()
            
        except Exception as e:
            print(f"❌ 应用程序启动失败: {e}")
            import traceback
            traceback.print_exc()


def main():
    """主函数"""
    app = ApplicationManager()
    app.start()


if __name__ == "__main__":
    main()