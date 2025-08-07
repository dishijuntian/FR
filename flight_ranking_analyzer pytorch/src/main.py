"""
主程序入口 - 重构版

专注于：
- 用户交互
- 程序流程控制
- 错误处理
- 结果展示

作者: Flight Ranking Team
版本: 4.0 (重构版)
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import torch

# 导入模块
from config import Config
from analyzer import FlightRankingAnalyzer
from predictor import FlightRankingPredictor


class UserInterface:
    """用户交互界面"""
    #@staticmethod用于定义静态方法，不需要实例化类即可调用，
    #它类似于普通函数，但被封装在类的命名空间中，主要用于逻辑上的组织。
    @staticmethod
    def get_run_mode() -> str:
        """获取运行模式"""
        print("\n" + "="*60)
        print("航班排序分析系统 v4.0 (重构版)")
        print("="*60)
        
        print("\n运行模式:")
        print("1. 完整流程 (训练 + 预测)")
        print("2. 仅训练模型")
        print("3. 仅预测 (使用已保存的模型)")
        
        while True:
            choice = input("\n请选择运行模式 (1-3): ").strip()
            if choice in ['1', '2', '3']:
                return choice
            print("请输入有效选择 (1-3)")
    #获得了训练配置，包括使用的数据模式、模型选择、自动调参等选项。
    @staticmethod
    def get_training_config() -> Dict[str, Any]:
        """获取训练配置"""
        config = {}
        
        # 数据模式
        print("\n数据加载模式:")
        print("1. 抽样模式 (推荐)")
        print("2. 全量模式")
        
        data_mode = input("请选择 (1-2): ").strip()
        config['use_sampling'] = data_mode == '1'
        
        if config['use_sampling']:
            try:
                config['num_groups'] = int(input(f"抽样组数 (默认{Config.DEFAULT_NUM_GROUPS}): ") or Config.DEFAULT_NUM_GROUPS)
                config['min_group_size'] = int(input(f"最小组大小 (默认{Config.DEFAULT_MIN_GROUP_SIZE}): ") or Config.DEFAULT_MIN_GROUP_SIZE)
            except ValueError:
                config['num_groups'] = Config.DEFAULT_NUM_GROUPS
                config['min_group_size'] = Config.DEFAULT_MIN_GROUP_SIZE
        
        # 模型选择
        print("\n可用模型:")
        for i, model in enumerate(Config.AVAILABLE_MODELS, 1):
            model_type = "🔥 PyTorch" if Config.is_pytorch_model(model) else "📊 传统"
            print(f"{i}. {model} ({model_type})")
        print(f"{len(Config.AVAILABLE_MODELS) + 1}. 所有模型")
        
        model_input = input("\n选择模型 (用逗号分隔): ").strip()
        if model_input:
            try:
                choices = [int(x.strip()) for x in model_input.split(',')]
                if len(Config.AVAILABLE_MODELS) + 1 in choices:
                    config['selected_models'] = Config.AVAILABLE_MODELS
                else:
                    config['selected_models'] = [
                        Config.AVAILABLE_MODELS[i-1] 
                        for i in choices 
                        if 1 <= i <= len(Config.AVAILABLE_MODELS)
                    ]
            except:
                config['selected_models'] = ['XGBRanker', 'NeuralRanker']
        else:
            config['selected_models'] = ['XGBRanker', 'NeuralRanker']
        
        # 自动调参
        config['enable_auto_tuning'] = input("\n启用自动调参? (y/n): ").strip().lower() == 'y'
        if config['enable_auto_tuning']:
            try:
                config['auto_tuning_trials'] = int(input("调参试验次数 (默认50): ") or 50)
            except:
                config['auto_tuning_trials'] = 50
        
        # 模型保存
        config['save_models'] = input("保存训练的模型? (y/n): ").strip().lower() != 'n'
        
        return config
    
    @staticmethod
    def get_prediction_config() -> Dict[str, Any]:
        """获取预测配置"""
        config = {}
        
        # 初始化预测器检查可用模型
        predictor = FlightRankingPredictor(Config.DATA_BASE_PATH)
        available_models = predictor.get_available_models()
        
        if not available_models:
            print("❌ 没有找到保存的模型，请先训练模型")
            config['prediction_possible'] = False
            return config
        
        print("\n可用的已保存模型:")
        for model_name, segments in available_models.items():
            model_type = "🔥 PyTorch" if Config.is_pytorch_model(model_name) else "📊 传统"
            print(f"  {model_name} ({model_type}): 段 {segments}")
        
        # 选择模型
        model_names = list(available_models.keys())
        print(f"\n模型选择:")
        for i, model_name in enumerate(model_names, 1):
            print(f"{i}. {model_name}")
        print(f"{len(model_names) + 1}. 所有模型")
        
        model_input = input("选择模型 (用逗号分隔): ").strip()
        if model_input:
            try:
                choices = [int(x.strip()) for x in model_input.split(',')]
                if len(model_names) + 1 in choices:
                    config['prediction_models'] = model_names
                else:
                    config['prediction_models'] = [
                        model_names[i-1] for i in choices if 1 <= i <= len(model_names)
                    ]
            except:
                config['prediction_models'] = model_names
        else:
            config['prediction_models'] = model_names
        
        # 选择数据段
        all_segments = set()
        for segments in available_models.values():
            all_segments.update(segments)
        all_segments = sorted(list(all_segments))
        
        print(f"\n可预测数据段: {all_segments}")
        segment_input = input("选择数据段 (用逗号分隔，默认全部): ").strip()
        
        if segment_input:
            try:
                config['prediction_segments'] = [int(s.strip()) for s in segment_input.split(',')]
                config['prediction_segments'] = [s for s in config['prediction_segments'] if s in all_segments]
            except:
                config['prediction_segments'] = all_segments
        else:
            config['prediction_segments'] = all_segments
        
        # 集成方法
        print("\n集成方法:")
        print("1. 平均分数 (average)")
        print("2. 排名投票 (voting)")
        
        ensemble_choice = input("选择集成方法 (1-2): ").strip()
        config['ensemble_method'] = 'voting' if ensemble_choice == '2' else 'average'
        
        config['prediction_possible'] = True
        return config


class SystemChecker:
    """系统检查器"""
    
    @staticmethod
    def check_gpu() -> bool:
        """检查GPU可用性"""
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✓ GPU可用: {device_name} ({gpu_memory:.1f}GB)")
            return True
        else:
            print("⚠️ GPU不可用，将使用CPU")
            return False
    
    @staticmethod
    def validate_data_files() -> tuple[List[Path], List[Path]]:
        """验证数据文件"""
        train_files = []
        test_files = []
        
        # 查找训练文件
        train_dir = Config.TRAIN_DATA_PATH
        if train_dir.exists():
            train_files = sorted(train_dir.glob("*.parquet"))
        
        # 查找测试文件
        test_dir = Config.TEST_DATA_PATH
        if test_dir.exists():
            test_files = sorted(test_dir.glob("*.parquet"))
        
        print(f"\n数据文件检查:")
        print(f"训练文件: {len(train_files)} 个")
        print(f"测试文件: {len(test_files)} 个")
        
        if not train_files:
            raise FileNotFoundError("未找到训练文件")
        
        return train_files, test_files


class WorkflowManager:
    """工作流管理器"""
    
    def __init__(self):
        self.ui = UserInterface()
        self.checker = SystemChecker()
    
    def run(self):
        """运行主工作流"""
        try:
            # 系统检查
            print("🔍 系统检查...")
            use_gpu = self.checker.check_gpu()
            train_files, test_files = self.checker.validate_data_files()
            
            # 获取用户配置
            run_mode = self.ui.get_run_mode()
            
            if run_mode == '3':
                # 仅预测模式
                self._run_prediction_only()
            else:
                # 训练模式
                self._run_training_workflow(run_mode, use_gpu, train_files, test_files)
            
            print("\n🎉 程序执行完成!")
            
        except KeyboardInterrupt:
            print("\n⚠️ 程序被用户中断")
        except Exception as e:
            print(f"\n❌ 程序执行失败: {e}")
            import traceback
            traceback.print_exc()
        finally:
            input("\n按Enter键退出...")
    
    def _run_prediction_only(self):
        """仅预测模式"""
        print("\n📊 仅预测模式")
        
        config = self.ui.get_prediction_config()
        if not config.get('prediction_possible', False):
            return
        
        # 初始化预测器
        predictor = FlightRankingPredictor(Config.DATA_BASE_PATH)
        
        # 执行预测
        result = predictor.predict_all(
            segments=config['prediction_segments'],
            model_names=config['prediction_models'],
            ensemble_method=config['ensemble_method']
        )
        
        if result is not None:
            print(f"✅ 预测完成，结果记录数: {len(result)}")
        else:
            print("❌ 预测失败")
    
    def _run_training_workflow(self, run_mode: str, use_gpu: bool, 
                              train_files: List[Path], test_files: List[Path]):
        """训练工作流"""
        print(f"\n🚀 训练模式 ({'完整流程' if run_mode == '1' else '仅训练'})")
        
        config = self.ui.get_training_config()
        
        # 显示配置总结
        print(f"\n配置总结:")
        print(f"  GPU: {'启用' if use_gpu else '关闭'}")
        print(f"  数据模式: {'抽样' if config['use_sampling'] else '全量'}")
        print(f"  模型: {', '.join(config['selected_models'])}")
        print(f"  自动调参: {'启用' if config['enable_auto_tuning'] else '关闭'}")
        
        input("\n按Enter键开始...")
        
        # 初始化分析器
        analyzer = FlightRankingAnalyzer(
            use_gpu=use_gpu,
            selected_models=config['selected_models'],
            enable_auto_tuning=config['enable_auto_tuning'],
            auto_tuning_trials=config.get('auto_tuning_trials', 50),
            save_models=config['save_models']
        )
        
        # 训练阶段
        train_results = {}
        for i, train_file in enumerate(train_files):
            print(f"\n{'='*50}")
            print(f"训练段 {i}: {train_file.name}")
            print('='*50)
            
            try:
                result = analyzer.full_analysis(
                    train_file,
                    use_sampling=config['use_sampling'],
                    num_groups=config.get('num_groups', Config.DEFAULT_NUM_GROUPS),
                    min_group_size=config.get('min_group_size', Config.DEFAULT_MIN_GROUP_SIZE)
                )
                train_results[f'segment_{i}'] = result
                
                # 显示最佳模型
                if not result['model_results'].empty:
                    best_model = result['model_results'].loc[
                        result['model_results']['HitRate@3'].idxmax()
                    ]
                    print(f"✅ 最佳模型: {best_model['Model']} (HitRate@3: {best_model['HitRate@3']:.4f})")
                
            except Exception as e:
                print(f"❌ 训练段 {i} 失败: {e}")
                continue
        
        # 预测阶段 (仅完整流程)
        if run_mode == '1' and test_files:
            print(f"\n{'='*50}")
            print("预测阶段")
            print('='*50)
            
            # 使用保存的模型进行预测
            predictor = FlightRankingPredictor(Config.DATA_BASE_PATH)
            
            result = predictor.predict_all(
                segments=list(range(len(test_files))),
                model_names=config['selected_models'],
                ensemble_method='average'
            )
            
            if result is not None:
                print(f"✅ 预测完成，总记录数: {len(result)}")
            else:
                print("❌ 预测失败")
        
        # 显示训练总结
        self._show_training_summary(train_results, config)
    
    def _show_training_summary(self, train_results: Dict[str, Any], config: Dict[str, Any]):
        """显示训练总结"""
        print(f"\n{'='*60}")
        print("训练总结")
        print('='*60)
        
        print(f"训练段数: {len(train_results)}")
        print(f"使用模型: {', '.join(config['selected_models'])}")
        
        # 各段最佳性能
        if train_results:
            print(f"\n各段最佳模型性能:")
            for segment_name, result in train_results.items():
                if 'model_results' in result and not result['model_results'].empty:
                    best_model = result['model_results'].loc[
                        result['model_results']['HitRate@3'].idxmax()
                    ]
                    model_type = "🔥" if Config.is_pytorch_model(best_model['Model']) else "📊"
                    print(f"  {segment_name}: {best_model['Model']} {model_type} "
                          f"(HitRate@3: {best_model['HitRate@3']:.4f})")
        
        print(f"\n结果保存路径: {Config.OUTPUT_PATH}")


def main():
    """主函数"""
    # 确保配置正确
    Config.ensure_paths()
    
    # 运行工作流
    workflow = WorkflowManager()
    workflow.run()


if __name__ == "__main__":
    main()