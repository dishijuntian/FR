"""
主程序文件 - 改进版本

该程序提供了完整的航班排序分析流程
- 新增模型保存和加载功能
- 支持仅预测模式
- 改进的用户交互
- 更好的错误处理和结果展示

作者: Flight Ranking Team
版本: 3.0 (改进版)
"""

import os
import sys
import gc
import logging
from typing import List, Dict, Any

# 尝试相对导入，如果失败则使用绝对导入
try:
    from .config import Config
    from .analyzer import FlightRankingAnalyzer
    from .models import ModelFactory
    from .predictor import FlightRankingPredictor
except ImportError:
    from config import Config
    from analyzer import FlightRankingAnalyzer
    from models import ModelFactory
    from predictor import FlightRankingPredictor


def setup_logging() -> logging.Logger:
    """设置日志记录"""
    # 确保输出目录存在
    Config.ensure_output_dir()
    
    # 创建日志文件路径
    log_file = os.path.join(Config.OUTPUT_PATH, 'analysis.log')
    
    logging.basicConfig(
        level=getattr(logging, Config.LOG_LEVEL),
        format=Config.LOG_FORMAT,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding='utf-8')
        ]
    )
    return logging.getLogger(__name__)


def check_gpu_availability() -> bool:
    """检查GPU是否可用"""
    try:
        import cupy as cp
        import tensorflow as tf
        
        # 检查CUDA
        cupy_available = cp.cuda.is_available()
        
        if cupy_available:
            print("✓ 检测到CUDA GPU，将启用GPU加速")
            return True
        else:
            print("⚠ 未检测到完整的GPU支持，将使用CPU版本")
            return False
    except ImportError:
        print("⚠ 未安装GPU相关库，将使用CPU版本")
        return False


def get_user_choices() -> Dict[str, Any]:
    """获取用户选择"""
    print("\n" + "="*60)
    print("航班排序分析系统 v3.0 (改进版)")
    print("="*60)
    
    # 运行模式选择
    print("\n运行模式:")
    print("1. 完整流程 (训练 + 预测)")
    print("2. 仅训练模型")
    print("3. 仅预测 (使用已保存的模型)")
    
    while True:
        run_mode = input("\n请选择运行模式 (1-3): ").strip()
        if run_mode in ['1', '2', '3']:
            break
        print("请输入有效选择 (1-3)")
    
    choices = {'run_mode': run_mode}
    
    # 如果选择仅预测，询问预测设置
    if run_mode == '3':
        return get_prediction_choices(choices)
    
    # 训练相关选择
    choices.update(get_training_choices())
    
    return choices


def get_training_choices() -> Dict[str, Any]:
    """获取训练相关选择"""
    choices = {}
    
    # 数据加载选择
    print("\n数据加载模式:")
    print("1. 抽样模式 (推荐用于快速测试)")
    print("2. 全量模式 (使用所有数据)")
    
    while True:
        data_mode = input("\n请选择数据加载模式 (1-2): ").strip()
        if data_mode in ['1', '2']:
            break
        print("请输入有效选择 (1-2)")
    
    use_sampling = data_mode == '1'
    choices['use_sampling'] = use_sampling
    
    # 抽样参数设置
    num_groups = Config.DEFAULT_NUM_GROUPS
    min_group_size = Config.DEFAULT_MIN_GROUP_SIZE
    
    if use_sampling:
        print(f"\n当前抽样参数:")
        print(f"- 每个文件抽取组数: {num_groups}")
        print(f"- 每组最小数据条数: {min_group_size}")
        
        modify = input("\n是否修改抽样参数? (y/n): ").strip().lower()
        if modify == 'y':
            try:
                num_groups = int(input(f"请输入每个文件抽取的组数 (默认{num_groups}): ") or num_groups)
                min_group_size = int(input(f"请输入每组最小数据条数 (默认{min_group_size}): ") or min_group_size)
            except ValueError:
                print("使用默认参数")
    
    choices['num_groups'] = num_groups
    choices['min_group_size'] = min_group_size
    
    # 模型选择
    print("\n可用模型:")
    available_models = ModelFactory.get_available_models()
    for i, model in enumerate(available_models, 1):
        print(f"{i}. {model}")
    print(f"{len(available_models) + 1}. 所有模型")
    
    model_choices = input(f"\n请选择要训练的模型(用逗号分隔,如1,2,3): ").strip().split(',')
    selected_models = []
    
    if str(len(available_models) + 1) in model_choices:
        selected_models = available_models
    else:
        for choice in model_choices:
            try:
                idx = int(choice.strip()) - 1
                if 0 <= idx < len(available_models):
                    selected_models.append(available_models[idx])
            except ValueError:
                continue
    
    if not selected_models:
        print("未选择任何模型，将默认运行所有模型")
        selected_models = available_models
    
    choices['selected_models'] = selected_models
    
    # 自动调参选择
    print("\n自动调参设置:")
    print("自动调参可以优化模型性能，但会显著增加运行时间")
    enable_auto_tuning = input("是否启用自动调参? (y/n, 默认n): ").strip().lower() == 'y'
    
    auto_tuning_trials = Config.AUTO_TUNING_TRIALS
    if enable_auto_tuning:
        try:
            auto_tuning_trials = int(input(f"调参试验次数 (默认{auto_tuning_trials}): ") or auto_tuning_trials)
        except ValueError:
            pass
    
    choices['enable_auto_tuning'] = enable_auto_tuning
    choices['auto_tuning_trials'] = auto_tuning_trials
    
    # 模型保存选择
    save_models = input("\n是否保存训练好的模型? (y/n, 默认y): ").strip().lower() != 'n'
    choices['save_models'] = save_models
    
    return choices


def get_prediction_choices(choices: Dict[str, Any]) -> Dict[str, Any]:
    """获取预测相关选择"""
    print("\n预测设置:")
    
    # 初始化预测器来检查可用模型
    predictor = FlightRankingPredictor(data_path=Config.DATA_BASE_PATH)
    available_models = predictor.get_available_models()
    
    if not available_models:
        print("❌ 没有找到保存的模型，请先训练模型")
        choices['prediction_possible'] = False
        return choices
    
    print("可用的已保存模型:")
    predictor.print_model_summary()
    
    # 选择要使用的模型
    model_names = list(available_models.keys())
    print(f"\n模型选择:")
    for i, model_name in enumerate(model_names, 1):
        segments = available_models[model_name]
        print(f"{i}. {model_name} (段: {segments})")
    print(f"{len(model_names) + 1}. 所有模型")
    
    model_choices = input(f"\n请选择要使用的模型(用逗号分隔): ").strip().split(',')
    selected_models = []
    
    if str(len(model_names) + 1) in model_choices:
        selected_models = model_names
    else:
        for choice in model_choices:
            try:
                idx = int(choice.strip()) - 1
                if 0 <= idx < len(model_names):
                    selected_models.append(model_names[idx])
            except ValueError:
                continue
    
    if not selected_models:
        selected_models = model_names
    
    choices['prediction_models'] = selected_models
    
    # 选择要预测的数据段
    all_segments = set()
    for model_name in selected_models:
        all_segments.update(available_models[model_name])
    all_segments = sorted(list(all_segments))
    
    print(f"\n可预测的数据段: {all_segments}")
    segment_input = input(f"请选择要预测的数据段(用逗号分隔,默认全部): ").strip()
    
    if segment_input:
        try:
            selected_segments = [int(s.strip()) for s in segment_input.split(',')]
            selected_segments = [s for s in selected_segments if s in all_segments]
        except ValueError:
            selected_segments = all_segments
    else:
        selected_segments = all_segments
    
    choices['prediction_segments'] = selected_segments
    
    # 集成方法选择
    print("\n集成方法:")
    print("1. 平均分数 (average)")
    print("2. 排名投票 (voting)")
    print("3. 加权平均 (weighted)")
    
    ensemble_choice = input("请选择集成方法 (1-3, 默认1): ").strip()
    ensemble_methods = {'1': 'average', '2': 'voting', '3': 'weighted'}
    ensemble_method = ensemble_methods.get(ensemble_choice, 'average')
    
    choices['ensemble_method'] = ensemble_method
    choices['prediction_possible'] = True
    
    return choices


def validate_data_files() -> tuple[List[str], List[str]]:
    """验证数据文件是否存在"""
    train_files = Config.get_train_files()
    test_files = Config.get_test_files()
    
    print(f"\n检查数据文件...")
    print(f"训练文件: 找到 {len(train_files)} 个")
    for f in train_files:
        print(f"  ✓ {f}")
    
    print(f"测试文件: 找到 {len(test_files)} 个")
    for f in test_files:
        print(f"  ✓ {f}")
    
    if not train_files:
        raise FileNotFoundError("未找到训练文件")
    if not test_files:
        raise FileNotFoundError("未找到测试文件")
    
    return train_files, test_files


def run_training_phase(analyzer: FlightRankingAnalyzer, 
                      train_files: List[str],
                      use_sampling: bool,
                      num_groups: int,
                      min_group_size: int) -> Dict[str, Any]:
    """执行训练阶段"""
    print(f"\n" + "="*60)
    print("开始训练阶段")
    print("="*60)
    
    all_results = {}
    
    for i, train_path in enumerate(train_files):
        try:
            print(f"\n{'='*40}")
            print(f"训练段 {i}: {os.path.basename(train_path)}")
            if use_sampling:
                print(f"抽样参数: {num_groups}个组, 每组至少{min_group_size}条数据")
            else:
                print("使用全量数据")
            print('='*40)
            
            result = analyzer.full_analysis(
                train_path, 
                use_sampling=use_sampling,
                num_groups=num_groups, 
                min_group_size=min_group_size
            )
            all_results[f'train_segment_{i}'] = result
            
            # 显示重要特征
            if 'feature_importance' in result and result['feature_importance'] is not None:
                top_features = result['feature_importance'].head(10)
                print(f"\n训练段 {i} 中最重要的10个特征:")
                for feature, importance in top_features.items():
                    print(f"  {feature}: {importance:.4f}")
            
            # 释放内存
            gc.collect()
            
        except Exception as e:
            print(f"\n分析训练段 {train_path} 时出错: {str(e)}")
            continue
    
    return all_results


def run_prediction_phase_with_saved_models(user_choices: Dict[str, Any]) -> str:
    """使用保存的模型执行预测阶段"""
    print(f"\n" + "="*60)
    print("开始预测阶段 (使用保存的模型)")
    print("="*60)
    
    # 初始化预测器
    predictor = FlightRankingPredictor(data_path=Config.DATA_BASE_PATH)
    
    # 执行预测
    result = predictor.predict_all(
        segments=user_choices['prediction_segments'],
        model_names=user_choices['prediction_models'],
        ensemble_method=user_choices['ensemble_method']
    )
    
    if result is not None:
        model_suffix = "_".join(user_choices['prediction_models'])
        final_output = predictor.output_path / f"{model_suffix}_final_submission.csv"
        print(f"✅ 预测完成，结果已保存到: {final_output}")
        return str(final_output)
    else:
        raise ValueError("预测失败")


def run_prediction_phase_legacy(analyzer: FlightRankingAnalyzer, 
                               test_files: List[str]) -> List[str]:
    """执行预测阶段 (传统方法)"""
    print(f"\n" + "="*60)
    print("开始预测阶段 (传统方法)")
    print("="*60)
    
    prediction_files = []
    
    for i, test_path in enumerate(test_files):
        try:
            print(f"\n处理测试文件 {i}: {os.path.basename(test_path)}")
            
            result_file = analyzer.predict_test_data(test_path, i)
            
            if result_file and os.path.exists(result_file):
                prediction_files.append(result_file)
                print(f"✓ 预测完成: {os.path.basename(result_file)}")
            else:
                print(f"✗ 预测失败")
            
        except Exception as e:
            print(f"✗ 预测测试文件 {test_path} 时出错: {str(e)}")
            continue
    
    return prediction_files


def run_merge_phase(analyzer: FlightRankingAnalyzer, 
                   prediction_files: List[str]) -> str:
    """执行结果合并阶段"""
    print(f"\n" + "="*60)
    print("开始结果合并阶段")
    print("="*60)
    
    if not prediction_files:
        raise ValueError("没有可用的预测文件进行合并")
    
    print(f"找到 {len(prediction_files)} 个预测文件:")
    for f in prediction_files:
        print(f"  - {os.path.basename(f)}")
    
    # 合并预测结果
    output_file = os.path.join(Config.OUTPUT_PATH, Config.FINAL_PREDICTION_FILENAME)
    submission_file = Config.SUBMISSION_FILE_PATH
    
    try:
        final_result = analyzer.merge_all_predictions(
            prediction_files=prediction_files,
            submission_file=submission_file,
            output_file=output_file
        )
        
        print(f"✓ 最终预测结果已保存到: {final_result}")
        return final_result
        
    except Exception as e:
        print(f"✗ 合并预测结果时出错: {str(e)}")
        raise


def print_summary(train_results: Dict[str, Any] = None, 
                 prediction_files: List[str] = None,
                 final_result: str = None,
                 selected_models: List[str] = None,
                 enable_auto_tuning: bool = False,
                 run_mode: str = "1"):
    """打印分析总结"""
    print(f"\n" + "="*60)
    print("分析完成总结")
    print("="*60)
    
    if run_mode == "1":
        print("运行模式: 完整流程 (训练 + 预测)")
    elif run_mode == "2":
        print("运行模式: 仅训练模型")
    elif run_mode == "3":
        print("运行模式: 仅预测")
    
    if selected_models:
        print(f"使用的模型: {', '.join(selected_models)}")
    
    if run_mode in ["1", "2"]:
        print(f"自动调参: {'启用' if enable_auto_tuning else '关闭'}")
        if train_results:
            print(f"训练段数: {len(train_results)}")
            
            # 显示各段最佳模型性能
            print(f"\n各训练段最佳模型性能:")
            for segment_name, result in train_results.items():
                if 'model_results' in result and not result['model_results'].empty:
                    best_model = result['model_results'].loc[result['model_results']['HitRate@3'].idxmax()]
                    print(f"  {segment_name}: {best_model['Model']} (HitRate@3: {best_model['HitRate@3']:.4f})")
    
    if prediction_files:
        print(f"预测文件数: {len(prediction_files)}")
    
    if final_result:
        print(f"最终结果文件: {os.path.basename(final_result)}")
    
    print(f"\n所有文件已保存到: {Config.OUTPUT_PATH}")


def main():
    """主函数"""
    logger = None
    try:
        # 设置日志
        logger = setup_logging()
        
        # 验证配置
        Config.validate_config()
        
        # 检查GPU
        use_gpu = check_gpu_availability()
        
        # 获取用户选择
        user_choices = get_user_choices()
        
        # 验证数据文件
        train_files, test_files = validate_data_files()
        
        run_mode = user_choices['run_mode']
        
        # 根据运行模式执行不同流程
        if run_mode == '3':  # 仅预测模式
            if not user_choices.get('prediction_possible', False):
                print("❌ 无法执行预测，请先训练模型")
                return
            
            print(f"\n将使用模型: {', '.join(user_choices['prediction_models'])}")
            print(f"预测数据段: {user_choices['prediction_segments']}")
            print(f"集成方法: {user_choices['ensemble_method']}")
            
            input("\n按Enter键开始预测...")
            
            # 使用保存的模型进行预测
            final_result = run_prediction_phase_with_saved_models(user_choices)
            
            # 打印总结
            print_summary(
                final_result=final_result,
                selected_models=user_choices['prediction_models'],
                run_mode=run_mode
            )
            
        else:  # 训练模式 (1或2)
            # 初始化分析器
            analyzer = FlightRankingAnalyzer(
                use_gpu=use_gpu,
                logger=logger,
                selected_models=user_choices['selected_models'],
                enable_auto_tuning=user_choices['enable_auto_tuning'],
                auto_tuning_trials=user_choices['auto_tuning_trials'],
                save_models=user_choices.get('save_models', True)
            )
            
            print(f"\n将运行以下模型: {', '.join(user_choices['selected_models'])}")
            print(f"数据模式: {'抽样' if user_choices['use_sampling'] else '全量'}")
            print(f"自动调参: {'启用' if user_choices['enable_auto_tuning'] else '关闭'}")
            print(f"模型保存: {'启用' if user_choices.get('save_models', True) else '关闭'}")
            
            input("\n按Enter键开始分析...")
            
            # 执行训练阶段
            train_results = run_training_phase(
                analyzer=analyzer,
                train_files=train_files,
                use_sampling=user_choices['use_sampling'],
                num_groups=user_choices['num_groups'],
                min_group_size=user_choices['min_group_size']
            )
            
            final_result = None
            prediction_files = None
            
            # 如果是完整流程，执行预测
            if run_mode == '1':
                try:
                    # 尝试使用保存的模型进行预测
                    print("\n尝试使用保存的模型进行预测...")
                    available_models = analyzer.predictor.get_available_models()
                    
                    if available_models:
                        # 使用改进的预测方法
                        final_result = analyzer.predict_with_saved_models(
                            segments=list(range(len(test_files))),
                            model_names=user_choices['selected_models'],
                            ensemble_method='average'
                        )
                        if final_result is not None:
                            print("✅ 使用保存的模型预测成功")
                        else:
                            print("⚠️ 保存的模型预测失败，尝试传统方法...")
                            raise Exception("保存的模型预测失败")
                    else:
                        raise Exception("没有保存的模型")
                        
                except Exception as e:
                    print(f"⚠️ 使用保存的模型预测失败: {str(e)}")
                    print("使用传统方法进行预测...")
                    
                    # 使用传统方法预测
                    prediction_files = run_prediction_phase_legacy(
                        analyzer=analyzer,
                        test_files=test_files
                    )
                    
                    # 执行合并阶段
                    if prediction_files:
                        final_result = run_merge_phase(
                            analyzer=analyzer,
                            prediction_files=prediction_files
                        )
            
            # 打印总结
            print_summary(
                train_results=train_results,
                prediction_files=prediction_files,
                final_result=final_result,
                selected_models=user_choices['selected_models'],
                enable_auto_tuning=user_choices['enable_auto_tuning'],
                run_mode=run_mode
            )
        
    except KeyboardInterrupt:
        print("\n\n分析被用户中断")
        sys.exit(1)
    except Exception as e:
        error_msg = f"分析过程中发生错误: {str(e)}"
        print(f"\n\n{error_msg}")
        
        # 如果logger已经初始化，则记录详细错误
        if logger is not None:
            logger.error(f"分析失败: {str(e)}", exc_info=True)
        else:
            # 如果logger未初始化，至少打印详细错误信息
            import traceback
            print("\n详细错误信息:")
            traceback.print_exc()
        
        sys.exit(1)


if __name__ == "__main__":
    main()