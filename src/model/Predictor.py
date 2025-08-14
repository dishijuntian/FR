import os
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import json
import numpy as np
import pandas as pd
import torch
from .Manager import FlightRankingModelsManager

class FlightRankingPredictor:
    def __init__(self, config: Dict, logger=None):
        self.config = config
        self.logger = logger
        
        self.data_path = Path(config['paths']['model_input_dir'])
        self.model_save_path = Path(config['paths']['model_save_dir'])
        self.output_path = Path(config['paths']['output_dir'])
        
        prediction_config = config['prediction']
        # 从training配置中获取segments和group_categories，或使用默认值
        training_config = config.get('training', {})
        self.segments = training_config.get('segments', [0, 1, 2])
        self.group_categories = training_config.get('group_categories', ['small', 'medium', 'big'])
        
        self.use_gpu = prediction_config['use_gpu'] and torch.cuda.is_available()
        
        # 预测优化配置
        self.optimization_config = prediction_config.get('prediction_optimization', {})
        self.batch_size = self.optimization_config.get('batch_size', 10000)
        self.memory_efficient = self.optimization_config.get('memory_efficient', True)
        self.save_individual = self.optimization_config.get('save_individual_results', True)
        
        # 输出配置
        self.output_config = prediction_config.get('output', {})
        self.final_submission_file = self.output_config.get('final_submission_file', 'final_submission.csv')
        self.generate_report = self.output_config.get('generate_report', True)
        
        # 错误处理配置
        self.error_config = prediction_config.get('error_handling', {})
        self.continue_on_failure = self.error_config.get('continue_on_segment_failure', True)
        self.min_successful = self.error_config.get('min_successful_segments', 1)
        
        # 排名生成配置
        self.ranking_config = prediction_config.get('ranking', {})
        self.use_gpu_ranking = self.ranking_config.get('use_gpu_ranking', True)
        self.fallback_to_cpu = self.ranking_config.get('fallback_to_cpu', True)
        self.validate_rankings = self.ranking_config.get('validate_rankings', True)
        
        # 数据验证配置
        self.validation_config = prediction_config.get('validation', {})
        
        self.best_models_config = self._load_best_models_config()
        self.training_report_path = self.model_save_path / "training_report.json"
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        if self.use_gpu:
            self.logger.info(f"GPU预测: {torch.cuda.get_device_name(0)}")
        else:
            self.logger.info("CPU预测")
        
        self.logger.info(f"加载 {len(self.best_models_config)} 个分段模型配置")
    
    def _load_best_models_config(self) -> Dict[str, str]:
        """加载最佳模型配置"""
        config_path = self.model_save_path / "best_models_config.json"
        
        if not config_path.exists():
            self.logger.warning("配置文件未找到，尝试从训练报告推断")
            return self._infer_best_models_from_report()
        
        with open(config_path, 'r') as f:
            config = json.load(f)
            return config
    
    def _infer_best_models_from_report(self) -> Dict[str, str]:
        """从训练报告推断最佳模型"""
        report_path = self.model_save_path / "training_report.json"
        best_models = {}
        
        if report_path.exists():
            try:
                with open(report_path, 'r') as f:
                    report = json.load(f)
                
                segment_results = report.get('segment_results', {})
                for segment_key, results in segment_results.items():
                    best_model = results.get('best_model_name')
                    if best_model:
                        best_models[segment_key] = best_model
                
                self.logger.info(f"从训练报告推断出 {len(best_models)} 个模型配置")
            except Exception as e:
                self.logger.warning(f"读取训练报告失败: {e}")
        
        # 如果没有找到配置，使用默认值
        if not best_models:
            for segment_level in self.segments:
                for group_category in self.group_categories:
                    segment_key = f"segment_{segment_level}_{group_category}"
                    best_models[segment_key] = 'XGBoostRanker'  # 默认模型
            self.logger.warning("使用默认模型配置")
        
        return best_models
    
    def get_test_files(self) -> List[Tuple[str, int, str]]:
        """获取所有测试文件路径"""
        files = []
        for segment_level in self.segments:
            for group_category in self.group_categories:
                filename = f"test_segment_{segment_level}_{group_category}.parquet"
                filepath = self.data_path / "test" / filename
                if filepath.exists():
                    files.append((str(filepath), segment_level, group_category))
                else:
                    if self.logger:
                        self.logger.warning(f"测试文件不存在: {filepath}")
        return files
    
    def load_segment_model(self, segment_level: int, group_category: str) -> FlightRankingModelsManager:
        """加载特定段的模型"""
        segment_key = f"segment_{segment_level}_{group_category}"
        segment_model_dir = self.model_save_path / segment_key
        
        if not segment_model_dir.exists():
            raise FileNotFoundError(f"段模型目录不存在: {segment_model_dir}")
        
        # 获取该段的最佳模型
        best_model_name = self.best_models_config.get(segment_key)
        if not best_model_name:
            raise ValueError(f"未找到段 {segment_key} 的最佳模型配置")
        
        model_file = segment_model_dir / f"{best_model_name}.pkl"
        if not model_file.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_file}")
        
        models_manager = FlightRankingModelsManager(use_gpu=self.use_gpu, logger=self.logger)
        
        # 设置训练报告路径，用于获取训练时的特征
        models_manager.training_report_path = self.training_report_path
        
        success = models_manager.load_model(best_model_name, str(segment_model_dir))
        
        if not success:
            raise ValueError(f"模型加载失败: {best_model_name} from {segment_model_dir}")
        
        self.logger.debug(f"段模型加载成功: {segment_key} -> {best_model_name}")
        return models_manager, best_model_name

    def predict_single_segment(self, filepath: str, segment_level: int, group_category: str) -> pd.DataFrame:
        """预测单个段"""
        segment_key = f"segment_{segment_level}_{group_category}"
        self.logger.info(f"预测 {segment_key}")
        
        try:
            # 加载段模型
            models_manager, best_model_name = self.load_segment_model(segment_level, group_category)
            
            # 加载测试数据
            df = pd.read_parquet(filepath)
            df['segment_level'] = segment_level
            df['group_category'] = group_category
            
            # 准备数据 - 传递segment_key用于特征匹配
            X, _, groups, _, _ = models_manager.prepare_data(
                df, target_col='selected', is_training=False, segment_key=segment_key
            )
            
            # 预测
            if best_model_name in ['GraphRanker', 'TransformerRanker']:
                predictions = models_manager.predict_model(best_model_name, X, groups)
            else:
                predictions = models_manager.predict_model(best_model_name, X)
            
            # 生成排名
            rankings = self.generate_rankings(predictions, groups)
            
            # 验证排名
            if self.validate_rankings:
                self._validate_segment_rankings(rankings, groups)
            
            # 创建结果
            results = df[['Id', 'ranker_id']].copy()
            results['selected'] = rankings
            
            self.logger.info(f"✓ {segment_key} - "
                           f"{len(results)}样本, {results['ranker_id'].nunique()}组, 模型: {best_model_name}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"✗ {segment_key} 预测失败: {e}")
            
            if self.continue_on_failure:
                # 随机回退
                df = pd.read_parquet(filepath)
                results = df[['Id', 'ranker_id']].copy()
                
                # 生成随机排名
                rankings = np.zeros(len(df), dtype=int)
                for group_id in df['ranker_id'].unique():
                    group_mask = df['ranker_id'] == group_id
                    group_size = group_mask.sum()
                    rankings[group_mask] = np.random.permutation(range(1, group_size + 1))
                results['selected'] = rankings
                
                self.logger.warning(f"使用随机排名: {segment_key}")
                return results
            else:
                raise e
    def generate_rankings(self, scores: np.ndarray, groups: np.ndarray) -> np.ndarray:
        """生成排名"""
        if torch.cuda.is_available() and self.use_gpu and self.use_gpu_ranking:
            try:
                scores_tensor = torch.FloatTensor(scores).cuda()
                groups_tensor = torch.LongTensor(groups).cuda()
                rankings = torch.zeros(len(scores), dtype=torch.long).cuda()
                
                unique_groups = torch.unique(groups_tensor)
                for group_id in unique_groups:
                    group_mask = groups_tensor == group_id
                    group_scores = scores_tensor[group_mask]
                    group_indices = torch.where(group_mask)[0]
                    
                    _, sort_indices = torch.sort(group_scores, descending=True)
                    group_rankings = torch.arange(1, len(group_scores) + 1, device=group_scores.device)
                    rankings[group_indices[sort_indices]] = group_rankings
                
                result = rankings.cpu().numpy()
                del scores_tensor, groups_tensor, rankings
                torch.cuda.empty_cache()
                return result
                
            except Exception as e:
                if self.fallback_to_cpu:
                    self.logger.warning(f"GPU排名失败，使用CPU: {e}")
                    torch.cuda.empty_cache()
                else:
                    raise e
        
        # CPU回退
        unique_groups = np.unique(groups)
        rankings = np.zeros(len(scores), dtype=int)
        
        for group_id in unique_groups:
            group_mask = groups == group_id
            group_scores = scores[group_mask]
            group_indices = np.where(group_mask)[0]
            
            sort_indices = np.argsort(-group_scores)
            group_rankings = np.arange(1, len(group_scores) + 1)
            rankings[group_indices[sort_indices]] = group_rankings
        
        return rankings
    
    def _validate_segment_rankings(self, rankings: np.ndarray, groups: np.ndarray):
        """验证段排名的正确性"""
        for group_id in np.unique(groups):
            group_mask = groups == group_id
            group_rankings = rankings[group_mask]
            expected = list(range(1, len(group_rankings) + 1))
            actual = sorted(group_rankings.tolist())
            
            if actual != expected:
                raise ValueError(f"组 {group_id} 排名不正确: expected {expected}, got {actual}")
    
    def predict_all_files(self) -> pd.DataFrame:
        """预测所有测试文件"""
        self.logger.info("开始分段预测所有测试文件")
        start_time = time.time()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 获取测试文件
        test_files = self.get_test_files()
        if not test_files:
            raise FileNotFoundError("未找到测试文件")
        
        self.logger.info(f"找到 {len(test_files)} 个测试文件")
        
        all_results = []
        successful_files = 0
        failed_files = 0
        segment_predictions = {}
        
        for filepath, segment_level, group_category in test_files:
            try:
                file_start_time = time.time()
                
                # 预测单个段
                results = self.predict_single_segment(filepath, segment_level, group_category)
                
                all_results.append(results)
                successful_files += 1
                
                # 保存单独的结果文件（如果启用）
                if self.save_individual:
                    segment_key = f"segment_{segment_level}_{group_category}"
                    individual_file = self.output_path / f"prediction_{segment_key}.csv"
                    results.to_csv(individual_file, index=False)
                    self.logger.debug(f"保存单独结果: {individual_file}")
                
                # 记录预测统计
                segment_key = f"segment_{segment_level}_{group_category}"
                segment_predictions[segment_key] = {
                    'samples': len(results),
                    'groups': results['ranker_id'].nunique(),
                    'model_used': self.best_models_config.get(segment_key, 'unknown'),
                    'time': time.time() - file_start_time
                }
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                self.logger.error(f"✗ 文件预测失败 {filepath}: {e}")
                failed_files += 1
                
                if not self.continue_on_failure:
                    raise e
        
        # 检查最小成功数量
        if successful_files < self.min_successful:
            raise ValueError(f"成功预测的文件数量 ({successful_files}) 少于最小要求 ({self.min_successful})")
        
        if not all_results:
            raise ValueError("所有文件预测失败")
        
        # 合并结果
        final_submission = pd.concat(all_results, ignore_index=True)
        final_submission = final_submission.sort_values('Id').reset_index(drop=True)
        
        total_time = time.time() - start_time
        
        # 数据验证
        if self.validation_config.get('check_required_columns', True):
            self._validate_submission(final_submission)
        
        # 保存结果
        output_file = self.output_path / self.final_submission_file
        final_submission.to_csv(output_file, index=False)
        
        # 生成报告
        if self.generate_report:
            self._generate_prediction_report(
                final_submission, total_time, successful_files, failed_files, 
                len(test_files), segment_predictions
            )
        
        self.logger.info(f"✓ 分段预测完成 ({total_time:.1f}s)")
        self.logger.info(f"✓ 结果文件: {output_file}")
        self.logger.info(f"✓ 总记录数: {len(final_submission):,}")
        self.logger.info(f"✓ 成功率: {successful_files}/{len(test_files)}")
        
        return final_submission
    
    def _validate_submission(self, df: pd.DataFrame):
        """验证提交数据格式"""
        # 检查必需列
        if self.validation_config.get('check_required_columns', True):
            required_cols = ['Id', 'ranker_id', 'selected']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"缺少必需列: {missing_cols}")
        
        # 验证组完整性
        if self.validation_config.get('verify_group_completeness', True):
            total_samples = len(df)
            total_groups = df['ranker_id'].nunique()
            
            if total_samples == 0 or total_groups == 0:
                raise ValueError("数据为空")
        
        # 检查排名是否正确
        if self.validation_config.get('validate_ranking_sequence', True):
            invalid_groups = []
            for group_id in df['ranker_id'].unique():
                group_data = df[df['ranker_id'] == group_id]
                rankings = sorted(group_data['selected'].values)
                expected = list(range(1, len(group_data) + 1))
                if rankings != expected:
                    invalid_groups.append(group_id)
                    if len(invalid_groups) <= 5:  # 只记录前5个错误
                        self.logger.error(f"组 {group_id} 排名不正确: expected {expected}, got {rankings}")
            
            if invalid_groups:
                raise ValueError(f"发现 {len(invalid_groups)} 个组的排名不正确")
        
        total_samples = len(df)
        total_groups = df['ranker_id'].nunique()
        
        self.logger.info(f"数据验证通过: {total_samples} 样本, {total_groups} 组")
    
    def _generate_prediction_report(self, results: pd.DataFrame, total_time: float,
                                   successful_files: int, failed_files: int, total_files: int,
                                   segment_predictions: Dict):
        """生成预测报告"""
        total_samples = len(results)
        total_rankers = int(results['ranker_id'].nunique())
        group_sizes = results.groupby('ranker_id').size()
        
        # 按模型类型统计
        model_usage = {}
        for segment_key, stats in segment_predictions.items():
            model_name = stats['model_used']
            if model_name not in model_usage:
                model_usage[model_name] = {
                    'segments': 0,
                    'total_samples': 0,
                    'total_groups': 0,
                    'total_time': 0
                }
            model_usage[model_name]['segments'] += 1
            model_usage[model_name]['total_samples'] += stats['samples']
            model_usage[model_name]['total_groups'] += stats['groups']
            model_usage[model_name]['total_time'] += stats['time']
        
        report = {
            'prediction_summary': {
                'total_files': total_files,
                'successful_files': successful_files,
                'failed_files': failed_files,
                'success_rate': successful_files / total_files if total_files > 0 else 0,
                'total_time': total_time,
                'prediction_mode': 'segment_based'
            },
            'data_statistics': {
                'total_samples': total_samples,
                'total_rankers': total_rankers,
                'avg_options_per_ranker': total_samples / total_rankers if total_rankers > 0 else 0,
                'min_options': int(group_sizes.min()) if len(group_sizes) > 0 else 0,
                'max_options': int(group_sizes.max()) if len(group_sizes) > 0 else 0
            },
            'segment_predictions': segment_predictions,
            'model_usage': model_usage,
            'configuration': {
                'segments': self.segments,
                'group_categories': self.group_categories,
                'use_gpu': self.use_gpu,
                'gpu_ranking': self.use_gpu_ranking,
                'validation_enabled': self.validation_config
            }
        }
        
        report_path = self.output_path / "prediction_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"预测报告已保存: {report_path}")
        
        # 打印模型使用统计
        self.logger.info("模型使用统计:")
        for model_name, stats in model_usage.items():
            self.logger.info(f"  {model_name}: {stats['segments']}段, "
                           f"{stats['total_samples']}样本, "
                           f"{stats['total_time']:.1f}s")
    
    def predict_single_file(self, segment_level: int, group_category: str) -> pd.DataFrame:
        """预测单个文件（便于调试和测试）"""
        filename = f"test_segment_{segment_level}_{group_category}.parquet"
        filepath = self.data_path / "test" / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"测试文件不存在: {filepath}")
        
        return self.predict_single_segment(str(filepath), segment_level, group_category)