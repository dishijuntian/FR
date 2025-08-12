import os
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import json
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from .Manager import FlightRankingModelsManager

class FlightRankingTrainer:
    def __init__(self, config: Dict, logger=None):
        self.config = config
        self.logger = logger
        
        self.data_path = Path(config['paths']['model_input_dir'])
        self.model_save_path = Path(config['paths']['model_save_dir'])
        
        training_config = config['training']
        self.segments = training_config['segments']
        # 简化模型列表，重点关注效果好的模型
        self.model_names = ['XGBoostRanker', 'LightGBMRanker', 'RankNet']
        self.use_gpu = training_config['use_gpu'] and torch.cuda.is_available()
        self.random_state = training_config['random_state']
        self.validation_split = 0.15  # 减少验证集比例以提高训练效率
        
        # 简化数据分批配置
        self.max_groups_per_batch = 10000  # 减少批次大小
        self.max_samples_per_batch = 300000
        
        self.model_save_path.mkdir(parents=True, exist_ok=True)
        
        if self.use_gpu:
            self.logger.info(f"GPU训练: {torch.cuda.get_device_name(0)}")
        else:
            self.logger.info("CPU训练模式")
    
    def load_segment_data_batches(self, segment_id: int) -> List[pd.DataFrame]:
        """按组分批加载数据"""
        train_file = self.data_path / "train" / f"train_segment_{segment_id}.parquet"
        if not train_file.exists():
            raise FileNotFoundError(f"训练文件未找到: {train_file}")
        
        df = pd.read_parquet(train_file)
        self.logger.info(f"加载segment_{segment_id}: {df.shape}")
        
        # 按组计算数据量
        unique_groups = df['ranker_id'].unique()
        total_groups = len(unique_groups)
        
        # 如果组数较少，直接返回
        if total_groups <= self.max_groups_per_batch:
            return [df]
        
        # 计算需要分成几批（按组数计算）
        num_batches = (total_groups + self.max_groups_per_batch - 1) // self.max_groups_per_batch
        
        # 随机打乱组ID以确保每批数据分布均匀
        np.random.shuffle(unique_groups)
        
        batches = []
        groups_per_batch = (total_groups + num_batches - 1) // num_batches
        
        for i in range(num_batches):
            start_idx = i * groups_per_batch
            end_idx = min((i + 1) * groups_per_batch, total_groups)
            batch_groups = unique_groups[start_idx:end_idx]
            
            # 按组提取完整数据
            batch_df = df[df['ranker_id'].isin(batch_groups)].copy()
            
            # 如果单批数据量仍然过大，随机选择部分组
            if len(batch_df) > self.max_samples_per_batch:
                current_batch_groups = batch_df['ranker_id'].unique()
                # 估算需要保留的组数
                avg_group_size = len(batch_df) / len(current_batch_groups)
                target_groups = int(self.max_samples_per_batch / avg_group_size)
                target_groups = min(target_groups, len(current_batch_groups))
                
                selected_groups = np.random.choice(current_batch_groups, 
                                                 size=target_groups, 
                                                 replace=False)
                batch_df = batch_df[batch_df['ranker_id'].isin(selected_groups)]
            
            if len(batch_df) > 0:
                batches.append(batch_df)
                self.logger.info(f"批次{i+1}: {len(batch_df)}样本, {batch_df['ranker_id'].nunique()}组")
        
        return batches
    
    def create_validation_split_by_groups(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """按组创建验证集分割"""
        unique_groups = df['ranker_id'].unique()
        train_groups, val_groups = train_test_split(
            unique_groups, test_size=self.validation_split, random_state=self.random_state
        )
        
        # 按组分割数据
        train_df = df[df['ranker_id'].isin(train_groups)].copy()
        val_df = df[df['ranker_id'].isin(val_groups)].copy()
        
        self.logger.info(f"数据分割 - 训练组: {len(train_groups)}, 验证组: {len(val_groups)}")
        self.logger.info(f"数据分割 - 训练样本: {len(train_df)}, 验证样本: {len(val_df)}")
        
        return train_df, val_df
    
    def create_validation_split(self, groups: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """创建验证集分割（保持向后兼容）"""
        unique_groups = np.unique(groups)
        train_groups, val_groups = train_test_split(
            unique_groups, test_size=self.validation_split, random_state=self.random_state
        )
        
        train_mask = np.isin(groups, train_groups)
        val_mask = np.isin(groups, val_groups)
        
        return np.where(train_mask)[0], np.where(val_mask)[0]
    
    def calculate_hitrate_at_3(self, y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray) -> float:
        """计算HitRate@3指标"""
        unique_groups = np.unique(groups)
        hit_count = 0
        total_groups = 0
        
        for group_id in unique_groups:
            group_mask = groups == group_id
            group_y_true = y_true[group_mask]
            group_y_pred = y_pred[group_mask]
            
            if len(group_y_true) > 1 and np.sum(group_y_true) > 0:
                # 获取预测分数最高的前3个
                top_3_indices = np.argsort(group_y_pred)[::-1][:3]
                # 检查前3个中是否有正样本
                hit = np.any(group_y_true[top_3_indices] > 0)
                if hit:
                    hit_count += 1
                total_groups += 1
        
        return hit_count / total_groups if total_groups > 0 else 0.0
    
    def evaluate_model(self, model, model_name: str, X_val: np.ndarray, 
                      y_val: np.ndarray, groups_val: np.ndarray) -> float:
        """评估模型性能"""
        try:
            if model_name in ['GraphRanker', 'TransformerRanker']:
                val_pred = model.predict(X_val, groups_val)
            else:
                val_pred = model.predict(X_val)
            
            score = self.calculate_hitrate_at_3(y_val, val_pred, groups_val)
            return score
            
        except Exception as e:
            self.logger.warning(f"{model_name}评估失败: {e}")
            return 0.0
    
    def train_model_on_batches(self, models_manager: FlightRankingModelsManager, 
                              model_name: str, batches: List[Tuple], n_features: int) -> bool:
        """在多个数据批次上训练模型"""
        try:
            model = models_manager.create_model(model_name, n_features)
            if model is None:
                return False
            
            # 对每个批次进行训练
            for batch_idx, (X_batch, y_batch, groups_batch) in enumerate(batches):
                self.logger.debug(f"{model_name} - 训练批次 {batch_idx+1}/{len(batches)}")
                
                # 为RankNet减少epochs
                if model_name == 'RankNet':
                    epochs = max(15, 30 // len(batches))
                    success = models_manager.train_model(model_name, X_batch, y_batch, groups_batch, epochs=epochs)
                else:
                    success = models_manager.train_model(model_name, X_batch, y_batch, groups_batch)
                
                if not success:
                    self.logger.warning(f"{model_name} 批次{batch_idx+1}训练失败")
                    return False
                
                # 清理GPU缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            return True
            
        except Exception as e:
            self.logger.error(f"{model_name}批次训练失败: {e}")
            return False
    
    def train_segment(self, segment_id: int) -> Dict:
        """训练单个segment"""
        self.logger.info(f"开始训练segment_{segment_id}")
        start_time = time.time()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 分批加载数据
        data_batches = self.load_segment_data_batches(segment_id)
        
        # 使用第一批数据初始化，并按组分割训练/验证集
        first_batch = data_batches[0]
        
        # 按组分割训练和验证数据
        train_batch_df, val_batch_df = self.create_validation_split_by_groups(first_batch)
        
        models_manager = FlightRankingModelsManager(use_gpu=self.use_gpu, logger=self.logger)
        
        # 准备训练数据
        X_train, y_train, groups_train, feature_cols, _ = models_manager.prepare_data(train_batch_df, is_training=True)
        
        # 准备验证数据（使用相同的特征选择器）
        X_val, y_val, groups_val, _, _ = models_manager.prepare_data(val_batch_df, is_training=False)
        
        n_samples, n_features = X_train.shape
        n_rankers_train = len(np.unique(groups_train))
        n_rankers_val = len(np.unique(groups_val))
        
        self.logger.info(f"训练数据: {n_samples}样本, {n_rankers_train}组, {n_features}特征")
        self.logger.info(f"验证数据: {len(X_val)}样本, {n_rankers_val}组")
        
        # 准备所有批次的训练数据
        training_batches = []
        total_samples = 0
        total_groups = 0
        
        # 添加第一批的训练部分
        training_batches.append((X_train, y_train, groups_train))
        total_samples += len(X_train)
        total_groups += n_rankers_train
        
        # 处理其他批次（如果有的话）
        for batch_idx, batch_df in enumerate(data_batches[1:], 1):
            # 其他批次完整用于训练（不再分割验证集）
            X_batch, y_batch, groups_batch, _, _ = models_manager.prepare_data(batch_df, is_training=False)
            
            if len(X_batch) > 0:
                training_batches.append((X_batch, y_batch, groups_batch))
                total_samples += len(X_batch)
                total_groups += len(np.unique(groups_batch))
        
        self.logger.info(f"总训练数据: {total_samples}样本, {total_groups}组, {len(training_batches)}批次")
        
        # 训练所有模型
        model_results = {}
        best_model_name = None
        best_score = -1
        
        for model_name in self.model_names:
            self.logger.info(f"训练{model_name}...")
            model_start_time = time.time()
            
            try:
                success = self.train_model_on_batches(models_manager, model_name, training_batches, n_features)
                
                if success:
                    score = self.evaluate_model(models_manager.models[model_name], model_name, X_val, y_val, groups_val)
                    training_time = time.time() - model_start_time
                    
                    model_results[model_name] = {
                        'model': models_manager.models[model_name],
                        'hitrate_at_3': score,
                        'training_time': training_time,
                        'success': True
                    }
                    
                    if score > best_score:
                        best_score = score
                        best_model_name = model_name
                    
                    self.logger.info(f"✓ {model_name} - HitRate@3: {score:.4f} ({training_time:.1f}s)")
                else:
                    model_results[model_name] = {
                        'model': None,
                        'hitrate_at_3': 0.0,
                        'training_time': time.time() - model_start_time,
                        'success': False
                    }
                    self.logger.info(f"✗ {model_name} - 训练失败")
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                self.logger.error(f"✗ {model_name} 异常: {e}")
                model_results[model_name] = {
                    'model': None,
                    'hitrate_at_3': 0.0,
                    'training_time': time.time() - model_start_time,
                    'success': False,
                    'error': str(e)
                }
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # 最终训练最佳模型
        final_best_model = None
        if best_model_name and model_results[best_model_name]['success']:
            self.logger.info(f"最佳模型: {best_model_name} (HitRate@3: {best_score:.4f})")
            self.logger.info(f"在所有批次上最终训练{best_model_name}...")
            
            try:
                success = self.train_model_on_batches(models_manager, best_model_name, training_batches, n_features)
                if success:
                    final_best_model = models_manager.models[best_model_name]
                    self.logger.info(f"✓ {best_model_name}最终训练完成")
            except Exception as e:
                self.logger.error(f"✗ {best_model_name}最终训练失败: {e}")
        
        total_time = time.time() - start_time
        results = {
            'segment_id': segment_id,
            'n_samples': total_samples,
            'n_rankers': total_groups,
            'n_features': n_features,
            'n_batches': len(training_batches),
            'n_val_samples': len(X_val),
            'n_val_rankers': n_rankers_val,
            'model_results': {name: {
                'hitrate_at_3': result['hitrate_at_3'],
                'training_time': result['training_time'],
                'success': result['success']
            } for name, result in model_results.items()},
            'best_model_name': best_model_name,
            'best_hitrate_at_3': best_score,
            'final_model': final_best_model,
            'total_time': total_time,
            'feature_names': feature_cols
        }
        
        self._save_segment_results(segment_id, results, models_manager)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.logger.info(f"✓ segment_{segment_id} 完成 ({total_time:.1f}s)")
        return results
    
    def _save_segment_results(self, segment_id: int, results: Dict, models_manager: FlightRankingModelsManager):
        """保存segment结果"""
        segment_dir = self.model_save_path / f"segment_{segment_id}"
        segment_dir.mkdir(exist_ok=True)
        
        # 保存最佳模型
        if results['final_model'] and results['best_model_name']:
            try:
                original_models = models_manager.models.copy()
                models_manager.models = {results['best_model_name']: results['final_model']}
                models_manager.save_model(results['best_model_name'], str(segment_dir))
                models_manager.models = original_models
                self.logger.info(f"已保存{results['best_model_name']}")
            except Exception as e:
                self.logger.error(f"模型保存失败: {e}")
        
        # 简化的训练报告
        report = {
            'segment_id': results['segment_id'],
            'best_model_name': results['best_model_name'],
            'best_hitrate_at_3': results['best_hitrate_at_3'],
            'total_time': results['total_time'],
            'n_samples': results['n_samples'],
            'n_features': results['n_features'],
            'n_batches': results['n_batches'],
            'model_performance': results['model_results']
        }
        
        report_path = segment_dir / "training_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
    
    def train_all_segments(self) -> Dict:
        """训练所有segments"""
        self.logger.info(f"开始训练所有segments: {self.segments}")
        self.logger.info(f"候选模型: {self.model_names}")
        self.logger.info(f"评估指标: HitRate@3")
        
        all_results = {}
        total_start_time = time.time()
        successful_segments = 0
        failed_segments = 0
        
        for segment_id in self.segments:
            try:
                results = self.train_segment(segment_id)
                all_results[f'segment_{segment_id}'] = results
                
                if results['best_model_name']:
                    successful_segments += 1
                else:
                    failed_segments += 1
                
            except Exception as e:
                self.logger.error(f"✗ segment_{segment_id} 失败: {e}")
                failed_segments += 1
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
        
        total_time = time.time() - total_start_time
        self._generate_final_report(all_results, total_time, successful_segments, failed_segments)
        
        return all_results
    
    def _generate_final_report(self, all_results: Dict, total_time: float,
                              successful_segments: int, failed_segments: int):
        """生成最终报告"""
        total_segments = successful_segments + failed_segments
        
        # 最佳模型分布
        best_model_counts = {}
        segment_best_models = {}
        
        for segment_name, results in all_results.items():
            if results.get('best_model_name'):
                best_model = results['best_model_name']
                best_model_counts[best_model] = best_model_counts.get(best_model, 0) + 1
                segment_best_models[segment_name] = {
                    'best_model': best_model,
                    'hitrate_at_3': results['best_hitrate_at_3'],
                    'training_time': results['total_time']
                }
        
        # 模型平均性能
        model_avg_scores = {}
        for segment_name, results in all_results.items():
            for model_name, model_result in results.get('model_results', {}).items():
                if model_result['success']:
                    if model_name not in model_avg_scores:
                        model_avg_scores[model_name] = []
                    model_avg_scores[model_name].append(model_result['hitrate_at_3'])
        
        for model_name in model_avg_scores:
            scores = model_avg_scores[model_name]
            model_avg_scores[model_name] = {
                'avg_hitrate_at_3': np.mean(scores),
                'std_hitrate_at_3': np.std(scores),
                'segments_count': len(scores)
            }
        
        # 简化的最终报告
        report = {
            'summary': {
                'total_segments': total_segments,
                'successful_segments': successful_segments,
                'success_rate': successful_segments / total_segments if total_segments > 0 else 0,
                'total_time': total_time,
                'evaluation_metric': 'HitRate@3',
                'candidate_models': self.model_names
            },
            'best_models': segment_best_models,
            'model_distribution': best_model_counts,
            'model_performance': model_avg_scores
        }
        
        # 保存报告
        report_path = self.model_save_path / "training_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # 保存预测配置
        prediction_config = {}
        for segment_name, best_info in segment_best_models.items():
            segment_id = segment_name.split('_')[1]
            prediction_config[segment_id] = best_info['best_model']
        
        pred_config_path = self.model_save_path / "best_models_config.json"
        with open(pred_config_path, 'w', encoding='utf-8') as f:
            json.dump(prediction_config, f, indent=2, ensure_ascii=False)
        
        # 打印简化摘要
        self.logger.info(f"\n{'='*50}")
        self.logger.info("训练完成摘要")
        self.logger.info(f"{'='*50}")
        self.logger.info(f"总耗时: {total_time:.1f}s")
        self.logger.info(f"成功率: {successful_segments}/{total_segments}")
        self.logger.info(f"评估指标: HitRate@3")
        
        if best_model_counts:
            self.logger.info(f"\n最佳模型分布:")
            for model_name, count in sorted(best_model_counts.items(), key=lambda x: x[1], reverse=True):
                self.logger.info(f"  {model_name}: {count} segments")
        
        if model_avg_scores:
            self.logger.info(f"\n模型性能排名:")
            sorted_models = sorted(model_avg_scores.items(), key=lambda x: x[1]['avg_hitrate_at_3'], reverse=True)
            for i, (model_name, perf) in enumerate(sorted_models, 1):
                self.logger.info(f"  {i}. {model_name}: {perf['avg_hitrate_at_3']:.4f}±{perf['std_hitrate_at_3']:.4f}")
        
        self.logger.info(f"\n报告已保存: {report_path}")
        self.logger.info(f"预测配置: {pred_config_path}")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.logger.info("GPU缓存已清理")