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
        
        self.training_config = config['training']
        self.segment_levels = self.training_config.get('segments', [0, 1, 2])
        self.group_categories = self.training_config.get('group_categories', ['small', 'medium', 'big'])
        self.model_names = self.training_config.get('model_names', ['XGBoostRanker', 'LightGBMRanker', 'RankNet', 'GraphRanker', 'CNNRanker', 'TransformerRanker'])
        self.model_configs = self.training_config.get('model_configs', {})
        self.use_gpu = self.training_config['use_gpu'] and torch.cuda.is_available()
        self.random_state = self.training_config['random_state']
        
        # 数据分割比例
        self.validation_split = self.training_config.get('validation_split', 0.15)
        
        # 早停参数
        self.early_stopping_rounds = self.training_config.get('early_stopping_rounds', 20)
        
        self.model_save_path.mkdir(parents=True, exist_ok=True)
        
        if self.use_gpu:
            self.logger.info(f"GPU训练: {torch.cuda.get_device_name(0)}")
        else:
            self.logger.info("CPU训练模式")
    
    def get_data_file(self, data_type: str, segment_level: int, group_category: str) -> str:
        """获取特定数据文件路径"""
        filename = f"{data_type}_segment_{segment_level}_{group_category}.parquet"
        filepath = self.data_path / data_type / filename
        return str(filepath) if filepath.exists() else None
    
    def load_single_dataset(self, data_type: str, segment_level: int, group_category: str) -> pd.DataFrame:
        """加载单个数据集"""
        filepath = self.get_data_file(data_type, segment_level, group_category)
        if not filepath:
            raise FileNotFoundError(f"数据文件不存在: {data_type}_segment_{segment_level}_{group_category}")
        
        df = pd.read_parquet(filepath)
        df['segment_level'] = segment_level
        df['group_category'] = group_category
        
        self.logger.debug(f"加载 {filepath}: {df.shape}")
        return df
    
    def create_validation_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """创建验证集分割"""
        unique_groups = df['ranker_id'].unique()
        
        # 按组分割
        train_groups, val_groups = train_test_split(
            unique_groups, 
            test_size=self.validation_split,
            random_state=self.random_state,
            shuffle=True
        )
        
        train_df = df[df['ranker_id'].isin(train_groups)].copy()
        val_df = df[df['ranker_id'].isin(val_groups)].copy()
        
        self.logger.debug(f"数据分割 - 训练: {len(train_df)} ({len(train_groups)}组), "
                         f"验证: {len(val_df)} ({len(val_groups)}组)")
        
        return train_df, val_df
    
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
                top_3_indices = np.argsort(group_y_pred)[::-1][:3]
                hit = np.any(group_y_true[top_3_indices] > 0)
                if hit:
                    hit_count += 1
                total_groups += 1
        
        return hit_count / total_groups if total_groups > 0 else 0.0
    
    def train_model_with_early_stopping(self, models_manager: FlightRankingModelsManager, 
                                       model_name: str, X_train: np.ndarray, y_train: np.ndarray, 
                                       groups_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, 
                                       groups_val: np.ndarray, n_features: int) -> Tuple[bool, float, int]:
        """使用早停策略训练模型"""
        try:
            # 获取模型配置
            model_config = self.model_configs.get(model_name, {})
            model = models_manager.create_model(model_name, n_features, model_config)
            if model is None:
                return False, 0.0, 0
            
            best_score = -1
            patience_counter = 0
            best_epoch = 0
            
            if model_name in ['XGBoostRanker', 'LightGBMRanker']:
                # 对于XGBoost和LightGBM，使用内置早停
                success = models_manager.train_model(
                    model_name, X_train, y_train, groups_train,
                    early_stopping_rounds=self.early_stopping_rounds,
                    eval_set=[(X_val, y_val)],
                    eval_group=[groups_val]
                )
                
                if success:
                    val_pred = models_manager.predict_model(model_name, X_val)
                    val_score = self.calculate_hitrate_at_3(y_val, val_pred, groups_val)
                    return True, val_score, 0
                else:
                    return False, 0.0, 0
            
            elif model_name in ['RankNet', 'GraphRanker', 'CNNRanker', 'TransformerRanker']:
                # 对于神经网络模型，使用自定义早停
                max_epochs = 80 if model_name == 'RankNet' else 30
                
                for epoch in range(max_epochs):
                    # 训练一个epoch
                    success = models_manager.train_model(model_name, X_train, y_train, groups_train, epochs=1)
                    if not success:
                        break
                    
                    # 验证
                    if model_name in ['GraphRanker', 'TransformerRanker']:
                        val_pred = models_manager.predict_model(model_name, X_val, groups_val)
                    else:
                        val_pred = models_manager.predict_model(model_name, X_val)
                    
                    val_score = self.calculate_hitrate_at_3(y_val, val_pred, groups_val)
                    
                    if val_score > best_score:
                        best_score = val_score
                        patience_counter = 0
                        best_epoch = epoch + 1
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= self.early_stopping_rounds:
                        self.logger.debug(f"{model_name} 早停于epoch {epoch+1}")
                        break
                        
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                return True, best_score, best_epoch
            else:
                # 简单训练其他模型
                success = models_manager.train_model(model_name, X_train, y_train, groups_train)
                if success:
                    val_pred = models_manager.predict_model(model_name, X_val)
                    val_score = self.calculate_hitrate_at_3(y_val, val_pred, groups_val)
                    return True, val_score, 0
                else:
                    return False, 0.0, 0
                    
        except Exception as e:
            self.logger.error(f"{model_name}训练失败: {e}")
            return False, 0.0, 0
    
    def train_single_segment(self, segment_level: int, group_category: str) -> Dict:
        """训练单个段和组别的模型"""
        self.logger.info(f"训练 segment_{segment_level}_{group_category}")
        
        start_time = time.time()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 加载训练数据
        try:
            train_df = self.load_single_dataset('train', segment_level, group_category)
        except FileNotFoundError as e:
            self.logger.warning(f"跳过 segment_{segment_level}_{group_category}: {e}")
            return None
        
        # 创建验证集分割
        train_data, val_data = self.create_validation_split(train_df)
        
        # 初始化模型管理器
        models_manager = FlightRankingModelsManager(
            use_gpu=self.use_gpu, 
            logger=self.logger, 
            feature_selection_config=self.training_config.get('feature_selection', {})
        )
        
        # 准备数据
        X_train, y_train, groups_train, feature_cols, _ = models_manager.prepare_data(train_data, is_training=True)
        X_val, y_val, groups_val, _, _ = models_manager.prepare_data(val_data, is_training=False)
        
        n_samples, n_features = X_train.shape
        self.logger.info(f"segment_{segment_level}_{group_category} - 特征数量: {n_features}, 训练样本: {n_samples}")
        
        # 训练所有模型
        model_results = {}
        best_model_name = None
        best_score = -1
        
        for model_name in self.model_names:
            self.logger.info(f"  训练 {model_name}...")
            model_start_time = time.time()
            
            success, val_score, best_epoch = self.train_model_with_early_stopping(
                models_manager, model_name, X_train, y_train, groups_train,
                X_val, y_val, groups_val, n_features
            )
            
            training_time = time.time() - model_start_time
            
            if success:
                model_results[model_name] = {
                    'val_hitrate_at_3': val_score,
                    'training_time': training_time,
                    'best_epoch': best_epoch,
                    'success': True
                }
                
                if val_score > best_score:
                    best_score = val_score
                    best_model_name = model_name
                
                self.logger.info(f"    ✓ {model_name} - Val: {val_score:.4f} ({training_time:.1f}s)")
            else:
                model_results[model_name] = {
                    'val_hitrate_at_3': 0.0,
                    'training_time': training_time,
                    'success': False
                }
                self.logger.info(f"    ✗ {model_name} - 训练失败")
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        total_time = time.time() - start_time
        
        # 构建结果
        segment_results = {
            'segment_level': segment_level,
            'group_category': group_category,
            'best_model_name': best_model_name,
            'best_val_score': best_score,
            'model_results': model_results,
            'training_config': {
                'n_features': n_features,
                'n_train_samples': len(X_train),
                'n_val_samples': len(X_val),
                'total_time': total_time,
                'feature_names': feature_cols
            }
        }
        
        # 保存最佳模型
        if best_model_name and best_model_name in models_manager.models:
            try:
                segment_model_dir = self.model_save_path / f"segment_{segment_level}_{group_category}"
                segment_model_dir.mkdir(parents=True, exist_ok=True)
                models_manager.save_model(best_model_name, str(segment_model_dir))
                self.logger.info(f"  ✓ 已保存最佳模型: {best_model_name}")
            except Exception as e:
                self.logger.error(f"  ✗ 模型保存失败: {e}")
        
        self.logger.info(f"✓ segment_{segment_level}_{group_category} 完成 ({total_time:.1f}s)")
        self.logger.info(f"  最佳模型: {best_model_name} (Val HitRate@3: {best_score:.4f})")
        
        return segment_results
    
    def train_all_segments(self) -> Dict:
        """训练所有数据段和组别"""
        self.logger.info("开始分段训练所有数据组合")
        self.logger.info(f"数据段: {self.segment_levels}")
        self.logger.info(f"组别: {self.group_categories}")
        self.logger.info(f"候选模型: {self.model_names}")
        
        start_time = time.time()
        
        all_results = {}
        successful_segments = 0
        total_segments = 0
        
        # 为每个段和组别训练模型
        for segment_level in self.segment_levels:
            for group_category in self.group_categories:
                total_segments += 1
                
                segment_results = self.train_single_segment(segment_level, group_category)
                
                if segment_results:
                    segment_key = f"segment_{segment_level}_{group_category}"
                    all_results[segment_key] = segment_results
                    successful_segments += 1
                else:
                    self.logger.warning(f"跳过 segment_{segment_level}_{group_category}")
        
        total_time = time.time() - start_time
        
        # 汇总结果
        summary_results = {
            'training_summary': {
                'total_segments': total_segments,
                'successful_segments': successful_segments,
                'success_rate': successful_segments / total_segments if total_segments > 0 else 0,
                'total_time': total_time
            },
            'segment_results': all_results,
            'best_models_config': {}
        }
        
        # 为每个段生成最佳模型配置
        for segment_key, results in all_results.items():
            if results['best_model_name']:
                summary_results['best_models_config'][segment_key] = results['best_model_name']
        
        # 保存汇总结果
        self._save_results(summary_results)
        
        self.logger.info(f"✓ 分段训练完成 ({total_time:.1f}s)")
        self.logger.info(f"成功率: {successful_segments}/{total_segments}")
        
        return summary_results
    
    def _save_results(self, results: Dict):
        """保存训练结果"""
        # 保存训练报告
        report_path = self.model_save_path / "training_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 保存最佳模型配置（用于预测）
        best_models_config = results.get('best_models_config', {})
        if best_models_config:
            config_path = self.model_save_path / "best_models_config.json"
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(best_models_config, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"训练报告已保存: {report_path}")
        if best_models_config:
            self.logger.info(f"最佳模型配置已保存: {config_path}")