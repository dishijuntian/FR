import json
import os
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
import joblib

from .Models import XGBoostRanker, LightGBMRanker, RankNet, GraphRanker, CNNRanker, TransformerRanker
from .FeatureSelector import FeatureSelector

class FlightRankingModelsManager:
    def __init__(self, use_gpu: bool = True, logger=None, feature_selection_config: Dict = None):
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.logger = logger or logging.getLogger(__name__)
        self.models: Dict[str, object] = {}
        self.feature_names: List[str] = []
        self.feature_selector: Optional[FeatureSelector] = None
        self.feature_selection_config = feature_selection_config or {'enabled': True, 'max_features': 50}
        self.training_report_path = 'data/aeroclub-recsys-2025/models/training_report.json'
        if self.use_gpu:
            self.logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.logger.info("CPU模式")
    
    def get_features(self, df: pd.DataFrame, target_col: str = 'selected') -> List[str]:
        """获取特征列"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['Id', target_col, 'ranker_id', 'profileId', 'companyID', 'segment_level', 'group_category']
        return [col for col in numeric_cols if col not in exclude_cols]
    
    def _load_training_features(self, segment_key: str) -> List[str]:
        """从训练报告中加载特定segment的特征列表"""
        if not self.training_report_path:
            return []
            
        try:
            with open(self.training_report_path, 'r') as f:
                report = json.load(f)
            
            segment_results = report.get('segment_results', {})
            if segment_key in segment_results:
                training_config = segment_results[segment_key].get('training_config', {})
                features = training_config.get('feature_names', [])
                if self.logger:
                    self.logger.info(f"从训练报告加载 {segment_key} 的 {len(features)} 个特征")
                return features
        except Exception as e:
            if self.logger:
                self.logger.warning(f"无法加载训练报告特征: {e}")
        
        return []

    def prepare_data(self, df: pd.DataFrame, target_col: str = 'selected', 
                    is_training: bool = False, segment_key: str = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], pd.DataFrame]:
        """准备训练/预测数据 - 修复特征不匹配问题"""
        original_shape = df.shape
        
        # 清理训练数据中的无效组
        if is_training and target_col in df.columns:
            selected_per_group = df.groupby('ranker_id')[target_col].sum()
            invalid_groups = selected_per_group[selected_per_group != 1].index
            if len(invalid_groups) > 0:
                df = df[~df['ranker_id'].isin(invalid_groups)]
                self.logger.info(f"移除 {len(invalid_groups)} 个无效组")
        
        # 预测模式：直接使用训练时的特征列表
        if not is_training and segment_key:
            training_features = self._load_training_features(segment_key)
            if training_features:
                # 检查特征可用性并填充缺失特征
                missing_features = [f for f in training_features if f not in df.columns]
                if missing_features:
                    if self.logger:
                        self.logger.warning(f"预测数据缺少 {len(missing_features)} 个训练特征，用默认值填充")
                    # 用-1填充缺失特征
                    for feature in missing_features:
                        df[feature] = -1
                
                # 使用训练时的确切特征
                df_features = df[training_features].copy().fillna(-1)
                self.feature_names = training_features
                
                X = df_features.values.astype(np.float32)
                y = df[target_col].values if target_col in df.columns else np.zeros(len(df))
                
                ranker_ids = df['ranker_id'].values
                unique_ranker_ids = pd.Series(ranker_ids).unique()
                ranker_id_to_numeric = {rid: i for i, rid in enumerate(unique_ranker_ids)}
                groups = np.array([ranker_id_to_numeric[rid] for rid in ranker_ids])
                
                self.logger.info(f"预测数据准备: {original_shape} → {X.shape}, 使用训练特征: {len(training_features)}")
                return X, y, groups, training_features, df
        
        # 训练模式或特征选择模式
        feature_cols = self.get_features(df, target_col)
        df_features = df[feature_cols].copy().fillna(-1)
        
        # 特征选择
        if is_training and self.feature_selection_config.get('enabled', False):
            if target_col in df.columns:
                self.feature_selector = FeatureSelector(config=self.feature_selection_config, logger=self.logger)
                df_features_selected = self.feature_selector.fit_transform(df_features, df[target_col])
                self.feature_names = self.feature_selector.get_selected_features()
                df_features = df_features_selected
                feature_cols = self.feature_names
                if self.logger:
                    self.logger.info(f"训练时特征选择: {len(feature_cols)} 个特征被选中")
            else:
                self.feature_names = feature_cols
        else:
            if not is_training and self.feature_selector:
                df_features = self.feature_selector.transform(df_features)
                feature_cols = self.feature_selector.get_selected_features()
            else:
                self.feature_names = feature_cols
        
        X = df_features.values.astype(np.float32)
        y = df[target_col].values if target_col in df.columns else np.zeros(len(df))
        
        ranker_ids = df['ranker_id'].values
        unique_ranker_ids = pd.Series(ranker_ids).unique()
        ranker_id_to_numeric = {rid: i for i, rid in enumerate(unique_ranker_ids)}
        groups = np.array([ranker_id_to_numeric[rid] for rid in ranker_ids])
        
        self.logger.info(f"数据准备完成: {original_shape} → {X.shape}")
        return X, y, groups, feature_cols, df
    
    def create_model(self, model_name: str, input_dim: int, model_config: Dict = None) -> object:
        """创建模型实例"""
        if model_config is None:
            model_config = {}
        
        model_config.update({'use_gpu': self.use_gpu, 'logger': self.logger})
        
        try:
            if model_name == 'XGBoostRanker':
                model = XGBoostRanker(**model_config)
            elif model_name == 'LightGBMRanker':
                model = LightGBMRanker(**model_config)
            elif model_name == 'RankNet':
                model = RankNet(input_dim=input_dim, **model_config)
            elif model_name == 'GraphRanker':
                model = GraphRanker(input_dim=input_dim, **model_config)
            elif model_name == 'CNNRanker':
                model = CNNRanker(input_dim=input_dim, **model_config)
            elif model_name == 'TransformerRanker':
                model = TransformerRanker(input_dim=input_dim, **model_config)
            else:
                raise ValueError(f"不支持的模型: {model_name}")
            
            self.models[model_name] = model
            self.logger.info(f"✓ 创建模型: {model_name}")
            return model
            
        except Exception as e:
            self.logger.error(f"✗ 创建模型失败 {model_name}: {e}")
            return None
    
    def train_model(self, model_name: str, X: np.ndarray, y: np.ndarray, 
                   groups: np.ndarray, **kwargs) -> bool:
        """训练模型"""
        if model_name not in self.models:
            self.logger.error(f"模型 {model_name} 未创建")
            return False
        
        try:
            model = self.models[model_name]
            
            if model_name in ['RankNet', 'GraphRanker', 'CNNRanker', 'TransformerRanker']:
                epochs = kwargs.get('epochs', 50)
                if model_name == 'RankNet':
                    model.fit(X, y, groups, epochs=epochs)
                else:
                    model.fit(X, y, groups)
            else:
                # XGBoost和LightGBM支持早停
                early_stopping_rounds = kwargs.get('early_stopping_rounds')
                eval_set = kwargs.get('eval_set')
                eval_group = kwargs.get('eval_group')
                
                model.fit(
                    X, y, groups,
                    early_stopping_rounds=early_stopping_rounds,
                    eval_set=eval_set,
                    eval_group=eval_group
                )
            
            return True
            
        except Exception as e:
            self.logger.error(f"✗ 训练 {model_name} 失败: {e}")
            return False
    
    def predict_model(self, model_name: str, X: np.ndarray, groups: np.ndarray = None) -> np.ndarray:
        """模型预测"""
        if model_name not in self.models:
            raise ValueError(f"模型 {model_name} 未找到")
        
        model = self.models[model_name]
        if not hasattr(model, 'is_fitted') or not model.is_fitted:
            raise ValueError(f"模型 {model_name} 未训练")
        
        if model_name in ['GraphRanker', 'TransformerRanker']:
            return model.predict(X, groups)
        else:
            return model.predict(X)
    
    def save_model(self, model_name: str, save_dir: str):
        """保存模型"""
        if model_name not in self.models:
            raise ValueError(f"模型 {model_name} 未找到")
        
        model = self.models[model_name]
        if not hasattr(model, 'is_fitted') or not model.is_fitted:
            raise ValueError(f"模型 {model_name} 未训练")
        
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存模型
        model_path = os.path.join(save_dir, f"{model_name}.pkl")
        model.save_model(model_path)
        
        # 保存特征名称
        if self.feature_names:
            feature_path = os.path.join(save_dir, "features.pkl")
            joblib.dump(self.feature_names, feature_path)
        
        # 保存特征选择器
        if self.feature_selector:
            selector_path = os.path.join(save_dir, "feature_selector.pkl")
            self.feature_selector.save_selector(selector_path)
        
        self.logger.info(f"✓ 保存模型: {model_name}")
    
    def load_model(self, model_name: str, save_dir: str) -> bool:
        """加载模型"""
        model_path = os.path.join(save_dir, f"{model_name}.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件未找到: {model_path}")
        
        try:
            # 加载模型
            if model_name == 'XGBoostRanker':
                self.models[model_name] = XGBoostRanker.load_model(model_path)
            elif model_name == 'LightGBMRanker':
                self.models[model_name] = LightGBMRanker.load_model(model_path)
            elif model_name == 'RankNet':
                self.models[model_name] = RankNet.load_model(model_path)
            elif model_name == 'GraphRanker':
                self.models[model_name] = GraphRanker.load_model(model_path)
            elif model_name == 'CNNRanker':
                self.models[model_name] = CNNRanker.load_model(model_path)
            elif model_name == 'TransformerRanker':
                self.models[model_name] = TransformerRanker.load_model(model_path)
            else:
                raise ValueError(f"未知模型: {model_name}")
            
            # 加载特征名称
            feature_path = os.path.join(save_dir, "features.pkl")
            if os.path.exists(feature_path):
                self.feature_names = joblib.load(feature_path)
            
            # 加载特征选择器
            selector_path = os.path.join(save_dir, "feature_selector.pkl")
            if os.path.exists(selector_path):
                self.feature_selector = FeatureSelector.load_selector(selector_path, self.logger)
            
            self.logger.info(f"✓ 加载模型: {model_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"✗ 加载 {model_name} 失败: {e}")
            return False