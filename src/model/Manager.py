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
    def __init__(self, use_gpu: bool = True, logger=None):
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.logger = logger or logging.getLogger(__name__)
        self.models: Dict[str, object] = {}
        self.feature_names: List[str] = []
        self.feature_selector: Optional[FeatureSelector] = None
        
        if self.use_gpu:
            self.logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.logger.info("CPU mode")
    
    def create_model(self, model_name: str, input_dim: int, model_config: Dict = None) -> object:
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
                raise ValueError(f"Unsupported model: {model_name}")
            
            self.models[model_name] = model
            self.logger.info(f"✓ Created {model_name}")
            return model
            
        except Exception as e:
            self.logger.error(f"✗ Failed to create {model_name}: {e}")
            return None
    
    def get_features(self, df: pd.DataFrame, target_col: str = 'selected') -> List[str]:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['Id', target_col, 'ranker_id', 'profileId', 'companyID']
        return [col for col in numeric_cols if col not in exclude_cols]
    
    def prepare_data(self, df: pd.DataFrame, target_col: str = 'selected', is_training: bool = False) -> Tuple:
        original_shape = df.shape
        
        # Clean invalid groups in training
        if is_training and target_col in df.columns:
            selected_per_group = df.groupby('ranker_id')[target_col].sum()
            invalid_groups = selected_per_group[selected_per_group != 1].index
            if len(invalid_groups) > 0:
                df = df[~df['ranker_id'].isin(invalid_groups)]
                self.logger.info(f"Removed {len(invalid_groups)} invalid groups")
        
        # Get features
        feature_cols = self.get_features(df, target_col)
        df_features = df[feature_cols].copy().fillna(-1)
        
        # Feature selection
        if is_training:
            if target_col in df.columns:
                self.feature_selector = FeatureSelector(logger=self.logger)
                selected_features = self.feature_selector.select_features(df, df[target_col])
                self.feature_names = selected_features
                df_features = df_features[selected_features]
                feature_cols = selected_features
                self.logger.info(f"Selected {len(feature_cols)} features")
            else:
                self.feature_names = feature_cols
        else:
            if self.feature_names:
                available_features = [f for f in self.feature_names if f in df_features.columns]
                df_features = df_features[available_features]
                feature_cols = available_features
                if len(available_features) < len(self.feature_names):
                    missing = len(self.feature_names) - len(available_features)
                    self.logger.warning(f"Missing {missing} features, using {len(available_features)}")
            else:
                self.feature_names = feature_cols
        
        X = df_features.values.astype(np.float32)
        y = df[target_col].values if target_col in df.columns else np.zeros(len(df))
        groups = df['ranker_id'].values
        
        self.logger.info(f"Data: {original_shape} → {X.shape}")
        return X, y, groups, feature_cols, df
    
    def train_model(self, model_name: str, X: np.ndarray, y: np.ndarray, groups: np.ndarray, **kwargs) -> bool:
        if model_name not in self.models:
            return False
        
        try:
            model = self.models[model_name]
            if model_name in ['RankNet']:
                model.fit(X, y, groups, epochs=kwargs.get('epochs', 80))
            else:
                model.fit(X, y, groups)
            return True
        except Exception as e:
            self.logger.error(f"✗ Training {model_name} failed: {e}")
            return False
    
    def predict_model(self, model_name: str, X: np.ndarray, groups: np.ndarray = None) -> np.ndarray:
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        if not hasattr(model, 'is_fitted') or not model.is_fitted:
            raise ValueError(f"Model {model_name} not trained")
        
        if model_name in ['GraphRanker', 'TransformerRanker']:
            return model.predict(X, groups)
        else:
            return model.predict(X)
    
    def save_model(self, model_name: str, save_dir: str):
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        if not hasattr(model, 'is_fitted') or not model.is_fitted:
            raise ValueError(f"Model {model_name} not trained")
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(save_dir, f"{model_name}.pkl")
        model.save_model(model_path)
        
        # Save features
        if self.feature_names:
            feature_path = os.path.join(save_dir, "features.pkl")
            joblib.dump(self.feature_names, feature_path)
        
        # Save feature selector
        if self.feature_selector:
            selector_path = os.path.join(save_dir, "feature_selector.pkl")
            joblib.dump(self.feature_selector, selector_path)
        
        self.logger.info(f"✓ Saved {model_name}")
    
    def load_model(self, model_name: str, save_dir: str) -> bool:
        model_path = os.path.join(save_dir, f"{model_name}.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            # Load model
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
                raise ValueError(f"Unknown model: {model_name}")
            
            # Load features
            feature_path = os.path.join(save_dir, "features.pkl")
            if os.path.exists(feature_path):
                self.feature_names = joblib.load(feature_path)
            
            # Load feature selector
            selector_path = os.path.join(save_dir, "feature_selector.pkl")
            if os.path.exists(selector_path):
                self.feature_selector = joblib.load(selector_path)
            
            self.logger.info(f"✓ Loaded {model_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"✗ Loading {model_name} failed: {e}")
            return False