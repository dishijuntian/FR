import pandas as pd
import numpy as np
from xgboost import XGBRanker
from lightgbm import LGBMRanker
import joblib
import warnings
warnings.filterwarnings('ignore')

class FlightRankingAnalyzer:
    def __init__(self, use_gpu=False, random_state=42):
        self.use_gpu = use_gpu
        self.random_state = random_state
        
        self.ranking_models = {
            'XGBRanker': XGBRanker(
                n_estimators=200,
                random_state=random_state,
                max_depth=8,
                learning_rate=0.05,
                verbosity=0,
                **({'tree_method': 'gpu_hist'} if use_gpu else {'n_jobs': -1})
            ),
            'LGBMRanker': LGBMRanker(
                n_estimators=200,
                random_state=random_state,
                max_depth=8,
                learning_rate=0.05,
                verbose=-1,
                **({'device': 'gpu'} if use_gpu else {'device': 'cpu', 'n_jobs': -1})
            )
        }
        
        self.trained_models = {}
        self.feature_names = []
    
    def prepare_data(self, df, target_col='selected'):
        # 验证每组只有一个选中项
        if target_col in df.columns and df[target_col].sum() > 0:
            selected_per_group = df.groupby('ranker_id')[target_col].sum()
            invalid_groups = selected_per_group[selected_per_group != 1].index
            if len(invalid_groups) > 0:
                df = df[~df['ranker_id'].isin(invalid_groups)]
        
        # 选择数值特征
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['Id', target_col, 'ranker_id', 'profileId', 'companyID']
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # 处理缺失值
        missing_ratios = df[feature_cols].isnull().mean()
        valid_features = missing_ratios[missing_ratios < 0.8].index.tolist()
        feature_cols = [col for col in feature_cols if col in valid_features]
        
        for col in feature_cols:
            if df[col].dtype in ['float32', 'float64']:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(0)
        
        X = df[feature_cols].values.astype(np.float32)
        y = df[target_col].values if target_col in df.columns else np.zeros(len(df))
        groups = df['ranker_id'].values
        
        self.feature_names = feature_cols
        return X, y, groups, feature_cols, df
    
    def calculate_hitrate_at_k(self, y_true, y_pred_scores, groups, k=3):
        unique_groups = np.unique(groups)
        hits = 0
        total_groups = 0
        
        for group in unique_groups:
            group_mask = groups == group
            group_y_true = y_true[group_mask]
            group_scores = y_pred_scores[group_mask]
            
            true_selected_indices = np.where(group_y_true == 1)[0]
            if len(true_selected_indices) > 0:
                sorted_indices = np.argsort(group_scores)[::-1]
                top_k_indices = sorted_indices[:k]
                if any(idx in top_k_indices for idx in true_selected_indices):
                    hits += 1
                total_groups += 1
        
        return hits / total_groups if total_groups > 0 else 0
    
    def train_models(self, X_train, X_test, y_train, y_test, groups_train, groups_test):
        results = []
        unique_groups_train = np.unique(groups_train)
        group_sizes_train = [np.sum(groups_train == g) for g in unique_groups_train]
        
        for name, model in self.ranking_models.items():
            try:
                model.fit(X_train, y_train, group=group_sizes_train)
                y_pred_scores = model.predict(X_test)
                hitrate_3 = self.calculate_hitrate_at_k(y_test, y_pred_scores, groups_test, k=3)
                
                results.append({
                    'Model': name,
                    'HitRate@3': f'{hitrate_3:.4f}'
                })
                
                self.trained_models[name] = {
                    'model': model,
                    'hitrate': hitrate_3
                }
                
            except Exception as e:
                print(f"训练{name}失败: {e}")
                continue
        
        return results
    
    def predict_ranks(self, X, groups, model_name):
        model = self.trained_models[model_name]['model']
        scores = model.predict(X)
        
        ranks = np.zeros_like(scores, dtype=int)
        unique_groups = np.unique(groups)
        
        for group in unique_groups:
            group_mask = groups == group
            group_scores = scores[group_mask]
            sorted_indices = np.argsort(group_scores)[::-1]
            group_ranks = np.empty_like(sorted_indices)
            group_ranks[sorted_indices] = np.arange(1, len(sorted_indices) + 1)
            ranks[group_mask] = group_ranks
        
        return ranks
    
    def save_model(self, filepath, model_name):
        joblib.dump(self.trained_models[model_name], filepath)
    
    def load_model(self, filepath, model_name):
        self.trained_models[model_name] = joblib.load(filepath)