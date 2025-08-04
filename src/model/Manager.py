"""
航班排名模型集合管理器
"""

import os
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
import warnings
import joblib

from .Models import LightGBMRanker, XGBoostRanker, RankNet

warnings.filterwarnings('ignore')


class FlightRankingModelsManager:
    """优化后的航班排名模型管理器"""
    
    def __init__(self, use_gpu: bool = True, logger=None):
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.logger = logger or logging.getLogger(__name__)
        self.models: Dict[str, object] = {}
        self.feature_names: List[str] = []
        self._model_configs: Dict = {}
        
        if self.use_gpu:
            self.logger.info(f"GPU可用: {torch.cuda.get_device_name(0)}")
        else:
            self.logger.info("使用CPU模式")
    
    def create_models(self, input_dim: int, model_configs: Dict = None) -> Dict:
        """创建模型 - 只在需要时创建，避免重复"""
        if model_configs is None:
            model_configs = {}
        
        # 保存配置以便后续使用
        self._model_configs = model_configs
        
        default_configs = {
            'XGBRanker': {
                'n_estimators': 200, 
                'max_depth': 8, 
                'learning_rate': 0.05
            },
            'LGBMRanker': {
                'n_estimators': 200, 
                'max_depth': 8, 
                'learning_rate': 0.05
            },
            'RankNet': {
                'input_dim': input_dim, 
                'hidden_dims': [128, 64, 32],
                'learning_rate': 0.001,
                'dropout_rate': 0.2
            }
        }
        
        # 合并用户配置
        for model_name in default_configs:
            if model_name in model_configs:
                default_configs[model_name].update(model_configs[model_name])
        
        created_models = {}
        
        # 创建模型实例
        for model_name in ['XGBRanker', 'LGBMRanker']:
            try:
                if model_name == 'XGBRanker':
                    created_models[model_name] = XGBoostRanker(
                        use_gpu=self.use_gpu, 
                        logger=self.logger,
                        **default_configs[model_name]
                    )
                elif model_name == 'LGBMRanker':
                    created_models[model_name] = LightGBMRanker(
                        use_gpu=self.use_gpu,
                        logger=self.logger,
                        **default_configs[model_name]
                    )
                elif model_name == 'RankNet':
                    created_models[model_name] = RankNet(
                        use_gpu=self.use_gpu,
                        logger=self.logger,
                        **default_configs[model_name]
                    )
                
                self.logger.info(f"✓ {model_name}模型创建成功")
                
            except Exception as e:
                self.logger.warning(f"✗ {model_name}创建失败: {e}")
                continue
        
        # 只更新成功创建的模型
        self.models.update(created_models)
        return created_models
    
    def create_single_model(self, model_name: str, input_dim: int, 
                           model_configs: Dict = None) -> Optional[object]:
        """创建单个模型实例（用于重新训练等场景）"""
        if model_configs is None:
            model_configs = self._model_configs
        
        default_configs = {
            'XGBRanker': {
                'n_estimators': 200, 
                'max_depth': 8, 
                'learning_rate': 0.05
            },
            'LGBMRanker': {
                'n_estimators': 200, 
                'max_depth': 8, 
                'learning_rate': 0.05
            },
            'RankNet': {
                'input_dim': input_dim, 
                'hidden_dims': [128, 64, 32],
                'learning_rate': 0.001,
                'dropout_rate': 0.2
            }
        }
        
        # 合并配置
        config = default_configs.get(model_name, {})
        if model_name in model_configs:
            config.update(model_configs[model_name])
        
        try:
            if model_name == 'XGBRanker':
                return XGBoostRanker(
                    use_gpu=self.use_gpu, 
                    logger=self.logger,
                    **config
                )
            elif model_name == 'LGBMRanker':
                return LightGBMRanker(
                    use_gpu=self.use_gpu,
                    logger=self.logger,
                    **config
                )
            elif model_name == 'RankNet':
                return RankNet(
                    use_gpu=self.use_gpu,
                    logger=self.logger,
                    **config
                )
            else:
                raise ValueError(f"不支持的模型类型: {model_name}")
                
        except Exception as e:
            self.logger.error(f"创建 {model_name} 失败: {e}")
            return None
    
    def prepare_data(self, df: pd.DataFrame, target_col: str = 'selected') -> Tuple:
        """数据预处理 - 优化内存使用"""
        original_shape = df.shape
        
        # 数据清理 - 移除无效组
        if target_col in df.columns:
            selected_per_group = df.groupby('ranker_id')[target_col].sum()
            invalid_groups = selected_per_group[selected_per_group != 1].index
            if len(invalid_groups) > 0:
                df = df[~df['ranker_id'].isin(invalid_groups)]
                self.logger.info(f"移除 {len(invalid_groups)} 个无效组")
        
        # 特征选择 - 更智能的特征筛选
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['Id', target_col, 'ranker_id', 'profileId', 'companyID']
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # 过滤掉方差为0的特征
        feature_data = df[feature_cols]
        zero_var_cols = feature_data.columns[feature_data.var() == 0].tolist()
        if zero_var_cols:
            feature_cols = [col for col in feature_cols if col not in zero_var_cols]
            self.logger.info(f"移除 {len(zero_var_cols)} 个零方差特征")
        
        # 处理缺失值 - 使用更快的方法
        df_features = df[feature_cols].copy()
        
        # 批量填充缺失值
        numeric_medians = df_features.median()
        df_features = df_features.fillna(numeric_medians)
        
        # 转换为numpy数组
        X = df_features.values.astype(np.float32)
        y = df[target_col].values if target_col in df.columns else np.zeros(len(df))
        groups = df['ranker_id'].values
        
        self.feature_names = feature_cols
        
        self.logger.info(f"数据预处理完成: {original_shape} → {X.shape}, "
                        f"特征数: {len(feature_cols)}")
        
        return X, y, groups, feature_cols, df
    
    def train_models(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                    model_names: List[str] = None, **training_kwargs) -> Dict:
        """训练模型 - 避免重复训练"""
        if model_names is None:
            model_names = list(self.models.keys())
        
        trained_models = {}
        
        for name in model_names:
            if name not in self.models:
                self.logger.warning(f"模型 {name} 不存在，跳过")
                continue
            
            try:
                self.logger.info(f"开始训练 {name}...")
                model = self.models[name]
                
                # 根据模型类型选择训练参数
                if name == 'RankNet':
                    epochs = training_kwargs.get('epochs', 50)
                    model.fit(X, y, groups, epochs=epochs)
                else:
                    model.fit(X, y, groups)
                
                trained_models[name] = model
                self.logger.info(f"✓ {name} 训练完成")
                
            except Exception as e:
                self.logger.error(f"✗ {name} 训练失败: {e}")
                continue
        
        return trained_models
    
    def predict_ensemble(self, X: np.ndarray, model_names: List[str] = None,
                        weights: List[float] = None) -> np.ndarray:
        """集成预测 - 优化性能"""
        if model_names is None:
            model_names = [name for name, model in self.models.items() 
                          if hasattr(model, 'is_fitted') and model.is_fitted]
        
        if not model_names:
            raise ValueError("没有已训练的模型可用于预测")
        
        if weights is None:
            weights = [1.0] * len(model_names)
        
        if len(weights) != len(model_names):
            raise ValueError("权重数量必须与模型数量相同")
        
        # 标准化权重
        weights = np.array(weights) / np.sum(weights)
        
        predictions = []
        valid_weights = []
        
        for i, name in enumerate(model_names):
            if name in self.models and hasattr(self.models[name], 'is_fitted') and self.models[name].is_fitted:
                try:
                    pred = self.models[name].predict(X)
                    predictions.append(pred)
                    valid_weights.append(weights[i])
                    self.logger.debug(f"{name} 预测完成")
                except Exception as e:
                    self.logger.warning(f"✗ {name} 预测失败: {e}")
        
        if not predictions:
            raise ValueError("所有模型预测都失败")
        
        # 重新标准化有效权重
        valid_weights = np.array(valid_weights) / np.sum(valid_weights)
        
        # 集成预测
        ensemble_pred = np.average(predictions, axis=0, weights=valid_weights)
        
        self.logger.info(f"集成预测完成，使用 {len(predictions)} 个模型")
        return ensemble_pred
    
    def get_model_summary(self) -> Dict:
        """获取模型概要信息"""
        summary = {
            'total_models': len(self.models),
            'fitted_models': [],
            'unfitted_models': [],
            'feature_count': len(self.feature_names),
            'gpu_enabled': self.use_gpu
        }
        
        for name, model in self.models.items():
            if hasattr(model, 'is_fitted') and model.is_fitted:
                summary['fitted_models'].append(name)
            else:
                summary['unfitted_models'].append(name)
        
        return summary
    
    def save_models(self, save_dir: str):
        """保存模型 - 只保存已训练的模型"""
        os.makedirs(save_dir, exist_ok=True)
        
        saved_count = 0
        for name, model in self.models.items():
            if hasattr(model, 'is_fitted') and model.is_fitted:
                filepath = os.path.join(save_dir, f"{name}.pkl")
                try:
                    model.save_model(filepath)
                    saved_count += 1
                    self.logger.info(f"✓ 保存模型: {name}")
                except Exception as e:
                    self.logger.error(f"✗ 保存模型失败 {name}: {e}")
        
        # 保存特征名称
        if self.feature_names:
            feature_path = os.path.join(save_dir, "features.pkl")
            joblib.dump(self.feature_names, feature_path)
        
        # 保存模型配置
        config_path = os.path.join(save_dir, "model_configs.pkl")
        joblib.dump(self._model_configs, config_path)
        
        self.logger.info(f"已保存 {saved_count} 个模型到: {save_dir}")
    
    def load_models(self, save_dir: str, model_names: List[str] = None):
        """加载模型 - 优化加载逻辑"""
        if not os.path.exists(save_dir):
            raise FileNotFoundError(f"模型目录不存在: {save_dir}")
        
        if model_names is None:
            model_files = [f for f in os.listdir(save_dir) 
                          if f.endswith('.pkl') and f not in ['features.pkl', 'model_configs.pkl']]
            model_names = [f.replace('.pkl', '') for f in model_files]
        
        loaded_count = 0
        for name in model_names:
            filepath = os.path.join(save_dir, f"{name}.pkl")
            if os.path.exists(filepath):
                try:
                    if name == 'XGBRanker':
                        self.models[name] = XGBoostRanker.load_model(filepath)
                    elif name == 'LGBMRanker':
                        self.models[name] = LightGBMRanker.load_model(filepath)
                    elif name == 'RankNet':
                        self.models[name] = RankNet.load_model(filepath)
                    else:
                        self.logger.warning(f"未知模型类型: {name}")
                        continue
                    
                    loaded_count += 1
                    self.logger.info(f"✓ 加载模型: {name}")
                except Exception as e:
                    self.logger.warning(f"✗ 加载模型失败 {name}: {e}")
        
        # 加载特征名称
        feature_path = os.path.join(save_dir, "features.pkl")
        if os.path.exists(feature_path):
            try:
                self.feature_names = joblib.load(feature_path)
                self.logger.info(f"✓ 加载特征名称: {len(self.feature_names)} 个")
            except Exception as e:
                self.logger.warning(f"加载特征名称失败: {e}")
        
        # 加载模型配置
        config_path = os.path.join(save_dir, "model_configs.pkl")
        if os.path.exists(config_path):
            try:
                self._model_configs = joblib.load(config_path)
                self.logger.info("✓ 加载模型配置")
            except Exception as e:
                self.logger.warning(f"加载模型配置失败: {e}")
        
        self.logger.info(f"成功加载 {loaded_count} 个模型")
        return loaded_count > 0
    
    def clear_models(self):
        """清空模型缓存"""
        self.models.clear()
        self.feature_names.clear()
        self._model_configs.clear()
        self.logger.info("模型缓存已清空")
    
    def validate_models(self) -> Dict[str, bool]:
        """验证模型状态"""
        validation_results = {}
        
        for name, model in self.models.items():
            try:
                # 检查模型是否已训练
                is_fitted = hasattr(model, 'is_fitted') and model.is_fitted
                
                # 检查模型是否可以预测（使用小数据测试）
                if is_fitted:
                    test_X = np.random.random((10, len(self.feature_names) or 5)).astype(np.float32)
                    _ = model.predict(test_X)
                    validation_results[name] = True
                else:
                    validation_results[name] = False
                    
            except Exception as e:
                self.logger.warning(f"模型 {name} 验证失败: {e}")
                validation_results[name] = False
        
        return validation_results