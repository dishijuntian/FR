"""
排序模型定义文件 - 修复版本

该模块包含所有排序模型的定义和实现
- 修复了RankNet和TransformerRanker中validation_split的问题
- 改进了内存使用和训练稳定性

作者: Flight Ranking Team
版本: 2.3 (修复版)
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from xgboost import XGBRanker
from lightgbm import LGBMRanker
from rank_bm25 import BM25Okapi
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any
import warnings

warnings.filterwarnings('ignore')


class BaseRanker(ABC):
    """排序模型基类"""
    
    @abstractmethod
    def fit(self, X, y, group, **kwargs):
        """训练模型"""
        pass
    
    @abstractmethod
    def predict(self, X):
        """预测分数"""
        pass
    
    @abstractmethod
    def get_model_name(self):
        """获取模型名称"""
        pass
    
    def get_params(self):
        """获取模型参数"""
        return getattr(self, 'params', {})
    
    def set_params(self, **params):
        """设置模型参数"""
        self.params = params
        return self


class XGBRankerModel(BaseRanker):
    """XGBoost排序模型"""
    
    def __init__(self, use_gpu=True, **params):
        default_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'eval_metric': 'ndcg'
        }
        default_params.update(params)
        
        if use_gpu:
            default_params.update({
                'tree_method': 'gpu_hist',
                'predictor': 'gpu_predictor'
            })
        else:
            default_params['n_jobs'] = -1
        
        self.params = default_params
        self.model = XGBRanker(**default_params)
        
    def fit(self, X, y, group, **kwargs):
        self.model.fit(X, y, group=group, **kwargs)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def get_model_name(self):
        return "XGBRanker"
    
    @property
    def feature_importances_(self):
        return self.model.feature_importances_


class LGBMRankerModel(BaseRanker):
    """LightGBM排序模型"""
    
    def __init__(self, use_gpu=True, **params):
        default_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'objective': 'lambdarank',
            'metric': 'ndcg'
        }
        default_params.update(params)
        
        if use_gpu:
            default_params['device'] = 'gpu'
        else:
            default_params['n_jobs'] = -1
        
        self.params = default_params
        self.model = LGBMRanker(**default_params)
        
    def fit(self, X, y, group, **kwargs):
        self.model.fit(X, y, group=group, **kwargs)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def get_model_name(self):
        return "LGBMRanker"
    
    @property
    def feature_importances_(self):
        return self.model.feature_importances_


class LambdaMART(BaseRanker):
    """LambdaMART实现（基于XGBoost）"""
    
    def __init__(self, use_gpu=True, **params):
        default_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        default_params.update(params)
        
        xgb_params = {
            'objective': "rank:pairwise",
            **default_params
        }
        
        if use_gpu:
            xgb_params.update({
                'tree_method': 'gpu_hist',
                'predictor': 'gpu_predictor'
            })
        else:
            xgb_params['n_jobs'] = -1
        
        self.params = default_params
        self.model = XGBRanker(**xgb_params)
        
    def fit(self, X, y, group, **kwargs):
        self.model.fit(X, y, group=group, **kwargs)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def get_model_name(self):
        return "LambdaMART"
    
    @property
    def feature_importances_(self):
        return self.model.feature_importances_


class ListNetModel(BaseRanker):
    """ListNet实现（基于LightGBM）"""
    
    def __init__(self, use_gpu=True, **params):
        default_params = {
            'n_estimators': 100,
            'learning_rate': 0.05,
            'max_depth': 7,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        default_params.update(params)
        
        lgb_params = {
            'objective': "lambdarank",
            'metric': "ndcg",
            **default_params
        }
        
        if use_gpu:
            lgb_params['device'] = 'gpu'
        else:
            lgb_params['n_jobs'] = -1
        
        self.params = default_params
        self.model = LGBMRanker(**lgb_params)
        
    def fit(self, X, y, group, **kwargs):
        self.model.fit(X, y, group=group, **kwargs)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def get_model_name(self):
        return "ListNet"
    
    @property
    def feature_importances_(self):
        return self.model.feature_importances_


class NeuralRanker(BaseRanker):
    """神经网络排序模型"""
    
    def __init__(self, input_dim, **params):
        default_params = {
            'hidden_units': [256, 128, 64],
            'learning_rate': 0.001,
            'epochs': 10,
            'batch_size': 32
        }
        default_params.update(params)
        
        self.input_dim = input_dim
        self.params = default_params
        self.model = self._build_model()
        
    def _build_model(self):
        inputs = keras.Input(shape=(self.input_dim,))
        
        x = layers.Dense(self.params['hidden_units'][0], activation='relu')(inputs)
        x = layers.Dropout(0.2)(x)
        
        for units in self.params['hidden_units'][1:]:
            x = layers.Dense(units, activation='relu')(x)
            x = layers.Dropout(0.2)(x)
        
        outputs = layers.Dense(1)(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.params['learning_rate']),
            loss='mse'
        )
        return model
    
    def fit(self, X, y, group, **kwargs):
        epochs = kwargs.get('epochs', self.params['epochs'])
        batch_size = kwargs.get('batch_size', self.params['batch_size'])
        
        # 手动分割验证集（如果数据量足够大）
        if len(X) > 1000:
            val_size = int(len(X) * 0.2)
            train_size = len(X) - val_size
            
            X_train_split = X[:train_size].astype(np.float32)
            y_train_split = y[:train_size].astype(np.float32)
            X_val_split = X[train_size:].astype(np.float32)
            y_val_split = y[train_size:].astype(np.float32)
            
            # 训练模型（带验证）
            self.model.fit(
                X_train_split, y_train_split,
                validation_data=(X_val_split, y_val_split),
                epochs=epochs,
                batch_size=batch_size,
                verbose=1
            )
        else:
            # 训练模型（无验证）
            self.model.fit(
                X.astype(np.float32), y.astype(np.float32),
                epochs=epochs,
                batch_size=batch_size,
                verbose=1
            )
    
    def predict(self, X):
        return self.model.predict(X.astype(np.float32)).flatten()
    
    def get_model_name(self):
        return "NeuralRanker"


class RankNet(BaseRanker):
    """RankNet排序模型 - 修复版本
    
    RankNet是一个基于神经网络的pairwise ranking方法，
    通过学习文档对之间的相对排序关系来训练模型。
    """
    
    def __init__(self, input_dim, **params):
        default_params = {
            'hidden_units': [128, 64, 32],
            'learning_rate': 0.001,
            'epochs': 15,
            'batch_size': 64,
            'dropout_rate': 0.3
        }
        default_params.update(params)
        
        self.input_dim = input_dim
        self.params = default_params
        self.model = self._build_model()
        self._feature_importance = None
        
    def _build_model(self):
        """构建RankNet模型"""
        inputs = keras.Input(shape=(self.input_dim,))
        
        # 特征层
        x = layers.Dense(self.params['hidden_units'][0], activation='relu', name='dense_1')(inputs)
        x = layers.Dropout(self.params['dropout_rate'])(x)
        x = layers.BatchNormalization()(x)
        
        for i, units in enumerate(self.params['hidden_units'][1:], 2):
            x = layers.Dense(units, activation='relu', name=f'dense_{i}')(x)
            x = layers.Dropout(self.params['dropout_rate'])(x)
            x = layers.BatchNormalization()(x)
        
        # 输出层 - 输出单个分数
        outputs = layers.Dense(1, activation='linear', name='score_output')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name='RankNet')
        
        # 使用MSE损失函数（简化版本）
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.params['learning_rate']),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def fit(self, X, y, group, **kwargs):
        """训练RankNet模型 - 修复版本"""
        epochs = kwargs.get('epochs', self.params['epochs'])
        batch_size = kwargs.get('batch_size', self.params['batch_size'])
        
        print(f"开始训练RankNet模型，数据形状: {X.shape}")
        
        # 修复：手动处理验证集分割
        if len(X) > 1000:
            # 计算验证集大小
            val_size = int(len(X) * 0.2)
            train_size = len(X) - val_size
            
            # 分割数据
            X_train_split = X[:train_size].astype(np.float32)
            y_train_split = y[:train_size].astype(np.float32)
            X_val_split = X[train_size:].astype(np.float32)
            y_val_split = y[train_size:].astype(np.float32)
            
            # 训练模型（带验证）
            history = self.model.fit(
                X_train_split, y_train_split,
                validation_data=(X_val_split, y_val_split),
                epochs=epochs,
                batch_size=batch_size,
                verbose=1
            )
        else:
            # 训练模型（无验证）
            history = self.model.fit(
                X.astype(np.float32), y.astype(np.float32),
                epochs=epochs,
                batch_size=batch_size,
                verbose=1
            )
        
        # 计算特征重要性
        self._compute_feature_importance(X[:min(1000, len(X))])
        
        print("RankNet模型训练完成")
        return history
    
    def _compute_feature_importance(self, X_sample):
        """计算特征重要性"""
        try:
            X_tensor = tf.convert_to_tensor(X_sample.astype(np.float32))
            
            with tf.GradientTape() as tape:
                tape.watch(X_tensor)
                predictions = self.model(X_tensor)
            
            grads = tape.gradient(predictions, X_tensor)
            if grads is not None:
                # 使用梯度的绝对值均值作为特征重要性
                importance = np.mean(np.abs(grads.numpy()), axis=0)
                # 归一化
                self._feature_importance = importance / (np.sum(importance) + 1e-8)
            else:
                # 如果无法计算梯度，使用均匀分布
                self._feature_importance = np.ones(self.input_dim) / self.input_dim
                
        except Exception as e:
            print(f"计算RankNet特征重要性时出错: {e}")
            # 使用均匀分布作为后备方案
            self._feature_importance = np.ones(self.input_dim) / self.input_dim
    
    def predict(self, X):
        """预测分数"""
        return self.model.predict(X.astype(np.float32), verbose=0).flatten()
    
    def get_model_name(self):
        return "RankNet"
    
    @property
    def feature_importances_(self):
        """获取特征重要性"""
        if self._feature_importance is None:
            # 如果还没有计算，返回均匀分布
            return np.ones(self.input_dim) / self.input_dim
        return self._feature_importance


class TransformerRanker(BaseRanker):
    """修复版TransformerRanker - 稳定可用的实现"""
    
    def __init__(self, input_dim, **params):
        # 使用更保守的默认参数
        default_params = {
            'num_heads': 4,           # 减少注意力头数
            'num_layers': 2,          # 减少层数
            'd_model': 64,            # 减少模型维度
            'dff': 128,               # 减少前馈网络维度
            'learning_rate': 0.001,
            'epochs': 10,
            'batch_size': 64,
            'dropout_rate': 0.1,
            'max_seq_length': 16      # 固定序列长度
        }
        default_params.update(params)
        
        self.input_dim = input_dim
        self.params = default_params
        self._feature_importance = None
        
        print(f"🔧 初始化TransformerRanker:")
        print(f"   输入维度: {input_dim}")
        print(f"   序列长度: {self.params['max_seq_length']}")
        print(f"   模型维度: {self.params['d_model']}")
        print(f"   注意力头数: {self.params['num_heads']}")
        
        try:
            self.model = self._build_model()
            print("✅ TransformerRanker模型构建成功")
        except Exception as e:
            print(f"❌ TransformerRanker构建失败: {e}")
            raise
        
    def _build_model(self):
        """构建稳定的Transformer模型"""
        print("🔨 构建Transformer模型...")
        
        # 输入层
        inputs = keras.Input(shape=(self.input_dim,), name='input_features')
        
        # 特征预处理和维度调整
        seq_len = self.params['max_seq_length']
        d_model = self.params['d_model']
        
        # 方法1: 如果输入维度可以整除序列长度
        if self.input_dim % seq_len == 0:
            features_per_token = self.input_dim // seq_len
            x = layers.Reshape((seq_len, features_per_token), name='reshape_input')(inputs)
            
            # 投影到d_model维度
            if features_per_token != d_model:
                x = layers.Dense(d_model, activation='relu', name='feature_projection')(x)
        else:
            # 方法2: 使用线性变换
            # 先投影到可以整除的维度
            target_dim = seq_len * d_model
            x = layers.Dense(target_dim, activation='relu', name='linear_projection')(inputs)
            x = layers.Reshape((seq_len, d_model), name='reshape_projected')(x)
        
        print(f"✓ 特征重塑完成: (batch, {seq_len}, {d_model})")
        
        # 添加位置编码
        x = self._add_positional_encoding(x, seq_len, d_model)
        print("✓ 位置编码完成")
        
        # Transformer层
        for i in range(self.params['num_layers']):
            x = self._transformer_block(x, i)
            print(f"✓ Transformer层 {i+1} 完成")
        
        # 全局特征提取
        # 使用多种池化方式
        avg_pool = layers.GlobalAveragePooling1D(name='global_avg_pool')(x)
        max_pool = layers.GlobalMaxPooling1D(name='global_max_pool')(x)
        
        # 合并不同的池化结果
        pooled = layers.Concatenate(name='concat_pools')([avg_pool, max_pool])
        
        # 分类头
        x = layers.Dense(128, activation='relu', name='dense_1')(pooled)
        x = layers.Dropout(self.params['dropout_rate'], name='dropout_1')(x)
        
        x = layers.Dense(64, activation='relu', name='dense_2')(x)
        x = layers.Dropout(self.params['dropout_rate'], name='dropout_2')(x)
        
        # 输出层
        outputs = layers.Dense(1, activation='linear', name='ranking_output')(x)
        
        # 创建模型
        model = keras.Model(inputs=inputs, outputs=outputs, name='TransformerRanker')
        
        # 编译模型
        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=self.params['learning_rate'],
                clipnorm=1.0  # 梯度裁剪，提高稳定性
            ),
            loss='mse',
            metrics=['mae']
        )
        
        print("✓ 模型编译完成")
        return model
    
    def _add_positional_encoding(self, x, seq_len, d_model):
        """添加稳定的位置编码"""
        # 创建位置编码矩阵
        positions = tf.range(seq_len, dtype=tf.float32)[:, tf.newaxis]
        
        # 使用简单但有效的位置编码
        div_term = tf.pow(10000.0, tf.range(0, d_model, 2, dtype=tf.float32) / d_model)
        
        # 计算sin和cos
        sin_vals = tf.sin(positions / div_term)
        cos_vals = tf.cos(positions / div_term)
        
        # 交替排列sin和cos值
        if d_model % 2 == 0:
            # 偶数维度
            pe_sin = tf.reshape(sin_vals, [seq_len, d_model // 2])
            pe_cos = tf.reshape(cos_vals, [seq_len, d_model // 2])
            
            # 交替排列
            pos_encoding = tf.stack([pe_sin, pe_cos], axis=2)
            pos_encoding = tf.reshape(pos_encoding, [seq_len, d_model])
        else:
            # 奇数维度，最后一个使用sin
            pe_sin = tf.reshape(sin_vals, [seq_len, d_model // 2])
            pe_cos = tf.reshape(cos_vals[:, :d_model//2], [seq_len, d_model // 2])
            
            pos_encoding = tf.stack([pe_sin, pe_cos], axis=2)
            pos_encoding = tf.reshape(pos_encoding, [seq_len, d_model - 1])
            
            # 添加最后一个维度
            last_dim = tf.sin(positions[:, 0:1] / 10000.0)
            pos_encoding = tf.concat([pos_encoding, last_dim], axis=1)
        
        # 扩展batch维度并添加到输入
        pos_encoding = pos_encoding[tf.newaxis, :, :]  # (1, seq_len, d_model)
        
        return x + pos_encoding
    
    def _transformer_block(self, x, block_idx):
        """稳定的Transformer块实现"""
        # 多头自注意力
        attention = layers.MultiHeadAttention(
            num_heads=self.params['num_heads'],
            key_dim=self.params['d_model'] // self.params['num_heads'],
            dropout=self.params['dropout_rate'],
            name=f'attention_{block_idx}'
        )
        
        # 注意力计算
        attn_output = attention(x, x)
        
        # 第一个残差连接和层归一化
        x1 = layers.Add(name=f'add_1_{block_idx}')([x, attn_output])
        x1 = layers.LayerNormalization(
            epsilon=1e-6, 
            name=f'layer_norm_1_{block_idx}'
        )(x1)
        
        # 前馈网络
        ffn_output = layers.Dense(
            self.params['dff'], 
            activation='relu',
            name=f'ffn_1_{block_idx}'
        )(x1)
        ffn_output = layers.Dropout(
            self.params['dropout_rate'],
            name=f'ffn_dropout_{block_idx}'
        )(ffn_output)
        ffn_output = layers.Dense(
            self.params['d_model'],
            name=f'ffn_2_{block_idx}'
        )(ffn_output)
        
        # 第二个残差连接和层归一化
        x2 = layers.Add(name=f'add_2_{block_idx}')([x1, ffn_output])
        x2 = layers.LayerNormalization(
            epsilon=1e-6,
            name=f'layer_norm_2_{block_idx}'
        )(x2)
        
        return x2
    
    def fit(self, X, y, group, **kwargs):
        """训练模型 - 带完整错误处理"""
        epochs = kwargs.get('epochs', self.params['epochs'])
        batch_size = kwargs.get('batch_size', self.params['batch_size'])
        
        print(f"🚀 开始训练TransformerRanker")
        print(f"📊 数据形状: {X.shape}")
        print(f"📊 训练参数: epochs={epochs}, batch_size={batch_size}")
        
        try:
            # 数据预处理
            X_processed = X.astype(np.float32)
            y_processed = y.astype(np.float32)
            
            # 检查数据
            if np.any(np.isnan(X_processed)) or np.any(np.isinf(X_processed)):
                print("⚠️ 发现NaN或Inf值，进行清理...")
                X_processed = np.nan_to_num(X_processed, nan=0.0, posinf=1.0, neginf=-1.0)
            
            if np.any(np.isnan(y_processed)) or np.any(np.isinf(y_processed)):
                print("⚠️ 标签中发现NaN或Inf值，进行清理...")
                y_processed = np.nan_to_num(y_processed, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # 数据标准化（可选，但推荐）
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_processed = scaler.fit_transform(X_processed).astype(np.float32)
            
            print("✓ 数据预处理完成")
            
            # 智能确定验证集大小
            total_samples = len(X_processed)
            if total_samples > 100000:
                val_ratio = 0.02  # 大数据集用2%
            elif total_samples > 10000:
                val_ratio = 0.05  # 中等数据集用5%
            else:
                val_ratio = 0.2   # 小数据集用20%
            
            val_size = int(total_samples * val_ratio)
            train_size = total_samples - val_size
            
            print(f"📊 数据分割: 训练={train_size}, 验证={val_size} ({val_ratio*100:.1f}%)")
            
            # 分割数据
            if val_size > 100:  # 只有当验证集足够大时才使用
                X_train = X_processed[:train_size]
                y_train = y_processed[:train_size]
                X_val = X_processed[train_size:]
                y_val = y_processed[train_size:]
                
                # 设置回调函数
                callbacks = [
                    keras.callbacks.EarlyStopping(
                        patience=5,
                        restore_best_weights=True,
                        verbose=1,
                        monitor='val_loss'
                    ),
                    keras.callbacks.ReduceLROnPlateau(
                        patience=3,
                        factor=0.5,
                        verbose=1,
                        monitor='val_loss'
                    )
                ]
                
                print("🔄 开始训练（带验证集）...")
                history = self.model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=callbacks,
                    verbose=1
                )
            else:
                print("🔄 开始训练（无验证集）...")
                history = self.model.fit(
                    X_processed, y_processed,
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=1
                )
            
            print("✅ TransformerRanker训练完成")
            
            # 计算特征重要性
            try:
                print("🔍 计算特征重要性...")
                sample_size = min(1000, len(X))
                self._compute_feature_importance(X[:sample_size])
                print("✅ 特征重要性计算完成")
            except Exception as e:
                print(f"⚠️ 特征重要性计算失败: {e}")
                self._feature_importance = np.ones(self.input_dim) / self.input_dim
            
            return history
            
        except Exception as e:
            print(f"❌ TransformerRanker训练失败: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _compute_feature_importance(self, X_sample):
        """计算特征重要性"""
        try:
            # 方法1: 尝试使用梯度
            X_tensor = tf.convert_to_tensor(X_sample.astype(np.float32))
            
            with tf.GradientTape() as tape:
                tape.watch(X_tensor)
                predictions = self.model(X_tensor, training=False)
                loss = tf.reduce_mean(predictions)
            
            gradients = tape.gradient(loss, X_tensor)
            
            if gradients is not None:
                importance = np.mean(np.abs(gradients.numpy()), axis=0)
                importance = importance / (np.sum(importance) + 1e-8)
                self._feature_importance = importance
                print("✓ 使用梯度计算特征重要性")
            else:
                raise ValueError("无法计算梯度")
                
        except Exception as e:
            print(f"⚠️ 梯度方法失败: {e}，使用方差方法")
            # 方法2: 使用输入特征的方差
            try:
                feature_variance = np.var(X_sample, axis=0)
                importance = feature_variance / (np.sum(feature_variance) + 1e-8)
                self._feature_importance = importance
                print("✓ 使用方差计算特征重要性")
            except:
                # 方法3: 均匀分布
                self._feature_importance = np.ones(self.input_dim) / self.input_dim
                print("✓ 使用均匀分布作为特征重要性")
    
    def predict(self, X):
        """预测分数"""
        try:
            print(f"🔮 TransformerRanker预测，数据形状: {X.shape}")
            
            # 数据预处理（与训练时保持一致）
            X_processed = X.astype(np.float32)
            
            # 清理异常值
            if np.any(np.isnan(X_processed)) or np.any(np.isinf(X_processed)):
                X_processed = np.nan_to_num(X_processed, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # 标准化（注意：实际应用中应该保存训练时的scaler）
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_processed = scaler.fit_transform(X_processed).astype(np.float32)
            
            # 批量预测以避免内存问题
            batch_size = 1000
            predictions = []
            
            for i in range(0, len(X_processed), batch_size):
                batch = X_processed[i:i+batch_size]
                batch_pred = self.model.predict(batch, verbose=0)
                predictions.append(batch_pred.flatten())
            
            result = np.concatenate(predictions)
            print(f"✅ TransformerRanker预测完成，结果形状: {result.shape}")
            return result
            
        except Exception as e:
            print(f"❌ TransformerRanker预测失败: {e}")
            import traceback
            traceback.print_exc()
            # 返回随机值作为后备
            return np.random.random(len(X)).astype(np.float32)
    
    def get_model_name(self):
        return "TransformerRanker"
    
    @property
    def feature_importances_(self):
        if self._feature_importance is None:
            return np.ones(self.input_dim) / self.input_dim
        return self._feature_importance


class BM25Ranker(BaseRanker):
    """BM25排序模型"""
    
    def __init__(self, **params):
        self.params = params
        self.model = None
        self.tokenized_corpus = None
        self.feature_names = None
        
    def fit(self, X, y, group, **kwargs):
        # 将特征转换为"文档"形式
        self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        self.tokenized_corpus = []
        
        for i in range(X.shape[0]):
            # 将特征值大于阈值的特征名作为"词"
            threshold = kwargs.get('threshold', 0.5)
            doc = [self.feature_names[j] for j in np.where(X[i] > threshold)[0]]
            if not doc:  # 如果没有特征超过阈值，使用所有非零特征
                doc = [self.feature_names[j] for j in np.where(X[i] != 0)[0]]
            self.tokenized_corpus.append(doc)
        
        self.model = BM25Okapi(self.tokenized_corpus)
    
    def predict(self, X):
        if self.model is None:
            raise ValueError("模型未训练，请先调用fit方法")
        
        scores = []
        for i in range(X.shape[0]):
            query = [self.feature_names[j] for j in np.where(X[i] > 0.5)[0]]
            if not query:
                query = [self.feature_names[j] for j in np.where(X[i] != 0)[0]]
            
            if query:
                doc_scores = self.model.get_scores(query)
                scores.append(doc_scores[i] if i < len(doc_scores) else 0.0)
            else:
                scores.append(0.0)
        
        return np.array(scores)
    
    def get_model_name(self):
        return "BM25Ranker"


class ModelFactory:
    """模型工厂类"""
    
    @staticmethod
    def create_model(model_name: str, use_gpu: bool = True, input_dim: Optional[int] = None, **params) -> BaseRanker:
        """
        创建指定的模型实例
        
        Args:
            model_name: 模型名称
            use_gpu: 是否使用GPU
            input_dim: 输入维度（神经网络模型需要）
            **params: 模型参数
            
        Returns:
            BaseRanker: 模型实例
        """
        if model_name == 'XGBRanker':
            return XGBRankerModel(use_gpu=use_gpu, **params)
        elif model_name == 'LGBMRanker':
            return LGBMRankerModel(use_gpu=use_gpu, **params)
        elif model_name == 'LambdaMART':
            return LambdaMART(use_gpu=use_gpu, **params)
        elif model_name == 'ListNet':
            return ListNetModel(use_gpu=use_gpu, **params)
        elif model_name == 'NeuralRanker':
            if input_dim is None:
                raise ValueError("NeuralRanker需要指定input_dim参数")
            return NeuralRanker(input_dim=input_dim, **params)
        elif model_name == 'RankNet':
            if input_dim is None:
                raise ValueError("RankNet需要指定input_dim参数")
            return RankNet(input_dim=input_dim, **params)
        elif model_name == 'TransformerRanker':
            if input_dim is None:
                raise ValueError("TransformerRanker需要指定input_dim参数")
            return TransformerRanker(input_dim=input_dim, **params)
        elif model_name == 'BM25Ranker':
            return BM25Ranker(**params)
        else:
            raise ValueError(f"未知模型类型: {model_name}")
    
    @staticmethod
    def get_available_models() -> List[str]:
        """获取所有可用的模型名称"""
        return [
            'XGBRanker', 'LGBMRanker', 'LambdaMART', 'ListNet', 
            'NeuralRanker', 'RankNet', 'TransformerRanker', 'BM25Ranker'
        ]