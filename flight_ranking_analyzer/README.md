# 航班排序分析器 (Flight Ranking Analyzer) v2.1

一个基于机器学习的航班排序分析系统，支持多种排序模型、自动超参数调优和预测结果合并。

## ✨ 新功能 (v2.1)

- ✅ **不抽样选项**: 支持对所有segment文件的全部数据进行训练
- ✅ **预测结果合并**: 自动合并多个预测文件并与submission文件对应
- ✅ **自动调参功能**: 基于Optuna的超参数自动优化（默认关闭）
- ✅ **模块化架构**: 代码重构为多个独立模块，便于维护和扩展

## 🚀 功能特性

### 排序模型
- **XGBRanker**: XGBoost排序模型
- **LGBMRanker**: LightGBM排序模型  
- **LambdaMART**: 基于XGBoost的LambdaMART实现
- **ListNet**: 基于LightGBM的ListNet实现
- **NeuralRanker**: 深度神经网络排序模型
- **BM25Ranker**: 传统BM25算法

### 核心功能
- 📊 **HitRate@3评估**: 专业的排序模型评估指标
- 🎯 **特征重要性分析**: 多种方法分析特征贡献度
- 🔍 **SHAP可解释性**: 模型决策的可视化解释
- ⚡ **GPU加速**: 自动检测并使用GPU进行训练
- 🎛️ **自动调参**: 基于Optuna的贝叶斯优化
- 🔄 **结果合并**: 多模型集成和预测结果整合

### 数据处理
- 📈 **智能抽样**: 基于ranker_id的分组抽样
- 💾 **内存优化**: 自动内存管理和数据类型优化
- 🔧 **特征工程**: 自动特征选择和预处理
- 📋 **缺失值处理**: 智能填充策略

## 📁 项目结构

```
flight_ranking_analyzer/
├── src/
│   ├── __init__.py           # 包初始化
│   ├── config.py             # 配置管理
│   ├── models.py             # 模型定义
│   ├── data_processor.py     # 数据处理
│   ├── auto_tuner.py         # 自动调参
│   └── analyzer.py           # 主分析器
├── main.py                   # 主程序入口
├── requirements.txt          # 依赖包列表
└── README.md                # 项目说明
```

## 🛠️ 安装说明

### 1. 环境要求
- Python 3.8+
- CUDA 11.x (可选，用于GPU加速)

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. GPU支持 (可选)
如需GPU加速，请安装CUDA相关库：
```bash
# 对于CUDA 11.x
pip install cupy-cuda11x

# 确保TensorFlow GPU支持
pip install tensorflow[and-cuda]
```

## 📝 配置说明

### 数据路径配置
在 `src/config.py` 中修改数据路径：

```python
class Config:
    # 数据路径配置
    DATA_BASE_PATH = "你的数据根目录"
    TRAIN_DATA_PATH = os.path.join(DATA_BASE_PATH, "encode/train")
    TEST_DATA_PATH = os.path.join(DATA_BASE_PATH, "encode/test")
    SUBMISSION_FILE_PATH = os.path.join(DATA_BASE_PATH, "submission_template.csv")
```

### 抽样参数配置
```python
# 抽样配置
DEFAULT_NUM_GROUPS = 2000      # 每个文件抽取的ranker_id组数
DEFAULT_MIN_GROUP_SIZE = 20    # 每组最小数据条数
USE_SAMPLING = True            # 是否使用抽样
```

### 自动调参配置
```python
# 自动调参配置
ENABLE_AUTO_TUNING = False     # 默认关闭自动调参
AUTO_TUNING_TRIALS = 50        # 调参试验次数
AUTO_TUNING_TIMEOUT = 3600     # 调参超时时间(秒)
```

## 🎮 使用方法

### 基本使用
```bash
python main.py
```

### 交互式选项
运行程序后，会出现交互式选择界面：

1. **数据加载模式**
   - 抽样模式：快速测试，适合开发调试
   - 全量模式：使用所有数据，适合最终训练

2. **模型选择**
   - 可选择单个或多个模型
   - 支持所有6种排序模型

3. **自动调参**
   - 可选择是否启用自动调参
   - 可设置调参试验次数

### 编程式使用
```python
from src import FlightRankingAnalyzer, Config

# 初始化分析器
analyzer = FlightRankingAnalyzer(
    use_gpu=True,
    selected_models=['XGBRanker', 'LGBMRanker'],
    enable_auto_tuning=True,
    auto_tuning_trials=30
)

# 训练模型
results = analyzer.full_analysis(
    file_path="path/to/train_segment_0_encoded.parquet",
    use_sampling=False  # 使用全量数据
)

# 预测
prediction_file = analyzer.predict_test_data(
    test_file_path="path/to/test_segment_0_encoded.parquet",
    segment_idx=0
)

# 合并结果
final_result = analyzer.merge_all_predictions(
    prediction_files=[prediction_file],
    submission_file="path/to/submission_template.csv",
    output_file="final_predictions.parquet"
)
```

## 📊 输出说明

### 训练阶段输出
- **模型性能比较**: 各模型的HitRate@3分数
- **特征重要性图表**: Top 30重要特征的可视化
- **SHAP分析图**: 模型决策的可解释性分析

### 预测阶段输出
- **预测文件**: 每个测试segment的预测结果
  - 格式: `test_segment_X_encoded_predictions_train_segment_Y_encoded.parquet`
  - 包含: Id, ranker_id, 各模型的分数和排名

### 最终输出
- **合并预测文件**: `final_predictions.parquet`
  - 与submission模板对应的最终结果
  - 包含集成预测和最终排名

## ⚙️ 高级配置

### 模型参数调整
在 `src/config.py` 中修改 `DEFAULT_MODEL_PARAMS`:

```python
DEFAULT_MODEL_PARAMS = {
    'XGBRanker': {
        'n_estimators': 200,        # 树的数量
        'learning_rate': 0.1,       # 学习率
        'max_depth': 6,             # 最大深度
        # ... 其他参数
    }
}
```

### 自动调参搜索空间
在 `src/config.py` 中修改 `TUNING_SEARCH_SPACES`:

```python
TUNING_SEARCH_SPACES = {
    'XGBRanker': {
        'n_estimators': [50, 100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        # ... 其他参数范围
    }
}
```

## 🔧 故障排除

### 常见问题

1. **GPU不可用**
   ```
   解决方法: 检查CUDA安装，或在config.py中设置FORCE_USE_CPU=True
   ```

2. **内存不足**
   ```
   解决方法: 启用抽样模式，减少抽样组数，或增加系统内存
   ```

3. **文件路径错误**
   ```
   解决方法: 检查config.py中的路径配置，确保数据文件存在
   ```

4. **模型训练失败**
   ```
   解决方法: 检查数据格式，确保包含必需的列(Id, ranker_id, selected等)
   ```

### 性能优化建议

1. **启用GPU加速**: 安装CUDA和相关库
2. **合理设置抽样参数**: 平衡速度和准确性
3. **选择关键模型**: 避免运行所有模型以节省时间
4. **调整自动调参参数**: 根据时间预算设置试验次数

## 📈 版本历史

### v2.1 (当前版本)
- ✨ 新增不抽样选项
- ✨ 新增预测结果自动合并功能
- ✨ 新增自动调参功能
- 🔧 重构为模块化架构
- 📝 完善文档和配置管理

### v2.0
- ✨ 支持多种排序模型
- ✨ 特征重要性和SHAP分析
- ✨ GPU加速支持

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进这个项目！

## 📄 许可证

本项目采用MIT许可证 - 详见LICENSE文件

## 👥 团队

Flight Ranking Team

---

如有问题或建议，请创建Issue或联系开发团队。