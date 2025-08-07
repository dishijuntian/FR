# 使用指南

## 🚀 快速开始

### 步骤 1: 安装依赖
```bash
pip install -r requirements.txt
```
*新增了 `tqdm` 和 `rich` 库用于美观的进度条显示*

### 步骤 2: 配置数据路径
```bash
python fix_config.py
```
这个脚本会：
- 检查当前配置
- 自动查找数据目录
- 帮助你更新配置文件

### 步骤 3: 运行程序
```bash
python start.py
```

### 🎯 新功能: 进度条演示
```bash
python demo_progress.py
```
查看各种进度条的效果

## 📁 项目文件说明

### 🌟 推荐使用的文件
1. **start.py** - 最可靠的启动脚本
2. **fix_config.py** - 配置修复脚本
3. **demo_progress.py** - 🆕 进度条演示脚本
4. **src/config.py** - 主配置文件

### 📄 完整文件列表
```
flight_ranking_analyzer/
├── src/                        # 核心代码目录
│   ├── __init__.py            # 包初始化
│   ├── config.py              # 🔧 配置文件（需要修改）
│   ├── models.py              # 模型定义
│   ├── data_processor.py      # 数据处理
│   ├── auto_tuner.py          # 自动调参
│   ├── analyzer.py            # 主分析器
│   ├── progress_utils.py      # 🆕 进度条工具
│   └── main.py                # 主程序
├── start.py                   # 🌟 推荐启动脚本
├── fix_config.py              # 🔧 配置修复脚本
├── demo_progress.py           # 🆕 进度条演示
├── test_gpu.py                # GPU测试脚本
├── quick_start.py             # 备用启动脚本
├── run.py                     # 简化启动脚本
├── requirements.txt           # 依赖文件
└── 说明文档...
```

## 🎯 新增进度条功能

### 📊 进度条类型

1. **数据加载进度条**
   ```
   📂 读取文件 ████████████ 100% 3/3 [00:15<00:00]
   ```

2. **模型训练进度条**
   ```
   🔧 训练 XGBRanker ████████ 80% 4/5 [01:23<00:21]
   总体训练进度 ██████████ 100% 3/3 [05:45<00:00]
   ```

3. **文件处理进度条**
   ```
   📁 处理训练文件 ████████ 100% 3/3 [02:30<00:00]
   🔮 执行预测 ████████████ 100% 3/3 [01:15<00:00]
   ```

4. **数据预处理进度条**
   ```
   🎯 执行数据抽样 ████████ 100% [00:05<00:00]
   🔄 特征工程 ██████████████ 100% [00:12<00:00]
   ```

### 🎨 进度条特性

- ✨ **美观显示**: 使用rich库的现代化进度条
- ⏱️ **时间估计**: 显示已用时间和剩余时间
- 📈 **实时更新**: 显示当前处理状态
- 🔄 **嵌套进度**: 支持多层级进度显示
- 📊 **完成总结**: 任务完成后的详细统计

### 💡 进度条配置# 使用指南

## 🚀 快速开始

### 步骤 1: 安装依赖
```bash
pip install -r requirements.txt
```

### 步骤 2: 配置数据路径
```bash
python fix_config.py
```
这个脚本会：
- 检查当前配置
- 自动查找数据目录
- 帮助你更新配置文件

### 步骤 3: 运行程序
```bash
python start.py
```

## 📁 项目文件说明

### 🌟 推荐使用的文件
1. **start.py** - 最可靠的启动脚本
2. **fix_config.py** - 配置修复脚本
3. **src/config.py** - 主配置文件

### 📄 完整文件列表
```
flight_ranking_analyzer/
├── src/                     # 核心代码目录
│   ├── __init__.py         # 包初始化
│   ├── config.py           # 🔧 配置文件（需要修改）
│   ├── models.py           # 模型定义
│   ├── data_processor.py   # 数据处理
│   ├── auto_tuner.py       # 自动调参
│   ├── analyzer.py         # 主分析器
│   └── main.py             # 主程序
├── start.py                # 🌟 推荐启动脚本
├── fix_config.py           # 🔧 配置修复脚本
├── quick_start.py          # 备用启动脚本
├── run.py                  # 简化启动脚本
├── requirements.txt        # 依赖文件
└── 说明文档...
```

## ⚙️ 配置说明

### 数据路径配置
在 `src/config.py` 中修改：
```python
class Config:
    # 🔧 修改这个路径为你的数据目录
    DATA_BASE_PATH = "你的数据路径/aeroclub-recsys-2025"
```

### 数据文件结构
确保你的数据目录结构如下：
```
你的数据路径/
├── encode/
│   ├── train/
│   │   ├── train_segment_0_encoded.parquet
│   │   ├── train_segment_1_encoded.parquet
│   │   └── train_segment_2_encoded.parquet
│   └── test/
│       ├── test_segment_0_encoded.parquet
│       ├── test_segment_1_encoded.parquet
│       └── test_segment_2_encoded.parquet
├── submission_template.csv
└── results/                 # 自动创建
```

## 🎯 运行选项

### 数据模式选择
- **抽样模式**: 快速测试，处理部分数据
- **全量模式**: 完整训练，使用所有数据

### 模型选择
可用模型：
1. XGBRanker (XGBoost)
2. LGBMRanker (LightGBM)
3. LambdaMART
4. ListNet
5. NeuralRanker (神经网络)
6. BM25Ranker (传统算法)

### 自动调参
- **关闭**: 使用默认参数，快速运行
- **开启**: 自动优化参数，提升性能但耗时更长

## 📊 输出结果

### 运行过程输出
- 模型训练进度
- 性能评估结果
- 特征重要性分析
- SHAP可解释性分析

### 最终输出文件
- `results/final_predictions.parquet` - 最终预测结果
- `results/analysis.log` - 详细运行日志
- 各种中间预测文件

## 🔧 常见问题解决

### 问题 1: 找不到模块
```
ImportError: No module named 'xxx'
```
**解决方案**: 
1. 运行 `pip install -r requirements.txt`
2. 使用 `start.py` 启动

### 问题 2: 数据路径错误
```
FileNotFoundError: 数据文件不存在
```
**解决方案**:
1. 运行 `python fix_config.py` 修复配置
2. 确保数据文件在正确位置

### 问题 3: 权限错误
```
PermissionError: 无法创建目录
```
**解决方案**:
1. 以管理员身份运行
2. 修改输出路径到有权限的目录

### 问题 4: GPU相关错误
```
CUDA相关错误
```
**解决方案**:
1. 安装CUDA和cuDNN
2. 在配置中设置 `FORCE_USE_CPU = True`

### 问题 5: 内存不足
```
MemoryError
```
**解决方案**:
1. 使用抽样模式
2. 减少抽样参数
3. 选择更少的模型

## 📈 性能优化建议

### 快速测试配置
- 数据模式: 抽样
- 抽样组数: 500-1000
- 模型选择: XGBRanker
- 自动调参: 关闭

### 最佳性能配置
- 数据模式: 全量
- 模型选择: XGBRanker + LGBMRanker + NeuralRanker
- 自动调参: 开启
- GPU加速: 启用

### 平衡配置
- 数据模式: 抽样 (2000组)
- 模型选择: XGBRanker + LGBMRanker
- 自动调参: 关闭
- 运行时间: 约30-60分钟

## 🎮 交互式界面说明

运行 `start.py` 后会出现交互式选择：

1. **数据加载模式选择**
   ```
   1. 抽样模式 (推荐用于快速测试)
   2. 全量模式 (使用所有数据)
   ```

2. **模型选择**
   ```
   1. XGBRanker
   2. LGBMRanker
   ...
   7. 所有模型
   ```

3. **自动调参设置**
   ```
   是否启用自动调参? (y/n, 默认n)
   ```

## 📞 获取帮助

如果遇到问题：
1. 🔧 先运行 `python fix_config.py` 检查配置
2. 📋 查看 `results/analysis.log` 日志文件
3. 📖 阅读错误信息中的解决建议
4. 🔄 尝试重新安装依赖包

## 🎉 成功运行的标志

看到以下输出表示运行成功：
```
✅ 所有依赖包检查通过
✅ Python路径已设置
✅ 数据路径配置正确
✅ 输出目录已创建
...
✅ 程序执行完成!
🎉 所有任务完成!
```

最终会在 `results/` 目录下生成预测结果文件。