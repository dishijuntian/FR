# 项目结构说明

## 📁 完整目录结构

```
flight_ranking_analyzer/
├── src/                          # 源代码目录
│   ├── __init__.py              # 包初始化文件
│   ├── config.py                # 配置管理
│   ├── models.py                # 模型定义
│   ├── data_processor.py        # 数据处理
│   ├── auto_tuner.py            # 自动调参
│   ├── analyzer.py              # 主分析器
│   └── main.py                  # 主程序（在src内）
├── quick_start.py               # 快速启动脚本（推荐）
├── run.py                       # 简化启动脚本
├── requirements.txt             # 依赖文件
├── setup.py                     # 安装脚本
├── README.md                    # 项目说明
└── PROJECT_STRUCTURE.md         # 本文件
```

## 🚀 运行方式

### 推荐方式 1: 快速启动（最简单）
```bash
python quick_start.py
```
- ✅ 自动检查环境和依赖
- ✅ 自动配置数据路径
- ✅ 提供错误诊断和解决建议

### 推荐方式 2: 简化启动
```bash
python run.py
```
- ✅ 处理模块导入问题
- ✅ 基本的错误处理

### 高级方式: 直接运行
```bash
cd src
python main.py
```
- 需要手动配置路径
- 适合开发调试

## ⚙️ 首次运行配置

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 配置数据路径
编辑 `src/config.py` 文件中的路径配置：

```python
class Config:
    # 修改为你的数据根目录
    DATA_BASE_PATH = "你的数据路径/aeroclub-recsys-2025"
    
    # 以下路径会自动计算，通常不需要修改
    TRAIN_DATA_PATH = os.path.join(DATA_BASE_PATH, "encode/train")
    TEST_DATA_PATH = os.path.join(DATA_BASE_PATH, "encode/test")
    SUBMISSION_FILE_PATH = os.path.join(DATA_BASE_PATH, "submission_template.csv")
    OUTPUT_PATH = os.path.join(DATA_BASE_PATH, "results")
```

### 3. 准备数据文件
确保以下文件存在：

**训练文件:**
- `{TRAIN_DATA_PATH}/train_segment_0_encoded.parquet`
- `{TRAIN_DATA_PATH}/train_segment_1_encoded.parquet`
- `{TRAIN_DATA_PATH}/train_segment_2_encoded.parquet`

**测试文件:**
- `{TEST_DATA_PATH}/test_segment_0_encoded.parquet`
- `{TEST_DATA_PATH}/test_segment_1_encoded.parquet`
- `{TEST_DATA_PATH}/test_segment_2_encoded.parquet`

**提交模板:**
- `{DATA_BASE_PATH}/submission_template.csv`

## 🔧 故障排除

### 问题 1: 找不到模块
```
ModuleNotFoundError: No module named 'config'
```
**解决方案:** 使用 `quick_start.py` 或 `run.py` 启动

### 问题 2: 找不到数据文件
```
FileNotFoundError: 文件不存在
```
**解决方案:** 
1. 检查 `src/config.py` 中的路径配置
2. 确保数据文件存在于正确位置
3. 使用 `quick_start.py` 进行自动配置

### 问题 3: 权限错误
```
PermissionError: 无法创建输出目录
```
**解决方案:**
1. 确保输出目录有写入权限
2. 尝试以管理员身份运行
3. 修改输出路径到有权限的目录

### 问题 4: 内存不足
```
MemoryError: 内存不足
```
**解决方案:**
1. 启用抽样模式
2. 减少抽样组数
3. 选择较少的模型进行训练

## 📝 运行流程

### 1. 启动程序
```bash
python quick_start.py
```

### 2. 选择数据模式
- **抽样模式**: 适合快速测试和开发
- **全量模式**: 使用所有数据，适合最终训练

### 3. 选择模型
- 可选择单个或多个模型
- 建议首次运行选择1-2个模型测试

### 4. 自动调参设置
- **关闭**: 快速运行，使用默认参数
- **开启**: 优化性能，但需要更多时间

### 5. 查看结果
- 训练过程会显示模型性能
- 最终结果保存在 `{OUTPUT_PATH}/final_predictions.parquet`

## 📊 输出文件说明

### 训练阶段输出
- 模型性能比较表
- 特征重要性图表
- SHAP分析图（如果启用）

### 预测阶段输出
- 每个segment的预测文件
- 格式: `test_segment_X_encoded_predictions_train_segment_Y_encoded.parquet`

### 最终输出
- `final_predictions.parquet`: 与submission模板对应的最终结果
- `analysis.log`: 详细的运行日志

## 🎯 快速测试

### 最小化测试运行
1. 使用抽样模式
2. 设置较小的抽样参数（如500组，每组10条）
3. 只选择XGBRanker模型
4. 关闭自动调参

这样可以在几分钟内完成整个流程测试。

## 📞 获取帮助

如果遇到问题：
1. 查看运行日志 `{OUTPUT_PATH}/analysis.log`
2. 使用 `quick_start.py` 的自动诊断功能
3. 检查本文档的故障排除部分
4. 确保所有依赖正确安装