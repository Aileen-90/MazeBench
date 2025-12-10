# MazeBench 多模型迷宫测试系统

这个系统允许你使用多线程同时测试多个AI模型在不同大小迷宫中的路径找到能力。

## 功能特点

- **多线程并发**: 支持同时运行多个测试任务，提高测试效率
- **多模型支持**: 可以同时测试多个AI模型（如GPT-4、GPT-3.5等）
- **多迷宫大小**: 支持测试不同大小的迷宫（5x5, 9x9, 15x15等）
- **重复测试**: 每个迷宫可以进行多次测试，计算成功率和平均步数
- **结果组织**: 结果按迷宫大小和模型名称自动组织到不同文件夹
- **详细报告**: 生成详细的汇总报告和统计信息

## 项目文件结构

```
mazebench/                           # 项目根目录
├── multi_model_test/                # 多模型测试模块
│   ├── __init__.py
│   ├── multi_model_maze_test.py     # 主测试脚本
│   ├── run_multi_test.sh            # Shell运行脚本
│   ├── test_multi_test.py           # 功能验证脚本
│   └── MULTI_MODEL_TEST_README.md   # 本文档
├── run_multi_test.py                # 便捷启动脚本
└── ...

multi_model_results/                 # 测试结果目录（自动生成）
├── 5x5/                            # 迷宫大小
│   ├── gpt-4/                      # 模型名称
│   │   ├── maze_5x5_0_trial_0.json # 单个测试结果
│   │   ├── maze_5x5_0_trial_1.json
│   │   └── ...
│   └── doubao-seed-1-6-251015/
│       └── ...
├── 9x9/
│   └── ...
├── summary_report.json             # 详细汇总报告
└── summary_report.txt             # 人类可读汇总报告
```

## 快速开始

### 方法1：使用便捷启动脚本（推荐）

```bash
# 测试GPT-4和GPT-3.5-turbo在5x5和9x9迷宫上，每迷宫10次测试
python run_multi_test.py --models gpt-4 gpt-3.5-turbo --sizes 5x5 9x9 --trials 10

# 指定并发线程数为2
python run_multi_test.py --models gpt-4 --sizes 5x5 --trials 5 --workers 2

# 自定义输出目录
python run_multi_test.py --models gpt-4 --sizes 9x9 --output-dir my_test_results
```

### 方法2：直接使用测试脚本

```bash
# 测试GPT-4和GPT-3.5-turbo在5x5和9x9迷宫上，每迷宫10次测试
python multi_model_test/multi_model_maze_test.py --models gpt-4 gpt-3.5-turbo --sizes 5x5 9x9 --trials 10

# 指定并发线程数为2
python multi_model_test/multi_model_maze_test.py --models gpt-4 --sizes 5x5 --trials 5 --workers 2

# 自定义输出目录
python multi_model_test/multi_model_maze_test.py --models gpt-4 --sizes 9x9 --output-dir my_test_results
```

### 方法3：使用Shell脚本

```bash
# 设置环境变量（可选）
export MODELS="gpt-4,doubao-seed-1-6-251015"
export SIZES="5x5,9x9,15x15"
export TRIALS=10
export WORKERS=4
export OUTPUT_DIR="multi_model_results"

# 运行测试（从项目根目录运行）
./multi_model_test/run_multi_test.sh
```

## 参数说明

### Python脚本参数

- `--models`: 要测试的模型列表（必需）
  - 示例: `--models gpt-4 gpt-3.5-turbo`

- `--sizes`: 迷宫大小列表（必需）
  - 示例: `--sizes 5x5 9x9 15x15`

- `--trials`: 每个迷宫的测试次数（默认: 10）
  - 示例: `--trials 5`

- `--workers`: 并发线程数（默认: 4）
  - 示例: `--workers 8`

- `--output-dir`: 结果输出目录（默认: multi_model_results）
  - 示例: `--output-dir my_results`

### Shell脚本环境变量

- `MODELS`: 模型列表，用逗号分隔（默认: gpt-4,doubao-seed-1-6-251015）
- `SIZES`: 迷宫大小列表，用逗号分隔（默认: 5x5,9x9,15x15）
- `TRIALS`: 每迷宫测试次数（默认: 10）
- `WORKERS`: 并发线程数（默认: 4）
- `OUTPUT_DIR`: 输出目录（默认: multi_model_results）

## 输出结果

### 单个测试结果文件

每个测试都会生成一个JSON文件，包含：

```json
{
  "model": "gpt-4",
  "maze_size": "5x5",
  "maze_name": "maze_5x5_0",
  "trial_id": 0,
  "success": true,
  "steps": 8,
  "total_steps": 8,
  "error": null,
  "timestamp": 1703123456.789,
  "result_file": "outputs/ai_sandbox_maze_5x5_0_gpt-4_1703123456.json"
}
```

### 汇总报告

系统会生成两个汇总报告：

1. **summary_report.json**: 详细的JSON格式报告
2. **summary_report.txt**: 人类可读的文本格式报告

汇总报告包含：
- 各模型的成功率和平均步数
- 各迷宫大小的统计信息
- 总体测试统计

## 示例输出

```
测试配置:
  模型: gpt-4 doubao-seed-1-6-251015
  迷宫大小: 5x5 9x9 15x15
  每迷宫测试次数: 10
  并发线程数: 4
  输出目录: multi_model_results

开始测试，总共需要运行 60 个测试任务
进度: 10/60 (16.7%)
...
进度: 60/60 (100.0%)

测试完成！总耗时: 125.3秒

关键统计:
gpt-4: 成功率 95.0%, 平均步数 12.3
doubao-seed-1-6-251015: 成功率 87.5%, 平均步数 15.7

总测试数: 60
总体成功率: 91.2%
```

## 注意事项

1. **API密钥**: 确保在`config/local.yaml`中配置了正确的API密钥
2. **迷宫文件**: 确保`mazes_{size}/`目录下有对应的迷宫JSON文件
3. **并发限制**: 根据你的API速率限制调整`--workers`参数
4. **内存使用**: 大迷宫和大量并发可能会消耗较多内存
5. **错误处理**: 系统会自动处理测试失败的情况，并在报告中记录

## 故障排除

### 常见问题

1. **迷宫文件不存在**
   - 检查`mazes_{size}/`目录是否存在
   - 运行`python main.py`先生成迷宫

2. **API调用失败**
   - 检查API密钥配置
   - 检查网络连接
   - 降低并发线程数

3. **内存不足**
   - 减少并发线程数
   - 测试较小的迷宫

### 日志查看

系统会输出详细的日志信息，包括：
- 测试进度
- 每个测试的详细结果
- 错误信息

如果需要更详细的调试信息，可以修改`config/local.yaml`中的`log_level`。