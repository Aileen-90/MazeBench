# MazeBench

迷宫评测基准测试框架 - 用于评测大语言模型在迷宫路径规划任务上的表现。

## 架构特点

- **模块化设计**：核心功能、模型适配器、评测执行器完全解耦
- **可扩展性**：轻松支持新的模型、新的评测任务类型
- **标准化接口**：统一的迷宫数据格式，标准化的评测流程

## 目录结构

```
mazebench/
├── adapters/          # 模型适配器层
│   ├── base.py                # 适配器接口
│   ├── openai_adapter.py      # OpenAI API 适配器
│   ├── azure_adapter.py       # Azure OpenAI API 适配器
│   ├── transformers_adapter.py # Transformers 本地模型适配器
│   ├── ollama_adapter.py      # Ollama 服务适配器
│   └── ark_adapter.py         # ARK API 适配器
├── core/             # 核心算法层
│   ├── config.py     # 配置类
│   ├── generator.py  # 迷宫生成器
│   ├── validator.py  # 路径验证器
│   ├── metrics.py    # 评分计算器
│   ├── parser.py     # 输出解析器
│   ├── anti_cheat.py # 防作弊系统
│   └── report.py     # 报告生成器
├── runners/          # 评测执行器层
│   ├── text2d_runner.py
│   └── image2d_runner.py
├── utils/            # 工具函数层
│   ├── io.py         # 文件I/O
│   └── logging.py    # 日志工具
├── config/           # 配置文件
│   ├── config.yaml   # 主配置
│   └── local.yaml    # 本地配置
├── outputs/          # 输出结果
├── mazes/            # 迷宫数据
├── sandbox/          # 沙盒模式
├── main.py           # 主入口
└── requirements.txt  # 依赖包
```

## 快速开始

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 模型配置：

MazeBench 支持多种模型运行方式：

#### 云服务模型 (OpenAI/Azure/ARK)

配置 API 密钥（以 OpenAI 为例）：

```bash
export OPENAI_API_KEY="your-api-key"
```

或在配置文件中设置：

```yaml
PROVIDER: "openai"
OPENAI_API_KEY: "your-api-key"
```

#### 本地模型 (Transformers)

直接使用 Hugging Face Transformers 加载本地模型：

```yaml
PROVIDER: "transformers"
model: "meta-llama/Llama-2-7b-chat-hf"
device_map: "auto"  # 自动检测 GPU
```

#### Ollama 服务

通过 Ollama 服务运行本地模型：

```yaml
PROVIDER: "ollama"
model: "llama3.1:8b-instruct"
base_url: "http://localhost:11434/v1"
```

3. 配置迷宫生成（可选）：
编辑 `config/config.yaml` 启用迷宫自动生成：
```yaml
generate_mazes: true
maze_count: 5
text2d:
  size: "9x9"
  algorithm: "dfs"
```

4. 运行评测：
```bash
# 使用启动脚本（推荐）
python run.py

# 使用指定配置（本地模型示例）
python run.py --config config/llama_local_config.yaml

# 使用 Ollama 服务示例
python run.py --config config/ollama_config.yaml

# 或者作为模块运行
python -m mazebench.main

# Windows 用户
run.bat
```

### 故障排除

如果遇到导入错误，请使用 `python run.py` 而不是直接运行 `python main.py`。

## 沙盒模式 (Sandbox Mode)

MazeBench 提供了沙盒模式，支持真人玩家和AI模型进行交互式迷宫探索。

### CLI模式 (真人玩家)

使用命令行界面亲自体验迷宫：

```bash
# 启动CLI沙盒（推荐方式）
python run_sandbox.py cli

# 或直接运行模块
python -m mazebench.sandbox.cli

# 指定迷宫目录
python run_sandbox.py cli --mazes-dir ./mazes
```

**操作说明：**
- `w/a/s/d` 或 `up/down/left/right` - 移动
- `h/help` - 显示帮助
- `r/reset` - 重置迷宫
- `i/info` - 显示迷宫信息
- `p/path` - 显示最短路径
- `q/quit` - 退出游戏

### 批量测试模型

**迷宫目录结构：**
```
mazes/              # 迷宫父目录（可通过 --mazes-dir 指定）
├── 5x5/            # 5x5 尺寸迷宫目录（精确匹配）
│   ├── maze_5x5_0.json
│   ├── maze_5x5_1.json
│   └── ...
├── mazes9x9/       # 9x9 尺寸迷宫目录（模糊匹配，包含"9x9"即可）
│   ├── maze_9x9_0.json
│   └── ...
└── maze_15x15/     # 15x15 尺寸迷宫目录（模糊匹配）
    └── ...
```

**目录名匹配规则：**
- 优先精确匹配：如果存在与尺寸名完全相同的目录（如 `5x5/`），则使用该目录
- 模糊匹配：如果精确匹配失败，会查找所有包含尺寸名的子目录（如 `mazes5x5/`、`maze_5x5/` 等）
- 例如：指定 `--sizes 5x5` 时，可以匹配到 `5x5/`、`mazes5x5/`、`maze_5x5/` 等目录

**使用示例：**

```bash
# 从配置文件读取模型（推荐），测试三种迷宫大小，每迷宫10次测试
python run_multi_test.py --sizes 5x5 9x9 15x15 --trials 10

# 测试两个模型在三种迷宫大小上，每迷宫10次测试
python run_multi_test.py --models gpt-4 gpt-3.5-turbo --sizes 5x5 9x9 15x15 --trials 10

# 指定并发线程数
python run_multi_test.py --models gpt-4 --sizes 5x5 --trials 5 --workers 2

# 自定义输出目录和迷宫目录
python run_multi_test.py --models gpt-4 --sizes 9x9 --output-dir my_results --mazes-dir my_mazes

# 完整示例：涵盖所有字段
python run_multi_test.py --models gpt-4 gpt-3.5-turbo --sizes 5x5 9x9 15x15 --trials 10 --workers 4 --output-dir my_results --mazes-dir mazes

```

**注意：**
- 如果未指定 `--models` 参数，程序会自动从配置文件（`config/config.yaml` 或 `config/local.yaml`）中读取 `model` 或 `models` 字段。配置文件中的 `model` 字段（单个模型）会被转换为列表使用。
- 如果未指定 `--mazes-dir` 参数，程序会从配置文件的 `sandbox.mazes_path` 读取，如果配置中也没有，则默认使用 `mazes/` 目录。

**API接口：**

| 方法 | 端点 | 描述 |
|------|------|------|
| GET | `/health` | 健康检查 |
| GET | `/mazes` | 列出可用迷宫 |
| POST | `/load` | 加载指定迷宫 |
| POST | `/reset` | 重置环境 |
| POST | `/step` | 执行动作 |
| GET | `/state` | 获取当前状态 |
| GET | `/info` | 获取迷宫信息 |
| GET | `/render` | 获取ASCII渲染 |

**使用示例：**

```bash
# 加载迷宫
curl -X POST http://localhost:5000/load \
  -H "Content-Type: application/json" \
  -d '{"maze_name": "maze_9x9_0"}'

# 执行动作
curl -X POST http://localhost:5000/step \
  -H "Content-Type: application/json" \
  -d '{"action": "right"}'

# 获取状态
curl http://localhost:5000/state
```

**响应格式：**
```json
{
  "action": "right",
  "state": {
    "position": [0, 1],
    "done": false,
    "steps": 1
  },
  "reward": -0.1,
  "success": true
}
```

## 配置文件

- `config/config.yaml`：主配置文件
- `config/local.yaml`：本地配置（覆盖主配置）

支持通过环境变量设置敏感信息。

### 模型配置

```yaml
# 模型提供者配置
PROVIDER: "openai"  # 可选值: openai, azure, transformers, ollama, ark

# OpenAI 适配器配置
OPENAI_API_KEY: "your-api-key"
model: "gpt-4"
temperature: 0.0
max_new_tokens: 1000

# Azure OpenAI 适配器配置
AZURE_OPENAI_KEY: "your-azure-key"
AZURE_OPENAI_ENDPOINT: "your-azure-endpoint"
AZURE_OPENAI_VERSION: "2023-05-15"

# Transformers 本地模型配置
# model: "meta-llama/Llama-2-7b-chat-hf"
device_map: "auto"  # 自动检测 GPU
load_in_4bit: false  # 4-bit 量化
load_in_8bit: false  # 8-bit 量化

# Ollama 适配器配置
# model: "llama3.1:8b-instruct"
base_url: "http://localhost:11434/v1"

# ARK 适配器配置
ARK_API_KEY: "your-ark-key"
ARK_BASE_URL: "https://api.ark.com/v1"
```

### 迷宫生成配置

MazeBench 支持自动生成迷宫，无需手动准备数据：

```yaml
# 迷宫生成配置
generate_mazes: true       # 启用自动生成
maze_output_dir: "mazes/"  # 输出目录
maze_count: 5             # 生成数量

# 文本迷宫配置
text2d:
  size: "9x9"              # 尺寸 (高x宽)
  seed: 42                 # 随机种子
  start_goal: "corner"     # 起点位置: "corner"/"random"
  algorithm: "dfs"         # 算法: "dfs"/"prim"

# 图片迷宫配置
image2d:
  size: "9x9"
  seed: 42
  start_goal: "corner"
  algorithm: "dfs"
  cell_px: 24              # 单元格像素大小
```

## 迷宫数据格式

```json
{
  "grid": [[1, 1, 1], [1, 0, 1], [1, 1, 1]],
  "start": [1, 1],
  "goal": [1, 2],
  "shortest_path": [[1, 1], [1, 2]],
  "nonce": 12345
}
```

## 扩展开发

### 添加新模型适配器

1. 继承 `BaseAdapter` 类
2. 实现 `generate()` 方法
3. 在 `adapters/__init__.py` 中注册

### 添加新评测类型

1. 在 `core/` 中实现相应的验证器、解析器等
2. 在 `runners/` 中创建新的runner
3. 在 `main.py` 中添加支持