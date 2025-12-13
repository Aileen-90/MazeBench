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
│   ├── base.py       # 适配器接口
│   ├── openai_adapter.py
│   └── azure_adapter.py
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

2. 配置API密钥：
```bash
export OPENAI_API_KEY="your-api-key"
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

# 或者作为模块运行
python -m mazebench.main

# Windows 用户
run.bat
```

### 故障排除

如果遇到导入错误，请使用 `python run.py` 而不是直接运行 `python main.py`。

## 沙盒模式 (Sandbox Mode)

MazeBench 提供了多种沙盒模式，支持真人玩家和AI模型进行交互式迷宫探索。所有模式都通过 `run_sandbox.py` 脚本启动。

### 1. CLI模式（真人玩家交互）

使用命令行界面亲自体验迷宫，适合手动测试和调试。

**启动命令：**
```bash
# 基本启动（使用默认迷宫目录）
python run_sandbox.py cli

# 指定迷宫目录
python run_sandbox.py cli --mazes-dir ./mazes

# 或直接运行模块
python -m mazebench.sandbox.cli --mazes-dir ./mazes
```

**参数说明：**
- `--mazes-dir <目录>`: 指定迷宫文件所在目录（默认：从配置文件读取或使用 `mazes/`）

**操作说明：**
- `w/a/s/d` 或 `up/down/left/right` - 移动
- `h/help` - 显示帮助
- `r/reset` - 重置迷宫
- `i/info` - 显示迷宫信息
- `p/path` - 显示最短路径
- `q/quit` - 退出游戏

**配置参数（在 `config/config.yaml` 中）：**
- `sandbox.visibility`: 可视范围（-1=全部地图，0=只看当前位置，正数=k格范围）
- `sandbox.symbols`: 自定义迷宫渲染符号
- `sandbox.mazes_path`: 默认迷宫路径

### 2. API模式（RESTful API服务）

提供HTTP API接口，允许AI模型通过HTTP请求与迷宫环境交互，适合远程调用和自动化测试。

**启动命令：**
```bash
# 基本启动（默认：127.0.0.1:5000）
python run_sandbox.py api

# 自定义主机和端口
python run_sandbox.py api --host 0.0.0.0 --port 8000

# 启用调试模式
python run_sandbox.py api --debug

# 指定迷宫目录
python run_sandbox.py api --mazes-dir ./mazes --port 8000
```

**参数说明：**
- `--mazes-dir <目录>`: 指定迷宫文件目录（默认：从配置文件读取或使用 `mazes/`）
- `--host <地址>`: 服务器主机地址（默认：`127.0.0.1`）
- `--port <端口>`: 服务器端口（默认：`5000`）
- `--debug`: 启用Flask调试模式

**API接口：**

| 方法 | 端点 | 描述 | 请求体/参数 |
|------|------|------|------------|
| GET | `/health` | 健康检查 | - |
| GET | `/mazes` | 列出可用迷宫 | - |
| POST | `/load` | 加载指定迷宫 | `{"maze_name": "maze_9x9_0"}` |
| POST | `/reset` | 重置环境 | - |
| POST | `/step` | 执行动作 | `{"action": "right"}` (up/down/left/right) |
| GET | `/state` | 获取当前状态 | - |
| GET | `/info` | 获取迷宫信息 | - |
| GET | `/render` | 获取ASCII渲染 | `?show_path=true/false` |

**使用示例：**

```bash
# 1. 启动API服务器
python run_sandbox.py api --port 5000

# 2. 在另一个终端使用curl测试
# 列出可用迷宫
curl http://localhost:5000/mazes

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

# 获取ASCII渲染
curl http://localhost:5000/render?show_path=true
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
  "success": true,
  "info": {
    "current_position": [0, 1],
    "goal": [8, 8],
    "available_actions": ["up", "down", "left", "right"]
  }
}
```

### 3. AI模式（AI模型直接测试）

AI模型直接与迷宫环境交互，适合批量测试和模型性能评估。AI可以看到完整地图，并使用坐标或方向指令进行移动。

**启动命令：**
```bash
# 使用默认配置（从配置文件读取模型和迷宫）
python run_sandbox.py ai

# 指定迷宫名称
python run_sandbox.py ai maze_9x9_0

# 指定模型和最大步数
python run_sandbox.py ai maze_9x9_0 --model gpt-4 --max-steps 100

# 指定迷宫目录（如果迷宫不在默认路径）
python run_sandbox.py ai maze_15x15_0 --model gpt-3.5-turbo
```

**参数说明：**
- `maze` (位置参数): 迷宫名称，可选（如不指定则使用默认迷宫）
- `--model <模型名>`: 指定AI模型（默认：从配置文件读取或 `gpt-4`）
- `--max-steps <步数>`: 最大执行步数（默认：从配置文件读取或 `50`）

**配置参数（在 `config/config.yaml` 中）：**
- `sandbox.max_steps`: 最大步数（默认：`50`）
- `sandbox.memory`: AI记忆长度，最近k次动作（默认：`5`，0=无记忆，-1=全部记忆）
- `sandbox.visibility`: 可视范围（默认：`-1`，表示可见全部地图）
- `sandbox.mazes_path`: 迷宫文件路径（默认：`mazes/`）
- `sandbox.symbols`: 迷宫符号配置
- `model`: 默认AI模型名称
- `temperature`: 模型温度参数

**AI行为说明：**
- AI可以看到完整迷宫地图（如果 `visibility=-1`）
- AI可以输出路径坐标序列，如：`(1,2),(1,3),(2,3)`（使用 (y,x) 格式）
- AI也可以输出单个方向指令：`up`, `down`, `left`, `right`
- 结果会保存到 `outputs/ai_sandbox_{maze}_{model}_{timestamp}.json`

### 4. Partial-Observe模式（部分观察模式）

AI模型在有限视野下探索迷宫，只能看到周围k个格子，视线会被墙壁阻挡。此模式更接近真实场景，AI不知道自己的坐标，只能通过方向+步数移动。

**启动命令：**
```bash
# 使用默认配置
python run_sandbox.py partial-observe maze_9x9_0

# 指定模型和最大步数
python run_sandbox.py partial-observe maze_9x9_0 --model gpt-4 --max-steps 100

# 使用其他模型
python run_sandbox.py partial-observe maze_15x15_0 --model gpt-3.5-turbo --max-steps 150
```

**参数说明：**
- `maze` (位置参数): 迷宫名称（可选，如不指定则使用默认迷宫）
- `--model <模型名>`: 指定AI模型（默认：从配置文件读取）
- `--max-steps <步数>`: 最大执行步数（默认：从配置文件读取）

**配置参数（在 `config/config.yaml` 中）：**
- `sandbox.max_steps`: 最大步数（默认：`50`）
- `sandbox.memory`: AI记忆长度（默认：`5`）
- `sandbox.visibility`: **必须为正数**（如：`3` 表示可见周围3格），这是部分观察模式的关键参数
- `sandbox.mazes_path`: 迷宫文件路径
- `sandbox.symbols`: 迷宫符号配置（`masked` 符号用于标记不可见区域）

**AI行为说明：**
- AI只能看到周围 `visibility` 格内的区域（曼哈顿距离）
- 视线会被墙壁阻挡（无法看穿墙壁）
- AI不知道自己的坐标和目标的坐标
- AI使用"方向+步数"格式移动，如：`up 3`, `right 2, down 1`
- 步数不能超过可视范围
- 结果保存格式与AI模式相同

**可见性规则：**
- 使用曼哈顿距离（上下左右移动的距离）判断是否在可视范围内
- 如果从当前位置到目标位置的直线路径上有墙壁阻挡，则目标不可见
- 墙壁本身如果可见，也会显示在视野中

### 配置参数说明

所有沙盒模式的配置都在 `config/config.yaml` 的 `sandbox` 节点下：

```yaml
sandbox:
  enabled: true              # 是否启用沙盒功能
  max_steps: 50              # AI测试最大步数
  memory: 5                  # AI记忆长度（最近k次动作），0=无记忆，-1=全部记忆
  visibility: -1             # 可视范围：-1=全部地图，0=只看当前位置，正数=k格范围（partial-observe模式必须>0）
  mazes_path: "mazes/"       # 迷宫文件路径
  symbols:                   # 迷宫符号配置
    wall: "█"               # 墙壁符号
    path: " "               # 路径符号
    start: "S"              # 起点符号
    goal: "G"               # 终点符号
    agent: "A"              # 当前位置符号
    masked: "?"             # 不可见区域符号（partial-observe模式使用）
```

**配置优先级：**
1. 命令行参数（最高优先级）
2. 环境变量
3. `config/local.yaml`（本地配置）
4. `config/config.yaml`（默认配置）

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

## 配置文件

- `config/config.yaml`：主配置文件
- `config/local.yaml`：本地配置（覆盖主配置）

支持通过环境变量设置敏感信息。

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
