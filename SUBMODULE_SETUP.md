# MazeBenchmark 子模块安装指南

本文档说明如何将MazeBenchmark作为子模块集成到其他项目中。

## 快速开始

### 1. 添加为Git子模块

在你的项目根目录执行：

```bash
git submodule add <mazebenchmark-repo-url> submodules/MazeBenchmark
git submodule update --init --recursive
```

### 2. 安装依赖

确保安装了MazeBenchmark的依赖：

```bash
pip install -r submodules/MazeBenchmark/requirements.txt
```

### 3. 基本使用

在你的代码中：

```python
import sys
from pathlib import Path

# 添加子模块路径
submodule_path = Path(__file__).parent / "submodules" / "MazeBenchmark"
sys.path.insert(0, str(submodule_path))

# 导入MazeBenchmark
from mazebenchmark import MazeBenchmarkAPI, quick_test

# 快速测试
result = quick_test("maze_9x9_0", "gpt-4")
print(f"测试结果: {result}")

# 使用API
api = MazeBenchmarkAPI()
mazes = api.list_available_mazes()
print(f"可用迷宫: {mazes}")
```

## 详细集成方案

### 方案1：统一入口脚本

创建 `main.py` 作为项目统一入口：

```python
#!/usr/bin/env python3
"""
统一入口脚本 - 集成MazeBenchmark和自定义代码
"""

import sys
from pathlib import Path

# 添加子模块路径
submodule_path = Path(__file__).parent / "submodules" / "MazeBenchmark"
sys.path.insert(0, str(submodule_path))

from mazebenchmark import MazeBenchmarkAPI


def main():
    """主入口函数"""
    print("=== 迷宫求解项目 ===")
    print("1. 运行MazeBenchmark测试")
    print("2. 运行自定义求解器")
    
    choice = input("请选择功能: ")
    
    if choice == "1":
        run_benchmark()
    elif choice == "2":
        run_custom_solver()
    else:
        print("无效选择")


def run_benchmark():
    """运行基准测试"""
    api = MazeBenchmarkAPI()
    mazes = api.list_available_mazes()
    
    if mazes:
        result = api.run_partial_observe_test(mazes[0])
        print(f"基准测试结果: {result}")


def run_custom_solver():
    """运行自定义求解器"""
    # 你的自定义逻辑
    pass


if __name__ == "__main__":
    main()
```

### 方案2：API包装器模式

创建 `maze_integration.py`：

```python
"""
MazeBenchmark集成包装器
"""

import sys
from pathlib import Path

class MazeIntegration:
    def __init__(self, config_path=None):
        # 设置子模块路径
        submodule_path = Path(__file__).parent / "submodules" / "MazeBenchmark"
        sys.path.insert(0, str(submodule_path))
        
        from mazebenchmark import MazeBenchmarkAPI
        self.api = MazeBenchmarkAPI(config_path)
    
    def benchmark_solver(self, maze_name, solver_type="ai"):
        """基准测试求解器"""
        if solver_type == "ai":
            return self.api.run_partial_observe_test(maze_name)
        elif solver_type == "custom":
            # 集成你的自定义求解器
            return self._run_custom_solver(maze_name)
    
    def _run_custom_solver(self, maze_name):
        """运行自定义求解器"""
        # 你的自定义求解逻辑
        pass
```

## 配置管理

### 使用默认配置

MazeBenchmark会自动查找配置文件，优先级：
1. 当前目录的 `config/config.yaml`
2. 项目根目录的 `config/config.yaml`
3. 子模块目录的 `config/config.yaml`
4. 内置默认配置

### 自定义配置

创建 `config/custom_config.yaml`：

```yaml
# 自定义配置
model: gpt-4
sandbox:
  max_steps: 100
  visibility: 5
  memory: 10

# API密钥（可选）
OPENAI_API_KEY: your-api-key
OPENAI_API_BASE: your-api-base
```

然后在代码中：

```python
api = MazeBenchmarkAPI("config/custom_config.yaml")
```

## 主要API功能

### MazeBenchmarkAPI 类

- `list_available_mazes()` - 获取可用迷宫列表
- `load_maze(maze_name)` - 加载迷宫环境
- `generate_maze(width, height)` - 生成新迷宫
- `run_partial_observe_test()` - 运行部分观察模式测试
- `run_full_observe_test()` - 运行完整观察模式测试
- `run_multi_model_test()` - 运行多模型批量测试

### 便捷函数

- `quick_test(maze_name, model)` - 快速测试
- `create_api(config_path)` - 创建API实例

## 目录结构建议

```
my_maze_project/
├── README.md
├── requirements.txt
├── main.py                    # 统一入口
├── config/
│   ├── config.yaml           # 主配置
│   └── custom_config.yaml    # 自定义配置
├── src/                      # 你的代码
│   ├── __init__.py
│   ├── maze_solver.py
│   └── utils.py
└── submodules/
    └── MazeBenchmark/        # Git子模块
        ├── api.py
        ├── __init__.py
        └── requirements.txt
```

## 常见问题

### Q: 导入失败怎么办？
A: 确保已正确添加子模块路径到Python路径：
```python
import sys
sys.path.insert(0, "submodules/MazeBenchmark")
```

### Q: 配置文件找不到怎么办？
A: MazeBenchmark有内置默认配置，或手动指定配置文件路径。

### Q: 如何更新子模块？
A: 在项目根目录执行：
```bash
git submodule update --remote
```

## 示例代码

更多使用示例请参考 `example_usage.py`。

---

通过以上步骤，你可以轻松将MazeBenchmark作为子模块集成到你的项目中！