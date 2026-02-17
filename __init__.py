"""
MazeBenchmark - 迷宫评测基准测试框架

作为子模块使用时，请通过 MazeBenchmarkAPI 类访问功能
"""
import sys
from pathlib import Path

__version__ = "1.0.0"

# 自动设置项目根目录到 Python 路径
_project_root = Path(__file__).parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# 子模块API入口
from .api import MazeBenchmarkAPI

# 导出主要功能类
from .sandbox.env import MazeEnvironment
from .sandbox.partial_observe import run_ai_sandbox_partial_observe
from .utils.io import load_config

__all__ = [
    'MazeBenchmarkAPI',
    'MazeEnvironment', 
    'run_ai_sandbox_partial_observe',
    'load_config',
    '__version__'
]