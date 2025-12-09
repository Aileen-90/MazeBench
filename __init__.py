# mazebench/__init__.py
"""
MazeBench - 迷宫评测基准测试框架
"""
import sys
from pathlib import Path

__version__ = "1.0.0"

# 自动设置项目根目录到 Python 路径
_project_root = Path(__file__).parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
