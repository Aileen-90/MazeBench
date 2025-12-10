#!/usr/bin/env python3
"""
MazeBench 多模型迷宫测试启动脚本

这个脚本是 multi_model_test/multi_model_maze_test.py 的快捷启动方式
"""

import sys
import os
from pathlib import Path

# 获取项目根目录
project_root = Path(__file__).parent
test_dir = project_root / "multi_model_test"
test_script = test_dir / "multi_model_maze_test.py"

if not test_script.exists():
    print(f"错误: 测试脚本不存在: {test_script}")
    sys.exit(1)

# 将测试脚本路径添加到Python路径
sys.path.insert(0, str(test_dir))

# 执行测试脚本
exec(open(test_script).read())