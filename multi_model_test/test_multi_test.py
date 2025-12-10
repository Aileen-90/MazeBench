#!/usr/bin/env python3
"""
简单的测试脚本，验证多模型测试系统的基本功能
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """测试导入是否正常"""
    try:
        from sandbox.ai import run_ai_sandbox
        from utils.io import load_config, apply_env_keys
        from utils.logging import setup_logging, get_logger
        print("✓ 所有导入正常")
        return True
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        return False

def test_maze_discovery():
    """测试迷宫发现功能"""
    import glob
    maze_sizes = ['5x5', '9x9', '15x15']

    for size in maze_sizes:
        maze_dir = Path(f"mazes_{size}")
        if maze_dir.exists():
            maze_files = list(maze_dir.glob("*.json"))
            print(f"✓ {size}: 找到 {len(maze_files)} 个迷宫文件")
        else:
            print(f"✗ {size}: 目录不存在")

def test_config_loading():
    """测试配置加载"""
    try:
        from utils.io import load_config
        cfg = load_config()
        print(f"✓ 配置加载成功，模型: {cfg.get('model', 'unknown')}")
        return True
    except Exception as e:
        print(f"✗ 配置加载失败: {e}")
        return False

def main():
    """主测试函数"""
    print("MazeBench 多模型测试系统 - 功能验证")
    print("=" * 50)

    # 测试导入
    if not test_imports():
        return

    # 测试配置加载
    if not test_config_loading():
        return

    # 测试迷宫发现
    test_maze_discovery()

    print("\n基本功能验证完成！")
    print("可以使用以下命令运行完整测试:")
    print("python multi_model_maze_test.py --models gpt-4 --sizes 5x5 --trials 2 --workers 1")

if __name__ == '__main__':
    main()