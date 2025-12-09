#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MazeBench Sandbox 启动脚本

提供便捷的命令行接口来启动CLI或API沙盒模式。
"""

import argparse
import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def run_cli(args):
    """启动CLI模式"""
    from sandbox.cli import main as cli_main
    # 传递命令行参数
    sys.argv = ['cli.py'] + args
    cli_main()


def run_api(args):
    """启动API模式"""
    from sandbox.api import main as api_main
    # 传递命令行参数
    sys.argv = ['api.py'] + args
    api_main()


def run_ai(args):
    """启动AI沙盒模式"""
    from sandbox.ai import run_ai_sandbox

    parser = argparse.ArgumentParser(description="AI Sandbox")
    parser.add_argument("maze", nargs="?", help="迷宫名称")
    parser.add_argument("--model", help="AI模型")
    parser.add_argument("--max-steps", type=int, help="最大步数")

    ai_args = parser.parse_args(args)

    # 默认迷宫
    if not ai_args.maze:
        from sandbox.env import MazeEnvironment
        mazes = MazeEnvironment.list_available_mazes("mazes")
        ai_args.maze = mazes[0] if mazes else None

    if not ai_args.maze:
        print("错误: 未找到可用迷宫")
        return

    result = run_ai_sandbox(ai_args.maze, ai_args.model, ai_args.max_steps)

    if 'error' in result:
        print(f"❌ {result['error']}")
        return

    print(f"🎯 成功: {result['success']}, 步数: {result['steps']}, 结果: {result['result_file']}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="MazeBench Sandbox - 迷宫交互沙盒",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:

  # 启动CLI模式（真人玩家）
  python run_sandbox.py cli

  # 启动API模式（AI模型通过HTTP）
  python run_sandbox.py api

  # 启动AI沙盒模式（AI直接测试，使用默认配置）
  python run_sandbox.py ai

  # 指定迷宫进行AI测试
  python run_sandbox.py ai maze_9x9_0

  # 指定迷宫目录
  python run_sandbox.py cli --mazes-dir ./my_mazes

  # API模式自定义端口
  python run_sandbox.py api --port 8000 --host 0.0.0.0

  # AI模式覆盖默认配置
  python run_sandbox.py ai maze_9x9_0 --model gpt-3.5-turbo --max-steps 30

  # 查看详细帮助
  python run_sandbox.py cli --help
  python run_sandbox.py api --help
  python run_sandbox.py ai --help
        """
    )

    parser.add_argument(
        'mode',
        choices=['cli', 'api', 'ai'],
        help='沙盒模式: cli(命令行交互), api(RESTful API), ai(AI模型测试)'
    )

    # 解析已知参数，剩余参数传递给子命令
    args, remaining = parser.parse_known_args()

    if args.mode == 'cli':
        run_cli(remaining)
    elif args.mode == 'api':
        run_api(remaining)
    elif args.mode == 'ai':
        run_ai(remaining)


if __name__ == '__main__':
    main()
