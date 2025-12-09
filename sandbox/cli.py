# mazebench/sandbox/cli.py
"""
MazeBench Sandbox CLI交互程序

提供命令行界面，让真人玩家可以交互式地探索迷宫。
"""

import sys
import os
from pathlib import Path
from typing import Optional

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from .env import MazeEnvironment


class MazeCLI:
    """迷宫命令行交互界面"""

    def __init__(self, mazes_dir: str = "mazes"):
        """
        初始化CLI

        Args:
            mazes_dir: 迷宫文件目录
        """
        self.mazes_dir = mazes_dir
        self.env: Optional[MazeEnvironment] = None

    def list_mazes(self) -> None:
        """列出所有可用的迷宫"""
        mazes = MazeEnvironment.list_available_mazes(self.mazes_dir)
        if not mazes:
            print(f"未找到迷宫文件在目录: {self.mazes_dir}")
            print("请先运行主程序生成迷宫，或指定正确的迷宫目录。")
            return

        print("可用的迷宫:")
        for i, maze_name in enumerate(mazes, 1):
            print(f"  {i}. {maze_name}")

    def select_maze(self) -> Optional[str]:
        """
        让用户选择迷宫

        Returns:
            Optional[str]: 选择的迷宫文件名（不含扩展名）
        """
        mazes = MazeEnvironment.list_available_mazes(self.mazes_dir)
        if not mazes:
            return None

        while True:
            try:
                choice = input("请选择迷宫编号 (或输入 'q' 退出): ").strip().lower()
                if choice == 'q':
                    return None

                idx = int(choice) - 1
                if 0 <= idx < len(mazes):
                    return mazes[idx]
                else:
                    print(f"无效选择，请输入 1-{len(mazes)} 之间的数字。")

            except ValueError:
                print("请输入有效的数字或 'q' 退出。")

    def load_maze(self, maze_name: str) -> bool:
        """
        加载指定的迷宫

        Args:
            maze_name: 迷宫文件名（不含扩展名）

        Returns:
            bool: 是否加载成功
        """
        maze_path = Path(self.mazes_dir) / f"{maze_name}.json"
        try:
            self.env = MazeEnvironment(str(maze_path))
            print(f"已加载迷宫: {maze_name}")
            return True
        except Exception as e:
            print(f"加载迷宫失败: {e}")
            return False

    def print_help(self) -> None:
        """打印帮助信息"""
        print("\n=== 迷宫沙盒帮助 ===")
        print("移动命令:")
        print("  w 或 up    - 向上移动")
        print("  s 或 down  - 向下移动")
        print("  a 或 left  - 向左移动")
        print("  d 或 right - 向右移动")
        print("其他命令:")
        print("  h 或 help  - 显示此帮助")
        print("  r 或 reset - 重置迷宫")
        print("  i 或 info  - 显示迷宫信息")
        print("  p 或 path  - 显示最短路径")
        print("  q 或 quit  - 退出游戏")
        print("==================\n")

    def print_status(self, message: str = "") -> None:
        """
        打印当前状态

        Args:
            message: 额外消息
        """
        if self.env is None:
            return

        print("\n" + "="*50)
        if message:
            print(message)

        # 显示迷宫
        print("\n当前迷宫:")
        print(self.env.render_ascii())

        # 显示状态信息
        info = self.env.get_info()
        print("状态信息:")
        print(f"  当前位置: {info['current_position']}")
        print(f"  终点位置: {info['goal']}")
        print(f"  已走步数: {info['steps']}")
        print(f"  可用动作: {', '.join(info['available_actions'])}")

        if info['done']:
            print("  🎉 恭喜！你到达了终点！")
        print("="*50)

    def handle_command(self, cmd: str) -> bool:
        """
        处理用户命令

        Args:
            cmd: 用户输入的命令

        Returns:
            bool: 是否继续游戏
        """
        cmd = cmd.strip().lower()

        # 移动命令映射
        action_map = {
            'w': 'up', 'up': 'up',
            's': 'down', 'down': 'down',
            'a': 'left', 'left': 'left',
            'd': 'right', 'right': 'right'
        }

        if cmd in action_map:
            # 执行移动动作
            action = action_map[cmd]
            state, info = self.env.step(action)

            if 'error' in info:
                self.print_status(f"❌ {info['error']}")
            elif 'success' in info and info['success']:
                if 'message' in info:
                    self.print_status(f"✅ {info['message']}")
                else:
                    self.print_status(f"移动成功！当前位置: {state.position}")
            else:
                self.print_status("移动执行完毕")

        elif cmd in ['h', 'help']:
            self.print_help()
            return True

        elif cmd in ['r', 'reset']:
            self.env.reset()
            self.print_status("迷宫已重置！")

        elif cmd in ['i', 'info']:
            info = self.env.get_info()
            print("\n=== 迷宫信息 ===")
            print(f"大小: {info['size']}")
            print(f"起点: {info['start']}")
            print(f"终点: {info['goal']}")
            print(f"当前位置: {info['current_position']}")
            print(f"步数: {info['steps']}")
            print(f"状态: {'已完成' if info['done'] else '进行中'}")
            print("===============\n")

        elif cmd in ['p', 'path']:
            print("\n显示最短路径:")
            print(self.env.render_ascii(show_path=True))
            print()

        elif cmd in ['q', 'quit']:
            print("感谢游玩！再见！")
            return False

        else:
            print(f"未知命令: {cmd}")
            print("输入 'h' 或 'help' 查看可用命令")

        return True

    def run(self) -> None:
        """运行CLI界面"""
        print("🎮 欢迎来到 MazeBench 沙盒模式！")
        print("在这里你可以亲自探索迷宫，测试你的寻路技能。")
        print()

        # 列出迷宫
        self.list_mazes()

        # 选择迷宫
        maze_name = self.select_maze()
        if maze_name is None:
            return

        # 加载迷宫
        if not self.load_maze(maze_name):
            return

        # 显示初始状态
        self.print_help()
        self.print_status("游戏开始！使用 WASD 或方向键移动，输入 'h' 查看帮助。")

        # 主游戏循环
        try:
            while True:
                cmd = input("请输入命令: ").strip()
                if not self.handle_command(cmd):
                    break

        except KeyboardInterrupt:
            print("\n\n游戏被中断。感谢游玩！")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="MazeBench Sandbox - CLI模式")
    parser.add_argument(
        "--mazes-dir",
        default="mazes",
        help="迷宫文件目录 (默认: mazes)"
    )

    args = parser.parse_args()

    # 检查目录是否存在
    if not Path(args.mazes_dir).exists():
        print(f"错误: 迷宫目录不存在: {args.mazes_dir}")
        print("请先运行主程序生成迷宫，或使用 --mazes-dir 指定正确的目录。")
        sys.exit(1)

    # 启动CLI
    cli = MazeCLI(args.mazes_dir)
    cli.run()


if __name__ == "__main__":
    main()
