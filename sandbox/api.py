# mazebench/sandbox/api.py
"""
MazeBench Sandbox API服务器

提供RESTful API接口，让AI模型可以与迷宫环境交互。
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any
from flask import Flask, request, jsonify
import json

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from .env import MazeEnvironment


class MazeAPIServer:
    """迷宫API服务器"""

    def __init__(self, mazes_dir: str = "mazes", host: str = "127.0.0.1", port: int = 5000):
        """
        初始化API服务器

        Args:
            mazes_dir: 迷宫文件目录
            host: 服务器主机地址
            port: 服务器端口
        """
        self.mazes_dir = mazes_dir
        self.host = host
        self.port = port
        self.env: Optional[MazeEnvironment] = None
        self.app = Flask(__name__)

        # 设置路由
        self._setup_routes()

    def _setup_routes(self):
        """设置API路由"""

        @self.app.route('/health', methods=['GET'])
        def health():
            """健康检查"""
            return jsonify({"status": "healthy", "service": "MazeBench Sandbox API"})

        @self.app.route('/mazes', methods=['GET'])
        def list_mazes():
            """列出所有可用迷宫"""
            try:
                mazes = MazeEnvironment.list_available_mazes(self.mazes_dir)
                return jsonify({
                    "mazes": mazes,
                    "count": len(mazes)
                })
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self.app.route('/load', methods=['POST'])
        def load_maze():
            """加载指定的迷宫"""
            try:
                data = request.get_json()
                if not data or 'maze_name' not in data:
                    return jsonify({"error": "Missing 'maze_name' in request body"}), 400

                maze_name = data['maze_name']
                maze_path = Path(self.mazes_dir) / f"{maze_name}.json"

                self.env = MazeEnvironment(str(maze_path))

                return jsonify({
                    "message": f"Maze '{maze_name}' loaded successfully",
                    "maze_info": self.env.get_info()
                })

            except FileNotFoundError:
                return jsonify({"error": f"Maze '{maze_name}' not found"}), 404
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self.app.route('/reset', methods=['POST'])
        def reset():
            """重置当前环境"""
            try:
                if self.env is None:
                    return jsonify({"error": "No maze loaded. Use /load first"}), 400

                state = self.env.reset()
                return jsonify({
                    "message": "Environment reset",
                    "state": {
                        "position": state.position,
                        "done": state.done,
                        "steps": state.steps
                    },
                    "info": self.env.get_info()
                })

            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self.app.route('/step', methods=['POST'])
        def step():
            """执行一步动作"""
            try:
                if self.env is None:
                    return jsonify({"error": "No maze loaded. Use /load first"}), 400

                data = request.get_json()
                if not data or 'action' not in data:
                    return jsonify({"error": "Missing 'action' in request body"}), 400

                action = data['action']
                if action not in ['up', 'down', 'left', 'right']:
                    return jsonify({
                        "error": f"Invalid action: {action}",
                        "valid_actions": ['up', 'down', 'left', 'right']
                    }), 400

                state, info = self.env.step(action)

                response = {
                    "action": action,
                    "state": {
                        "position": state.position,
                        "done": state.done,
                        "steps": state.steps
                    },
                    "info": info
                }

                # 添加奖励信息（可选，用于强化学习）
                if 'error' in info:
                    response["reward"] = -1  # 撞墙惩罚
                    response["success"] = False
                elif state.done:
                    response["reward"] = 100  # 到达终点奖励
                    response["success"] = True
                else:
                    response["reward"] = -0.1  # 每步小惩罚
                    response["success"] = True

                return jsonify(response)

            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self.app.route('/state', methods=['GET'])
        def get_state():
            """获取当前状态"""
            try:
                if self.env is None:
                    return jsonify({"error": "No maze loaded. Use /load first"}), 400

                info = self.env.get_info()
                return jsonify({
                    "state": {
                        "position": info["current_position"],
                        "done": info["done"],
                        "steps": info["steps"]
                    },
                    "available_actions": info["available_actions"]
                })

            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self.app.route('/info', methods=['GET'])
        def get_info():
            """获取迷宫信息"""
            try:
                if self.env is None:
                    return jsonify({"error": "No maze loaded. Use /load first"}), 400

                return jsonify(self.env.get_info())

            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self.app.route('/render', methods=['GET'])
        def render():
            """获取迷宫的ASCII渲染"""
            try:
                if self.env is None:
                    return jsonify({"error": "No maze loaded. Use /load first"}), 400

                show_path = request.args.get('show_path', 'false').lower() == 'true'
                ascii_maze = self.env.render_ascii(show_path=show_path)

                return jsonify({
                    "render": ascii_maze,
                    "show_path": show_path
                })

            except Exception as e:
                return jsonify({"error": str(e)}), 500

    def run(self, debug: bool = False):
        """
        启动API服务器

        Args:
            debug: 是否启用调试模式
        """
        print(f"🚀 启动 MazeBench Sandbox API 服务器")
        print(f"📍 服务地址: http://{self.host}:{self.port}")
       
        self.app.run(host=self.host, port=self.port, debug=debug)


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="MazeBench Sandbox - API模式")
    parser.add_argument(
        "--mazes-dir",
        default="mazes",
        help="迷宫文件目录 (默认: mazes)"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="服务器主机地址 (默认: 127.0.0.1)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="服务器端口 (默认: 5000)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="启用调试模式"
    )

    args = parser.parse_args()

    # 检查目录是否存在
    if not Path(args.mazes_dir).exists():
        print(f"错误: 迷宫目录不存在: {args.mazes_dir}")
        print("请先运行主程序生成迷宫，或使用 --mazes-dir 指定正确的目录。")
        sys.exit(1)

    # 检查Flask是否安装
    try:
        import flask
    except ImportError:
        print("错误: 需要安装Flask。运行: pip install flask")
        sys.exit(1)

    # 启动API服务器
    server = MazeAPIServer(args.mazes_dir, args.host, args.port)
    server.run(args.debug)


if __name__ == "__main__":
    main()
