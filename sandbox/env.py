# mazebench/sandbox/env.py
"""
MazeBench Sandbox环境类

提供迷宫环境的完整管理，包括加载、状态追踪、动作执行和重置功能。
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass


@dataclass
class MazeState:
    """迷宫状态"""
    position: Tuple[int, int]  # 当前位置 (y, x)
    done: bool = False  # 是否到达终点
    steps: int = 0  # 已走的步数


class MazeEnvironment:
    """
    迷宫环境类

    支持加载JSON格式的迷宫，管理agent状态，执行动作等。
    """

    def __init__(self, maze_path: str):
        """
        初始化迷宫环境

        Args:
            maze_path: 迷宫JSON文件路径
        """
        self.maze_path = Path(maze_path)
        self.maze_data = None
        self.grid = None
        self.start = None
        self.goal = None
        self.state = None

        self._load_maze()

    def _load_maze(self) -> None:
        """加载迷宫数据"""
        if not self.maze_path.exists():
            raise FileNotFoundError(f"迷宫文件不存在: {self.maze_path}")

        try:
            with open(self.maze_path, 'r', encoding='utf-8') as f:
                self.maze_data = json.load(f)

            # 解析迷宫数据
            self.grid = np.array(self.maze_data['grid'])
            self.start = tuple(self.maze_data['start'])
            self.goal = tuple(self.maze_data['goal'])

            # 初始化状态
            self.reset()

        except Exception as e:
            raise ValueError(f"加载迷宫失败: {e}")

    def reset(self) -> MazeState:
        """
        重置环境到初始状态

        Returns:
            MazeState: 重置后的状态
        """
        self.state = MazeState(position=self.start, done=False, steps=0)
        return self.state

    def step(self, action: str) -> Tuple[MazeState, Dict[str, Any]]:
        """
        执行一步动作

        Args:
            action: 动作 ('up', 'down', 'left', 'right')

        Returns:
            Tuple[MazeState, Dict]: (新状态, 信息字典)
        """
        if self.state.done:
            return self.state, {'info': '游戏已结束'}

        # 计算新位置
        y, x = self.state.position
        if action == 'up':
            new_pos = (y - 1, x)
        elif action == 'down':
            new_pos = (y + 1, x)
        elif action == 'left':
            new_pos = (y, x - 1)
        elif action == 'right':
            new_pos = (y, x + 1)
        else:
            return self.state, {'error': f'无效动作: {action}', 'valid_actions': ['up', 'down', 'left', 'right']}

        # 检查边界
        height, width = self.grid.shape
        new_y, new_x = new_pos
        if not (0 <= new_y < height and 0 <= new_x < width):
            return self.state, {'error': '撞墙了！超出边界', 'position': self.state.position}

        # 检查是否是墙壁
        if self.grid[new_y, new_x] == 1:
            return self.state, {'error': '撞墙了！这是墙壁', 'position': self.state.position}

        # 更新状态
        self.state.position = new_pos
        self.state.steps += 1

        # 检查是否到达终点
        if new_pos == self.goal:
            self.state.done = True
            return self.state, {'success': True, 'message': f'恭喜！到达终点！总共走了{self.state.steps}步'}

        return self.state, {'success': True, 'position': new_pos}

    def step_path(self, path: List[Tuple[int, int]]) -> Tuple[MazeState, Dict[str, Any]]:
        """
        沿着路径移动，确保路径连续且每步都合法

        Args:
            path: 坐标路径列表 [(y1, x1), (y2, x2), ...]

        Returns:
            Tuple[MazeState, Dict]: (新状态, 信息字典)
        """
        if self.state.done:
            return self.state, {'info': '游戏已结束'}

        if not path:
            return self.state, {'error': '路径为空'}

        current_position = self.state.position
        valid_steps = 0

        for pos in path:
            y, x = pos

            # 检查是否是当前相邻位置（不允许跳步）
            current_y, current_x = current_position
            y_diff = abs(y - current_y)
            x_diff = abs(x - current_x)

            # 必须是相邻移动（上下左右），不允许跳步
            if not ((y_diff == 1 and x_diff == 0) or (y_diff == 0 and x_diff == 1)):
                return self.state, {'error': f'跳步检测：从{current_position}到{pos}不是相邻移动', 'position': self.state.position}

            # 检查边界
            height, width = self.grid.shape
            if not (0 <= y < height and 0 <= x < width):
                return self.state, {'error': f'撞墙检测：位置{pos}超出边界', 'position': self.state.position}

            # 检查是否是墙壁
            if self.grid[y, x] == 1:
                return self.state, {'error': f'撞墙检测：位置{pos}是墙壁', 'position': self.state.position}

            # 更新当前位置和有效步数
            current_position = pos
            valid_steps += 1

        # 如果没有有效移动
        if valid_steps == 0:
            return self.state, {'error': '无法移动到路径中的任何位置', 'position': self.state.position}

        # 更新状态
        self.state.position = current_position
        self.state.steps += valid_steps

        # 检查是否到达终点
        if current_position == self.goal:
            self.state.done = True
            return self.state, {'success': True, 'message': f'恭喜！到达终点！总共走了{self.state.steps}步', 'steps_moved': valid_steps}

        return self.state, {'success': True, 'position': current_position, 'steps_moved': valid_steps}

    def get_available_actions(self) -> List[str]:
        """
        获取当前位置可用的动作

        Returns:
            List[str]: 可用的动作列表
        """
        if self.state.done:
            return []

        actions = []
        y, x = self.state.position
        height, width = self.grid.shape

        # 检查每个方向
        directions = [
            ('up', (-1, 0)),
            ('down', (1, 0)),
            ('left', (0, -1)),
            ('right', (0, 1))
        ]

        for action, (dy, dx) in directions:
            new_y, new_x = y + dy, x + dx
            if (0 <= new_y < height and 0 <= new_x < width and
                self.grid[new_y, new_x] == 0):
                actions.append(action)

        return actions

    def render_ascii(self, show_path: bool = False, symbols: Dict[str, str] = None) -> str:
        """
        以ASCII字符渲染迷宫

        Args:
            show_path: 是否显示最短路径
            symbols: 符号配置字典，包含 wall, path, start, goal, agent 键

        Returns:
            str: ASCII迷宫字符串
        """
        if self.maze_data is None:
            return "迷宫未加载"

        # 默认符号
        default_symbols = {
            'wall': '#',
            'path': ' ',
            'start': 'S',
            'goal': 'G',
            'agent': 'A'
        }

        # 使用提供的符号或默认符号
        if symbols is None:
            symbols = default_symbols
        else:
            symbols = {**default_symbols, **symbols}

        height, width = self.grid.shape
        lines = []

        for y in range(height):
            line = ""
            for x in range(width):
                if (y, x) == self.state.position:
                    line += symbols['agent']  # Agent当前位置
                elif (y, x) == self.start:
                    line += symbols['start']  # 起点
                elif (y, x) == self.goal:
                    line += symbols['goal']  # 终点
                elif show_path and self.maze_data.get('shortest_path') and (y, x) in [tuple(p) for p in self.maze_data['shortest_path']]:
                    line += "*"  # 最短路径
                elif self.grid[y, x] == 1:
                    line += symbols['wall']  # 墙壁
                else:
                    line += symbols['path']  # 空地
            lines.append(line)

        return "\n".join(lines)

    def render_tensor(self, symbols: Dict[str, str] = None) -> np.ndarray:
        """
        以张量格式渲染迷宫状态，使用符号表示

        Args:
            symbols: 符号配置字典，包含 wall, path, start, goal, agent 键

        Returns:
            np.ndarray: 形状为(height, width)的字符串张量
        """
        if self.maze_data is None:
            return np.array([])

        # 默认符号配置
        default_symbols = {
            'wall': '#',
            'path': ' ',
            'start': 'S',
            'goal': 'G',
            'agent': 'A'
        }

        # 使用提供的符号或默认符号
        if symbols is None:
            symbols = default_symbols
        else:
            symbols = {**default_symbols, **symbols}

        height, width = self.grid.shape
        tensor = np.full((height, width), symbols['path'], dtype='<U10')  # 使用字符串类型

        # 设置墙壁
        wall_mask = self.grid == 1
        tensor[wall_mask] = symbols['wall']

        # 设置起点 (y, x)
        if self.start:
            tensor[self.start] = symbols['start']

        # 设置终点 (y, x)
        if self.goal:
            tensor[self.goal] = symbols['goal']

        # 设置agent位置（覆盖其他标记）(y, x)
        tensor[self.state.position] = symbols['agent']

        return tensor

    def get_info(self) -> Dict[str, Any]:
        """
        获取迷宫信息

        Returns:
            Dict: 迷宫基本信息
        """
        if self.maze_data is None:
            return {}

        return {
            'size': f"{self.maze_data['height']}x{self.maze_data['width']}",
            'start': self.start,
            'goal': self.goal,
            'current_position': self.state.position,
            'steps': self.state.steps,
            'done': self.state.done,
            'available_actions': self.get_available_actions()
        }

    @staticmethod
    def list_available_mazes(mazes_dir: str = None) -> List[str]:
        """
        列出可用的迷宫文件

        Args:
            mazes_dir: 迷宫目录路径

        Returns:
            List[str]: 迷宫文件名列表
        """
        if mazes_dir is None:
            from utils.io import load_config
            cfg = load_config()
            mazes_dir = cfg.get('sandbox', {}).get('mazes_path', 'mazes/')

        mazes_path = Path(mazes_dir)
        if not mazes_path.exists():
            return []

        return [f.stem for f in mazes_path.glob("*.json") if f.is_file()]
