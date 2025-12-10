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
    position: Tuple[int, int]  # 当前位置 (row, col)
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
        row, col = self.state.position
        if action == 'up':
            new_pos = (row - 1, col)
        elif action == 'down':
            new_pos = (row + 1, col)
        elif action == 'left':
            new_pos = (row, col - 1)
        elif action == 'right':
            new_pos = (row, col + 1)
        else:
            return self.state, {'error': f'无效动作: {action}', 'valid_actions': ['up', 'down', 'left', 'right']}

        # 检查边界
        height, width = self.grid.shape
        new_row, new_col = new_pos
        if not (0 <= new_row < height and 0 <= new_col < width):
            return self.state, {'error': '撞墙了！超出边界', 'position': self.state.position}

        # 检查是否是墙壁
        if self.grid[new_row, new_col] == 1:
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
        沿着路径移动，找到最后一个合法位置

        Args:
            path: 坐标路径列表 [(row1, col1), (row2, col2), ...]

        Returns:
            Tuple[MazeState, Dict]: (新状态, 信息字典)
        """
        if self.state.done:
            return self.state, {'info': '游戏已结束'}

        if not path:
            return self.state, {'error': '路径为空'}

        # 找到最后一个合法位置
        valid_position = self.state.position
        last_valid_index = -1

        for i, pos in enumerate(path):
            row, col = pos

            # 检查边界
            height, width = self.grid.shape
            if not (0 <= row < height and 0 <= col < width):
                break  # 超出边界，停止

            # 检查是否是墙壁
            if self.grid[row, col] == 1:
                break  # 撞墙，停止

            valid_position = pos
            last_valid_index = i

        # 如果没有移动到任何新位置
        if valid_position == self.state.position:
            return self.state, {'error': '无法移动到路径中的任何位置', 'position': self.state.position}

        # 计算移动的步数（到最后一个合法位置）
        steps_moved = last_valid_index + 1

        # 更新状态
        self.state.position = valid_position
        self.state.steps += steps_moved

        # 检查是否到达终点
        if valid_position == self.goal:
            self.state.done = True
            return self.state, {'success': True, 'message': f'恭喜！到达终点！总共走了{self.state.steps}步', 'steps_moved': steps_moved}

        return self.state, {'success': True, 'position': valid_position, 'steps_moved': steps_moved}

    def get_available_actions(self) -> List[str]:
        """
        获取当前位置可用的动作

        Returns:
            List[str]: 可用的动作列表
        """
        if self.state.done:
            return []

        actions = []
        row, col = self.state.position
        height, width = self.grid.shape

        # 检查每个方向
        directions = [
            ('up', (-1, 0)),
            ('down', (1, 0)),
            ('left', (0, -1)),
            ('right', (0, 1))
        ]

        for action, (dr, dc) in directions:
            new_row, new_col = row + dr, col + dc
            if (0 <= new_row < height and 0 <= new_col < width and
                self.grid[new_row, new_col] == 0):
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

        for i in range(height):
            line = ""
            for j in range(width):
                if (i, j) == self.state.position:
                    line += symbols['agent']  # Agent当前位置
                elif (i, j) == self.start:
                    line += symbols['start']  # 起点
                elif (i, j) == self.goal:
                    line += symbols['goal']  # 终点
                elif show_path and self.maze_data.get('shortest_path') and (i, j) in [tuple(p) for p in self.maze_data['shortest_path']]:
                    line += "*"  # 最短路径
                elif self.grid[i, j] == 1:
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

        # 设置起点
        if self.start:
            tensor[self.start] = symbols['start']

        # 设置终点
        if self.goal:
            tensor[self.goal] = symbols['goal']

        # 设置agent位置（覆盖其他标记）
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
