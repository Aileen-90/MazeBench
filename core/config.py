# core/config.py
from dataclasses import dataclass
from typing import Optional

Coord = tuple[int, int]

@dataclass
class MazeConfig:
    """迷宫生成配置"""
    width: int
    height: int
    seed: Optional[int] = None
    start_goal: str = 'corner'  # 'corner' or 'random'
    algorithm: str = 'dfs'  # 'dfs' or 'prim'

@dataclass
class TextMazeConfig(MazeConfig):
    """文本迷宫配置"""
    pass

@dataclass
class ImageMazeConfig(MazeConfig):
    """图片迷宫配置"""
    cell_px: int = 24
