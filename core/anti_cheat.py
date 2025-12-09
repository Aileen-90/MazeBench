# core/anti_cheat.py
import random
import re
from typing import Dict, List

class AntiCheat:
    """防作弊系统，扰动输入和清理输出"""

    def __init__(self, seed: int):
        random.seed(seed)

    def perturb_input(self, maze: Dict) -> Dict:
        """
        扰动迷宫输入，防止模型记住标准模式
        返回扰动后的迷宫副本
        """
        perturbed = maze.copy()

        # 随机扰动网格（小概率翻转一些可通行的墙壁）
        if random.random() < 0.1:  # 10%概率进行扰动
            grid = [row[:] for row in maze['grid']]  # 深拷贝
            height, width = len(grid), len(grid[0])

            # 随机选择一些位置进行扰动
            perturb_count = random.randint(1, 3)
            for _ in range(perturb_count):
                x = random.randint(1, height - 2)
                y = random.randint(1, width - 2)

                # 只扰动非起点终点的墙壁
                if (x, y) not in [maze['start'], maze['goal']]:
                    grid[x][y] = 1 - grid[x][y]  # 翻转0/1

            perturbed['grid'] = grid

        return perturbed

    def sandbox_output(self, text: str) -> str:
        """
        清理模型输出，移除潜在的作弊信息
        """
        # 移除可能的系统提示泄露
        cleaned = text

        # 移除可能的坐标提示
        patterns_to_remove = [
            r'\b(start|end|goal|target)\b.*?\n',
            r'坐标.*?\n',
            r'位置.*?\n',
        ]

        for pattern in patterns_to_remove:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)

        return cleaned.strip()
