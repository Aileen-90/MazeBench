# core/validator.py
from typing import Dict, List, Tuple

class MazeValidator:
    """迷宫路径验证器"""

    def __init__(self, grid: List[List[int]], start: Tuple[int, int], goal: Tuple[int, int],
                 shortest_path: List[Tuple[int, int]]):
        self.grid = grid
        self.start = tuple(start) if not isinstance(start, tuple) else start  # 确保是元组
        self.goal = tuple(goal) if not isinstance(goal, tuple) else goal  # 确保是元组
        self.shortest_path = shortest_path
        self.height = len(grid)
        self.width = len(grid[0]) if grid else 0

    def validate(self, path: List[Tuple[int, int]]) -> Dict:
        """
        验证路径是否有效
        返回验证结果字典
        """
        if not path:
            return {'ok': False, 'error': 'Path is empty'}

        # 检查起点和终点
        start_pos = tuple(path[0]) if not isinstance(path[0], tuple) else path[0]
        goal_pos = tuple(path[-1]) if not isinstance(path[-1], tuple) else path[-1]

        if start_pos != self.start:
            return {'ok': False, 'error': f'Path start incorrect, expected {self.start}, got {start_pos}'}

        if goal_pos != self.goal:
            return {'ok': False, 'error': f'Path goal incorrect, expected {self.goal}, got {goal_pos}'}

        # 检查路径连通性（相邻检查）
        for i in range(len(path) - 1):
            current = tuple(path[i]) if not isinstance(path[i], tuple) else path[i]
            next_pos = tuple(path[i + 1]) if not isinstance(path[i + 1], tuple) else path[i + 1]

            # 检查是否相邻（上下左右）
            dx = abs(next_pos[0] - current[0])
            dy = abs(next_pos[1] - current[1])

            if not ((dx == 1 and dy == 0) or (dx == 0 and dy == 1)):
                return {'ok': False, 'error': f'Path disconnected: {current} -> {next_pos}'}

        # 检查路径上的点是否都在网格内且可通行
        for pos in path:
            pos_tuple = tuple(pos) if not isinstance(pos, tuple) else pos
            x, y = pos_tuple
            if not (0 <= x < self.height and 0 <= y < self.width):
                return {'ok': False, 'error': f'Path point out of bounds: {pos}'}

            if self.grid[x][y] != 0:
                return {'ok': False, 'error': f'Path point is a wall: {pos}'}

        # 检查是否有重复访问（简单循环检测）
        seen = set()
        for pos in path:
            pos_tuple = tuple(pos)  # 转换为元组以便哈希
            if pos_tuple in seen:
                return {'ok': False, 'error': f'Path has duplicate point: {pos}'}
            seen.add(pos_tuple)

        # 计算路径效率指标
        path_length = len(path) - 1  # 步数
        optimal_length = len(self.shortest_path) - 1 if self.shortest_path else 0
        
        # 计算路径相似度（重合度）
        path_similarity = self._calculate_path_similarity(path)

        return {
            'ok': True,
            'path_length': path_length,
            'optimal_length': optimal_length,
            'efficiency': optimal_length / path_length if path_length > 0 else 0,
            'path_similarity': path_similarity,  # 新增：路径相似度
            'error': None
        }

    def _calculate_path_similarity(self, model_path: List[Tuple[int, int]]) -> float:
        """计算模型路径与最短路径的相似度（重合度）"""
        if not self.shortest_path or not model_path:
            return 0.0
        
        # 转换为集合以便计算交集
        model_path_set = set(model_path)
        shortest_path_set = set(self.shortest_path)
        
        # 计算交集和并集
        intersection = model_path_set.intersection(shortest_path_set)
        union = model_path_set.union(shortest_path_set)
        
        # 使用Jaccard相似度：交集大小 / 并集大小
        if len(union) == 0:
            return 0.0
            
        jaccard_similarity = len(intersection) / len(union)
        
        # 同时考虑路径长度的比例
        model_length = len(model_path)
        optimal_length = len(self.shortest_path)
        
        # 如果模型路径过长，降低相似度
        length_penalty = min(1.0, optimal_length / model_length if model_length > 0 else 1.0)
        
        # 综合相似度：Jaccard相似度 * 长度惩罚
        final_similarity = jaccard_similarity * length_penalty
        
        return round(final_similarity, 4)