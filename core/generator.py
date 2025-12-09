# core/generator.py
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Set, Callable
import numpy as np
from .config import MazeConfig, Coord

class MazeGenerator:
    """迷宫生成器基类"""
    def __init__(self, cfg: MazeConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        # 可扩展算法注册表：处理器接受 (grid, start, goal)
        self._algo_map: Dict[str, Callable[[np.ndarray, Coord, Coord], None]] = {
            'dfs': self._apply_dfs,
            'prim': self._apply_prim,
        }

    def register_algo(self, name: str, handler: Callable[[np.ndarray, Coord, Coord], None]):
        """注册新的迷宫雕刻算法。
        Handler 签名: handler(grid: np.ndarray, start: Coord, goal: Coord) -> None
        """
        self._algo_map[name] = handler

    def _in_bounds(self, r: int, c: int, grid: np.ndarray) -> bool:
        h, w = grid.shape
        return 0 <= r < h and 0 <= c < w

    def _free_neighbors(self, r: int, c: int, grid: np.ndarray) -> List[Coord]:
        res = []
        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nr, nc = r + dr, c + dc
            if self._in_bounds(nr, nc, grid) and grid[nr, nc] == 0:
                res.append((nr, nc))
        return res

    def _shortest_path(self, grid: np.ndarray, start: Coord, goal: Coord) -> List[Coord]:
        """使用BFS计算最短路径"""
        from collections import deque
        q = deque([start])
        prev = {start: None}
        while q:
            r, c = q.popleft()
            if (r, c) == goal:
                break
            for nr, nc in self._free_neighbors(r, c, grid):
                if (nr, nc) not in prev:
                    prev[(nr, nc)] = (r, c)
                    q.append((nr, nc))
        if goal not in prev:
            return []
        path = []
        cur = goal
        while cur is not None:
            path.append(cur)
            cur = prev[cur]
        return list(reversed(path))

    # ====== DFS MAZE (stride-2 recursive backtracker to create walls) ======
    def _dfs_maze(self, grid: np.ndarray) -> None:
        h, w = grid.shape
        visited = np.zeros((h, w), dtype=bool)
        stack: List[Coord] = []
        # 使用步长为2的格点，这样中间的单元格充当墙壁
        start_r, start_c = 0, 0
        stack.append((start_r, start_c))
        grid[start_r, start_c] = 0
        visited[start_r, start_c] = True
        directions = [(2, 0), (-2, 0), (0, 2), (0, -2)]
        while stack:
            r, c = stack[-1]
            unvisited = []
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc]:
                    unvisited.append((nr, nc, dr, dc))
            if unvisited:
                idx = int(self.rng.integers(0, len(unvisited)))
                nr, nc, dr, dc = unvisited[idx]
                # 雕刻通往邻居的通道和之间的墙壁
                grid[r + dr//2, c + dc//2] = 0
                grid[nr, nc] = 0
                visited[nr, nc] = True
                stack.append((nr, nc))
            else:
                stack.pop()

    def _apply_dfs(self, grid: np.ndarray, start: Coord, goal: Coord) -> None:
        # 将网格清空为墙壁，然后使用步长为2的DFS雕刻
        grid.fill(1)
        self._dfs_maze(grid)
        grid[start] = 0
        grid[goal] = 0

    def _carve_prim_tree(self, grid: np.ndarray, start: Coord) -> None:
        # 步长为2的Prim算法：在步长为2的格点上雕刻中间的墙壁
        carved: Set[Coord] = set()
        grid[start] = 0
        carved.add(start)
        frontier: Set[Coord] = set()
        directions = [(2, 0), (-2, 0), (0, 2), (0, -2)]
        for dr, dc in directions:
            nr, nc = start[0] + dr, start[1] + dc
            if self._in_bounds(nr, nc, grid):
                frontier.add((nr, nc))
        h, w = grid.shape
        max_iters = h * w * 8
        iters = 0
        while frontier and iters < max_iters:
            iters += 1
            cell = list(frontier)[int(self.rng.integers(0, len(frontier)))]
            frontier.discard(cell)
            # 在已雕刻的邻居中查找（步长为2）
            frn = []
            for dr, dc in directions:
                nr, nc = cell[0] + dr, cell[1] + dc
                if self._in_bounds(nr, nc, grid) and (nr, nc) in carved:
                    frn.append((nr, nc, dr, dc))
            if frn:
                nr, nc, dr, dc = frn[int(self.rng.integers(0, len(frn)))]
                # 雕刻之间的墙壁和单元格
                grid[cell[0] + dr//2, cell[1] + dc//2] = 0
                grid[cell[0], cell[1]] = 0
                carved.add(cell)
                # 添加新的前沿邻居
                for dr2, dc2 in directions:
                    xr, xc = cell[0] + dr2, cell[1] + dc2
                    if self._in_bounds(xr, xc, grid) and (xr, xc) not in carved and (xr, xc) not in frontier:
                        frontier.add((xr, xc))

    def _apply_prim(self, grid: np.ndarray, start: Coord, goal: Coord) -> None:
        grid.fill(1)
        self._carve_prim_tree(grid, start)
        grid[start] = 0
        grid[goal] = 0

    def generate(self, start: Optional[Coord] = None, goal: Optional[Coord] = None) -> Dict:
        """生成迷宫"""
        h, w = self.cfg.height, self.cfg.width
        if start is None or goal is None:
            if self.cfg.start_goal == 'random':
                start = (int(self.rng.integers(0, h)), int(self.rng.integers(0, w)))
                while True:
                    goal = (int(self.rng.integers(0, h)), int(self.rng.integers(0, w)))
                    if goal != start:
                        break
            else:
                start = (0, 0)
                goal = (h - 1, w - 1)
        # 为步长为2的雕刻调整到偶数坐标，以确保连通性
        if self.cfg.algorithm in ('dfs', 'prim'):
            start = (start[0] - start[0] % 2, start[1] - start[1] % 2)
            goal = (goal[0] - goal[0] % 2, goal[1] - goal[1] % 2)
        grid = np.ones((h, w), dtype=np.int8)

        handler = self._algo_map.get(self.cfg.algorithm, self._algo_map['dfs'])
        handler(grid, start, goal)

        # 最终安全检查：确保起点和终点是开放的（以防算法遗漏）
        grid[start] = 0
        grid[goal] = 0

        sp = self._shortest_path(grid, start, goal)
        return {
            'width': w,
            'height': h,
            'grid': grid.tolist(),
            'start': start,
            'goal': goal,
            'shortest_path': sp,
            'nonce': int(self.cfg.seed) if self.cfg.seed is not None else 0,
        }


class TextMazeGenerator(MazeGenerator):
    """文本迷宫生成器"""
    pass
