from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
from .traps import TrapInjector
from common.maze_generator import CommonMazeConfig, CommonMazeGenerator

@dataclass
class MazeConfig:
    width: int
    height: int
    trap_ratio: float = 0.2
    seed: Optional[int] = None

class MazeGenerator:
    def __init__(self, cfg: MazeConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.traps = TrapInjector(self.rng)
        self.core = CommonMazeGenerator(CommonMazeConfig(width=cfg.width, height=cfg.height, seed=cfg.seed))

    def generate(self) -> Dict:
        maze = self.core.generate()
        grid = np.array(maze['grid'], dtype=np.int8)
        trap_zones = self.traps.inject(grid, ratio=self.cfg.trap_ratio)
        sp = self.core._shortest_path(grid, maze['start'], maze['goal'])
        return {
            'width': maze['width'],
            'height': maze['height'],
            'grid': grid.tolist(),
            'start': maze['start'],
            'goal': maze['goal'],
            'trap_zones': trap_zones,
            'shortest_path': sp,
        }
            coords = np.argwhere(grid == 1)
            self.rng.shuffle(coords)
            for k in range(min(carves, len(coords))):
                r, c = map(int, coords[k])
                # prefer carving where orthogonal neighbors are also walls to break blocks
                neigh = 0
                for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == 1:
                        neigh += 1
                if neigh >= 2:
                    grid[r, c] = 0

    def _shortest_path(self, grid: np.ndarray, start: Coord, goal: Coord) -> List[Coord]:
        from collections import deque
        q = deque([start])
        prev = {start: None}
        while q:
            r, c = q.popleft()
            if (r, c) == goal:
                break
            for nr, nc in self._neighbors(r, c, grid):
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

if __name__ == '__main__':
    cfg = MazeConfig(width=10, height=10, trap_ratio=0.2, seed=42)
    gen = MazeGenerator(cfg)
    maze = gen.generate()
    print(json.dumps(maze)[:200])
