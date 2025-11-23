import json
from typing import List, Tuple, Dict, Optional
import numpy as np
from dataclasses import dataclass
from .traps import TrapInjector

Coord = Tuple[int, int]

@dataclass
class MazeConfig:
    width: int
    height: int
    # density is kept for config compatibility but intentionally unused
    density: float = 0.0
    trap_ratio: float = 0.2
    seed: Optional[int] = None

class MazeGenerator:
    def __init__(self, cfg: MazeConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.traps = TrapInjector(self.rng)

    def generate(self) -> Dict:
        w, h = self.cfg.width, self.cfg.height
        # Start with all walls; carve paths
        grid = np.ones((h, w), dtype=np.int8)  # 0 free, 1 wall
        start, goal = (0, 0), (h-1, w-1)
        path = self._carve_main_path(grid, start, goal)
        self._carve_branches(grid, path)
        self._disperse_walls(grid, limit_ratio=0.08)
        # ensure endpoints are free
        grid[start] = 0
        grid[goal] = 0
        # Inject traps into carved grid
        trap_zones = self.traps.inject(grid, ratio=self.cfg.trap_ratio)
        # shortest path
        sp = self._shortest_path(grid, start, goal)
        if not sp:
            # carve a minimal zigzag connection to guarantee solvability
            cr, cc = start
            while (cr, cc) != goal:
                if cr != goal[0]:
                    step = int(np.sign(goal[0]-cr))
                    nr, nc = cr+step, cc
                else:
                    step = int(np.sign(goal[1]-cc))
                    nr, nc = cr, cc+step
                if 0 <= nr < h and 0 <= nc < w:
                    grid[nr, nc] = 0
                    cr, cc = nr, nc
            sp = self._shortest_path(grid, start, goal)
        return {
            'width': w,
            'height': h,
            'grid': grid.tolist(),
            'start': start,
            'goal': goal,
            'trap_zones': trap_zones,
            'shortest_path': sp
        }

    def _in_bounds(self, r: int, c: int, grid: np.ndarray) -> bool:
        h, w = grid.shape
        return 0 <= r < h and 0 <= c < w

    def _neighbors(self, r: int, c: int, grid: np.ndarray) -> List[Coord]:
        h, w = grid.shape
        for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == 0:
                yield (nr, nc)

    def _carve_main_path(self, grid: np.ndarray, start: Coord, goal: Coord) -> List[Coord]:
        r, c = start
        grid[r, c] = 0
        path = [start]
        prev_dir = None
        h, w = grid.shape
        max_steps = h*w*4
        steps = 0
        while (r, c) != goal and steps < max_steps:
            steps += 1
            # candidate moves 4-neigh
            cand = [(1,0),(-1,0),(0,1),(0,-1)]
            # weights: prefer turning, slight bias toward reducing manhattan distance
            weights = []
            for dr, dc in cand:
                nr, nc = r+dr, c+dc
                if not self._in_bounds(nr, nc, grid):
                    weights.append(0.0)
                    continue
                # discourage carving over existing free cells too much to avoid corridors merging
                base = 1.0 if grid[nr, nc] == 1 else 0.2
                # bias toward goal if it reduces manhattan distance
                md_now = abs(goal[0]-r)+abs(goal[1]-c)
                md_next = abs(goal[0]-nr)+abs(goal[1]-nc)
                toward = 1.3 if md_next < md_now else 0.8
                # prefer turns over straight
                if prev_dir is not None and (dr, dc) == prev_dir:
                    turn = 0.6  # penalize straight continuation
                else:
                    turn = 1.6  # encourage turns
                weights.append(base * toward * turn)
            # normalize and sample
            s = sum(weights)
            if s == 0:
                # fallback: force a turn if possible
                for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
                    nr, nc = r+dr, c+dc
                    if self._in_bounds(nr, nc, grid):
                        grid[nr, nc] = 0
                        r, c = nr, nc
                        prev_dir = (dr, dc)
                        path.append((r, c))
                        break
                continue
            probs = [w/s for w in weights]
            idx = int(self.rng.choice(len(cand), p=probs))
            dr, dc = cand[idx]
            nr, nc = r+dr, c+dc
            # enforce turn after 2 straight steps if possible
            if prev_dir is not None and (dr, dc) == prev_dir:
                # check straight run length
                run_len = 1
                k = len(path)-1
                while k > 0 and (path[k][0]-path[k-1][0], path[k][1]-path[k-1][1]) == prev_dir:
                    run_len += 1
                    k -= 1
                if run_len >= 2:
                    # try lateral turn
                    lats = [(-prev_dir[1], prev_dir[0]), (prev_dir[1], -prev_dir[0])]
                    self.rng.shuffle(lats)
                    turned = False
                    for lr, lc in lats:
                        tr, tc = r+lr, c+lc
                        if self._in_bounds(tr, tc, grid) and grid[tr, tc] == 1:
                            grid[tr, tc] = 0
                            r, c = tr, tc
                            path.append((r, c))
                            prev_dir = (lr, lc)
                            turned = True
                            break
                    if turned:
                        continue
            # carve next
            grid[nr, nc] = 0
            r, c = nr, nc
            prev_dir = (dr, dc)
            path.append((r, c))
        # If not reached, force zigzag to goal
        if (r, c) != goal:
            cr, cc = r, c
            while (cr, cc) != goal:
                if cr != goal[0]:
                    step = np.sign(goal[0]-cr)
                    nr, nc = cr+step, cc
                else:
                    step = np.sign(goal[1]-cc)
                    nr, nc = cr, cc+step
                if self._in_bounds(nr, nc, grid):
                    grid[nr, nc] = 0
                    path.append((nr, nc))
                    cr, cc = nr, nc
        return path

    def _carve_branches(self, grid: np.ndarray, main_path: List[Coord]) -> None:
        h, w = grid.shape
        if len(main_path) < 4:
            return
        # choose 20-30% of main path cells excluding endpoints
        idxs = list(range(1, len(main_path)-1))
        self.rng.shuffle(idxs)
        take = max(2, int(0.25 * len(idxs)))
        for k in idxs[:take]:
            r, c = main_path[k]
            n_branches = self.rng.integers(1, 4)  # 1-3 branches
            for _ in range(n_branches):
                br_len = int(self.rng.integers(3, 8))
                # start direction: prefer turn from local direction
                prev_dir = None
                for i in range(br_len):
                    cand = [(1,0),(-1,0),(0,1),(0,-1)]
                    self.rng.shuffle(cand)
                    carved = False
                    for dr, dc in cand:
                        nr, nc = r+dr, c+dc
                        if not self._in_bounds(nr, nc, grid):
                            continue
                        if grid[nr, nc] == 1:
                            # avoid immediate back to main path neighbor except origin
                            grid[nr, nc] = 0
                            r, c = nr, nc
                            prev_dir = (dr, dc)
                            carved = True
                            break
                    if not carved:
                        break

    def _disperse_walls(self, grid: np.ndarray, limit_ratio: float = 0.1) -> None:
        # Break up large continuous 3x3 wall blocks and reduce huge clusters
        h, w = grid.shape
        # local 3x3 fixes
        for r in range(h-2):
            for c in range(w-2):
                block = grid[r:r+3, c:c+3]
                if np.all(block == 1):
                    if self.rng.random() < 0.7:
                        block[1,1] = 0
                    else:
                        i = int(self.rng.integers(0, 3))
                        j = int(self.rng.integers(0, 3))
                        block[i, j] = 0
        # global ratio: ensure walls not dominating contiguous mass
        total = h*w
        wall_count = int(np.sum(grid == 1))
        target = int(limit_ratio * total)
        if wall_count > target:
            # randomly carve some walls to reduce large blocks and create fragmentation
            carves = wall_count - target
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
