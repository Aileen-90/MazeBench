from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import numpy as np
from PIL import Image, ImageDraw

Coord = Tuple[int, int]

@dataclass
class MazeConfig:
    width: int
    height: int
    # density kept for compatibility but unused
    density: float = 0.0
    trap_ratio: float = 0.0
    seed: Optional[int] = None
    cell_px: int = 24

class MazeGenerator:
    def __init__(self, cfg: MazeConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

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
            cand = [(1,0),(-1,0),(0,1),(0,-1)]
            weights = []
            for dr, dc in cand:
                nr, nc = r+dr, c+dc
                if not self._in_bounds(nr, nc, grid):
                    weights.append(0.0)
                    continue
                base = 1.0 if grid[nr, nc] == 1 else 0.2
                md_now = abs(goal[0]-r)+abs(goal[1]-c)
                md_next = abs(goal[0]-nr)+abs(goal[1]-nc)
                toward = 1.2 if md_next < md_now else 0.8
                turn = 1.5 if (prev_dir is None or (dr, dc) != prev_dir) else 0.7
                weights.append(base * toward * turn)
            s = sum(weights)
            if s == 0:
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
            # enforce turn after long straight
            if prev_dir is not None and (dr, dc) == prev_dir:
                run_len = 1
                k = len(path)-1
                while k > 0 and (path[k][0]-path[k-1][0], path[k][1]-path[k-1][1]) == prev_dir:
                    run_len += 1
                    k -= 1
                if run_len >= 2:
                    lats = [(-prev_dir[1], prev_dir[0]), (prev_dir[1], -prev_dir[0])]
                    self.rng.shuffle(lats)
                    for lr, lc in lats:
                        tr, tc = r+lr, c+lc
                        if self._in_bounds(tr, tc, grid) and grid[tr, tc] == 1:
                            grid[tr, tc] = 0
                            r, c = tr, tc
                            path.append((r, c))
                            prev_dir = (lr, lc)
                            break
                    else:
                        grid[nr, nc] = 0
                        r, c = nr, nc
                        path.append((r, c))
                        prev_dir = (dr, dc)
                        continue
            grid[nr, nc] = 0
            r, c = nr, nc
            path.append((r, c))
            prev_dir = (dr, dc)
        if (r, c) != goal:
            cr, cc = r, c
            while (cr, cc) != goal:
                if cr != goal[0]:
                    step = int(np.sign(goal[0]-cr))
                    nr, nc = cr+step, cc
                else:
                    step = int(np.sign(goal[1]-cc))
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
        idxs = list(range(1, len(main_path)-1))
        self.rng.shuffle(idxs)
        take = max(2, int(0.25 * len(idxs)))
        for k in idxs[:take]:
            r, c = main_path[k]
            n_branches = int(self.rng.integers(1, 4))
            for _ in range(n_branches):
                br_len = int(self.rng.integers(3, 8))
                prev_dir = None
                for i in range(br_len):
                    cand = [(1,0),(-1,0),(0,1),(0,-1)]
                    self.rng.shuffle(cand)
                    for dr, dc in cand:
                        nr, nc = r+dr, c+dc
                        if not self._in_bounds(nr, nc, grid):
                            continue
                        if grid[nr, nc] == 1:
                            grid[nr, nc] = 0
                            r, c = nr, nc
                            prev_dir = (dr, dc)
                            break
                    else:
                        break

    def _disperse_walls(self, grid: np.ndarray) -> None:
        h, w = grid.shape
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

    def generate(self) -> Dict:
        w, h = self.cfg.width, self.cfg.height
        grid = np.ones((h, w), dtype=np.int8)
        start, goal = (0, 0), (h-1, w-1)
        path = self._carve_main_path(grid, start, goal)
        self._carve_branches(grid, path)
        self._disperse_walls(grid)
        sp = self._shortest_path(grid, start, goal)
        return {
            'width': w,
            'height': h,
            'grid': grid.tolist(),
            'start': start,
            'goal': goal,
            'shortest_path': sp
        }

    def render_image(self, maze: Dict) -> Image.Image:
        cell = self.cfg.cell_px
        h, w = maze['height'], maze['width']
        img = Image.new('RGB', (w*cell, h*cell), (255,255,255))
        draw = ImageDraw.Draw(img)
        for r in range(h):
            for c in range(w):
                x0, y0 = c*cell, r*cell
                x1, y1 = x0+cell-1, y0+cell-1
                if maze['grid'][r][c] == 1:
                    draw.rectangle([x0, y0, x1, y1], fill=(0,0,0))
                else:
                    draw.rectangle([x0, y0, x1, y1], outline=(200,200,200))
        sx, sy = maze['start'][1]*cell, maze['start'][0]*cell
        gx, gy = maze['goal'][1]*cell, maze['goal'][0]*cell
        draw.rectangle([sx+2, sy+2, sx+cell-3, sy+cell-3], fill=(0,255,0))
        draw.rectangle([gx+2, gy+2, gx+cell-3, gy+cell-3], fill=(255,0,0))
        return img

if __name__ == '__main__':
    cfg = MazeConfig(width=10, height=10, trap_ratio=0.0, seed=42)
    gen = MazeGenerator(cfg)
    maze = gen.generate()
    img = gen.render_image(maze)
    img.save('maze_10x10.png')
