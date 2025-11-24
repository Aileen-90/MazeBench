from typing import List, Tuple, Dict, Any
import numpy as np

Coord = Tuple[int, int]

class Validator:
    def __init__(self, grid: List[List[int]], start: Coord, goal: Coord, shortest_path: List[Coord]):
        self.grid = np.array(grid, dtype=np.int8)
        self.start = tuple(start)
        self.goal = tuple(goal)
        self.shortest_path = [tuple(p) for p in shortest_path]

    def validate(self, path: List[Coord]) -> Dict[str, Any]:
        # Compute overlap regardless of validity
        overlap = self._overlap_f1(path)
        # Check validity
        err = self._check_validity(path)
        if err:
            return {'ok': False, 'error': f'validity_failure: {err}', 'overlap': overlap}
        # Optimality only if valid
        optimal = len(path) == len(self.shortest_path)
        return {'ok': True, 'optimal': optimal, 'overlap': overlap}

    def _check_validity(self, path: List[Coord]) -> str:
        if not path:
            return 'empty_path'
        if tuple(path[0]) != self.start:
            return 'wrong_start'
        if tuple(path[-1]) != self.goal:
            return 'wrong_goal'
        # consecutive steps must be 4-neighbors and not through walls
        for i in range(1, len(path)):
            r0, c0 = path[i-1]
            r1, c1 = path[i]
            if abs(r0-r1) + abs(c0-c1) != 1:
                return 'illegal_move'
            if self.grid[r1, c1] == 1:
                return 'wall_collision'

    def _overlap_f1(self, path: List[Coord]) -> float:
        # F1 of precision and recall over path nodes
        if not path or not self.shortest_path:
            return 0.0
        sp = set(self.shortest_path)
        inter = sum(1 for p in path if tuple(p) in sp)
        if inter == 0:
            return 0.0
        precision = inter / len(path)
        recall = inter / len(sp)
        return (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
