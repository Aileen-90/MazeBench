"""
探索区域追踪器

用于跟踪agent在迷宫中的探索进度，包括已探索区域大小和探索度指标。
"""

from typing import Tuple, Set, List


class ExplorationTracker:
    """探索区域追踪器"""
    
    def __init__(self, start_position: Tuple[int, int], max_no_exploration_steps: int = 20):
        """
        初始化探索追踪器
        
        Args:
            start_position: 起始位置 (y, x)
            max_no_exploration_steps: 连续未探索新区域的最大API调用次数
        """
        self.explored_positions: Set[Tuple[int, int]] = {start_position}  # 已探索位置集合
        self.max_no_exploration_steps = max_no_exploration_steps
        self.no_exploration_count = 0  # 连续未探索新区域的API调用次数
    
    def update(self, new_position: Tuple[int, int]) -> bool:
        """
        更新探索状态（单个位置）
        
        Args:
            new_position: 新位置 (y, x)
            
        Returns:
            bool: 是否探索了新区域
        """
        is_new = new_position not in self.explored_positions
        if is_new:
            self.explored_positions.add(new_position)
            self.no_exploration_count = 0  # 重置计数器
        else:
            self.no_exploration_count += 1
        return is_new
    
    def update_path(self, path: List[Tuple[int, int]]) -> bool:
        """
        更新探索状态（路径上的所有位置）
        
        Args:
            path: 移动路径上的所有位置列表 [(y1, x1), (y2, x2), ...]
            
        Returns:
            bool: 是否探索了新区域（路径上至少有一个新位置）
        """
        has_new = False
        for position in path:
            if position not in self.explored_positions:
                self.explored_positions.add(position)
                has_new = True
        
        if has_new:
            self.no_exploration_count = 0  # 重置计数器
        else:
            self.no_exploration_count += 1
        
        return has_new
    
    def increment_no_exploration(self) -> None:
        """增加未探索计数器（用于API调用时）"""
        self.no_exploration_count += 1
    
    def should_fail(self) -> bool:
        """检查是否应该因未探索而失败"""
        return self.no_exploration_count >= self.max_no_exploration_steps
    
    def get_explored_area(self, grid=None) -> int:
        """
        获取已探索的唯一位置数量（只统计可通行位置）
        
        Args:
            grid: 可选的迷宫网格，用于过滤墙壁位置（1=墙壁，0=可通行）
            
        Returns:
            int: 已探索的可通行位置数量
        """
        if grid is not None:
            # 只统计可通行位置（grid[y][x] == 0）
            return sum(1 for y, x in self.explored_positions 
                      if 0 <= y < len(grid) and 0 <= x < len(grid[0]) and grid[y][x] == 0)
        return len(self.explored_positions)
    
    def get_exploration_rate(self, total_reachable: int, grid=None) -> float:
        """
        计算探索度（已探索/可通行区域）
        
        Args:
            total_reachable: 可通行区域总数
            grid: 可选的迷宫网格，用于过滤墙壁位置
            
        Returns:
            float: 探索度（0.0-1.0）
        """
        explored = self.get_explored_area(grid)
        return explored / total_reachable if total_reachable > 0 else 0.0

