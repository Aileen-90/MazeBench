# /home/aileen/MazeBenchmark/sandbox/spatial_memory.py
"""
Spatial Memory Map - Advanced spatial memory system for maze navigation

Provides spatial understanding, position mapping, feature extraction,
and intelligent path planning based on accumulated spatial knowledge.
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Set, Optional
import time
from collections import defaultdict, deque
import json


class SpatialMemoryMap:
    """
    Advanced spatial memory system that builds a comprehensive map of the environment
    including position observations, connectivity, features, and navigation patterns.
    """
    
    def __init__(self, grid_height: int, grid_width: int, max_memory_size: int = 1000):
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.max_memory_size = max_memory_size
        
        # Core spatial data structures
        self.position_observations = {}  # {(y, x): {'observation': str, 'timestamp': int, 'visit_count': int, 'features': dict}}
        self.position_connections = {}   # {(y, x): {'neighbors': set(), 'reachable': bool, 'distance_from_start': int, 'access_directions': set()}}
        self.position_features = {}      # {(y, x): {'is_corner': bool, 'is_corridor': bool, 'is_intersection': bool, 'wall_count': int}}
        
        # Advanced spatial reasoning
        self.regions = {}               # {region_id: {'positions': set(), 'center': (y, x), 'connections': set(), 'type': str}}
        self.landmarks = {}             # {landmark_id: {'position': (y, x), 'type': str, 'significance': float}}
        
        # Navigation patterns
        self.successful_paths = []      # [{'path': [(y, x), ...], 'efficiency': float, 'timestamp': int, 'context': dict}]
        self.failed_attempts = []       # [{'position': (y, x), 'action': str, 'reason': str, 'timestamp': int}]
        


        # 岔路口记忆和回溯功能
        self.intersection_memory = {}   # {intersection_pos: {'visited_directions': set(), 'unvisited_directions': set(), 'is_dead_end': bool, 'path_from_intersection': []}}
        self.current_path_from_intersection = []  # 当前从岔路口出发的路径
        self.last_intersection = None   # 上一个岔路口位置
        self.intersection_stack = []    # 岔路口栈，用于回溯
        
        # Spatial learning
        self.position_values = {}       # {(y, x): {'exploration_value': float, 'strategic_value': float, 'danger_level': float}}
        self.action_outcomes = defaultdict(list)  # {(position, action): [{'success': bool, 'new_position': (y, x), 'timestamp': int}]}
        
        # System state
        self.start_position = None
        self.current_position = None
        self.exploration_frontier = set()  # Positions adjacent to explored areas
        self.known_dead_ends = set()
        
    def update_observation(self, position: Tuple[int, int], observation: str, 
                          visible_positions: Set[Tuple[int, int]], grid: np.ndarray):
        """Update spatial memory with new observation"""
        timestamp = int(time.time())
        
        # Update position observation
        if position not in self.position_observations:
            self.position_observations[position] = {
                'observation': observation,
                'timestamp': timestamp,
                'visit_count': 1,
                'features': {}
            }
        else:
            self.position_observations[position]['visit_count'] += 1
            self.position_observations[position]['timestamp'] = timestamp
        
        # Update current position
        self.current_position = position
        if self.start_position is None:
            self.start_position = position
            
        # Analyze local connectivity and features
        self._analyze_local_connectivity(position, visible_positions, grid)
        self._extract_spatial_features(position, visible_positions, grid)
        self._update_exploration_frontier(visible_positions)
        
        # Maintain memory size
        self._maintain_memory_size()
        
    def _analyze_local_connectivity(self, position: Tuple[int, int], 
                                   visible_positions: Set[Tuple[int, int]], grid: np.ndarray):
        """Analyze connectivity patterns around a position"""
        y, x = position
        neighbors = set()
        access_directions = set()
        
        # Check all four directions
        directions = [('up', -1, 0), ('down', 1, 0), ('left', 0, -1), ('right', 0, 1)]
        
        for direction, dy, dx in directions:
            ny, nx = y + dy, x + dx
            if (0 <= ny < self.grid_height and 0 <= nx < self.grid_width and
                (ny, nx) in visible_positions and grid[ny, nx] == 0):
                neighbors.add((ny, nx))
                access_directions.add(direction)
        
        # Update connections
        self.position_connections[position] = {
            'neighbors': neighbors,
            'reachable': len(neighbors) > 0,
            'access_directions': access_directions,
            'distance_from_start': self._calculate_distance_from_start(position)
        }
        
        # Build bidirectional connections
        for neighbor in neighbors:
            if neighbor not in self.position_connections:
                self.position_connections[neighbor] = {'neighbors': set(), 'reachable': True}
            self.position_connections[neighbor]['neighbors'].add(position)
    
    def _extract_spatial_features(self, position: Tuple[int, int], 
                                 visible_positions: Set[Tuple[int, int]], grid: np.ndarray):
        """Extract spatial features for a position"""
        y, x = position
        
        # Count adjacent walls
        wall_count = 0
        open_directions = 0
        
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            if (0 <= ny < self.grid_height and 0 <= nx < self.grid_width):
                if grid[ny, nx] == 1:  # Wall
                    wall_count += 1
                elif (ny, nx) in visible_positions:  # Open and visible
                    open_directions += 1
        
        # Classify position type
        features = {
            'wall_count': wall_count,
            'open_directions': open_directions,
            'is_corner': wall_count >= 2 and open_directions <= 2,
            'is_corridor': wall_count >= 2 and open_directions == 2,
            'is_intersection': open_directions >= 3,
            'is_dead_end': wall_count >= 3 and open_directions == 1,
            'is_open_space': wall_count <= 1 and open_directions >= 3
        }
        
        self.position_features[position] = features
        
        # Identify landmarks
        if features['is_intersection'] or features['is_dead_end']:
            self._add_landmark(position, 'significant_junction' if features['is_intersection'] else 'dead_end')
            
        # 检测岔路口并初始化记忆
        if features['is_intersection']:
            access_directions = self.position_connections.get(position, {}).get('access_directions', set())
            if access_directions:
                self._record_intersection(position, access_directions)
    
    def _update_exploration_frontier(self, visible_positions: Set[Tuple[int, int]]):
        """Update the exploration frontier - positions adjacent to explored areas"""
        self.exploration_frontier = set()
        
        for position in self.position_observations.keys():
            y, x = position
            
            # Check adjacent positions
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                adjacent_pos = (ny, nx)
                
                if (0 <= ny < self.grid_height and 0 <= nx < self.grid_width and
                    adjacent_pos not in self.position_observations and
                    adjacent_pos in visible_positions):
                    self.exploration_frontier.add(adjacent_pos)
    
    def record_action_outcome(self, position: Tuple[int, int], action: str, 
                             success: bool, new_position: Optional[Tuple[int, int]], 
                             reason: str = ""):
        """Record the outcome of an action from a specific position"""
        timestamp = int(time.time())
        
        outcome = {
            'success': success,
            'new_position': new_position,
            'timestamp': timestamp,
            'reason': reason
        }
        
        self.action_outcomes[(position, action)].append(outcome)
        
        # Update position values based on outcomes
        self._update_position_values(position, action, success, reason)
        
        # Record failed attempts
        if not success:
            self.failed_attempts.append({
                'position': position,
                'action': action,
                'reason': reason,
                'timestamp': timestamp
            })
            
            # Identify dead ends
            if 'wall' in reason.lower() or 'collision' in reason.lower():
                self.known_dead_ends.add(position)
    
    def _update_position_values(self, position: Tuple[int, int], action: str, 
                               success: bool, reason: str):
        """Update position values based on action outcomes"""
        if position not in self.position_values:
            self.position_values[position] = {
                'exploration_value': 0.0,
                'strategic_value': 0.0,
                'danger_level': 0.0
            }
        
        values = self.position_values[position]
        
        if success:
            # Successful actions increase strategic value
            values['strategic_value'] = min(1.0, values['strategic_value'] + 0.1)
            values['danger_level'] = max(0.0, values['danger_level'] - 0.05)
        else:
            # Failed actions increase danger level
            values['danger_level'] = min(1.0, values['danger_level'] + 0.2)
            
            # Wall collisions reduce exploration value
            if 'wall' in reason.lower():
                values['exploration_value'] = max(0.0, values['exploration_value'] - 0.1)
    
    def get_spatial_context(self, position: Tuple[int, int]) -> Dict[str, Any]:
        """Get comprehensive spatial context for a position"""
        context = {
            'position': position,
            'observation': self.position_observations.get(position, {}),
            'connections': self.position_connections.get(position, {}),
            'features': self.position_features.get(position, {}),
            'values': self.position_values.get(position, {}),
            'nearby_landmarks': self._find_nearby_landmarks(position),
            'exploration_frontier': self._get_frontier_context(position),
            'recommended_actions': self._get_recommended_actions(position)
        }
        
        return context
    
    def _find_nearby_landmarks(self, position: Tuple[int, int], radius: int = 3) -> List[Dict[str, Any]]:
        """Find landmarks within specified radius"""
        nearby = []
        y, x = position
        
        for landmark_id, landmark_data in self.landmarks.items():
            ly, lx = landmark_data['position']
            distance = abs(ly - y) + abs(lx - x)
            
            if distance <= radius:
                nearby.append({
                    'id': landmark_id,
                    'position': (ly, lx),
                    'type': landmark_data['type'],
                    'distance': distance,
                    'direction': self._get_direction_to_position(position, (ly, lx))
                })
        
        return nearby
    
    def _get_frontier_context(self, position: Tuple[int, int]) -> Dict[str, Any]:
        """Get context about unexplored areas near the position"""
        frontier_positions = []
        y, x = position
        
        for fy, fx in self.exploration_frontier:
            distance = abs(fy - y) + abs(fx - x)
            if distance <= 5:  # Within reasonable range
                frontier_positions.append({
                    'position': (fy, fx),
                    'distance': distance,
                    'direction': self._get_direction_to_position(position, (fy, fx)),
                    'estimated_value': self._estimate_exploration_value((fy, fx))
                })
        
        return {
            'count': len(frontier_positions),
            'positions': frontier_positions,
            'closest': min(frontier_positions, key=lambda p: p['distance']) if frontier_positions else None
        }
    
    def _get_recommended_actions(self, position: Tuple[int, int]) -> List[Dict[str, Any]]:
        """Get AI-recommended actions based on spatial memory"""
        recommendations = []
        
        # 首先检查是否需要回溯
        backtrack_target, backtrack_path = self.get_backtrack_target()
        if backtrack_target and backtrack_target != position:
            # 计算到回溯目标的方向和距离
            direction = self._get_direction_to_position(position, backtrack_target)
            distance = abs(backtrack_target[0] - position[0]) + abs(backtrack_target[1] - position[1])
            
            # 添加回溯建议
            recommendations.append({
                'action': f"{direction} {min(distance, 3)}",  # 建议移动1-3步
                'position': backtrack_target,
                'confidence': 0.9,
                'reason': f"Backtrack to intersection with unexplored directions (distance: {distance})",
                'expected_value': 1.0,
                'is_backtrack': True
            })
        
        # Get connection info
        connections = self.position_connections.get(position, {})
        if not connections:
            return recommendations
        
        # 检查当前位置是否是岔路口
        intersection_context = self.get_intersection_context(position)
        if intersection_context:
            # 在岔路口，优先推荐未探索的方向
            unvisited_directions = set(intersection_context['unvisited_directions'])
        else:
            unvisited_directions = set()
        
        # Evaluate each possible direction
        for direction in connections.get('access_directions', []):
            # Calculate direction vector
            direction_vectors = {
                'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)
            }
            dy, dx = direction_vectors[direction]
            next_pos = (position[0] + dy, position[1] + dx)
            
            # 如果在岔路口且这个方向已经探索过，降低优先级
            is_visited_direction = direction in unvisited_directions and len(unvisited_directions) > 0
            
            if next_pos in self.position_observations:
                # Known position - use historical data
                visit_count = self.position_observations[next_pos]['visit_count']
                values = self.position_values.get(next_pos, {})
                
                # 如果是已访问过的方向且还有其他未探索方向，降低期望价值
                expected_value = values.get('strategic_value', 0) - values.get('danger_level', 0)
                if is_visited_direction and len(unvisited_directions) > 1:
                    expected_value *= 0.3  # 大幅降低已探索方向的优先级
                
                recommendation = {
                    'action': f"{direction} 1",
                    'position': next_pos,
                    'confidence': 0.8 if values.get('danger_level', 0) < 0.3 else 0.4,
                    'reason': f"Known position (visited {visit_count} times)",
                    'expected_value': expected_value,
                    'is_explored_direction': True
                }
            else:
                # Unknown position - estimate value
                frontier_context = self._get_frontier_context(position)
                frontier_info = next((p for p in frontier_context['positions'] if self._get_direction_to_position(position, p['position']) == direction), None)
                
                if frontier_info:
                    recommendation = {
                        'action': f"{direction} 1",
                        'position': next_pos,
                        'confidence': 0.8 if direction in unvisited_directions else 0.6,
                        'reason': "Unexplored area with high estimated value" + (" (unvisited direction)" if direction in unvisited_directions else ""),
                        'expected_value': 1.2 if direction in unvisited_directions else frontier_info['estimated_value'],
                        'is_unvisited_direction': direction in unvisited_directions
                    }
                else:
                    expected_value = 1.0 if direction in unvisited_directions else 0.5
                    recommendation = {
                        'action': f"{direction} 1",
                        'position': next_pos,
                        'confidence': 0.8 if direction in unvisited_directions else 0.3,
                        'reason': "Unknown area" + (" (unvisited direction)" if direction in unvisited_directions else ""),
                        'expected_value': expected_value,
                        'is_unvisited_direction': direction in unvisited_directions
                    }
            
            recommendations.append(recommendation)
        
        # Sort by expected value and confidence, 优先回溯和未探索方向
        recommendations.sort(key=lambda x: (x.get('is_backtrack', False), x.get('is_unvisited_direction', False), x['expected_value'], x['confidence']), reverse=True)
        return recommendations
    
    def _calculate_distance_from_start(self, position: Tuple[int, int]) -> int:
        """Calculate Manhattan distance from start position"""
        if self.start_position is None:
            return 0
        return abs(position[0] - self.start_position[0]) + abs(position[1] - self.start_position[1])
    
    def _get_direction_to_position(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> str:
        """Get direction from one position to another"""
        dy = to_pos[0] - from_pos[0]
        dx = to_pos[1] - from_pos[1]
        
        if abs(dy) > abs(dx):
            return 'down' if dy > 0 else 'up'
        else:
            return 'right' if dx > 0 else 'left'
    
    def _estimate_exploration_value(self, position: Tuple[int, int]) -> float:
        """Estimate the exploration value of an unknown position"""
        # Base exploration value
        value = 0.5
        
        # Increase value if near landmarks
        for landmark in self._find_nearby_landmarks(position, radius=2):
            if landmark['type'] == 'significant_junction':
                value += 0.2
            elif landmark['type'] == 'dead_end':
                value -= 0.1  # Dead ends are less valuable
        
        # Increase value if on exploration frontier
        if position in self.exploration_frontier:
            value += 0.1
        
        # Decrease value if near known dead ends
        if position in self.known_dead_ends:
            value -= 0.3
        
        return min(1.0, max(0.0, value))
    
    def _record_intersection(self, position: Tuple[int, int], access_directions: Set[str]):
        """记录岔路口信息"""
        if position not in self.intersection_memory:
            # 新岔路口，初始化记忆
            self.intersection_memory[position] = {
                'visited_directions': set(),
                'unvisited_directions': access_directions.copy(),
                'is_dead_end': False,
                'path_from_intersection': [],
                'timestamp': int(time.time())
            }
            # 将岔路口加入栈，用于回溯
            if position != self.start_position:  # 起始位置不入栈
                self.intersection_stack.append(position)
        
        # 更新当前岔路口
        self.last_intersection = position
        # 清空当前路径记录，准备记录从岔路口出发的新路径
        self.current_path_from_intersection = [position]
    
    def update_intersection_exploration(self, current_position: Tuple[int, int], action: str, success: bool):
        """更新岔路口探索状态"""
        if not self.last_intersection or self.last_intersection == current_position:
            return
            
        # 从动作中提取方向
        direction = self._extract_direction_from_action(action)
        if not direction:
            return
            
        intersection_data = self.intersection_memory.get(self.last_intersection)
        if not intersection_data:
            return
            
        # 记录已访问的方向
        intersection_data['visited_directions'].add(direction)
        intersection_data['unvisited_directions'].discard(direction)
        
        # 如果探索失败（撞墙），标记为死胡同方向
        if not success:
            # 这个方向是死路，但岔路口本身可能还有其他方向
            pass
        
        # 记录从岔路口出发的路径
        if current_position not in self.current_path_from_intersection:
            self.current_path_from_intersection.append(current_position)
    
    def _extract_direction_from_action(self, action: str) -> str:
        """从动作字符串中提取方向"""
        action_lower = action.lower()
        if 'up' in action_lower:
            return 'up'
        elif 'down' in action_lower:
            return 'down'
        elif 'left' in action_lower:
            return 'left'
        elif 'right' in action_lower:
            return 'right'
        return None
    
    def get_backtrack_target(self) -> Tuple[Tuple[int, int], List[Tuple[int, int]]]:
        """获取回溯目标：返回要回溯到的岔路口位置和路径"""
        if not self.intersection_stack:
            return None, []
            
        # 从栈顶开始找，找到第一个还有未探索方向的岔路口
        for i in range(len(self.intersection_stack) - 1, -1, -1):
            intersection_pos = self.intersection_stack[i]
            intersection_data = self.intersection_memory.get(intersection_pos)
            
            if intersection_data and intersection_data['unvisited_directions']:
                # 找到有未探索方向的岔路口
                return intersection_pos, intersection_data['path_from_intersection']
        
        return None, []
    
    def mark_intersection_dead_end(self, intersection_pos: Tuple[int, int]):
        """标记岔路口为死胡同（所有方向都已探索且都是死路）"""
        if intersection_pos in self.intersection_memory:
            self.intersection_memory[intersection_pos]['is_dead_end'] = True
            # 从栈中移除这个死胡同岔路口
            if intersection_pos in self.intersection_stack:
                self.intersection_stack.remove(intersection_pos)
    
    def _add_landmark(self, position: Tuple[int, int], landmark_type: str):
        """Add a landmark to spatial memory"""
        landmark_id = f"{landmark_type}_{position[0]}_{position[1]}"
        
        # Calculate significance based on type and context
        significance = 0.8 if landmark_type == 'significant_junction' else 0.6
        
        self.landmarks[landmark_id] = {
            'position': position,
            'type': landmark_type,
            'significance': significance,
            'timestamp': int(time.time())
        }
    
    def _maintain_memory_size(self):
        """Maintain memory size by removing old or less important memories"""
        if len(self.position_observations) <= self.max_memory_size:
            return
        
        # Calculate memory importance scores
        importance_scores = {}
        for pos, data in self.position_observations.items():
            score = 0.0
            
            # Higher score for frequently visited positions
            score += data['visit_count'] * 0.1
            
            # Higher score for recent positions
            time_diff = int(time.time()) - data['timestamp']
            score += max(0, 1.0 - time_diff / 3600)  # Decay over 1 hour
            
            # Higher score for landmark positions
            if pos in [landmark['position'] for landmark in self.landmarks.values()]:
                score += 0.5
            
            # Higher score for high-value positions
            if pos in self.position_values:
                values = self.position_values[pos]
                score += values.get('strategic_value', 0) * 0.3
            
            importance_scores[pos] = score
        
        # Remove positions with lowest scores
        positions_to_remove = sorted(importance_scores.items(), key=lambda x: x[1])[:10]
        
        for pos, _ in positions_to_remove:
            self.position_observations.pop(pos, None)
            self.position_connections.pop(pos, None)
            self.position_features.pop(pos, None)
            self.position_values.pop(pos, None)
    
    def get_intersection_context(self, position: Tuple[int, int]) -> Dict[str, Any]:
        """获取当前位置的岔路口上下文信息"""
        if position not in self.intersection_memory:
            return {}
            
        intersection_data = self.intersection_memory[position]
        backtrack_target, backtrack_path = self.get_backtrack_target()
        
        # 获取完整的岔路口栈信息，用于分析
        intersection_stack_info = []
        for i, pos in enumerate(self.intersection_stack):
            if pos in self.intersection_memory:
                data = self.intersection_memory[pos]
                intersection_stack_info.append({
                    'position': pos,
                    'depth': i,
                    'unvisited_count': len(data['unvisited_directions']),
                    'visited_count': len(data['visited_directions']),
                    'is_dead_end': data['is_dead_end']
                })
        
        return {
            'is_intersection': True,
            'visited_directions': list(intersection_data['visited_directions']),
            'unvisited_directions': list(intersection_data['unvisited_directions']),
            'is_dead_end': intersection_data['is_dead_end'],
            'path_from_intersection': intersection_data['path_from_intersection'],
            'can_backtrack': backtrack_target is not None,
            'backtrack_target': backtrack_target,
            'total_intersections': len(self.intersection_memory),
            'intersection_stack_size': len(self.intersection_stack),
            'intersection_stack_details': intersection_stack_info,  # 详细的栈信息
            'current_intersection_depth': intersection_stack_info[-1]['depth'] if intersection_stack_info else 0
        }
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get a summary of spatial memory contents"""
        return {
            'total_positions': len(self.position_observations),
            'total_landmarks': len(self.landmarks),
            'exploration_frontier_size': len(self.exploration_frontier),
            'known_dead_ends': len(self.known_dead_ends),
            'successful_paths': len(self.successful_paths),
            'failed_attempts': len(self.failed_attempts),
            'memory_usage': len(self.position_observations) / self.max_memory_size,
            'coverage_percentage': (len(self.position_observations) / (self.grid_height * self.grid_width)) * 100,
            'total_intersections': len(self.intersection_memory),
            'intersection_stack_size': len(self.intersection_stack),
            'backtrack_available': self.get_backtrack_target()[0] is not None
        }
    
    def export_memory_map(self) -> Dict[str, Any]:
        """Export complete spatial memory for analysis"""
        return {
            'position_observations': {str(k): v for k, v in self.position_observations.items()},
            'position_connections': {str(k): v for k, v in self.position_connections.items()},
            'position_features': {str(k): v for k, v in self.position_features.items()},
            'landmarks': self.landmarks,
            'successful_paths': self.successful_paths,
            'failed_attempts': self.failed_attempts,
            'position_values': {str(k): v for k, v in self.position_values.items()},
            'exploration_frontier': list(self.exploration_frontier),
            'known_dead_ends': list(self.known_dead_ends),
            'memory_summary': self.get_memory_summary()
        }