"""
Partial Observe Mode - AI Sandbox with limited visibility

Agent can only see surrounding k cells, cannot see through walls,
does not know coordinates, and moves by direction + steps.
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
import json
import time
import traceback
import numpy as np
import re

# Project path
sys.path.insert(0, str(Path(__file__).parent.parent))

from .env import MazeEnvironment
from .exploration import ExplorationTracker
from adapters import get_adapter
from utils.io import load_config, apply_env_keys


def _get_line_points(start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    使用 Bresenham 直线算法获取从起点到终点的所有中间点（不包括起点和终点）
    
    Args:
        start: 起点 (y, x)
        end: 终点 (y, x)
        
    Returns:
        List[Tuple[int, int]]: 路径上的所有中间点（不包括起点和终点）
    """
    y0, x0 = start
    y1, x1 = end
    
    # 如果起点和终点相同，返回空列表
    if y0 == y1 and x0 == x1:
        return []
    
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    
    x, y = x0, y0
    
    # 使用标准的 Bresenham 算法，但跳过起点和终点
    while True:
        # 移动到下一个点
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
        
        # 如果到达终点，停止（不包含终点）
        if x == x1 and y == y1:
            break
        
        # 添加到列表（不包含起点，因为我们在移动后才添加）
        points.append((y, x))
    
    return points


def _is_visible_with_wall_check(env: MazeEnvironment, pos: Tuple[int, int], visibility_range: int) -> bool:
    """
    判断指定位置是否在可见范围内（曼哈顿距离+直线路径墙壁阻挡检查）
    墙壁只要不被其他墙壁阻挡就可见
    使用直线路径（包括斜向）检查是否有墙壁阻挡
    
    Args:
        env: 迷宫环境实例
        pos: 目标位置 (y, x)
        visibility_range: 可见范围k（使用曼哈顿距离）
        
    Returns:
        bool: 是否可见
    """
    agent_y, agent_x = env.state.position
    target_y, target_x = pos
    
    # 计算曼哈顿距离（用于判断是否在可见范围内）
    manhattan_dist = abs(target_y - agent_y) + abs(target_x - agent_x)
    
    # 如果曼哈顿距离超过可见范围，不可见
    if manhattan_dist > visibility_range:
        return False
    
    # 如果距离为0，是自己位置，可见
    if manhattan_dist == 0:
        return True
    
    # 使用直线路径检查是否有其他墙壁阻挡
    # 获取从 agent 到目标位置的直线路径上的所有中间点（不包括起点和终点）
    line_points = _get_line_points((agent_y, agent_x), (target_y, target_x))
    
    # 检查路径上的每个点是否有墙壁阻挡
    for y, x in line_points:
        # 如果路径上有其他墙壁阻挡，目标位置不可见
        if env.grid[y, x] == 1:
            return False
    
    # 路径上没有阻挡，目标位置可见（无论目标位置本身是否是墙壁）
    return True


def render_partial_observe(env: MazeEnvironment, visibility_range: int, symbols: Dict[str, str] = None) -> np.ndarray:
    """
    只返回可见区域的子矩阵（复用 env.render_tensor 并提取可见子区域）
    
    Args:
        env: 迷宫环境实例
        visibility_range: 可见范围k
        symbols: 符号配置字典
        
    Returns:
        np.ndarray: 形状为 (visible_height, visible_width) 的numpy数组
    """
    height, width = env.grid.shape
    agent_y, agent_x = env.state.position
    
    # 找到所有可见位置（考虑墙壁阻挡）
    visible_positions = [
        (y, x) for y in range(height) for x in range(width)
        if _is_visible_with_wall_check(env, (y, x), visibility_range)
    ]
    
    if not visible_positions:
        visible_positions = [(agent_y, agent_x)]
    
    # 计算可见区域的边界
    min_y, max_y = min(p[0] for p in visible_positions), max(p[0] for p in visible_positions)
    min_x, max_x = min(p[1] for p in visible_positions), max(p[1] for p in visible_positions)
    
    # 复用 env.render_tensor 获取完整地图（env.render_tensor 已处理 symbols 默认值）
    full_tensor = env.render_tensor(symbols=symbols, visibility=-1)
    
    # 提取可见子矩阵
    visible_matrix = full_tensor[min_y:max_y+1, min_x:max_x+1].copy()
    
    # 将不可见位置标记为 masked（复用 env.render_tensor 的默认符号处理）
    default_symbols = {'masked': '?'}
    if symbols:
        default_symbols.update(symbols)
    masked_symbol = default_symbols.get('masked', '?')
    
    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            if not _is_visible_with_wall_check(env, (y, x), visibility_range):
                visible_matrix[y - min_y, x - min_x] = masked_symbol
    
    return visible_matrix


def step_direction_steps(env: MazeEnvironment, direction: str, steps: int) -> Tuple[Any, Dict[str, Any]]:
    """
    执行"方向+步数"的动作（复用 env.step）
    
    Args:
        env: 迷宫环境实例
        direction: 方向 ('up', 'down', 'left', 'right')
        steps: 步数
        
    Returns:
        Tuple[MazeState, Dict]: (新状态, 信息字典)，Dict中包含'path'键，值为移动路径
    """
    if env.state.done:
        return env.state, {'info': 'Game ended', 'path': []}
    
    if steps <= 0:
        return env.state, {'error': f'Invalid steps: {steps}, must be positive', 'path': []}
    
    if direction not in ['up', 'down', 'left', 'right']:
        return env.state, {'error': f'Invalid direction: {direction}', 'path': []}
    
    # 记录移动路径（包括起始位置）
    path = [env.state.position]  # 包含起始位置
    start_position = env.state.position
    
    # 复用 env.step 连续执行 steps 步
    valid_steps = 0
    for i in range(steps):
        state, step_info = env.step(direction)  # 复用 env.step
        
        if 'error' in step_info:
            return env.state, {
                'error': f'Wall collision after {valid_steps} steps: {step_info["error"]}',
                'steps_moved': valid_steps,
                'position': env.state.position,
                'path': path
            }
        
        # 记录路径上的位置（移动后的新位置）
        path.append(state.position)
        valid_steps = i + 1
        if state.done:
            return state, {
                'success': True,
                'steps_moved': valid_steps,
                'message': step_info.get('message', ''),
                'path': path
            }
    
    return env.state, {
        'success': True,
        'steps_moved': valid_steps,
        'position': env.state.position,
        'path': path
    }


def parse_direction_steps(response: str) -> List[Tuple[str, int]]:
    """
    解析AI返回的方向+步数字符串
    
    支持格式：
    - up 3, right 2
    - up 3
    - right 2
    - right3 (方向+数字连在一起)
    
    Args:
        response: AI返回的字符串
        
    Returns:
        List[Tuple[str, int]]: 每个元组是 (direction, steps)
    """
    response = response.strip().lower()
    
    # 方向列表
    directions = ['up', 'down', 'left', 'right']
    
    # 匹配模式1：方向 + 空格 + 数字
    pattern1 = r'\b(up|down|left|right)\s+(\d+)\b'
    matches1 = re.findall(pattern1, response)
    
    # 匹配模式2：方向 + 数字（连在一起）
    pattern2 = r'\b(up|down|left|right)(\d+)\b'
    matches2 = re.findall(pattern2, response)
    
    # 合并两种模式的匹配结果
    matches = matches1 + matches2
    
    result = []
    for direction, steps_str in matches:
        try:
            steps = int(steps_str)
            if steps > 0:
                result.append((direction, steps))
        except ValueError:
            continue
    
    return result


def run_ai_sandbox_partial_observe(maze: str, model: str = None, max_steps: int = None, cfg: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Run AI sandbox test in partial observe mode
    
    完全参考 run_ai_sandbox 的结构和代码风格
    """
    # If no config provided, load default config
    if cfg is None:
        cfg = load_config()
        apply_env_keys(cfg)

    # Default parameters
    model = model or cfg.get('model', 'gpt-4')
    max_steps = max_steps or cfg.get('sandbox', {}).get('max_steps', 50)
    memory = cfg.get('sandbox', {}).get('memory', 5)  # AI memory length

    # Read symbol config and visibility range
    symbols = cfg.get('sandbox', {}).get('symbols', {
        'wall': '█',
        'path': ' ',
        'start': 'S',
        'goal': 'G',
        'agent': 'A',
        'masked': '?'
    })
    visibility = cfg.get('sandbox', {}).get('visibility', -1)
    
    # 验证 visibility 必须为正数
    if visibility <= 0:
        return {'error': f'visibility must be positive in partial_observe mode, got: {visibility}'}
    
    # Load maze path from config
    mazes_path = cfg.get('sandbox', {}).get('mazes_path', 'mazes/')

    # Load maze - handle both full path and maze name
    maze_path = None
    maze_input = Path(maze)
    
    # If input contains path separators, treat it as a full path
    if '/' in str(maze) or '\\' in str(maze):
        # It's a path - check if it's a directory or file
        if maze_input.is_dir():
            # If it's a directory, list available mazes
            json_files = list(maze_input.glob("*.json"))
            if not json_files:
                return {'error': f'No maze files found in directory: {maze}'}
            # Use the first maze file
            maze_path = json_files[0]
        elif maze_input.exists():
            # It's an existing file
            maze_path = maze_input
        elif (maze_input.parent / f"{maze_input.name}.json").exists():
            # It's a path without .json extension
            maze_path = maze_input.parent / f"{maze_input.name}.json"
        else:
            # Try with .json extension
            maze_path = maze_input.with_suffix('.json')
    else:
        # It's just a maze name, use mazes_path from config
        maze_path = Path(mazes_path) / f"{maze}.json"
    
    if not maze_path or not maze_path.exists():
        # Provide helpful error message
        if '/' in str(maze) or '\\' in str(maze):
            return {'error': f'Maze does not exist: {maze_path}\n提示: 请检查路径是否正确，或使用迷宫文件名（如: maze_15x15_0）'}
        else:
            return {'error': f'Maze does not exist: {maze_path}\n提示: 迷宫文件应位于配置的 mazes_path 目录中: {mazes_path}'}

    env = MazeEnvironment(str(maze_path))
    env.reset()

    # Initialize exploration tracker
    max_no_exploration_steps = cfg.get('sandbox', {}).get('max_no_exploration_steps', 20)
    start_position = env.state.position
    exploration_tracker = ExplorationTracker(start_position, max_no_exploration_steps)
    
    # Calculate total reachable area (non-wall cells)
    total_reachable = int(np.sum(env.grid == 0))

    # AI adapter - directly use methods from /Adapter
    temperature = cfg.get('temperature', 0.1)
    adapter_cfg = {
        'PROVIDER': cfg.get('PROVIDER', 'openai').lower(),
        'ARK_API_KEY': cfg.get('ARK_API_KEY', ''),
        'ARK_API_BASE': cfg.get('ARK_API_BASE', ''),
        'AZURE_OPENAI_API_KEY': cfg.get('AZURE_OPENAI_API_KEY', ''),
        'AZURE_OPENAI_ENDPOINT': cfg.get('AZURE_OPENAI_ENDPOINT', ''),
        'AZURE_OPENAI_DEPLOYMENT': cfg.get('AZURE_OPENAI_DEPLOYMENT', ''),
        'AZURE_OPENAI_API_VERSION': cfg.get('AZURE_OPENAI_API_VERSION', '2023-12-01-preview'),
        'OPENAI_API_KEY': cfg.get('OPENAI_API_KEY', ''),
        'OPENAI_API_BASE': cfg.get('OPENAI_API_BASE'),
        'model': model,
        'temperature': temperature,
        'enable_thinking': cfg.get('enable_thinking', False)
    }
    adapter = get_adapter(adapter_cfg)

    # ============ Initialize statistics ============
    start_total_time = time.perf_counter()  # Use high-precision timer
    
    stats = {
        'api_calls': 0,                    # Number of API calls
        'errors': [],                      # Error list
        'action_errors': [],               # Action execution errors
        'wall_collisions': 0,              # Number of wall collisions
        'invalid_jumps': 0,                # Number of invalid jumps (illegal moves)
        'parse_errors': 0,                 # Parse errors
        'total_response_time': 0.0,        # Total response time
        'start_time': time.time(),         # Start timestamp
        'history': []                      # Detailed history
    }
    # ===========================================

    # AI test loop
    history = []
    action_memory = []  # Action memory list, each element is {'action': str, 'feedback': str}
    print(f"Starting AI test (Partial Observe Mode) - Model: {model}, Max steps: {max_steps}, Memory length: {memory}, Visibility: {visibility}")
    print(f"Maze: {maze}")
    print("-" * 50)

    for step in range(max_steps):
        if env.state.done:
            print(f"Step {step}: Task completed")
            break

        # AI decision
        try:
            info = env.get_info()
            print(f"Step {step + 1}")
            
            # 获取可见区域（复用 render_partial_observe）
            visible_matrix = render_partial_observe(env, visibility, symbols)
            maze_ascii = '\n'.join(''.join(row) for row in visible_matrix)

            # Build memory information
            memory_info = ""
            if memory != 0 and action_memory:
                if memory == -1:
                    # All memories
                    memory_actions = action_memory
                else:
                    # Recent k memories
                    memory_actions = action_memory[-memory:] if len(action_memory) > memory else action_memory

                if memory_actions:
                    memory_str = ", ".join([f"Step {i+1}: {mem['action']} ({mem['feedback']})" for i, mem in enumerate(memory_actions)])
                    memory_info = f"Recent action history: {memory_str}. "

            prompt = f"""You are a maze expert and need to navigate yourself to the goal. You can only see the surrounding {visibility} cells, and your vision cannot pass through walls. You do not know your current coordinates or the goal coordinates.

IMPORTANT HINTS:
- The goal is located in the bottom-right corner of the maze
- If you stay in already explored positions for too long without exploring new areas, you will be judged as failed

You can move by specifying direction and steps:
- Direction: up, down, left, or right
- Steps: a number between 1 and {visibility} (inclusive)
- Format: "direction steps", e.g., "up 3" or "right 2"
- You can specify multiple moves separated by commas, e.g., "up 3, right 2"

Visible maze layout (you are at position A):
{maze_ascii}
Symbol legend:
- {symbols['wall']}: Wall, cannot pass through
- {symbols['path']}: Path, can pass through
- {symbols['start']}: Start position (you have left)
- {symbols['goal']}: Goal position (target location)
- {symbols['agent']}: Current position (your location)
- {symbols['masked']}: Unknown area (not visible)
{memory_info}
Please reply with your move(s) in the format "direction steps" (e.g., "up 3" or "right 2, down 1"):
"""

            # print(f"  Prompt: {prompt}")
            print(env.render_ascii(symbols=symbols))
            print("Agent sight:")
            print(maze_ascii)
            print(memory_info)
        
            # ============ Record API call ============
            stats['api_calls'] += 1
            call_start_time = time.perf_counter()
            # ===========================================
            
            raw_response = adapter.generate(prompt)
            
            # ============ Record response time ============
            call_duration = time.perf_counter() - call_start_time
            stats['total_response_time'] += call_duration
            # ===========================================
            
            # Check exploration failure after API call
            if exploration_tracker.should_fail():
                print(f"  Exploration failed: {exploration_tracker.no_exploration_count} consecutive API calls without exploring new area")
                break
            
            duration = time.perf_counter() - call_start_time
            print(f"  Model response time: {duration:.2f} seconds")
            response = raw_response.strip().lower()

            # Parse direction + steps
            moves = parse_direction_steps(response)
            action_desc = None

            if moves:
                # Valid moves
                action_desc = f"Moves: {', '.join([f'{d} {s}' for d, s in moves])}"
                print(f"  AI decision: {action_desc}")
            else:
                print(f"  Invalid output, continuing test")
                # Record invalid feedback to memory
                action_memory.append({'action': response, 'feedback': 'Invalid output'})
                
                # ============ Record parse error ============
                stats['parse_errors'] += 1
                stats['errors'].append({
                    'step': step + 1,
                    'type': 'parse_error',
                    'response': response[:100],  # Only record first 100 characters
                    'timestamp': time.time(),
                    'description': 'AI response cannot be parsed as valid direction+steps format'
                })
                # ===========================================
                
                continue
                
        except Exception as e:
            print(f"  AI decision error: {type(e).__name__}: {e}, terminating test")
            print("=== Complete error information ===")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print("Complete stack trace:")
            traceback.print_exc()
            print("===================")
            
            # ============ Record AI decision error ============
            stats['errors'].append({
                'step': step + 1,
                'type': 'ai_decision_error',
                'error_type': type(e).__name__,
                'error_message': str(e),
                'timestamp': time.time()
            })
            # ===========================================
            
            break

        # Execute moves
        try:
            position_before = info['current_position']
            total_steps_moved = 0
            last_error = None
            all_paths = []  # 收集所有移动路径
            
            for direction, steps in moves:
                if env.state.done:
                    break
                
                state, step_info = step_direction_steps(env, direction, steps)
                
                if 'error' in step_info:
                    last_error = step_info['error']
                    total_steps_moved += step_info.get('steps_moved', 0)
                    # 即使出错，也记录已成功移动的路径
                    if 'path' in step_info and step_info['path']:
                        all_paths.extend(step_info['path'])
                    break
                
                total_steps_moved += step_info.get('steps_moved', steps)
                # 收集路径
                if 'path' in step_info and step_info['path']:
                    all_paths.extend(step_info['path'])
                
                if state.done:
                    break
            
            if action_desc is None:
                action_desc = f"Moves: {', '.join([f'{d} {s}' for d, s in moves])}"
            
            print(f"  Execute action: {action_desc} -> New position: {state.position}")

            # Determine feedback information
            feedback = ""
            if last_error:
                feedback = last_error
                print(f"  Feedback: {feedback}")
                
                # ============ Record action execution error ============
                error_info = {
                    'step': step + 1,
                    'type': 'action_error',
                    'action': action_desc,
                    'error_message': feedback,
                    'timestamp': time.time(),
                    'position_before': position_before,
                    'position_after': state.position if not state.done else None
                }
                
                # Classify error type
                if '墙' in feedback or 'wall' in feedback.lower():
                    stats['wall_collisions'] += 1
                    error_info['subtype'] = 'wall_collision'
                elif '跳步' in feedback or 'jump' in feedback.lower() or '不连续' in feedback:
                    stats['invalid_jumps'] += 1
                    error_info['subtype'] = 'invalid_jump'
                elif '超出' in feedback or '超出边界' in feedback or 'out of bounds' in feedback.lower():
                    stats['wall_collisions'] += 1  # Out of bounds also counts as wall collision
                    error_info['subtype'] = 'out_of_bounds'
                    
                stats['action_errors'].append(error_info)
                stats['errors'].append(error_info)
                # ===========================================
                
            elif 'success' in step_info and step_info.get('success'):
                feedback = f"Successfully moved {total_steps_moved} steps"
            else:
                feedback = f"Successfully moved {total_steps_moved} steps"

            # Update exploration tracker with path (即使失败也要更新已成功移动的部分)
            explored_before = exploration_tracker.get_explored_area(env.grid)
            if all_paths:
                # 计算更新前的新位置数量
                new_positions = [p for p in all_paths if p not in exploration_tracker.explored_positions]
                is_new = exploration_tracker.update_path(all_paths)
                if is_new:
                    explored_after = exploration_tracker.get_explored_area(env.grid)
                    print(f"  Explored new area: {len(new_positions)} new positions along path (Total explored: {explored_after})")
            else:
                # 如果没有路径信息，回退到只更新最终位置
                is_new = exploration_tracker.update(state.position)
                if is_new:
                    explored_after = exploration_tracker.get_explored_area(env.grid)
                    print(f"  Explored new area: {state.position} (Total explored: {explored_after})")
            
            # 如果移动完全失败（没有任何成功移动），增加未探索计数
            if last_error and total_steps_moved == 0:
                exploration_tracker.increment_no_exploration()
            
            # Print exploration rate every step
            explored_area = exploration_tracker.get_explored_area(env.grid)
            exploration_rate = exploration_tracker.get_exploration_rate(total_reachable, env.grid)
            print(f"  Exploration: {explored_area}/{total_reachable} ({exploration_rate*100:.2f}%)")

            # Update action memory
            action_memory.append({'action': action_desc, 'feedback': feedback})

            history.append({
                'step': step + 1, 
                'action': action_desc, 
                'position': state.position,
                'feedback': feedback
            })
            
            # ============ Record detailed history ============
            stats['history'].append({
                'step': step + 1,
                'api_call_duration': call_duration,
                'response': response[:100],  # Truncate response
                'action': action_desc,
                'position_before': position_before,
                'position_after': state.position,
                'feedback': feedback,
                'timestamp': time.time()
            })
            # ===========================================

            if state.done:
                print(f"Step {step + 1}: Reached goal!")
                # Update last step feedback to goal reached
                if action_memory:
                    action_memory[-1]['feedback'] = "Reached goal"
                break

        except Exception as e:
            print(f"  Action execution error: {type(e).__name__}: {e}")
            # ============ Record execution error ============
            stats['errors'].append({
                'step': step + 1,
                'type': 'execution_error',
                'action': action_desc if 'action_desc' in locals() else 'unknown',
                'error_type': type(e).__name__,
                'error_message': str(e),
                'timestamp': time.time()
            })
            # ===========================================
            break

        print()

    # ============ Calculate total time ============
    total_time = time.perf_counter() - start_total_time
    # ===========================================

    # Calculate exploration metrics
    explored_area = exploration_tracker.get_explored_area(env.grid)
    exploration_rate = exploration_tracker.get_exploration_rate(total_reachable, env.grid)
    exploration_failed = exploration_tracker.should_fail()

    # Determine error type
    success = env.state.done and not exploration_failed
    error = None
    if not success:
        if env.state.steps >= max_steps:
            error = 'max_steps_exceeded'
        elif exploration_failed:
            error = 'exploration_failed'
        else:
            error = 'unknown'

    # Evaluate results - enhanced result dictionary
    result = {
        'success': success,
        'steps': env.state.steps,
        'path': [h['position'] for h in history],
        'actions': [h['action'] for h in history],
        'error': error,
        
        # ============ Statistics ============
        'stats': {
            'api_calls': stats['api_calls'],
            'errors': stats['errors'],
            'action_errors': stats['action_errors'],
            'error_count': len(stats['errors']),
            'wall_collisions': stats['wall_collisions'],
            'invalid_jumps': stats['invalid_jumps'],
            'parse_errors': stats['parse_errors'],
            'avg_response_time': stats['total_response_time'] / stats['api_calls'] if stats['api_calls'] > 0 else 0,
            'total_response_time': stats['total_response_time'],
            'total_time': total_time,
            'start_time': stats['start_time'],
            'end_time': time.time(),
            'history': stats['history'],
            'explored_area': explored_area,
            'exploration_rate': exploration_rate,
            'exploration_failed': exploration_failed,
            'total_reachable': total_reachable
        },
        
        # For backward compatibility, also provide top-level fields
        'api_calls': stats['api_calls'],
        'error_count': len(stats['errors']),
        'total_time': total_time,
        'wall_collisions': stats['wall_collisions'],
        'invalid_jumps': stats['invalid_jumps'],
        'parse_errors': stats['parse_errors'],
        'explored_area': explored_area,
        'exploration_rate': exploration_rate,
        'exploration_failed': exploration_failed,
        'maze': maze,
        'model': model,
        'max_steps': max_steps,
        'timestamp': time.time()
        # ===========================================
    }

    # Save results - updated save logic
    Path("outputs").mkdir(exist_ok=True)
    maze_name = maze_path.stem  # Use pure filename without path and extension
    result_file = f"outputs/ai_sandbox_{maze_name}_{model}_{int(time.time())}.json"
    with open(result_file, 'w', encoding='utf-8') as f:  # Add encoding
        json.dump(result, f, indent=2, ensure_ascii=False)  # Add ensure_ascii=False

    result['result_file'] = result_file
    return result