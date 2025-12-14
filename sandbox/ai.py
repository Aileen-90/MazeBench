"""
AI Sandbox - Test models in mazes
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
import json
import time
import traceback
# Project path
sys.path.insert(0, str(Path(__file__).parent.parent))

from .env import MazeEnvironment
from .exploration import ExplorationTracker
from adapters import get_adapter
from utils.io import load_config, apply_env_keys
import re
import numpy as np


def run_ai_sandbox(maze: str, model: str = None, max_steps: int = None, cfg: Dict[str, Any] = None) -> Dict[str, Any]:
    """Run AI sandbox test"""
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

    # ============ New: Initialize statistics ============
    import time
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
    print(f"Starting AI test - Model: {model}, Max steps: {max_steps}, Memory length: {memory}")
    print(f"Maze: {maze}, Start position: {env.get_info()['current_position']}, Goal: {env.get_info()['goal']}")
    print("-" * 50)

    for step in range(max_steps):
        if env.state.done:
            print(f"Step {step}: Task completed")
            break

        # AI decision
        try:
            info = env.get_info()
            print(f"Step {step + 1}: Current position {info['current_position']}")
            # Print maze ASCII diagram
            maze_ascii = env.render_tensor(symbols=symbols, visibility=visibility)
            # print("Current maze state:")
            # print(maze_ascii)

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

            prompt = f"""You are a maze expert and need to navigate yourself to the goal. Current position A{info['current_position']}, goal {info['goal']}.
Coordinate format: All coordinates use (y, x) format, where y is the row number and x is the column number.

You can:
Output path coordinates: e.g., (1,2),(1,3),(2,3) - coordinate format is (y,x)
Note: Please start outputting from the next position's coordinate, do not output coordinates that duplicate the current position, and the output coordinates must be adjacent
Hint: First confirm your current position, then you can move to a safe position first, and then plan the next steps
Maze layout:
{maze_ascii}
Symbol legend:
- {symbols['wall']}: Wall, cannot pass through
- {symbols['path']}: Path, can pass through
- {symbols['start']}: Start position (you have left)
- {symbols['goal']}: Goal position (target location)
- {symbols['agent']}: Current position (your location)
{memory_info}Please reply with the path coordinate sequence:
"""

            print(f"  Prompt: {prompt}")
            
            # ============ New: Record API call ============
            stats['api_calls'] += 1
            call_start_time = time.perf_counter()
            # ===========================================
            
            raw_response = adapter.generate(prompt)
            
            # ============ New: Record response time ============
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

            # Try to parse as path coordinates
            path = parse_path_coordinates(response, info['current_position'])
            action = None

            if path:
                # Path coordinate mode
                action_desc = f"Path: {path}"
                print(f"  AI decision: {action_desc}")
            elif response in ['up', 'down', 'left', 'right']:
                # Single action mode
                action = response
                path = None
                print(f"  AI decision: {action}")
            else:
                print(f"  Invalid output, terminating test")
                # Record invalid feedback to memory
                action_memory.append({'action': response, 'feedback': 'Invalid output'})
                
                # ============ New: Record parse error ============
                stats['parse_errors'] += 1
                stats['errors'].append({
                    'step': step + 1,
                    'type': 'parse_error',
                    'response': response[:100],  # Only record first 100 characters
                    'timestamp': time.time(),
                    'description': 'AI response cannot be parsed as valid action or path'
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
            
            # ============ New: Record AI decision error ============
            stats['errors'].append({
                'step': step + 1,
                'type': 'ai_decision_error',
                'error_type': type(e).__name__,
                'error_message': str(e),
                'timestamp': time.time()
            })
            # ===========================================
            
            break

        # Execute action or path
        try:
            if path:
                # Path mode
                state, step_info = env.step_path(path)
                action_desc = f"Path move: {path}"
                print(f"  Execute action: {action_desc} -> New position: {state.position}")
            else:
                # Single action mode
                state, step_info = env.step(action)
                action_desc = action
                print(f"  Execute action: {action} -> New position: {state.position}")

            # Determine feedback information
            feedback = ""
            if 'error' in step_info:
                feedback = step_info['error']
                print(f"  Feedback: {feedback}")
                
                # ============ New: Record action execution error ============
                position_before = info['current_position']  # Default value
    
                # Simple extraction of "(number, number)" format
                import re
                # Find all (y, x) format coordinates
                all_coords = re.findall(r'\((\d+),\s*(\d+)\)', feedback)
                
                if all_coords:
                    try:
                        # Take first coordinate as position_before
                        y, x = map(int, all_coords[0])
                        position_before = (y, x)
                    except:
                        pass  # If conversion fails, keep default value
                        
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
                elif '超出' in feedback or '超出边界' in feedback:
                    stats['wall_collisions'] += 1  # Out of bounds also counts as wall collision
                    error_info['subtype'] = 'out_of_bounds'
                    
                stats['action_errors'].append(error_info)
                stats['errors'].append(error_info)
                # ===========================================
                
            elif 'success' in step_info and step_info.get('success'):
                if path:
                    steps_moved = step_info.get('steps_moved', 1)
                    feedback = f"Successfully moved {steps_moved} steps"
                else:
                    feedback = "Successfully moved"
            else:
                feedback = "Successfully moved"

            # Update exploration tracker
            if 'error' not in step_info:
                is_new = exploration_tracker.update(state.position)
                if is_new:
                    print(f"  Explored new area: {state.position} (Total explored: {exploration_tracker.get_explored_area()})")
            else:
                # If action failed, increment no exploration count
                exploration_tracker.increment_no_exploration()
            
            # Print exploration rate every step
            explored_area = exploration_tracker.get_explored_area()
            exploration_rate = exploration_tracker.get_exploration_rate(total_reachable)
            print(f"  Exploration: {explored_area}/{total_reachable} ({exploration_rate*100:.2f}%)")

            # Update action memory
            action_memory.append({'action': action_desc, 'feedback': feedback})

            history.append({
                'step': step + 1, 
                'action': action_desc, 
                'position': state.position,
                'feedback': feedback
            })
            
            # ============ New: Record detailed history ============
            stats['history'].append({
                'step': step + 1,
                'api_call_duration': call_duration,
                'response': response[:100],  # Truncate response
                'action': action_desc,
                'position_before': info['current_position'],
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
            # ============ New: Record execution error ============
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

    # ============ New: Calculate total time ============
    total_time = time.perf_counter() - start_total_time
    # ===========================================

    # Calculate exploration metrics
    explored_area = exploration_tracker.get_explored_area()
    exploration_rate = exploration_tracker.get_exploration_rate(total_reachable)
    exploration_failed = exploration_tracker.should_fail()
    
    # Evaluate results - enhanced result dictionary
    result = {
        'success': env.state.done and not exploration_failed,
        'steps': env.state.steps,
        'path': [h['position'] for h in history],
        'actions': [h['action'] for h in history],
        
        # ============ New: Statistics ============
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
    result_file = f"outputs/ai_sandbox_{maze}_{model}_{int(time.time())}.json"
    with open(result_file, 'w', encoding='utf-8') as f:  # Add encoding
        json.dump(result, f, indent=2, ensure_ascii=False)  # Add ensure_ascii=False

    result['result_file'] = result_file
    return result

def parse_path_coordinates(response: str, current_position: Tuple[int, int] = None) -> List[Tuple[int, int]]:
    """
    Parse path coordinates from AI response, coordinate format is (y, x)

    Supported formats:
    - (1,2),(3,4),(5,6)
    - (1, 2), (3, 4), (5, 6)

    If the first coordinate is the current position, it will be automatically skipped
    """
    response = response.strip().lower()

    # Use regex to find all coordinate pairs
    coord_pattern = r'\(\s*(\d+)\s*,\s*(\d+)\s*\)'
    matches = re.findall(coord_pattern, response)

    path = []
    for y_str, x_str in matches:
        try:
            y, x = int(y_str), int(x_str)
            path.append((y, x))
        except ValueError:
            continue

    # If current position is provided and first coordinate in path is current position, skip first coordinate
    if current_position and path and path[0] == current_position:
        path = path[1:]

    return path
