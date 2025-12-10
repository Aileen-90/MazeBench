"""
AI Sandbox - 让AI在迷宫中测试模型
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
import json
import time

# 项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from .env import MazeEnvironment
from adapters import get_adapter
from utils.io import load_config, apply_env_keys
import re


def parse_path_coordinates(response: str, current_position: Tuple[int, int] = None) -> List[Tuple[int, int]]:
    """
    解析AI响应中的路径坐标，坐标格式为 (y, x)

    支持格式：
    - (1,2),(3,4),(5,6)
    - (1, 2), (3, 4), (5, 6)

    如果第一个坐标是当前自身坐标，会自动跳过
    """
    response = response.strip().lower()

    # 使用正则表达式查找所有坐标对
    coord_pattern = r'\(\s*(\d+)\s*,\s*(\d+)\s*\)'
    matches = re.findall(coord_pattern, response)

    path = []
    for y_str, x_str in matches:
        try:
            y, x = int(y_str), int(x_str)
            path.append((y, x))
        except ValueError:
            continue

    # 如果提供了当前坐标且路径第一个坐标是自身坐标，则跳过第一个坐标
    if current_position and path and path[0] == current_position:
        path = path[1:]

    return path


def run_ai_sandbox(maze: str, model: str = None, max_steps: int = None) -> Dict[str, Any]:
    """运行AI沙盒测试"""
    # 加载配置
    cfg = load_config()
    apply_env_keys(cfg)

    # 默认参数
    model = model or cfg.get('model', 'gpt-4')
    max_steps = max_steps or cfg.get('sandbox', {}).get('max_steps', 50)
    memory = cfg.get('sandbox', {}).get('memory', 5)  # AI记忆长度

    # 读取符号配置
    symbols = cfg.get('sandbox', {}).get('symbols', {
        'wall': '█',
        'path': ' ',
        'start': 'S',
        'goal': 'G',
        'agent': 'A'
    })
    
    # 从config加载迷宫路径
    mazes_path = cfg.get('sandbox', {}).get('mazes_path', 'mazes/')

    # 加载迷宫
    maze_path = Path(f"{mazes_path}/{maze}.json")
    if not maze_path.exists():
        return {'error': f'迷宫不存在: {maze_path}'}

    env = MazeEnvironment(str(maze_path))
    env.reset()

    # AI适配器 直接使用/Adapter中的方法
    adapter_cfg = {
        'PROVIDER': 'azure' if model.startswith('azure') else 'openai',
        'AZURE_OPENAI_API_KEY': cfg.get('AZURE_OPENAI_API_KEY', ''),
        'AZURE_OPENAI_ENDPOINT': cfg.get('AZURE_OPENAI_ENDPOINT', ''),
        'AZURE_OPENAI_DEPLOYMENT': cfg.get('AZURE_OPENAI_DEPLOYMENT', ''),
        'AZURE_OPENAI_API_VERSION': cfg.get('AZURE_OPENAI_API_VERSION', '2023-12-01-preview'),
        'OPENAI_API_KEY': cfg.get('OPENAI_API_KEY', ''),
        'OPENAI_API_BASE': cfg.get('OPENAI_API_BASE'),
        'model': model
    }
    adapter = get_adapter(adapter_cfg)

    # AI测试循环
    history = []
    action_memory = []  # 动作记忆列表，每个元素为 {'action': str, 'feedback': str}
    print(f"开始AI测试 - 模型: {model}, 最大步数: {max_steps}, 记忆长度: {memory}")
    print(f"迷宫: {maze}, 起始位置: {env.get_info()['current_position']}, 目标: {env.get_info()['goal']}")
    print("-" * 50)

    for step in range(max_steps):
        if env.state.done:
            print(f"步骤 {step}: 任务已完成")
            break

        # AI决策
        try:
            info = env.get_info()
            print(f"步骤 {step + 1}: 当前位置 {info['current_position']}")
            # 打印迷宫ASCII图
            maze_ascii = env.render_tensor(symbols=symbols)
            # print("当前迷宫状态:")
            # print(maze_ascii)

            # 构建记忆信息
            memory_info = ""
            if memory != 0 and action_memory:
                if memory == -1:
                    # 全部记忆
                    memory_actions = action_memory
                else:
                    # 最近k次记忆
                    memory_actions = action_memory[-memory:] if len(action_memory) > memory else action_memory

                if memory_actions:
                    memory_str = ", ".join([f"第{i+1}步:{mem['action']}({mem['feedback']})" for i, mem in enumerate(memory_actions)])
                    memory_info = f"最近动作记录: {memory_str}。"

            prompt = f"""当前位置A{info['current_position']},目标{info['goal']}。

坐标格式：所有坐标都使用 (y, x) 格式，其中 y 是行号，x 是列号。

你可以：
1. 输入单个动作：up/down/left/right
2. 输入路径坐标：如 (1,2),(1,3),(2,3) - 坐标格式为 (y,x)，系统会移动到最后一个合法位置
   注意：如果路径第一个坐标是当前自身坐标 {info['current_position']}，系统会自动跳过第一个坐标

迷宫布局：
{maze_ascii}

符号说明：
- {symbols['wall']}：墙壁，不能穿过
- {symbols['path']}：道路，可以通行
- {symbols['start']}：起点（你已离开）
- {symbols['goal']}：终点（目标位置）
- {symbols['agent']}：当前位置（你所在位置）

{memory_info}回复动作或路径坐标："""

            print(f"  Prompt: {prompt}")
            raw_response = adapter.generate(prompt)
            response = raw_response.strip().lower()

            # 尝试解析为路径坐标
            path = parse_path_coordinates(response, info['current_position'])
            action = None

            if path:
                # 路径坐标模式
                action_desc = f"路径: {path}"
                print(f"  AI决策: {action_desc}")
            elif response in ['up', 'down', 'left', 'right']:
                # 单个动作模式
                action = response
                path = None
                print(f"  AI决策: {action}")
            else:
                print(f"  输出违规，终止测试")
                # 记录违规反馈到记忆中
                action_memory.append({'action': response, 'feedback': '输出违规'})
                continue
        except Exception as e:
            print(f"  AI决策出错: {e}，终止测试")
            break

        # 执行动作或路径
        if path:
            # 路径模式
            state, step_info = env.step_path(path)
            action_desc = f"路径移动: {path}"
            print(f"  执行动作: {action_desc} -> 新位置: {state.position}")
        else:
            # 单个动作模式
            state, step_info = env.step(action)
            action_desc = action
            print(f"  执行动作: {action} -> 新位置: {state.position}")

        # 确定反馈信息
        feedback = ""
        if 'error' in step_info:
            feedback = step_info['error']
            print(f"  反馈: {feedback}")
        elif 'success' in step_info and step_info.get('success'):
            if path:
                steps_moved = step_info.get('steps_moved', 1)
                feedback = f"成功移动{steps_moved}步"
            else:
                feedback = "成功移动"
        else:
            feedback = "成功移动"

        # 更新动作记忆
        action_memory.append({'action': action_desc, 'feedback': feedback})

        history.append({'step': step + 1, 'action': action_desc, 'position': state.position})

        if state.done:
            print(f"步骤 {step + 1}: 到达目标！")
            # 更新最后一步的反馈为到达目标
            if action_memory:
                action_memory[-1]['feedback'] = "到达目标"
            break

        print()

    # 评估结果
    result = {
        'success': env.state.done,
        'steps': env.state.steps,
        'path': [h['position'] for h in history],
        'actions': [h['action'] for h in history]
    }

    # 保存结果
    Path("outputs").mkdir(exist_ok=True)
    result_file = f"outputs/ai_sandbox_{maze}_{model}_{int(time.time())}.json"
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)

    result['result_file'] = result_file
    return result

