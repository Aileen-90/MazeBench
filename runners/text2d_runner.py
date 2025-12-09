# runners/text2d_runner.py

from adapters import get_adapter
from core.parser import OutputParser
from core.validator import MazeValidator
from core.metrics import Metrics
from core.anti_cheat import AntiCheat

def evaluate_text2d(maze: dict, cfg: dict, adapter=None) -> dict:
    """
    评测单个文本迷宫，输入为已生成好迷宫字典。
    不负责生成任何迷宫。
    """
    anti = AntiCheat(seed=maze.get('nonce', 0))
    maze_p = anti.perturb_input(maze)
    prompt = [{"role": "system", "content": "You are a helpful assistant that can help me solve mazes.\
        横轴为x轴，竖轴为y轴，坐标用（y，x）来表示\
        Please output the complete coordinate path list strictly."},
    {"role": "user", "content": 
        f"迷宫大小 {len(maze_p['grid'])}x{len(maze_p['grid'][0])}。"
        f"迷宫为{maze_p['grid']}。0为通路,1为墙壁。起点{maze_p['start']}，终点{maze_p['goal']}。"
        }
    ]
    if adapter is None:
        adapter = get_adapter(cfg)
    model_output = adapter.generate(prompt)
    model_output = anti.sandbox_output(model_output)
    parsed = OutputParser().parse_with_fallback(model_output, prompt=prompt)
    validator = MazeValidator(maze['grid'], maze['start'], maze['goal'], maze.get('shortest_path', []))
    vres = validator.validate(parsed.path)
    scores = Metrics().score(vres)
    return {
        "scores": scores,
        "parsed_path": parsed.path,
        "model_output": model_output,
        "maze_info": maze
    }
