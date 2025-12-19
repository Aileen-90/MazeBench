# runners/image2d_runner.py
from adapters import get_adapter
from core.parser import OutputParser
from core.validator import MazeValidator
from core.metrics import Metrics
from core.anti_cheat import AntiCheat

def evaluate_image2d(maze: dict, img_path: str, cfg: dict, adapter=None) -> dict:
    """
    评测单个图片迷宫，输入为已生成好迷宫字典和图片路径
    不负责生成任何迷宫。
    """
    anti = AntiCheat(seed=maze.get('nonce', 0))
    maze_p = anti.perturb_input(maze)
    prompt = [{"role": "system", "content": "You are a helpful assistant that can help me solve mazes.\
        横轴为x轴，竖轴为y轴，坐标用（y，x）来表示\
        Please output the complete coordinate path list strictly."},
    {"role": "user", "content":
        f"请根据图片中的迷宫，从绿色起点到红色终点输出坐标路径列表。"
        f"白色单元格为通路，黑色为墙壁。迷宫尺寸为{len(maze_p['grid'])}x{len(maze_p['grid'][0])}。"
        f"起点{maze_p['start']}，终点{maze_p['goal']}。"
        }
    ]
    if adapter is None:
        adapter = get_adapter(cfg, image=True)
    model_output = adapter.generate(prompt, image_path=img_path)
    model_output = anti.sandbox_output(model_output)
    parsed = OutputParser().parse_with_fallback(model_output, prompt=prompt)
    validator = MazeValidator(maze['grid'], maze['start'], maze['goal'], maze.get('shortest_path', []))
    vres = validator.validate(parsed.path)
    scores = Metrics().score(vres)
    return {
        "scores": scores,
        "parsed_path": parsed.path,
        "model_output": model_output,
        "maze_info": maze,
        "img_path": img_path,
        "validation_result": vres  # 新增：包含路径相似度等验证结果
    }