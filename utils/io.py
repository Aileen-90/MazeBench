# utils/io.py
import yaml
import json
from pathlib import Path
from typing import Dict, List, Any

def load_config(cfg_path='config/config.yaml') -> Dict[str, Any]:
    """
    加载项目主配置，并合并本地配置，返回dict。
    """
    path = Path(cfg_path)
    if not path.exists():
        # 默认配置
        config = {
            'model': 'gpt-4',
            'mode': 'text2d',
            'mazes_path': 'mazes/',
            'output_path': 'outputs/results.json',
            'OPENAI_API_KEY': None,
            'OPENAI_API_BASE': None,
        }
    else:
        with open(path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}

    # 合并本地配置
    local_path = Path('config/local.yaml')
    if local_path.exists():
        with open(local_path, 'r', encoding='utf-8') as f:
            local_config = yaml.safe_load(f) or {}
            # 本地配置覆盖主配置
            config.update(local_config)

    return config

def load_mazes(path) -> List[Dict]:
    """
    从json或yaml文件读取迷宫列表，返回list；支持文件路径或目录。
    """
    p = Path(path)
    mazes = []

    if p.is_dir():
        # 扫描目录下的所有json文件
        for file in p.glob('*.json'):
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        mazes.extend(data)
                    else:
                        mazes.append(data)
            except Exception as e:
                print(f"Warning: Failed to load {file}: {e}")
    elif p.suffix == '.json':
        try:
            with open(p, 'r', encoding='utf-8') as f:
                first = json.load(f)
                if isinstance(first, list):
                    mazes.extend(first)
                else:
                    mazes.append(first)
        except Exception as e:
            print(f"Warning: Failed to load {p}: {e}")
    elif p.suffix in ('.yaml', '.yml'):
        try:
            with open(p, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                if isinstance(data, list):
                    mazes.extend(data)
                else:
                    mazes.append(data)
        except Exception as e:
            print(f"Warning: Failed to load {p}: {e}")

    return mazes

def save_results(results: List[Dict], path: str) -> None:
    """
    保存评测结果为json文件。
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

def apply_env_keys(cfg: Dict) -> Dict:
    """
    将环境变量应用到配置中
    """
    import os

    # OpenAI相关
    if 'OPENAI_API_KEY' not in cfg and os.getenv('OPENAI_API_KEY'):
        cfg['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

    if 'OPENAI_API_BASE' not in cfg and os.getenv('OPENAI_API_BASE'):
        cfg['OPENAI_API_BASE'] = os.getenv('OPENAI_API_BASE')

    # Azure相关
    if 'AZURE_OPENAI_API_KEY' not in cfg and os.getenv('AZURE_OPENAI_API_KEY'):
        cfg['AZURE_OPENAI_API_KEY'] = os.getenv('AZURE_OPENAI_API_KEY')

    if 'AZURE_OPENAI_ENDPOINT' not in cfg and os.getenv('AZURE_OPENAI_ENDPOINT'):
        cfg['AZURE_OPENAI_ENDPOINT'] = os.getenv('AZURE_OPENAI_ENDPOINT')

    if 'AZURE_OPENAI_DEPLOYMENT' not in cfg and os.getenv('AZURE_OPENAI_DEPLOYMENT'):
        cfg['AZURE_OPENAI_DEPLOYMENT'] = os.getenv('AZURE_OPENAI_DEPLOYMENT')

    if 'AZURE_OPENAI_API_VERSION' not in cfg and os.getenv('AZURE_OPENAI_API_VERSION'):
        cfg['AZURE_OPENAI_API_VERSION'] = os.getenv('AZURE_OPENAI_API_VERSION')

    if 'PROVIDER' not in cfg and os.getenv('PROVIDER'):
        cfg['PROVIDER'] = os.getenv('PROVIDER')

    return cfg

def generate_mazes_to_dir(cfg: Dict, output_dir: str, mode: str = 'text2d', count: int = 5) -> None:
    """
    生成指定数量的迷宫并保存到目录（仅生成JSON格式，图片在运行时动态生成）

    Args:
        cfg: 配置字典
        output_dir: 输出目录路径
        mode: 模式 ('text2d' 或 'image2d')
        count: 生成的迷宫数量
    """
    from pathlib import Path
    from core.generator import MazeGenerator
    from core.config import TextMazeConfig, ImageMazeConfig
    import json

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for i in range(count):
        # 两种模式都使用相同的生成逻辑，仅文件名区分
        if mode == 'text2d':
            size_str = cfg.get('text2d', {}).get('size', '10x10')
            config_dict = cfg.get('text2d', {})
        else:  # image2d
            size_str = cfg.get('image2d', {}).get('size', '10x10')
            config_dict = cfg.get('image2d', {})

        h, w = map(int, size_str.split('x'))
        maze_cfg = TextMazeConfig(  # 统一使用TextMazeConfig，两种模式都先生成JSON
            width=w,
            height=h,
            seed=config_dict.get('seed', 0) + i,
            start_goal=config_dict.get('start_goal', 'corner'),
            algorithm=config_dict.get('algorithm', 'dfs')
        )
        generator = MazeGenerator(maze_cfg)
        maze = generator.generate()

        # 保存为JSON，统一命名
        filename = f"maze_{maze_cfg.height}x{maze_cfg.width}_{i}.json"
        filepath = output_path / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(maze, f, ensure_ascii=False, indent=2)

def render_maze_image(maze: Dict, cell_px: int = 24):
    """
    将迷宫数据渲染为图片

    Args:
        maze: 迷宫字典
        cell_px: 每个单元格的像素大小

    Returns:
        PIL Image对象
    """
    import numpy as np
    from PIL import Image

    grid = np.array(maze['grid'])
    height, width = grid.shape

    # 创建RGB图像
    img_array = np.zeros((height * cell_px, width * cell_px, 3), dtype=np.uint8)

    # 颜色定义
    wall_color = [0, 0, 0]      # 黑色墙壁
    path_color = [255, 255, 255]  # 白色通路
    start_color = [0, 255, 0]     # 绿色起点
    goal_color = [255, 0, 0]      # 红色终点

    for i in range(height):
        for j in range(width):
            color = wall_color if grid[i, j] == 1 else path_color
            img_array[i*cell_px:(i+1)*cell_px, j*cell_px:(j+1)*cell_px] = color

    # 标记起点和终点
    start_i, start_j = maze['start']
    goal_i, goal_j = maze['goal']

    # 起点
    img_array[start_i*cell_px:(start_i+1)*cell_px, start_j*cell_px:(start_j+1)*cell_px] = start_color

    # 终点
    img_array[goal_i*cell_px:(goal_i+1)*cell_px, goal_j*cell_px:(goal_j+1)*cell_px] = goal_color

    return Image.fromarray(img_array)
