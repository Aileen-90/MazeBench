# main.py
"""
MazeBench - 迷宫评测基准测试框架
主入口文件
"""
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
# 添加当前目录到Python路径，使相对导入正常工作
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.io import load_config, load_mazes, save_results, apply_env_keys, generate_mazes_to_dir
from utils.logging import setup_logging, get_logger
from runners.text2d_runner import evaluate_text2d
from runners.image2d_runner import evaluate_image2d
from core.report import ReportGenerator
from utils.visualization import create_summary_visualization_html
logger = get_logger(__name__)

def evaluate_text2d_mode(cfg):
    """运行text2d模式评测"""
    logger.info("Running text2d evaluation")
    mazes = load_mazes(cfg.get('mazes_path', 'mazes/'))

    # 为text2d迷宫准备图片路径
    img_paths = []
    from pathlib import Path
    maze_dir = Path(cfg.get('mazes_path', 'mazes/'))
    for i, maze in enumerate(mazes):
        # 尝试找到对应的图片文件
        possible_names = [
            f"text2d_maze_{maze.get('height', 10)}x{maze.get('width', 10)}_{i}.png",
            f"maze_{i}.png",
            f"generated_text_{maze.get('height', 10)}x{maze.get('width', 10)}_{i}.png",
            f"generated_{maze.get('height', 10)}x{maze.get('width', 10)}_{i}.png"
        ]

        img_path = None
        for name in possible_names:
            candidate = maze_dir / name
            if candidate.exists():
                img_path = str(candidate)
                break

        if img_path is None:
            # 如果没有找到图片，生成一个
            from utils.io import render_maze_image
            img = render_maze_image(maze, cfg.get('text2d', {}).get('cell_px', 24))
            img_path = str(maze_dir / f"generated_{maze.get('height', 10)}x{maze.get('width', 10)}_{i}.png")
            img.save(img_path)
            logger.info(f"Generated text2d image: {img_path}")

        img_paths.append(img_path)

    logger.info(f"Loaded {len(mazes)} mazes with {len(img_paths)} images")

    results = []
    report_gen = ReportGenerator()
    adapter = None

    # 使用多线程并行调用evaluate_text2d函数
    with ThreadPoolExecutor(max_workers=cfg.get('max_workers', 4)) as executor:
        # 提交所有评测任务
        future_to_index = {
            executor.submit(evaluate_text2d, maze, cfg, adapter): i
            for i, maze in enumerate(mazes)
        }

        # 收集结果并生成报告
        for future in as_completed(future_to_index):
            i = future_to_index[future]
            maze, img_path = mazes[i], img_paths[i]
            logger.info(f"Evaluating maze {i+1}/{len(mazes)}")
            try:
                result = future.result()
                results.append(result)
                logger.info(f"Maze {i+1} score: {result['scores']['total']}")

                # 生成单个迷宫的 HTML 报告
                model_name = cfg.get('model', 'gpt4').replace('-', '').replace('_', '').lower()
                maze_size = f"{maze.get('height', 10)}x{maze.get('width', 10)}"
                output_dir = cfg.get('output_dir', 'outputs/')
                report_path = f"{output_dir}{model_name}_{maze_size}_text_{i}.html"
                failure_info = result.get('error', '')
                report_gen.generate_html_report(
                    output_path=report_path,
                    maze=result.get('maze_info', maze),
                    parsed_path=result.get('parsed_path', []),
                    scores=result['scores'],
                    failure_info=failure_info,
                    image_path=img_path,
                    mode="text2d"
                )
                logger.info(f"Generated HTML report: {report_path}")

            except Exception as e:
                logger.error(f"Failed to evaluate maze {i+1}: {e}")
                results.append({
                    "scores": {"total": 0, "S": 0, "Q": 0, "O": 0, "A": 0},
                    "error": str(e),
                    "maze_info": maze
                })

    return results


def evaluate_image2d_mode(cfg):
    """运行image2d模式评测"""
    logger.info("Running image2d evaluation")
    mazes = load_mazes(cfg.get('mazes_path', 'mazes/'))

    # 为图片迷宫动态生成图片（不保存到磁盘，用于推理）
    img_paths = []
    from pathlib import Path
    maze_dir = Path(cfg.get('mazes_path', 'mazes/'))
    for i, maze in enumerate(mazes):
        # 尝试找到对应的图片文件
        possible_names = [
            f"image2d_maze_{maze.get('height', 10)}x{maze.get('width', 10)}_{i}.png",
            f"maze_{i}.png",
            f"generated_{maze.get('height', 10)}x{maze.get('width', 10)}_{i}.png"
        ]

        img_path = None
        for name in possible_names:
            candidate = maze_dir / name
            if candidate.exists():
                img_path = str(candidate)
                break

        if img_path is None:
            # 如果没有找到图片，生成一个
            from utils.io import render_maze_image
            img = render_maze_image(maze, cfg.get('image2d', {}).get('cell_px', 24))
            img_path = str(maze_dir / f"generated_{maze.get('height', 10)}x{maze.get('width', 10)}_{i}.png")
            img.save(img_path)
            logger.info(f"Generated image for maze {i+1}: {img_path}")

        img_paths.append(img_path)

    logger.info(f"Loaded {len(mazes)} mazes with {len(img_paths)} images")

    results = []
    report_gen = ReportGenerator()
    adapter = None

    # 使用多线程并行调用evaluate_image2d函数
    with ThreadPoolExecutor(max_workers=cfg.get('max_workers', 4)) as executor:
        # 提交所有评测任务
        future_to_index = {
            executor.submit(evaluate_image2d, maze, img_path, cfg, adapter): i
            for i, (maze, img_path) in enumerate(zip(mazes, img_paths))
        }

        # 收集结果并生成报告
        for future in as_completed(future_to_index):
            i = future_to_index[future]
            maze, img_path = mazes[i], img_paths[i]
            logger.info(f"Evaluating maze {i+1}/{len(mazes)}")
            try:
                result = future.result()
                results.append(result)
                logger.info(f"Maze {i+1} score: {result['scores']['total']}")

                # 生成单个迷宫的 HTML 报告
                model_name = cfg.get('model', 'gpt4').replace('-', '').replace('_', '').lower()
                maze_size = f"{maze.get('height', 10)}x{maze.get('width', 10)}"
                output_dir = cfg.get('output_dir', 'outputs/')
                report_path = f"{output_dir}{model_name}_{maze_size}_image_{i}.html"
                failure_info = result.get('error', '')
                report_gen.generate_html_report(
                    output_path=report_path,
                    maze=result.get('maze_info', maze),
                    parsed_path=result.get('parsed_path', []),
                    scores=result['scores'],
                    failure_info=failure_info,
                    image_path=img_path,
                    mode="image2d"
                )
                logger.info(f"Generated HTML report: {report_path}")

            except Exception as e:
                logger.error(f"Failed to evaluate maze {i+1}: {e}")
                results.append({
                    "scores": {"total": 0, "S": 0, "Q": 0, "O": 0, "A": 0},
                    "error": str(e),
                    "maze_info": maze,
                    "img_path": img_path
                })

    return results


def run_evaluation(cfg):
    """运行评测"""
    mode = cfg.get('mode', 'text2d')

    if mode == 'text2d':
        results = evaluate_text2d_mode(cfg)
    elif mode == 'image2d':
        results = evaluate_image2d_mode(cfg)
    else:
        logger.error(f"Unknown mode: {mode}")
        return []

    # 保存结果
    output_dir = cfg.get('output_dir', 'outputs/')
    model_name = cfg.get('model', 'gpt4').replace('-', '').replace('_', '').lower()
    mode_suffix = "text" if mode == "text2d" else "image"
    output_path = f"{output_dir}results_{model_name}_{mode_suffix}.json"
    save_results(results, output_path)
    logger.info(f"Results saved to {output_path}")

    # 生成汇总 HTML 报告
    summary_html_path = f"{output_dir}results_{model_name}_{mode_suffix}_summary.html"
    mode_title = f"MazeBench {mode.upper()} Summary"
    report_gen = ReportGenerator()
    report_gen.generate_summary_report(summary_html_path, results, mode_title)
    logger.info(f"Summary HTML report saved to {summary_html_path}")

    # 计算并显示汇总统计
    if results:
        valid_scores = [r['scores']['total'] for r in results if r['scores']['total'] > 0]
        if valid_scores:
            avg_score = sum(valid_scores) / len(valid_scores)
            logger.info(f"Average score: {avg_score:.2f} ({len(valid_scores)}/{len(results)} valid)")

    return results


def main():
    """主函数"""
    # 设置日志
    setup_logging()

    # 加载配置
    cfg = load_config()
    cfg = apply_env_keys(cfg)
    logger.info(f"Starting MazeBench with model: {cfg.get('model')}")

    # 检查是否需要生成迷宫
    if cfg.get('generate_mazes', False):
        logger.info("Generating mazes...")
        maze_output_dir = cfg.get('maze_output_dir', 'mazes/')
        maze_count = cfg.get('maze_count', 3)
        generate_mazes_to_dir(cfg, maze_output_dir, cfg.get('mode', 'text2d'), maze_count)
        logger.info(f"Generated {maze_count} mazes to {maze_output_dir}")
        # 更新mazes_path为新生成的迷宫目录
        cfg['mazes_path'] = maze_output_dir

    # 检查是否需要运行评测（默认运行，但可以通过配置跳过）
    if cfg.get('run_evaluation', True):
        # 运行评测
        run_evaluation(cfg)
    else:
        logger.info("Skipping evaluation (run_evaluation=False in config)")
        logger.info("Maze generation completed. Use run_evaluation=True to run tests.")

if __name__ == '__main__':
    main()
