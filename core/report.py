# core/report.py
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from utils.visualization import create_maze_visualization_html, create_summary_visualization_html

class ReportGenerator:
    """报告生成器"""

    def generate_html_report(self, output_path: str, maze: Dict, parsed_path: List[Tuple[int, int]],
                           scores: Dict, failure_info: str = "", image_path: Optional[str] = None, mode: str = "text2d") -> None:
        """
        生成HTML格式的可视化评测报告

        Args:
            output_path: 输出文件路径
            maze: 迷宫数据
            parsed_path: 解析后的路径
            scores: 评分结果
            failure_info: 失败信息（暂时未使用）
            image_path: 可选的图片路径（用于image2d模式）
            mode: 评测模式 ('text2d' 或 'image2d')
        """
        # 统一路径格式为 [[x,y], [x,y], ...]
        model_path = [[pos[0], pos[1]] for pos in parsed_path]

        # 确定标题
        title = f"MazeBench-{mode.upper()}"

        create_maze_visualization_html(
            maze=maze,
            model_path=model_path,
            scores=scores,
            output_path=output_path,
            title=title,
            image_path=image_path
        )

    def generate_summary_report(self, output_path: str, results: List[Dict], title: str = "MazeBench Summary") -> None:
        """
        生成汇总可视化报告

        Args:
            output_path: 输出文件路径
            results: 所有评测结果列表
            title: 报告标题
        """
        create_summary_visualization_html(results, output_path, title)
