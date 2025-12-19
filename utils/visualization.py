"""
简单的迷宫可视化工具
"""
import json
from pathlib import Path
from typing import Dict, List, Optional


def create_maze_visualization_html(maze: Dict, model_path: List[List[int]],
                                 scores: Dict, output_path: str,
                                 title: str = "Maze Visualization",
                                 image_path: Optional[str] = None):
    """
    创建迷宫可视化HTML报告

    Args:
        maze: 迷宫数据
        model_path: 模型输出的路径 [[x,y], [x,y], ...]
        scores: 评分结果
        output_path: 输出HTML文件路径
        title: 报告标题
        image_path: 可选的迷宫图片路径（用于image2d模式）
    """

    # 准备数据
    grid = maze['grid']
    shortest_path = maze.get('shortest_path', [])
    start = maze['start']
    goal = maze['goal']

    # 创建HTML
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .score {{ background: #e8f5e8; padding: 15px; border-radius: 8px; margin: 10px 0; }}
        .viz {{ margin: 20px 0; }}
        img {{ max-width: 100%; border: 1px solid #ccc; }}
    </style>
</head>
<body>
    <h1>{title}</h1>

    <div class="score">
        <h2>评分结果</h2>
        <p>总分: {scores.get('total', 0)}</p>
        <p>S (成功): {scores.get('S', 0)} | Q (质量): {scores.get('Q', 0)} | O (最优性): {scores.get('O', 0)} | P (路径相似度): {scores.get('P', 0)} | A (准确性): {scores.get('A', 0)}</p>
        <p>迷宫大小: {len(grid)}x{len(grid[0])} | 起点: {start} | 终点: {goal}</p>
    </div>"""

    # 如果有图片，添加图片显示
    if image_path and Path(image_path).exists():
        import base64
        with open(image_path, 'rb') as f:
            img_b64 = base64.b64encode(f.read()).decode('utf-8')
        html += f"""
    <div class="viz">
        <h3>迷宫图像</h3>
        <img src="data:image/png;base64,{img_b64}" alt="maze" style="image-rendering: pixelated;">
    </div>"""

    # 添加路径对比图
    html += f"""
    <div class="viz">
        <h3>路径对比</h3>
        <div id="pathplot"></div>
        <script>
            var sp = {json.dumps(shortest_path)};
            var mp = {json.dumps(model_path)};
            function toXY(path){{
                return [path.map(p=>p[1]), path.map(p=>p[0])];
            }}
            var spxy = toXY(sp);
            var mpxy = toXY(mp);
            var data = [
                {{x: spxy[0], y: spxy[1], mode: 'lines+markers', name: '最短路径'}},
                {{x: mpxy[0], y: mpxy[1], mode: 'lines+markers', name: '模型路径'}}
            ];
            Plotly.newPlot('pathplot', data, {{yaxis:{{autorange:'reversed'}}}});
        </script>
    </div>

</body>
</html>"""

    # 保存文件
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(html, encoding='utf-8')
    print(f"可视化报告已保存到: {output_path}")


def create_summary_visualization_html(results: List[Dict], output_path: str, title: str = "MazeBench Summary"):
    """
    创建汇总可视化HTML报告

    Args:
        results: 所有评测结果列表
        output_path: 输出HTML文件路径
        title: 报告标题
    """

    # 计算统计数据
    valid_scores = [r['scores']['total'] for r in results if r['scores']['total'] > 0]
    avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0

    # 创建HTML
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .summary {{ background: #e8f5e8; padding: 20px; border-radius: 8px; margin: 10px 0; }}
        table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background: #f2f2f2; }}
        .error {{ color: red; }}
    </style>
</head>
<body>
    <h1>{title}</h1>

    <div class="summary">
        <h2>汇总统计</h2>
        <p>平均总分: {avg_score:.2f}</p>
        <p>总项目数: {len(results)} | 有效项目数: {len(valid_scores)}</p>
    </div>

    <h2>详细结果</h2>
    <table>
        <tr>
            <th>#</th>
            <th>总分</th>
            <th>S</th>
            <th>Q</th>
            <th>O</th>
            <th>P</th>
            <th>A</th>
            <th>状态</th>
        </tr>"""

    for i, result in enumerate(results):
        status = "Normal"
        if 'error' in result:
            status = f'<span class="error">Error</span>'

        html += f"""
        <tr>
            <td>{i+1}</td>
            <td>{result['scores']['total']}</td>
            <td>{result['scores'].get('S', 0)}</td>
            <td>{result['scores'].get('Q', 0)}</td>
            <td>{result['scores'].get('O', 0)}</td>
            <td>{result['scores'].get('P', 0)}</td>
            <td>{result['scores'].get('A', 0)}</td>
            <td>{status}</td>
        </tr>"""

    html += """
    </table>
</body>
</html>"""

    # 保存文件
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(html, encoding='utf-8')
    print(f"汇总报告已保存到: {output_path}")