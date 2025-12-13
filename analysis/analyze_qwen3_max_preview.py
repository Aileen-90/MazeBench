from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def load_records(root: Path) -> List[Dict]:
    """读取目录下所有 json 结果，容错跳过坏文件。"""
    records: List[Dict] = []
    for p in root.glob("*.json"):
        if not p.is_file():
            continue
        try:
            with p.open(encoding="utf-8") as f:
                obj = json.load(f)
                if isinstance(obj, dict):
                    records.append(obj)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"警告: 跳过无法解析的文件 {p.name}: {e}")
    return records


def basic_stats(records: Iterable[Dict]) -> Dict[str, float]:
    """基础成功率与平均值统计。"""
    records = list(records)
    total = len(records)
    success = sum(1 for r in records if r.get("success"))
    rate = round(success / total, 4) if total else 0.0
    avg_steps = _avg(records, "steps")
    avg_calls = _avg(records, "api_calls")
    avg_calls_success = _avg([r for r in records if r.get("success")], "api_calls")
    total_time = sum(r.get("total_time", 0) for r in records)
    return {
        "total_cases": total,
        "success": success,
        "success_rate": rate,
        "avg_steps": avg_steps,
        "avg_api_calls": avg_calls,
        "avg_api_calls_success": avg_calls_success,
        "total_time": round(total_time, 2),
    }


def stats_with_errors(records: Iterable[Dict]) -> Dict:
    """基础统计 + 错误分布。"""
    records = list(records)
    return {
        **basic_stats(records),
        "errors": aggregate_errors(records),
    }


def aggregate_errors(records: Iterable[Dict], keys: Iterable[str] | None = None) -> Dict[str, int]:
    """聚合错误类型次数，默认根据字段名自动发现。"""
    records = list(records)
    if keys is None:
        keys = {
            k
            for r in records
            for k, v in r.items()
            if isinstance(v, (int, float)) and any(s in k for s in ("error", "collision", "invalid"))
        }
    return {k: int(sum(r.get(k, 0) for r in records if isinstance(r.get(k, 0), (int, float)))) for k in sorted(keys)}


def build_summary(records: List[Dict]) -> Dict:
    """汇总总览、错误分布、按尺寸/迷宫分组。"""
    return {
        "overview": basic_stats(records),
        "errors": aggregate_errors(records),
        "by_size": stats_by_size(records),
        "by_maze": stats_by_maze(records),
    }


def stats_by_maze(records: Iterable[Dict]) -> Dict[str, Dict[str, float]]:
    """按迷宫编号分组的成功率与错误。"""
    buckets: Dict[str, List[Dict]] = defaultdict(list)
    for r in records:
        buckets[r.get("maze", "unknown")].append(r)
    return {maze: stats_with_errors(rs) for maze, rs in buckets.items()}


def stats_by_size(records: Iterable[Dict]) -> Dict[str, Dict]:
    """按尺寸汇总（含下属迷宫次数一致性检查）。"""
    size_buckets: Dict[str, List[Dict]] = defaultdict(list)
    for r in records:
        size, _ = parse_maze_info(r.get("maze", "unknown"))
        size_buckets[size].append(r)

    result: Dict[str, Dict] = {}
    for size, rs in size_buckets.items():
        maze_counts: Dict[str, int] = defaultdict(int)
        for r in rs:
            maze_id = r.get("maze", "unknown")
            maze_counts[maze_id] += 1
        counts = list(maze_counts.values())
        result[size] = {
            **stats_with_errors(rs),
            "maze_counts": dict(sorted(maze_counts.items())),
            "count_min": min(counts) if counts else 0,
            "count_max": max(counts) if counts else 0,
            "count_consistent": len(set(counts)) <= 1,
        }
    return result


def parse_maze_info(maze: str) -> Tuple[str, str]:
    """解析迷宫名，返回 (尺寸, 编号)。"""
    try:
        parts = maze.split("_")
        size = parts[1] if len(parts) > 1 else "unknown"
        idx = parts[2] if len(parts) > 2 else "unknown"
        return size, idx
    except Exception:
        return "unknown", "unknown"


def _avg(records: Iterable[Dict], key: str) -> float:
    values = [r.get(key) for r in records if isinstance(r.get(key), (int, float))]
    return round(sum(values) / len(values), 2) if values else 0.0


def list_available_folders(analysis_dir: Path) -> List[str]:
    """列出 analysis 目录下所有可用的结果文件夹。"""
    folders = []
    for item in analysis_dir.iterdir():
        if item.is_dir() and not item.name.startswith("."):
            # 检查文件夹中是否有 json 文件
            if any(item.glob("*.json")):
                folders.append(item.name)
    return sorted(folders)


def main() -> None:
    analysis_dir = Path(__file__).parent
    
    # 列出可用的结果文件夹
    available_folders = list_available_folders(analysis_dir)
    if not available_folders:
        print("未找到包含 JSON 文件的结果文件夹")
        return
    
    # 让用户选择文件夹
    print("可用的结果文件夹:")
    for i, folder in enumerate(available_folders, 1):
        print(f"  {i}. {folder}")
    
    while True:
        try:
            choice = input(f"\n请选择要分析的文件夹 (1-{len(available_folders)}) 或直接输入文件夹名: ").strip()
            
            # 尝试作为数字解析
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(available_folders):
                    folder_name = available_folders[idx]
                    break
                else:
                    print(f"请输入 1-{len(available_folders)} 之间的数字")
                    continue
            
            # 尝试作为文件夹名解析
            if choice in available_folders:
                folder_name = choice
                break
            else:
                print(f"未找到文件夹 '{choice}'，请重新输入")
        except (KeyboardInterrupt, EOFError):
            print("\n已取消")
            return
    
    # 读取结果文件
    root = analysis_dir / folder_name
    print(f"\n正在分析文件夹: {folder_name}")
    records = load_records(root)
    if not records:
        print("未找到可用的结果文件")
        return
    
    print(f"找到 {len(records)} 条记录")
    
    # 生成汇总
    summary = build_summary(records)

    print("\n== 总览 ==")
    print(summary["overview"])
    print("\n== 错误分布 ==")
    print(summary["errors"])
    print("\n== 按迷宫分组成功率 ==")
    for maze, stat in sorted(summary["by_maze"].items()):
        print(maze, stat)

    # 创建 summaries 文件夹并保存结果
    summaries_dir = analysis_dir / "summaries"
    summaries_dir.mkdir(exist_ok=True)
    
    out_filename = f"{folder_name}_summary.json"
    out_path = summaries_dir / out_filename
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n汇总文件已生成: {out_path}")


if __name__ == "__main__":
    main()

