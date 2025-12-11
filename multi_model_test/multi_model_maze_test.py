#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MazeBench 多模型多线程迷宫测试脚本

功能：
- 多线程测试多个模型在不同大小迷宫中的路径找到能力
- 每个迷宫测试10次，计算成功率和平均步数
- 结果按迷宫大小和模型名称组织文件夹结构

使用示例：
python multi_model_maze_test.py --models gpt-4 gpt-3.5-turbo --sizes 5x5 9x9 15x15 --trials 10 --workers 4
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Any, Tuple
import sys
import glob

# 添加项目根目录到Python路径
# project_root = Path(__file__).parent
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sandbox.ai import run_ai_sandbox
from utils.io import load_config, apply_env_keys
from utils.logging import setup_logging, get_logger

logger = get_logger(__name__)


class MultiModelMazeTester:
    """多模型迷宫测试器"""

    def __init__(self, models: List[str], maze_sizes: List[str], trials_per_maze: int = 10,
                 max_workers: int = 4, output_base_dir: str = "multi_model_results"):
        """
        初始化测试器

        Args:
            models: 要测试的模型列表
            maze_sizes: 迷宫大小列表，如 ['5x5', '9x9', '15x15']
            trials_per_maze: 每个迷宫的测试次数
            max_workers: 最大并发线程数
            output_base_dir: 结果输出基础目录
        """
        self.models = models
        self.maze_sizes = maze_sizes
        self.trials_per_maze = trials_per_maze
        self.max_workers = max_workers
        self.output_base_dir = Path(output_base_dir)

        # 加载基础配置
        self.base_cfg = load_config()
        self.base_cfg = apply_env_keys(self.base_cfg)

        # 创建输出目录
        self.output_base_dir.mkdir(exist_ok=True)

        logger.info(f"初始化多模型测试器: {len(models)}个模型, {len(maze_sizes)}种迷宫大小, 每迷宫{trials_per_maze}次测试")

    def get_available_mazes(self) -> Dict[str, List[str]]:
        """获取所有可用迷宫，按大小分组"""
        mazes_by_size = {}

        for size in self.maze_sizes:
            maze_dir = Path(f"mazes_{size}")
            if not maze_dir.exists():
                logger.warning(f"迷宫目录不存在: {maze_dir}")
                continue

            # 查找该目录下的所有迷宫文件
            maze_files = list(maze_dir.glob("*.json"))
            maze_names = [f.stem for f in maze_files]  # 去掉.json后缀

            if maze_names:
                mazes_by_size[size] = maze_names
                logger.info(f"找到 {size} 迷宫: {len(maze_names)} 个")
            else:
                logger.warning(f"{size} 目录下未找到迷宫文件")

        return mazes_by_size

    def run_single_test(self, model: str, maze_size: str, maze_name: str, trial_id: int) -> Dict[str, Any]:
        """运行单个测试"""
        logger.info(f"开始测试: 模型={model}, 迷宫={maze_size}/{maze_name}, 尝试={trial_id+1}")

        try:
            # 创建模型特定的配置
            cfg = self.base_cfg.copy()
            cfg['model'] = model

            # 设置迷宫路径
            cfg['sandbox'] = cfg.get('sandbox', {})
            cfg['sandbox']['mazes_path'] = f"mazes_{maze_size}/"

            # 运行AI沙盒测试 - 现在会返回完整的统计信息
            result = run_ai_sandbox(maze_name, model, cfg.get('sandbox', {}).get('max_steps', 50), cfg)

            # 提取统计信息
            stats = result.get('stats', {})
            
            test_result = {
                'model': model,
                'maze_size': maze_size,
                'maze_name': maze_name,
                'trial_id': trial_id,
                'success': result.get('success', False),
                'steps': result.get('steps', 0),
                'total_steps': len(result.get('path', [])),
                'error': result.get('error', None),
                'timestamp': time.time(),
                'result_file': result.get('result_file', None),
                
                # ============ 新增：统计字段 ============
                'api_calls': result.get('api_calls', 0),
                'total_time': result.get('total_time', 0),
                'wall_collisions': result.get('wall_collisions', 0),
                'invalid_jumps': result.get('invalid_jumps', 0),
                'parse_errors': result.get('parse_errors', 0),
                'error_count': result.get('error_count', 0),
                'errors': result.get('stats', {}).get('errors', []),
                'error_types': self._extract_error_types(result),
                # =======================================
            }

            logger.info(f"测试完成: {model}/{maze_size}/{maze_name} 尝试{trial_id+1} - "
                    f"成功={test_result['success']}, 步数={test_result['steps']}, "
                    f"API调用={test_result['api_calls']}, 错误数={test_result['error_count']}, "
                    f"总时间={test_result['total_time']:.2f}秒")

            return test_result

        except Exception as e:
            logger.error(f"测试失败: {model}/{maze_size}/{maze_name} 尝试{trial_id+1} - {e}")
            return {
                'model': model,
                'maze_size': maze_size,
                'maze_name': maze_name,
                'trial_id': trial_id,
                'success': False,
                'steps': 0,
                'total_steps': 0,
                'error': str(e),
                'timestamp': time.time(),
                'result_file': None,
                'api_calls': 0,
                'total_time': 0,
                'wall_collisions': 0,
                'invalid_jumps': 0,
                'parse_errors': 0,
                'error_count': 0,
                'errors': [],
                'error_types': {}
            }

    def _extract_error_types(self, result: Dict[str, Any]) -> Dict[str, int]:
        """提取错误类型统计"""
        error_types = {}
        errors = result.get('stats', {}).get('errors', [])
        
        for error in errors:
            error_type = error.get('type', 'unknown')
            subtype = error.get('subtype', None)
            
            if subtype:
                key = f"{error_type}.{subtype}"
            else:
                key = error_type
                
            error_types[key] = error_types.get(key, 0) + 1
        
        return error_types







    # def run_single_test(self, model: str, maze_size: str, maze_name: str, trial_id: int) -> Dict[str, Any]:
    #     """运行单个测试"""
    #     logger.info(f"开始测试: 模型={model}, 迷宫={maze_size}/{maze_name}, 尝试={trial_id+1}")

    #     try:
    #         # 创建模型特定的配置
    #         cfg = self.base_cfg.copy()
    #         cfg['model'] = model

    #         # 设置迷宫路径
    #         cfg['sandbox'] = cfg.get('sandbox', {})
    #         cfg['sandbox']['mazes_path'] = f"mazes_{maze_size}/"

    #         # 运行AI沙盒测试
    #         result = run_ai_sandbox(maze_name, model, cfg.get('sandbox', {}).get('max_steps', 50), cfg)

    #         test_result = {
    #             'model': model,
    #             'maze_size': maze_size,
    #             'maze_name': maze_name,
    #             'trial_id': trial_id,
    #             'success': result.get('success', False),
    #             'steps': result.get('steps', 0),
    #             'total_steps': len(result.get('path', [])),
    #             'error': result.get('error', None),
    #             'timestamp': time.time(),
    #             'result_file': result.get('result_file', None)
    #         }

    #         logger.info(f"测试完成: {model}/{maze_size}/{maze_name} 尝试{trial_id+1} - 成功={test_result['success']}, 步数={test_result['steps']}")

    #         return test_result

    #     except Exception as e:
    #         logger.error(f"测试失败: {model}/{maze_size}/{maze_name} 尝试{trial_id+1} - {e}")
    #         return {
    #             'model': model,
    #             'maze_size': maze_size,
    #             'maze_name': maze_name,
    #             'trial_id': trial_id,
    #             'success': False,
    #             'steps': 0,
    #             'total_steps': 0,
    #             'error': str(e),
    #             'timestamp': time.time(),
    #             'result_file': None
    #         }

    def save_test_result(self, result: Dict[str, Any]) -> str:
        """保存单个测试结果到文件"""
        model = result['model']
        maze_size = result['maze_size']

        # 创建目录结构: results/{maze_size}/{model}/
        result_dir = self.output_base_dir / maze_size / model
        result_dir.mkdir(parents=True, exist_ok=True)

        # 文件名格式: maze_name_trial_id.json
        filename = f"{result['maze_name']}_trial_{result['trial_id']}.json"
        filepath = result_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        return str(filepath)

    def run_all_tests(self) -> Dict[str, Any]:
        """运行所有测试"""
        # 获取可用迷宫
        mazes_by_size = self.get_available_mazes()
        if not mazes_by_size:
            logger.error("未找到任何可用迷宫")
            return {}

        # 收集所有测试任务
        all_tasks = []
        for maze_size, maze_names in mazes_by_size.items():
            for maze_name in maze_names:
                for model in self.models:
                    for trial_id in range(self.trials_per_maze):
                        all_tasks.append((model, maze_size, maze_name, trial_id))

        logger.info(f"总共需要运行 {len(all_tasks)} 个测试任务")

        # 使用多线程执行测试
        results = []
        completed_count = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_task = {
                executor.submit(self.run_single_test, model, maze_size, maze_name, trial_id): (model, maze_size, maze_name, trial_id)
                for model, maze_size, maze_name, trial_id in all_tasks
            }

            # 收集结果
            for future in as_completed(future_to_task):
                task_info = future_to_task[future]
                model, maze_size, maze_name, trial_id = task_info

                try:
                    result = future.result()
                    results.append(result)

                    # 保存结果文件
                    result_file = self.save_test_result(result)
                    logger.info(f"结果已保存: {result_file}")

                except Exception as e:
                    logger.error(f"任务执行失败 {task_info}: {e}")
                    # 添加错误结果
                    error_result = {
                        'model': model,
                        'maze_size': maze_size,
                        'maze_name': maze_name,
                        'trial_id': trial_id,
                        'success': False,
                        'steps': 0,
                        'total_steps': 0,
                        'error': str(e),
                        'timestamp': time.time(),
                        'result_file': None
                    }
                    results.append(error_result)

                completed_count += 1
                if completed_count % 10 == 0:
                    logger.info(f"进度: {completed_count}/{len(all_tasks)} ({completed_count/len(all_tasks)*100:.1f}%)")

        # 生成汇总报告
        summary = self.generate_summary_report(results)
        self.save_summary_report(summary)

        logger.info("所有测试完成！")
        return summary

    def generate_summary_report(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成汇总报告"""
        summary = {
            'timestamp': time.time(),
            'total_tests': len(results),
            'models': self.models,
            'maze_sizes': self.maze_sizes,
            'trials_per_maze': self.trials_per_maze,
            'model_stats': {},
            'maze_size_stats': {},
            'overall_stats': {}
        }

        # 按模型统计
        for model in self.models:
            model_results = [r for r in results if r['model'] == model]
            successful_tests = [r for r in model_results if r['success']]
            avg_steps = sum(r['steps'] for r in successful_tests) / len(successful_tests) if successful_tests else 0

            summary['model_stats'][model] = {
                'total_tests': len(model_results),
                'successful_tests': len(successful_tests),
                'success_rate': len(successful_tests) / len(model_results) if model_results else 0,
                'avg_steps_successful': avg_steps,
                'failed_tests': len(model_results) - len(successful_tests)
            }

        # 按迷宫大小统计
        for maze_size in self.maze_sizes:
            size_results = [r for r in results if r['maze_size'] == maze_size]
            successful_tests = [r for r in size_results if r['success']]
            avg_steps = sum(r['steps'] for r in successful_tests) / len(successful_tests) if successful_tests else 0

            summary['maze_size_stats'][maze_size] = {
                'total_tests': len(size_results),
                'successful_tests': len(successful_tests),
                'success_rate': len(successful_tests) / len(size_results) if size_results else 0,
                'avg_steps_successful': avg_steps,
                'failed_tests': len(size_results) - len(successful_tests)
            }

        # 总体统计
        total_successful = sum(len([r for r in results if r['success'] and r['model'] == model]) for model in self.models)
        summary['overall_stats'] = {
            'total_successful': total_successful,
            'overall_success_rate': total_successful / len(results) if results else 0,
            'total_failed': len(results) - total_successful
        }

        return summary

    def save_summary_report(self, summary: Dict[str, Any]):
        """保存汇总报告"""
        # 保存JSON格式的详细报告
        summary_json_path = self.output_base_dir / "summary_report.json"
        with open(summary_json_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # 保存人类可读的文本报告
        summary_txt_path = self.output_base_dir / "summary_report.txt"
        with open(summary_txt_path, 'w', encoding='utf-8') as f:
            f.write("MazeBench 多模型迷宫测试汇总报告\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(summary['timestamp']))}\n")
            f.write(f"总测试数: {summary['total_tests']}\n")
            f.write(f"模型列表: {', '.join(summary['models'])}\n")
            f.write(f"迷宫大小: {', '.join(summary['maze_sizes'])}\n")
            f.write(f"每迷宫测试次数: {summary['trials_per_maze']}\n\n")

            f.write("各模型性能统计:\n")
            f.write("-" * 30 + "\n")
            for model, stats in summary['model_stats'].items():
                f.write(f"{model}:\n")
                f.write(".1f")
                f.write(".1f")
                f.write(f"  失败次数: {stats['failed_tests']}\n\n")

            f.write("各迷宫大小统计:\n")
            f.write("-" * 30 + "\n")
            for size, stats in summary['maze_size_stats'].items():
                f.write(f"{size} 迷宫:\n")
                f.write(".1f")
                f.write(".1f")
                f.write(f"  失败次数: {stats['failed_tests']}\n\n")

            f.write("总体统计:\n")
            f.write("-" * 30 + "\n")
            f.write(".1f")
            f.write(f"总失败数: {summary['overall_stats']['total_failed']}\n")

        logger.info(f"汇总报告已保存: {summary_json_path}, {summary_txt_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="MazeBench 多模型迷宫测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:

# 测试两个模型在三种迷宫大小上，每迷宫10次测试
python multi_model_maze_test.py --models gpt-4 gpt-3.5-turbo --sizes 5x5 9x9 15x15 --trials 10

# 指定并发线程数
python multi_model_maze_test.py --models gpt-4 --sizes 5x5 --trials 5 --workers 2

# 自定义输出目录
python multi_model_maze_test.py --models gpt-4 --sizes 9x9 --output-dir my_results
        """
    )

    parser.add_argument(
        '--models', nargs='+', required=True,
        help='要测试的模型列表'
    )
    parser.add_argument(
        '--sizes', nargs='+', required=True,
        help='迷宫大小列表，如 5x5 9x9 15x15'
    )
    parser.add_argument(
        '--trials', type=int, default=10,
        help='每个迷宫的测试次数 (默认: 10)'
    )
    parser.add_argument(
        '--workers', type=int, default=4,
        help='并发线程数 (默认: 4)'
    )
    parser.add_argument(
        '--output-dir', default='multi_model_results',
        help='结果输出目录 (默认: multi_model_results)'
    )

    args = parser.parse_args()

    # 设置日志
    setup_logging()

    # 创建测试器并运行
    tester = MultiModelMazeTester(
        models=args.models,
        maze_sizes=args.sizes,
        trials_per_maze=args.trials,
        max_workers=args.workers,
        output_base_dir=args.output_dir
    )

    start_time = time.time()
    summary = tester.run_all_tests()
    end_time = time.time()

    logger.info(f"测试完成！总耗时: {(end_time - start_time):.1f}秒")
    # 打印关键统计信息
    print("\n" + "="*60)
    print("测试完成！关键统计:")
    print("="*60)
    for model, stats in summary.get('model_stats', {}).items():
        print(f"{model}: 成功率 {stats['success_rate'] * 100:.1f}%, 平均步数 {stats['avg_steps_successful']:.1f}")
    print(f"\n总测试数: {summary.get('total_tests', 0)}")
    print(f"总体成功率: {summary.get('overall_stats', {}).get('overall_success_rate', 0):.1f}%")


if __name__ == '__main__':
    main()