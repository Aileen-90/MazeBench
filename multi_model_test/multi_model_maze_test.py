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
import psutil
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
                 max_workers: int = 4, output_base_dir: str = "multi_model_results",
                 mazes_base_dir: str = None, batch_size: int = 25, retry_count: int = 2):
        """
        初始化测试器

        Args:
            models: 要测试的模型列表
            maze_sizes: 迷宫大小列表，如 ['5x5', '9x9', '15x15']
            trials_per_maze: 每个迷宫的测试次数
            max_workers: 最大并发线程数
            output_base_dir: 结果输出基础目录
            mazes_base_dir: 迷宫父目录（默认从配置文件读取或使用 'mazes'）
            batch_size: 批量提交任务的大小
            retry_count: 失败任务的重试次数
        """
        self.models = models
        self.maze_sizes = maze_sizes
        self.trials_per_maze = trials_per_maze
        # 根据Kaggle环境优化线程数
        self.max_workers = min(max_workers, os.cpu_count() or 4, 8)  # Kaggle通常限制CPU资源
        self.output_base_dir = Path(output_base_dir)
        self.batch_size = batch_size
        self.retry_count = retry_count

        # 加载基础配置
        self.base_cfg = load_config()
        self.base_cfg = apply_env_keys(self.base_cfg)

        # 确定迷宫父目录
        if mazes_base_dir is None:
            # 从配置文件读取，默认使用 'mazes'
            self.mazes_base_dir = Path(self.base_cfg.get('sandbox', {}).get('mazes_path', 'mazes/'))
            # 如果配置中是相对路径，去掉末尾的斜杠
            if str(self.mazes_base_dir).endswith('/'):
                self.mazes_base_dir = Path(str(self.mazes_base_dir)[:-1])
        else:
            self.mazes_base_dir = Path(mazes_base_dir)

        # 创建输出目录
        self.output_base_dir.mkdir(exist_ok=True)

        logger.info(f"初始化多模型测试器: {len(models)}个模型, {len(maze_sizes)}种迷宫大小, 每迷宫{trials_per_maze}次测试, 迷宫目录: {self.mazes_base_dir}, 最大线程数: {self.max_workers}")
        self.size_to_dir = {}  # 缓存尺寸到实际目录名的映射

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

        # 批量处理任务，避免一次性提交过多任务
        for batch_start in range(0, len(all_tasks), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(all_tasks))
            batch_tasks = all_tasks[batch_start:batch_end]
            
            logger.info(f"处理批次 {batch_start//self.batch_size + 1}/{(len(all_tasks) + self.batch_size - 1)//self.batch_size}, 任务数: {len(batch_tasks)}")
            
            # 检查内存使用情况
            self._check_memory_usage()
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # 提交当前批次任务
                future_to_task = {
                    executor.submit(self._run_task_with_retry, task_info): task_info
                    for task_info in batch_tasks
                }

                # 收集结果
                for future in as_completed(future_to_task):
                    task_info = future_to_task[future]
                    model, maze_size, maze_name, trial_id = task_info

                    try:
                        result = future.result()
                        if result:
                            results.append(result)

                            # 保存结果文件
                            result_file = self.save_test_result(result)
                            logger.info(f"结果已保存: {result_file}")

                    except Exception as e:
                        logger.error(f"任务执行失败（包括重试） {task_info}: {e}")
                        # 添加最终错误结果
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

    def _run_task_with_retry(self, task_info):
        """带重试机制的任务执行"""
        model, maze_size, maze_name, trial_id = task_info
        
        for attempt in range(self.retry_count + 1):
            try:
                result = self.run_single_test(model, maze_size, maze_name, trial_id)
                return result
            except Exception as e:
                if attempt < self.retry_count:
                    wait_time = 2 ** attempt  # 指数退避
                    logger.warning(f"任务 {task_info} 执行失败 (尝试 {attempt + 1}/{self.retry_count + 1}): {e}, 将在 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    raise

    def _check_memory_usage(self):
        """检查内存使用情况，避免OOM错误"""
        process = psutil.Process()
        mem_info = process.memory_info()
        mem_used_gb = mem_info.rss / (1024 * 1024 * 1024)
        
        logger.info(f"内存使用情况: {mem_used_gb:.2f} GB")
        
        # 如果内存使用超过80%，适当暂停
        if mem_used_gb > 9.0:  # Kaggle通常提供16GB内存
            logger.warning(f"内存使用较高 ({mem_used_gb:.2f} GB)，暂停10秒...")
            time.sleep(10)

    def _find_dir(self, size: str) -> str:
        """查找匹配尺寸的目录名，支持模糊匹配"""
        if size in self.size_to_dir:
            return self.size_to_dir[size]
        
        if not self.mazes_base_dir.exists():
            return size
        
        # 精确匹配
        if (self.mazes_base_dir / size).exists():
            self.size_to_dir[size] = size
            return size
        
        # 模糊匹配
        for d in self.mazes_base_dir.iterdir():
            if d.is_dir() and size in d.name:
                self.size_to_dir[size] = d.name
                logger.info(f"模糊匹配: {size} -> {d.name}")
                return d.name
        
        return size

    def get_available_mazes(self) -> Dict[str, List[str]]:
        """获取所有可用迷宫，按大小分组，支持模糊匹配"""
        mazes_by_size = {}
        for size in self.maze_sizes:
            dir_name = self._find_dir(size)
            maze_dir = self.mazes_base_dir / dir_name
            if not maze_dir.exists():
                logger.warning(f"迷宫目录不存在: {maze_dir}")
                continue
            maze_files = list(maze_dir.glob("*.json"))
            maze_names = [f.stem for f in maze_files]
            if maze_names:
                mazes_by_size[size] = maze_names
                logger.info(f"找到 {size} 迷宫: {len(maze_names)} 个")
        return mazes_by_size

    def run_single_test(self, model: str, maze_size: str, maze_name: str, trial_id: int) -> Dict[str, Any]:
        """运行单个测试"""
        logger.info(f"开始测试: 模型={model}, 迷宫={maze_size}/{maze_name}, 尝试={trial_id+1}")

        try:
            # 创建模型特定的配置
            cfg = self.base_cfg.copy()
            cfg['model'] = model

            # 设置迷宫路径: 使用匹配到的目录名
            cfg['sandbox'] = cfg.get('sandbox', {})
            dir_name = self._find_dir(maze_size)
            cfg['sandbox']['mazes_path'] = str(self.mazes_base_dir / dir_name) + "/"

            # 根据模型名称选择合适的PROVIDER
            if model.startswith('llama') or model.startswith('mistral') or model.startswith('gemma'):
                cfg['PROVIDER'] = 'ollama'
                # 确保使用正确的Ollama配置
                cfg['base_url'] = 'http://localhost:11434/v1'
            elif cfg.get('USE_OPENAI_SDK', False):
                cfg['PROVIDER'] = 'openai'
            else:
                cfg['PROVIDER'] = cfg.get('PROVIDER', 'ark')  # 默认使用配置中的PROVIDER或ark
            
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
                
                # 新增：API响应时间指标
                'avg_response_time': result.get('stats', {}).get('avg_response_time', 0),
                'total_response_time': result.get('stats', {}).get('total_response_time', 0),
                
                # 新增：平均有效步数长度
                'avg_valid_steps_length': self._calculate_avg_valid_steps_length(result),
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
                'error_types': {},
                
                # 新增：API响应时间指标（测试失败时为0）
                'avg_response_time': 0,
                'total_response_time': 0,
                
                # 新增：平均有效步数长度（测试失败时为0）
                'avg_valid_steps_length': 0
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
        
    def _calculate_avg_valid_steps_length(self, result: Dict[str, Any]) -> float:
        """计算平均有效步数长度
        
        如果一次API调用没有error，记录移动path的长度，如果有错，记为0
        """
        history = result.get('stats', {}).get('history', [])
        if not history:
            return 0.0
            
        # 检查每步API调用是否有错误
        error_steps = set()
        errors = result.get('stats', {}).get('errors', [])
        for error in errors:
            error_steps.add(error.get('step', 0))
            
        valid_lengths = []
        for step_info in history:
            step = step_info.get('step', 0)
            # 如果该步没有错误，计算移动路径长度
            if step not in error_steps:
                position_before = step_info.get('position_before')
                position_after = step_info.get('position_after')
                # 如果位置信息存在且不同，则计算步数
                if position_before and position_after and position_before != position_after:
                    # 检查路径中该步的移动距离
                    path = result.get('path', [])
                    if len(path) >= 2:
                        # 查找该步在路径中的起始和结束位置
                        try:
                            start_idx = path.index(position_before)
                            end_idx = path.index(position_after)
                            if end_idx > start_idx:
                                step_length = end_idx - start_idx
                                valid_lengths.append(step_length)
                            else:
                                valid_lengths.append(1)  # 单步移动
                        except ValueError:
                            valid_lengths.append(1)  # 无法在路径中找到位置，假设为单步移动
                    else:
                        valid_lengths.append(1)  # 路径长度不足，假设为单步移动
                else:
                    valid_lengths.append(0)  # 位置相同，没有移动
            else:
                valid_lengths.append(0)  # 有错误，记为0
                
        if not valid_lengths:
            return 0.0
            
        # 计算平均值，包含所有步数（包括无效移动）
        return sum(valid_lengths) / len(valid_lengths)







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
            avg_api_calls = sum(r.get('api_calls', 0) for r in model_results) / len(model_results) if model_results else 0

            summary['model_stats'][model] = {
                'total_tests': len(model_results),
                'successful_tests': len(successful_tests),
                'success_rate': len(successful_tests) / len(model_results) if model_results else 0,
                'avg_steps_successful': avg_steps,
                'avg_api_calls': avg_api_calls,
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

    def collect_and_save_statistics(self):
        """
        收集并统计所有测试结果，保存到result文件夹
        """
        import json
        import os
        from datetime import datetime
        from pathlib import Path
        from typing import Dict, List, Any
        
        logger.info("开始收集并统计测试结果...")
        
        # 创建result文件夹
        result_dir = Path("result")
        result_dir.mkdir(exist_ok=True)
        
        for maze_size in self.maze_sizes:
            for model in self.models:
                # 结果文件路径
                results_dir = self.output_base_dir / maze_size / model
                
                if not results_dir.exists():
                    logger.warning(f"未找到结果目录 {results_dir}")
                    continue
                
                # 收集所有JSON文件
                json_files = list(results_dir.glob("*.json"))
                if not json_files:
                    logger.warning(f"在 {results_dir} 中未找到JSON文件")
                    continue
                
                logger.info(f"为 {model} {maze_size} 找到 {len(json_files)} 个测试结果文件")
                
                # 统计指标
                total_tests = 0
                successful_tests = 0
                total_avg_response_time = 0
                total_avg_valid_length = 0
                
                for file_path in json_files:
                    total_tests += 1
                    
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        
                        # 统计成功次数
                        if data.get("success", False):
                            successful_tests += 1
                        
                        # 统计平均API响应时间
                        avg_response_time = data.get("avg_response_time", 0)
                        total_avg_response_time += avg_response_time
                        
                        # 统计平均有效长度
                        avg_valid_length = data.get("avg_valid_steps_length", 0)
                        total_avg_valid_length += avg_valid_length
                        
                    except json.JSONDecodeError:
                        logger.warning(f"无法解析文件 {file_path}")
                        continue
                    except Exception as e:
                        logger.warning(f"处理文件 {file_path} 时出错: {str(e)}")
                        continue
                
                # 计算平均值
                avg_accuracy = successful_tests / total_tests if total_tests > 0 else 0
                avg_of_avg_response_time = total_avg_response_time / total_tests if total_tests > 0 else 0
                avg_of_avg_valid_length = total_avg_valid_length / total_tests if total_tests > 0 else 0
                
                # 构建结果字典
                result = {
                    "maze_size": maze_size,
                    "model": model,
                    "total_tests": total_tests,
                    "successful_tests": successful_tests,
                    "average_accuracy": round(avg_accuracy, 4),
                    "average_of_avg_response_time": round(avg_of_avg_response_time, 4),
                    "average_of_avg_valid_length": round(avg_of_avg_valid_length, 4),
                    "statistics_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                # 生成文件名：大小_模型_测试时间.json
                model_name = model.replace(":", "-")  # 替换不允许的文件名字符
                test_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                filename = f"{maze_size}_{model_name}_{test_time}.json"
                filepath = result_dir / filename
                
                # 保存结果
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                
                logger.info(f"统计结果已保存到: {filepath}")
    
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

# 从配置文件读取模型，测试三种迷宫大小，每迷宫10次测试
python run_multi_test.py --sizes 5x5 9x9 15x15 --trials 10

# 测试两个模型在三种迷宫大小上，每迷宫10次测试
python run_multi_test.py --models gpt-4 gpt-3.5-turbo --sizes 5x5 9x9 15x15 --trials 10

# 指定并发线程数
python run_multi_test.py --models gpt-4 --sizes 5x5 --trials 5 --workers 2

# 自定义输出目录和迷宫目录
python run_multi_test.py --models gpt-4 --sizes 9x9 --output-dir my_results --mazes-dir my_mazes

# 完整示例：涵盖所有字段
python run_multi_test.py --models gpt-4 gpt-3.5-turbo --sizes 5x5 9x9 15x15 --trials 10 --workers 4 --output-dir my_results --mazes-dir mazes
        """
    )

    parser.add_argument(
        '--models', nargs='+', required=False,
        help='要测试的模型列表（如果未指定，将从配置文件中读取）'
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
    parser.add_argument(
        '--mazes-dir', default='mazes/',
        help='迷宫父目录路径 (默认: 从配置文件读取或使用 mazes/)'
    )

    args = parser.parse_args()

    # 设置日志
    setup_logging()

    # 如果未指定模型，从配置文件读取
    if args.models is None:
        # 加载配置
        base_cfg = load_config()
        base_cfg = apply_env_keys(base_cfg)
        
        # 优先使用 models 字段（列表），如果没有则使用 model 字段（单个）
        if 'models' in base_cfg and base_cfg['models']:
            models = base_cfg['models']
            if isinstance(models, str):
                models = [models]
            elif not isinstance(models, list):
                models = [str(models)]
        elif 'model' in base_cfg and base_cfg['model']:
            model = base_cfg['model']
            models = [model] if isinstance(model, str) else [str(model)]
        else:
            logger.error("配置文件中未找到 'model' 或 'models' 字段，且命令行也未指定 --models")
            parser.print_help()
            sys.exit(1)
        
        logger.info(f"从配置文件读取模型列表: {models}")
        args.models = models

    # 创建测试器并运行
    tester = MultiModelMazeTester(
        models=args.models,
        maze_sizes=args.sizes,
        trials_per_maze=args.trials,
        max_workers=args.workers,
        output_base_dir=args.output_dir,
        mazes_base_dir=args.mazes_dir
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
        print(f"{model}: 成功率 {stats['success_rate']:.2f}, 平均步数 {stats['avg_steps_successful']:.1f}, 平均API调用 {stats.get('avg_api_calls', 0):.1f}")
    print(f"\n总测试数: {summary.get('total_tests', 0)}")
    print(f"总体成功率: {summary.get('overall_stats', {}).get('overall_success_rate', 0):.2f}")
    
    # 收集并统计测试结果
    tester.collect_and_save_statistics()


if __name__ == '__main__':
    main()