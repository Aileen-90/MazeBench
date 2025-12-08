#!/usr/bin/env python3
"""
迷宫算法批量基准测试脚本 - 同时测试text2d和image2d模式
自动化测试3种算法，8种迷宫大小（5x5到19x19的奇数）
每个配置运行10次，记录正确率
（测试开始之前需要将./config/config.yaml ./config/local.yaml移动到./config_backup文件夹）
"""

import yaml
import subprocess
import json
import time
import re
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
import shutil
import sys
import os

class MazeBenchmarkAutomator:
    def __init__(self, config_dir="config", config_backup_dir="config_backup", config_file="config.yaml", local_config_file="local.yaml"):
        self.config_dir = Path(config_dir)
        self.config_backup_dir = Path(config_backup_dir)
        self.config_file = self.config_backup_dir / config_file
        self.local_config_file = self.config_backup_dir / local_config_file
        self.base_config = self.load_merged_config()
        self.results_dir = Path("benchmark_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # 测试模式：text2d, image2d, 或 both
        self.test_modes = ["text2d", "image2d"]  # 两种模式都测试
        
        # 请在这里指定第三种算法名称
        self.algorithms = ["dfs", "prim", "prim_loops"]  # 请修改第三个算法名称
        
        # 奇数尺寸：5x5 到 19x19
        self.sizes = [f"{n}x{n}" for n in range(5, 20, 2)]  # [5,7,9,11,13,15,17,19]
        
        # image2d特有的配置
        self.cell_px_options = [24]  # 可以测试不同的cell像素大小
        self.image_start_goal_options = ["random"]  # image2d的起点终点模式
        
    def load_merged_config(self) -> Dict:
        """加载并合并配置"""
        config = {}
        
        if self.config_file.exists():
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
        else:
            print(f"警告: 主配置文件不存在: {self.config_file}")
        
        if self.local_config_file.exists():
            with open(self.local_config_file, 'r', encoding='utf-8') as f:
                local_config = yaml.safe_load(f) or {}
                self.merge_configs(config, local_config)
        
        return config
    
    def merge_configs(self, base: Dict, overlay: Dict, path: str = ""):
        """递归合并配置字典"""
        for key in overlay:
            if key in base:
                if isinstance(base[key], dict) and isinstance(overlay[key], dict):
                    self.merge_configs(base[key], overlay[key], f"{path}.{key}")
                else:
                    base[key] = overlay[key]
            else:
                base[key] = overlay[key]
    
    def save_config(self, config_data: Dict, config_file: Path):
        """保存配置到文件"""
        config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
    
    def backup_configs(self):
        """备份原始配置文件"""
        self.backups = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 备份config目录
        backup_dir = self.results_dir / f"config_backup_{timestamp}"
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        if self.config_file.exists():
            backup_file = backup_dir / "config.yaml"
            shutil.copy2(self.config_file, backup_file)
            self.backups["config"] = backup_file
        
        if self.local_config_file.exists():
            backup_file = backup_dir / "local.yaml"
            shutil.copy2(self.local_config_file, backup_file)
            self.backups["local"] = backup_file
        
        print(f"配置文件已备份到: {backup_dir}")
        return backup_dir
    
    def restore_configs(self, backup_dir: Path):
        """从备份恢复配置文件"""
        if backup_dir.exists():
            config_backup = backup_dir / "config.yaml"
            local_backup = backup_dir / "local.yaml"
            
            if config_backup.exists():
                shutil.copy2(config_backup, self.config_file)
            
            if local_backup.exists():
                shutil.copy2(local_backup, self.local_config_file)
            
            print("配置文件已恢复")
    
    def parse_summary_file(self, summary_file: Path) -> Dict:
        """解析summary文件获取评分"""
        try:
            with open(summary_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            scores = {
                "total_score": data.get("avg_total", 0),
                "success_rate": data.get("avg_total", 0),  # 假设avg_total就是成功率
                "item_count": len(data.get("items", [])),
                "raw_data": data
            }
            
            # 计算详细统计
            if "items" in data and data["items"]:
                items = data["items"]
                total_scores = []
                for item in items:
                    if isinstance(item, dict) and "scores" in item:
                        total_scores.append(item["scores"].get("total", 0))
                
                if total_scores:
                    scores["all_scores"] = total_scores
                    scores["success_count"] = sum(1 for s in total_scores if s >= 100)
                    scores["success_rate"] = (scores["success_count"] / len(total_scores)) * 100
                    scores["avg_score"] = np.mean(total_scores)
                    scores["std_score"] = np.std(total_scores)
            
            return scores
            
        except Exception as e:
            print(f"解析summary文件失败 {summary_file}: {e}")
            return {
                "total_score": 0,
                "success_rate": 0,
                "item_count": 0,
                "all_scores": []
            }
    
    def run_single_experiment(self, mode: str, algorithm: str, size: str, 
                             n_runs: int = 1, cell_px: int = 24) -> Dict:
        """运行单个实验配置"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if mode == "image2d":
            exp_name = f"{mode}_{algorithm}_{size}_{cell_px}px"
        else:
            exp_name = f"{mode}_{algorithm}_{size}"
        
        exp_dir = self.results_dir / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"开始实验: {exp_name}")
        print(f"时间: {timestamp}")
        print(f"模式: {mode}, 算法: {algorithm}, 大小: {size}, 运行次数: {n_runs}")
        if mode == "image2d":
            print(f"单元格像素: {cell_px}px")
        print(f"输出目录: {exp_dir}")
        
        # 准备配置
        config = self.base_config.copy()
        
        # 根据模式设置配置
        config["mode"] = mode
        config["output_dir"] = str(exp_dir)
        
        if mode == "text2d":
            config["text2d"]["algorithm"] = algorithm
            config["text2d"]["size"] = size
            config["text2d"]["n"] = n_runs
            config["text2d"]["start_goal"] = "random"
            config["text2d"]["workers"] = min(4, n_runs)  # 合理设置worker数
            
        elif mode == "image2d":
            config["image2d"]["algorithm"] = algorithm
            config["image2d"]["size"] = size
            config["image2d"]["n"] = n_runs
            config["image2d"]["start_goal"] = "random"
            config["image2d"]["cell_px"] = cell_px
            config["image2d"]["workers"] = min(4, n_runs)
        
        # 保存实验配置
        config_file = exp_dir / "experiment_config.yaml"
        self.save_config(config, config_file)
        
        # 临时保存到主配置位置供bench.py读取
        temp_config_file = self.config_dir / "config.yaml"
        self.save_config(config, temp_config_file)
        
        # 运行bench.py
        log_file = exp_dir / "run.log"
        start_time = time.time()
        exit_code = 0
        output = ""
        
        try:
            print(f"运行 bench.py (模式: {mode})...")
            
            # 构建命令
            cmd = [sys.executable, "bench.py"]
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                cwd=os.getcwd()  # 确保在正确目录运行
            )
            
            # 实时输出并记录
            output_lines = []
            with open(log_file, 'w', encoding='utf-8') as log:
                log.write(f"实验: {exp_name}\n")
                log.write(f"开始时间: {datetime.now().isoformat()}\n")
                log.write(f"配置:\n{yaml.dump(config, default_flow_style=False)}\n")
                log.write("="*50 + "\n\n")
                
                for line in process.stdout:
                    line = line.rstrip()
                    output_lines.append(line)
                    print(f"[{exp_name}] {line}")
                    log.write(line + "\n")
                    log.flush()
            
            exit_code = process.wait()
            output = "\n".join(output_lines)
            
        except Exception as e:
            exit_code = -1
            output = f"运行错误: {str(e)}"
            print(f"错误: {output}")
            with open(log_file, 'a', encoding='utf-8') as log:
                log.write(f"\n错误: {output}\n")
        
        finally:
            # 清理临时配置
            if temp_config_file.exists():
                temp_config_file.unlink()
        
        elapsed_time = time.time() - start_time
        
        # 解析结果
        if mode == "text2d":
            summary_file = exp_dir / "text2d_summary.json"
        else:  # image2d
            summary_file = exp_dir / "image2d_summary.json"
        
        scores = self.parse_summary_file(summary_file) if summary_file.exists() else {
            "total_score": 0,
            "success_rate": 0,
            "item_count": 0,
            "all_scores": []
        }
        
        # 收集实验结果
        result = {
            "experiment_name": exp_name,
            "test_mode": mode,
            "algorithm": algorithm,
            "maze_size": size,
            "size_numeric": int(size.split('x')[0]),
            "n_runs": n_runs,
            "cell_px": cell_px if mode == "image2d" else None,
            "timestamp": timestamp,
            "elapsed_time": elapsed_time,
            "exit_code": exit_code,
            "output_dir": str(exp_dir),
            "total_score": scores["total_score"],
            "success_rate": scores["success_rate"],
            "item_count": scores["item_count"],
            "success_count": scores.get("success_count", 0),
            "avg_score": scores.get("avg_score", 0),
            "std_score": scores.get("std_score", 0)
        }
        
        print(f"实验完成: {exp_name}")
        print(f"耗时: {elapsed_time:.2f}秒")
        print(f"正确率: {result['success_rate']:.2f}%")
        print(f"总分: {result['total_score']:.2f}")
        if "success_count" in scores:
            print(f"成功次数: {scores['success_count']}/{scores['item_count']}")
        
        return result
    
    def run_all_experiments(self, n_runs: int = 1):
        """运行所有实验配置"""
        all_results = []
        
        # 备份原始配置
        backup_dir = self.backup_configs()
        
        try:
            total_experiments = 0
            for mode in self.test_modes:
                if mode == "text2d":
                    total_experiments += len(self.algorithms) * len(self.sizes)
                elif mode == "image2d":
                    total_experiments += len(self.algorithms) * len(self.sizes) * len(self.cell_px_options)
            
            print(f"\n总实验数: {total_experiments}")
            print("预计总耗时: 约 {} 分钟".format(total_experiments * 2))
            
            current = 0
            
            # 测试text2d模式
            if "text2d" in self.test_modes:
                print(f"\n{'#'*60}")
                print("开始测试 TEXT2D 模式")
                print(f"{'#'*60}")
                
                for algorithm in self.algorithms:
                    algorithm_results = []
                    
                    for size in self.sizes:
                        current += 1
                        print(f"\n进度: {current}/{total_experiments}")
                        
                        # 运行text2d实验
                        result = self.run_single_experiment(
                            mode="text2d",
                            algorithm=algorithm,
                            size=size,
                            n_runs=n_runs
                        )
                        
                        algorithm_results.append(result)
                        all_results.append(result)
                        
                        # 短暂休息
                        time.sleep(0.5)
                    
                    # 打印算法小结
                    if algorithm_results:
                        avg_success = np.mean([r.get('success_rate', 0) for r in algorithm_results])
                        print(f"\n算法 {algorithm} (text2d) 平均正确率: {avg_success:.2f}%")
            
            # 测试image2d模式
            if "image2d" in self.test_modes:
                print(f"\n{'#'*60}")
                print("开始测试 IMAGE2D 模式")
                print(f"{'#'*60}")
                
                for algorithm in self.algorithms:
                    for cell_px in self.cell_px_options:
                        pixel_results = []
                        
                        for size in self.sizes:
                            current += 1
                            print(f"\n进度: {current}/{total_experiments}")
                            
                            # 运行image2d实验
                            result = self.run_single_experiment(
                                mode="image2d",
                                algorithm=algorithm,
                                size=size,
                                n_runs=n_runs,
                                cell_px=cell_px
                            )
                            
                            pixel_results.append(result)
                            all_results.append(result)
                            
                            # 短暂休息
                            time.sleep(0.5)
                        
                        # 打印像素大小小结
                        if pixel_results:
                            avg_success = np.mean([r.get('success_rate', 0) for r in pixel_results])
                            print(f"\n算法 {algorithm} (image2d, {cell_px}px) 平均正确率: {avg_success:.2f}%")
        
        except KeyboardInterrupt:
            print("\n\n用户中断测试")
            return all_results
        
        except Exception as e:
            print(f"\n测试过程中发生错误: {e}")
            import traceback
            traceback.print_exc()
            return all_results
        
        finally:
            # 恢复原始配置
            self.restore_configs(backup_dir)
        
        return all_results
    
    def generate_summary_report(self, results: List[Dict]):
        """生成汇总报告和分析图表"""
        if not results:
            print("没有测试结果可生成报告")
            return
        
        # 转换为DataFrame以便分析
        df = pd.DataFrame(results)
        
        # 生成Markdown报告
        report_file = self.results_dir / "benchmark_summary.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# 迷宫算法基准测试报告\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 测试配置
            f.write("## 测试配置\n\n")
            f.write(f"- 测试模式: {', '.join(self.test_modes)}\n")
            f.write(f"- 算法: {', '.join(self.algorithms)}\n")
            f.write(f"- 迷宫大小: {', '.join(self.sizes)}\n")
            f.write(f"- 每个配置运行次数: 10\n")
            f.write(f"- 起点终点: 随机\n")
            if "image2d" in self.test_modes:
                f.write(f"- 单元格像素: {', '.join(map(str, self.cell_px_options))}\n")
            f.write(f"- 总实验数: {len(results)}\n\n")
            
            # 详细结果表格
            f.write("## 详细结果\n\n")
            f.write("| 模式 | 算法 | 迷宫大小 | 像素 | 正确率(%) | 总分 | 成功次数/总数 | 耗时(秒) |\n")
            f.write("|------|------|----------|------|-----------|------|---------------|----------|\n")
            
            for result in results:
                mode = result['test_mode']
                algorithm = result['algorithm']
                size = result['maze_size']
                cell_px = result.get('cell_px', 'N/A')
                success_rate = result.get('success_rate', 0)
                total_score = result.get('total_score', 0)
                success_count = result.get('success_count', 0)
                item_count = result.get('item_count', 0)
                elapsed_time = result.get('elapsed_time', 0)
                
                f.write(f"| {mode} | {algorithm} | {size} | {cell_px} | "
                       f"{success_rate:.2f} | {total_score:.2f} | "
                       f"{success_count}/{item_count} | {elapsed_time:.2f} |\n")
            
            # 按模式统计
            f.write("\n## 模式性能对比\n\n")
            for mode in self.test_modes:
                mode_results = [r for r in results if r['test_mode'] == mode]
                if mode_results:
                    avg_success = np.mean([r.get('success_rate', 0) for r in mode_results])
                    avg_time = np.mean([r.get('elapsed_time', 0) for r in mode_results])
                    total_time = sum([r.get('elapsed_time', 0) for r in mode_results])
                    
                    f.write(f"### {mode.upper()}\n")
                    f.write(f"- 平均正确率: {avg_success:.2f}%\n")
                    f.write(f"- 平均耗时: {avg_time:.2f}秒\n")
                    f.write(f"- 总耗时: {total_time:.2f}秒\n\n")
            
            # 算法性能对比（分模式）
            f.write("## 算法性能对比\n\n")
            for mode in self.test_modes:
                mode_results = [r for r in results if r['test_mode'] == mode]
                if not mode_results:
                    continue
                
                f.write(f"### {mode.upper()} 模式\n\n")
                for algorithm in self.algorithms:
                    algo_results = [r for r in mode_results if r['algorithm'] == algorithm]
                    if algo_results:
                        avg_success = np.mean([r.get('success_rate', 0) for r in algo_results])
                        avg_time = np.mean([r.get('elapsed_time', 0) for r in algo_results])
                        
                        f.write(f"- **{algorithm}**: {avg_success:.2f}% 正确率, {avg_time:.2f}秒平均耗时\n")
                f.write("\n")
            
            # 迷宫大小影响分析
            f.write("## 迷宫大小影响分析\n\n")
            for mode in self.test_modes:
                mode_results = [r for r in results if r['test_mode'] == mode]
                if not mode_results:
                    continue
                
                f.write(f"### {mode.upper()} 模式\n\n")
                for size in self.sizes:
                    size_results = [r for r in mode_results if r['maze_size'] == size]
                    if size_results:
                        avg_success = np.mean([r.get('success_rate', 0) for r in size_results])
                        best_algo = max(size_results, key=lambda x: x.get('success_rate', 0))
                        
                        f.write(f"- **{size}**: {avg_success:.2f}% 平均正确率")
                        f.write(f" (最佳算法: {best_algo['algorithm']} - {best_algo['success_rate']:.2f}%)\n")
                f.write("\n")
            
            # 性能建议
            f.write("## 性能建议\n\n")
            
            # 最佳配置
            best_result = max(results, key=lambda x: x.get('success_rate', 0))
            f.write(f"- **最佳整体配置**: {best_result['test_mode']}模式, {best_result['algorithm']}算法, "
                   f"{best_result['maze_size']}大小")
            if best_result['test_mode'] == "image2d":
                f.write(f", {best_result.get('cell_px')}像素")
            f.write(f" (正确率: {best_result['success_rate']:.2f}%)\n")
            
            # 最快配置
            fastest_result = min(results, key=lambda x: x.get('elapsed_time', float('inf')))
            f.write(f"- **最快配置**: {fastest_result['test_mode']}模式, {fastest_result['algorithm']}算法, "
                   f"{fastest_result['maze_size']}大小 (耗时: {fastest_result['elapsed_time']:.2f}秒)\n")
            
            # 模式对比建议
            text2d_results = [r for r in results if r['test_mode'] == 'text2d']
            image2d_results = [r for r in results if r['test_mode'] == 'image2d']
            
            if text2d_results and image2d_results:
                text2d_avg = np.mean([r.get('success_rate', 0) for r in text2d_results])
                image2d_avg = np.mean([r.get('success_rate', 0) for r in image2d_results])
                
                f.write(f"- **模式对比**: TEXT2D平均{text2d_avg:.2f}% vs IMAGE2D平均{image2d_avg:.2f}%\n")
                if text2d_avg > image2d_avg:
                    f.write("  - 建议优先使用TEXT2D模式获得更高正确率\n")
                else:
                    f.write("  - 建议优先使用IMAGE2D模式获得更高正确率\n")
        
        # 保存详细数据
        csv_file = self.results_dir / "detailed_results.csv"
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        
        # 保存为JSON
        json_file = self.results_dir / "all_results.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 生成可视化图表
        self.generate_visualizations(df)
        
        print(f"\n报告已生成:")
        print(f"  - 汇总报告: {report_file}")
        print(f"  - 详细数据: {csv_file}")
        print(f"  - JSON数据: {json_file}")
        print(f"  - 图表目录: {self.results_dir / 'figures'}")
    
    def generate_visualizations(self, df):
        """生成可视化图表"""
        fig_dir = self.results_dir / "figures"
        fig_dir.mkdir(exist_ok=True)
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        if df.empty:
            print("没有数据生成图表")
            return
        
        try:
            # 1. 模式对比柱状图
            plt.figure(figsize=(10, 6))
            mode_groups = df.groupby('test_mode')['success_rate'].mean()
            mode_groups.plot(kind='bar', color=['skyblue', 'lightcoral'])
            plt.xlabel('测试模式')
            plt.ylabel('平均正确率(%)')
            plt.title('不同测试模式的正确率对比')
            plt.ylim(0, 100)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(fig_dir / 'mode_comparison.png', dpi=300)
            plt.close()
            
            # 2. 各算法在不同模式下的表现（分面图）
            fig, axes = plt.subplots(1, len(self.test_modes), figsize=(15, 6))
            if len(self.test_modes) == 1:
                axes = [axes]
            
            for idx, mode in enumerate(self.test_modes):
                mode_data = df[df['test_mode'] == mode]
                
                # 为每个算法收集数据
                algo_success = []
                algo_labels = []
                
                for algorithm in self.algorithms:
                    algo_data = mode_data[mode_data['algorithm'] == algorithm]
                    if not algo_data.empty:
                        algo_success.append(algo_data['success_rate'].mean())
                        algo_labels.append(algorithm)
                
                if algo_success:
                    axes[idx].bar(algo_labels, algo_success, color='steelblue')
                    axes[idx].set_xlabel('算法')
                    axes[idx].set_ylabel('平均正确率(%)')
                    axes[idx].set_title(f'{mode.upper()} 模式')
                    axes[idx].set_ylim(0, 100)
                    axes[idx].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(fig_dir / 'algorithm_by_mode.png', dpi=300)
            plt.close()
            
            # 3. 迷宫大小影响（分模式）
            for mode in self.test_modes:
                mode_data = df[df['test_mode'] == mode]
                if mode_data.empty:
                    continue
                
                plt.figure(figsize=(10, 6))
                
                for algorithm in self.algorithms:
                    algo_data = mode_data[mode_data['algorithm'] == algorithm]
                    if not algo_data.empty:
                        algo_data = algo_data.sort_values('size_numeric')
                        plt.plot(algo_data['size_numeric'], algo_data['success_rate'], 
                                marker='o', linewidth=2, markersize=8, label=algorithm)
                
                plt.xlabel('迷宫大小')
                plt.ylabel('正确率(%)')
                plt.title(f'{mode.upper()}模式 - 迷宫大小对正确率的影响')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.ylim(0, 100)
                plt.tight_layout()
                plt.savefig(fig_dir / f'size_impact_{mode}.png', dpi=300)
                plt.close()
            
            # 4. 耗时分析
            plt.figure(figsize=(12, 6))
            
            # 分组显示
            group_data = []
            group_labels = []
            
            for mode in self.test_modes:
                for algorithm in self.algorithms:
                    combo_data = df[(df['test_mode'] == mode) & (df['algorithm'] == algorithm)]
                    if not combo_data.empty:
                        group_data.append(combo_data['elapsed_time'].values)
                        group_labels.append(f'{mode}\n{algorithm}')
            
            if group_data:
                plt.boxplot(group_data, labels=group_labels)
                plt.ylabel('耗时(秒)')
                plt.title('不同配置的运行时间分布')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(fig_dir / 'runtime_distribution.png', dpi=300)
                plt.close()
            
            print(f"已生成 {len(list(fig_dir.glob('*.png')))} 个图表")
            
        except Exception as e:
            print(f"生成图表时出错: {e}")

def main():
    """主函数"""
    print("="*60)
    print("迷宫算法批量基准测试脚本")
    print("同时测试 TEXT2D 和 IMAGE2D 模式")
    print("="*60)
    
    # 创建自动化器
    automator = MazeBenchmarkAutomator()
    
    # 显示测试计划
    print("\n测试计划:")
    print(f"测试模式: {', '.join(automator.test_modes)}")
    print(f"算法: {', '.join(automator.algorithms)}")
    print(f"迷宫大小: {', '.join(automator.sizes)}")
    print(f"每个配置运行次数: 10")
    
    total_experiments = 0
    for mode in automator.test_modes:
        if mode == "text2d":
            total_experiments += len(automator.algorithms) * len(automator.sizes)
        elif mode == "image2d":
            total_experiments += len(automator.algorithms) * len(automator.sizes) * len(automator.cell_px_options)
    
    print(f"总实验数: {total_experiments}")
    print(f"预期总耗时: 约 {total_experiments * 2} 分钟")
    
    # 确认开始
    response = input("\n是否开始测试? (y/n): ")
    if response.lower() != 'y':
        print("测试已取消")
        return
    
    # 运行所有实验
    print("\n开始批量测试...")
    results = automator.run_all_experiments(n_runs=10)
    
    # 生成报告
    if results:
        print("\n生成报告和分析图表...")
        automator.generate_summary_report(results)
    
    print("\n" + "="*60)
    print("所有测试完成!")
    print(f"结果目录: {automator.results_dir}")
    print("="*60)

if __name__ == "__main__":
    main()