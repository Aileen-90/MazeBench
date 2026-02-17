"""
MazeBenchmark 子模块使用示例

演示如何将MazeBenchmark作为子模块集成到其他项目中
"""

import sys
from pathlib import Path

# 方法1: 直接导入（推荐）
from MazeBenchmark import MazeBenchmarkAPI, MazeEnvironment, quick_test

# 方法2: 如果作为子模块，可能需要添加路径
# submodule_path = Path(__file__).parent / "submodules" / "MazeBenchmark"
# sys.path.insert(0, str(submodule_path))
# from mazebenchmark import MazeBenchmarkAPI

def demo_basic_usage():
    """基础使用演示"""
    print("=== MazeBenchmark 子模块基础使用演示 ===")
    
    # 1. 创建API实例
    api = MazeBenchmarkAPI()
    
    # 2. 获取可用迷宫列表
    mazes = api.list_available_mazes()
    print(f"可用迷宫: {mazes[:3]}... (共{len(mazes)}个)")
    
    # 3. 运行快速测试
    if mazes:
        result = quick_test(mazes[0], "gpt-4")
        print(f"快速测试结果: 成功={result.get('success')}, 步数={result.get('steps')}")

def demo_advanced_usage():
    """高级使用演示"""
    print("\n=== MazeBenchmark 子模块高级使用演示 ===")
    
    # 1. 使用自定义配置
    api = MazeBenchmarkAPI("config/config.yaml")
    
    # 2. 更新配置
    api.update_config({
        'sandbox': {
            'max_steps': 100,
            'visibility': 5
        }
    })
    
    # 3. 运行部分观察模式测试
    mazes = api.list_available_mazes()
    if mazes:
        result = api.run_partial_observe_test(
            maze_name=mazes[0],
            model="gpt-4",
            max_steps=50,
            visibility=3
        )
        print(f"部分观察测试: 成功={result.get('success')}, 步数={result.get('steps')}")

def demo_multi_model_test():
    """多模型测试演示"""
    print("\n=== MazeBenchmark 多模型测试演示 ===")
    
    api = MazeBenchmarkAPI()
    
    # 运行多模型批量测试
    summary = api.run_multi_model_test(
        models=["gpt-4", "gpt-3.5-turbo"],
        maze_sizes=["9x9", "15x15"],
        trials_per_maze=2,
        mode="partial-observe",
        output_dir="multi_test_results"
    )
    
    print(f"测试完成: 总测试数={summary.get('total_tests')}")
    print(f"模型统计: {list(summary.get('model_stats', {}).keys())}")

def demo_custom_integration():
    """自定义集成演示"""
    print("\n=== MazeBenchmark 自定义集成演示 ===")
    
    # 在你的项目中集成MazeBenchmark
    class MyMazeSolver:
        def __init__(self):
            self.benchmark_api = MazeBenchmarkAPI()
            
        def solve_with_benchmark(self, maze_name):
            """使用MazeBenchmark进行求解"""
            # 运行基准测试
            result = self.benchmark_api.run_partial_observe_test(maze_name)
            
            # 分析结果
            if result.get('success'):
                print(f"AI成功求解迷宫，步数: {result.get('steps')}")
            else:
                print(f"AI求解失败，原因: {result.get('error', '未知错误')}")
            
            return result
    
    # 使用自定义求解器
    solver = MyMazeSolver()
    mazes = MazeEnvironment.list_available_mazes()
    if mazes:
        solver.solve_with_benchmark(mazes[0])

if __name__ == "__main__":
    # 运行演示
    demo_basic_usage()
    demo_advanced_usage()
    demo_multi_model_test()
    demo_custom_integration()
    
    print("\n=== 子模块集成完成 ===")
    print("现在你可以将MazeBenchmark作为子模块集成到你的项目中！")