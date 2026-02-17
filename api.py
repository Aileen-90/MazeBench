"""
MazeBenchmark API - 子模块统一入口

提供简洁的API接口，便于作为子模块集成到其他项目中
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# 确保项目根目录在Python路径中
_project_root = Path(__file__).parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from .utils.io import load_config, apply_env_keys
from .sandbox.env import MazeEnvironment
from .sandbox.partial_observe import run_ai_sandbox_partial_observe
from .sandbox.ai_sandbox import run_ai_sandbox
from .run_multi_test import MultiModelMazeTest


class MazeBenchmarkAPI:
    """MazeBenchmark API 主类"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化API
        
        Args:
            config_path: 配置文件路径，如果为None则使用默认配置
        """
        self.config = load_config(config_path) if config_path else load_config()
        apply_env_keys(self.config)
        
        # 缓存常用对象
        self._maze_env = None
    
    def list_available_mazes(self) -> List[str]:
        """获取可用迷宫列表"""
        return MazeEnvironment.list_available_mazes()
    
    def load_maze(self, maze_name: str) -> MazeEnvironment:
        """加载迷宫环境"""
        return MazeEnvironment(maze_name)
    
    def generate_maze(self, width: int, height: int, save_path: Optional[str] = None) -> MazeEnvironment:
        """生成新迷宫"""
        env = MazeEnvironment.generate_maze(width, height)
        if save_path:
            env.save(save_path)
        return env
    
    def run_partial_observe_test(self, 
                               maze_name: str, 
                               model: Optional[str] = None, 
                               max_steps: Optional[int] = None,
                               visibility: Optional[int] = None) -> Dict[str, Any]:
        """运行部分观察模式测试"""
        
        # 使用配置中的默认值或传入的参数
        model = model or self.config.get('model', 'gpt-4')
        max_steps = max_steps or self.config.get('sandbox', {}).get('max_steps', 50)
        
        # 部分观察模式需要正数的visibility
        if visibility is None:
            visibility = self.config.get('sandbox', {}).get('visibility', 3)
        
        if visibility <= 0:
            raise ValueError("Partial observe mode requires positive visibility")
        
        # 临时修改配置中的visibility
        original_visibility = self.config.get('sandbox', {}).get('visibility')
        if 'sandbox' not in self.config:
            self.config['sandbox'] = {}
        self.config['sandbox']['visibility'] = visibility
        
        try:
            result = run_ai_sandbox_partial_observe(
                maze_name, 
                model=model, 
                max_steps=max_steps, 
                cfg=self.config
            )
            return result
        finally:
            # 恢复原始配置
            if original_visibility is not None:
                self.config['sandbox']['visibility'] = original_visibility
            elif 'sandbox' in self.config and 'visibility' in self.config['sandbox']:
                del self.config['sandbox']['visibility']
    
    def run_full_observe_test(self, 
                            maze_name: str, 
                            model: Optional[str] = None, 
                            max_steps: Optional[int] = None) -> Dict[str, Any]:
        """运行完整观察模式测试"""
        
        model = model or self.config.get('model', 'gpt-4')
        max_steps = max_steps or self.config.get('sandbox', {}).get('max_steps', 50)
        
        # 设置完整观察模式（visibility为-1）
        if 'sandbox' not in self.config:
            self.config['sandbox'] = {}
        self.config['sandbox']['visibility'] = -1
        
        try:
            result = run_ai_sandbox(
                maze_name,
                model=model,
                max_steps=max_steps,
                cfg=self.config
            )
            return result
        finally:
            # 清理visibility设置
            if 'sandbox' in self.config and 'visibility' in self.config['sandbox']:
                del self.config['sandbox']['visibility']
    
    def run_multi_model_test(self, 
                           models: List[str], 
                           maze_sizes: List[str], 
                           trials_per_maze: int = 5,
                           mode: str = 'ai',
                           output_dir: str = 'results') -> Dict[str, Any]:
        """运行多模型批量测试"""
        
        # 创建测试实例
        test = MultiModelMazeTest(
            models=models,
            maze_sizes=maze_sizes,
            trials_per_maze=trials_per_maze,
            mode=mode,
            output_base_dir=output_dir
        )
        
        # 运行测试
        summary = test.run()
        return summary
    
    def get_config(self) -> Dict[str, Any]:
        """获取当前配置"""
        return self.config.copy()
    
    def update_config(self, updates: Dict[str, Any]):
        """更新配置"""
        self._deep_update(self.config, updates)
        apply_env_keys(self.config)
    
    def _deep_update(self, base: Dict[str, Any], updates: Dict[str, Any]):
        """深度更新字典"""
        for key, value in updates.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value


# 便捷函数
def create_api(config_path: Optional[str] = None) -> MazeBenchmarkAPI:
    """创建MazeBenchmark API实例"""
    return MazeBenchmarkAPI(config_path)


def quick_test(maze_name: str = "maze_9x9_0", model: str = "gpt-4") -> Dict[str, Any]:
    """快速测试函数"""
    api = MazeBenchmarkAPI()
    return api.run_partial_observe_test(maze_name, model)


if __name__ == "__main__":
    # 简单的演示
    api = MazeBenchmarkAPI()
    print("可用迷宫:", api.list_available_mazes()[:3])
    
    # 快速测试
    result = quick_test()
    print(f"测试结果: 成功={result.get('success')}, 步数={result.get('steps')}")