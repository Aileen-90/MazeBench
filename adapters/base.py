# adapters/base.py
"""
模型适配器接口基类，定义统一的方法签名
"""
from abc import ABC, abstractmethod

class BaseAdapter(ABC):
    @abstractmethod
    def generate(self, prompt: str, image_path: str = None) -> str:
        """模型推理接口，返回生成文本（可选图片输入）"""
        pass
