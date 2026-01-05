# adapters/ollama_adapter.py
"""
Ollama适配器，使用本地Ollama服务运行模型
支持GPU加速（依赖Ollama的GPU配置）
"""
import os
from typing import Optional
import openai
import httpx
from .base import BaseAdapter

class OllamaAdapter(BaseAdapter):
    def __init__(self, model_name: str = "llama3", temperature: float = 0.1, ollama_base_url: str = "http://localhost:11434/v1"):
        """
        初始化Ollama适配器
        
        Args:
            model_name: Ollama模型名称（如llama2, mistral, llama3等）
            temperature: 生成温度
            ollama_base_url: Ollama服务的基础URL（默认为Ollama默认地址）
        """
        # 使用httpx.Timeout设置超时控制
        custom_timeout = httpx.Timeout(
            connect=30.0,   # 30秒连接超时
            read=1200.0,    # 1200秒（20分钟）读取超时，适用于长时间生成
            write=30.0,     # 30秒写入超时
            pool=30.0       # 30秒连接池超时
        )
        
        # 创建OpenAI兼容客户端
        self.client = openai.OpenAI(
            base_url=ollama_base_url,
            api_key="ollama",  # Ollama不需要实际的API密钥，这里使用任意值
            timeout=custom_timeout,
            max_retries=10   # 设置最大重试次数
        )
        
        self.model_name = model_name
        self.temperature = temperature
    
    def generate(self, prompt, image_path: Optional[str] = None) -> str:
        """
        使用Ollama模型生成文本
        
        Args:
            prompt: 提示文本或消息列表
            image_path: 图片路径（暂不支持）
            
        Returns:
            生成的文本
        """
        if image_path:
            raise NotImplementedError("OllamaAdapter暂不支持图片输入")
        
        if isinstance(prompt, str):
            # prompt是字符串，转换为消息格式
            messages = [{"role": "user", "content": prompt}]
        else:
            # prompt已经是消息列表
            messages = prompt
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=1000,
            temperature=self.temperature
        )
        
        return response.choices[0].message.content