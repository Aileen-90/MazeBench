# adapters/transformers_adapter.py
"""
本地模型适配器，使用transformers库运行Llama等开源模型
支持GPU加速（自动检测CUDA）
"""
import os
from typing import Optional
from .base import BaseAdapter

class TransformersAdapter(BaseAdapter):
    def __init__(self, model_name: str = "meta-llama/Llama-2-7b-chat-hf", 
                 temperature: float = 0.1, 
                 device_map: str = "auto",
                 max_new_tokens: int = 512):
        """
        初始化Transformers适配器
        
        Args:
            model_name: 模型名称或本地路径
            temperature: 生成温度
            device_map: 设备映射策略（"auto"自动检测GPU）
            max_new_tokens: 最大生成token数
        """
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype="auto"
        )
        self.model_name = model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
    
    def generate(self, prompt, image_path: Optional[str] = None) -> str:
        """
        使用本地模型生成文本
        
        Args:
            prompt: 提示文本或消息列表
            image_path: 图片路径（暂不支持）
            
        Returns:
            生成的文本
        """
        if image_path:
            raise NotImplementedError("TransformersAdapter暂不支持图片输入")
        
        if isinstance(prompt, list):
            # 转换为模型所需的格式
            formatted_prompt = self.tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # 单轮对话
            formatted_prompt = prompt
        
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)
        
        with self.tokenizer.as_target_tokenizer():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 如果是多轮对话，提取模型回复部分
        if isinstance(prompt, list):
            # 查找模型回复的起始位置
            model_response_start = response.rfind("assistant")
            if model_response_start != -1:
                # 查找assistant: 后面的内容
                colon_pos = response.find(":", model_response_start)
                if colon_pos != -1:
                    response = response[colon_pos + 1:].strip()
        
        return response