# adapters/openai_adapter.py
import os
from typing import Optional
import openai
from .base import BaseAdapter

class OpenAIAdapter(BaseAdapter):
    def __init__(self, api_key: str, api_base: Optional[str] = None, model: str = "gpt-4", temperature: float = 0.1):
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=api_base,
            timeout=120.0,  # 增加超时时间到120秒
            max_retries=3   # 设置最大重试次数
        )
        self.model = model
        self.temperature = temperature

    def generate(self, prompt, image_path: Optional[str] = None) -> str:
        """调用OpenAI API生成文本，可选图片输入"""
        if isinstance(prompt, str):
            # prompt是字符串，转换为消息格式
            messages = [{"role": "user", "content": prompt}]
        else:
            # prompt已经是消息列表
            messages = prompt

        if image_path:
            # 处理图片输入（需要base64编码等）
            import base64
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')

            # 对于图片模式，修改用户消息的内容
            if isinstance(prompt, str):
                # prompt是字符串，直接替换内容
                messages[0]["content"] = [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                    }
                ]
            else:
                # prompt是消息列表，修改最后一个用户消息
                for msg in reversed(messages):
                    if msg.get("role") == "user":
                        if isinstance(msg["content"], str):
                            msg["content"] = [
                                {"type": "text", "text": msg["content"]},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                                }
                            ]
                        break

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=1000,
            temperature=self.temperature
        )

        return response.choices[0].message.content
