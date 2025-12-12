# adapters/azure_adapter.py
import os
from typing import Optional
from openai import AzureOpenAI
import httpx
from .base import BaseAdapter

class AzureAdapter(BaseAdapter):
    def __init__(self, api_key: str, endpoint: str, deployment: str, api_version: str = "2023-12-01-preview", temperature: float = 0.1):
        # 使用httpx.Timeout设置更细粒度的超时控制
        # connect: 连接超时, read: 读取响应超时, write: 写入请求超时
        custom_timeout = httpx.Timeout(
            connect=30.0,   # 30秒连接超时
            read=1200.0,    # 1200秒（20分钟）读取超时，适用于长时间生成
            write=30.0,     # 30秒写入超时
            pool=30.0       # 30秒连接池超时
        )
        self.client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            azure_deployment=deployment,
            api_version=api_version,
            timeout=custom_timeout,
            max_retries=10   # 设置最大重试次数
        )
        self.deployment = deployment
        self.temperature = temperature

    def generate(self, prompt, image_path: Optional[str] = None) -> str:
        """调用Azure OpenAI API生成文本，可选图片输入"""
        if isinstance(prompt, str):
            # prompt是字符串，转换为消息格式
            messages = [{"role": "user", "content": prompt}]
        else:
            # prompt已经是消息列表
            messages = prompt

        if image_path:
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
            model=self.deployment,
            messages=messages,
            max_tokens=1000,
            temperature=self.temperature
        )

        return response.choices[0].message.content
