# adapters/openai_adapter.py
import os
from typing import Optional
import openai
import httpx
from .base import BaseAdapter

class OpenAIAdapter(BaseAdapter):
    def __init__(self, api_key: str, api_base: Optional[str] = None, model: str = "gpt-4", temperature: float = 0.1, enable_thinking: bool = False):
        # 使用httpx.Timeout设置更细粒度的超时控制
        # connect: 连接超时, read: 读取响应超时, write: 写入请求超时
        custom_timeout = httpx.Timeout(
            connect=30.0,   # 30秒连接超时
            read=1200.0,    # 1200秒（20分钟）读取超时，适用于长时间生成
            write=30.0,     # 30秒写入超时
            pool=30.0       # 30秒连接池超时
        )
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=api_base,
            timeout=custom_timeout,
            max_retries=10   # 设置最大重试次数
        )
        self.model = model
        self.temperature = temperature
        self.enable_thinking = enable_thinking

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

        # 构建 API 调用参数
        create_params = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 1000,
            "temperature": self.temperature
        }
        
        # 只有当启用 thinking 模式时才添加 thinking 参数
        if self.enable_thinking:
            create_params["thinking"] = {"type": "enabled"}  # 火山deepseek
            # 如果需要支持百炼qwen3，可以使用：
            # create_params["extra_body"] = {"enable_thinking": True}
        
        response = self.client.chat.completions.create(**create_params)
        # completion = client.chat.completions.create(
        #     model="qwen-plus", # 选择模型
        #     messages=[{"role": "user", "content": "你是谁"}],    
        #     # 由于 enable_thinking 非 OpenAI 标准参数，需要通过 extra_body 传入
        #     extra_body={"enable_thinking":True},
        #     # 流式输出方式调用
        #     stream=True,
        #     # 使流式返回的最后一个数据包包含Token消耗信息
        #     stream_options={
        #         "include_usage": True
        #     }
        # )

        return response.choices[0].message.content
