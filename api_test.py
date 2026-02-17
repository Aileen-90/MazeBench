import os
# 通过命令安装方舟SDK pip install 'volcengine-python-sdk[ark]'
from volcenginesdkarkruntime import Ark 

# 初始化Ark客户端
client = Ark(
    # The base URL for model invocation
    base_url="https://ark.cn-beijing.volces.com/api/v3",    
    api_key=os.getenv('ARK_API_KEY'), 
    # 深度思考耗时更长，请设置更大的超时限制，推荐为1800 秒及以上
    timeout=1800,
)

# 创建一个对话请求
completion = client.chat.completions.create(
    # Replace with Model ID
    model = "deepseek-v3-2-251201",
    messages=[
        {"role": "user", "content": "我要研究深度思考模型与非深度思考模型区别的课题，怎么体现我的专业性"}
    ],
    thinking={"type":"enabled"}
)
# 当触发深度思考时，打印思维链内容
if hasattr(completion.choices[0].message, 'reasoning_content'):
    print(completion.choices[0].message.reasoning_content)
print(completion.choices[0].message.content)