# test_openai.py
import os
from openai import OpenAI

client = OpenAI(api_key='6a9d3803-a391-4fb1-a526-14b9fcc0ae81', base_url='https://ark.cn-beijing.volces.com/api/v3')

try:
    response = client.chat.completions.create(
        model="doubao-seed-1-6-251015",
        
        messages=[{"role": "user", "content": "Say hello in 3 words"}],
        max_tokens=10,
        temperature=0
    )
    print("API OK:", response.choices[0].message.content.strip())
except Exception as e:
    print("API Error:", e)