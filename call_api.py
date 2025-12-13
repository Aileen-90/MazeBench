# test_openai.py
import os
from openai import OpenAI

client = OpenAI()

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
