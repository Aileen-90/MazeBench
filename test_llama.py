#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Llama模型测试脚本
功能：验证Llama模型是否可以通过OpenAI兼容API正常工作
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.io import load_config, apply_env_keys
from adapters import get_adapter

def test_llama_model():
    """测试Llama模型"""
    # 加载配置
    cfg = load_config("config/llama_config.yaml")
    cfg = apply_env_keys(cfg)
    
    print("使用的配置:")
    print(f"- 模型: {cfg['model']}")
    print(f"- API基础URL: {cfg['OPENAI_API_BASE']}")
    print(f"- API密钥: {'***' if cfg['OPENAI_API_KEY'] else 'None'}")
    
    # 获取适配器
    adapter = get_adapter(cfg)
    
    print("\n正在测试Llama模型...")
    
    # 测试简单的文本生成
    prompt = "你好，请简单介绍一下你自己"
    try:
        response = adapter.generate(prompt)
        print(f"\n模型响应:")
        print(response)
        print("\n✅ Llama模型测试成功!")
    except Exception as e:
        print(f"\n❌ Llama模型测试失败: {str(e)}")
        print("请检查Ollama服务是否正在运行，以及配置是否正确。")
        print("启动Ollama服务的命令: ollama serve")
        print("拉取Llama模型的命令: ollama pull llama3")
        return False
    
    return True

if __name__ == "__main__":
    test_llama_model()