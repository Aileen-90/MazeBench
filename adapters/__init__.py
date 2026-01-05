# adapters/__init__.py
from .base import BaseAdapter
from .openai_adapter import OpenAIAdapter
from .azure_adapter import AzureAdapter
from .ark_adapter import ArkAdapter
from .transformers_adapter import TransformersAdapter
from .ollama_adapter import OllamaAdapter

def get_adapter(cfg: dict, image: bool = False) -> BaseAdapter:
    """
    适配器工厂函数，根据配置返回对应的适配器实例
    """
    provider = cfg.get('PROVIDER', 'openai').lower()

    temperature = cfg.get('temperature', 0.1)

    if provider == 'azure':
        return AzureAdapter(
            api_key=cfg.get('AZURE_OPENAI_API_KEY'),
            endpoint=cfg.get('AZURE_OPENAI_ENDPOINT'),
            deployment=cfg.get('AZURE_OPENAI_DEPLOYMENT'),
            api_version=cfg.get('AZURE_OPENAI_API_VERSION', '2023-12-01-preview'),
            temperature=temperature
        )
    elif provider == 'ark':
        return ArkAdapter(
            api_key=cfg.get('ARK_API_KEY'),
            api_base=cfg.get('ARK_API_BASE'),
            model=cfg.get('model', 'ark-model'),
            temperature=temperature
        )
    elif provider == 'ollama':
        model = cfg.get('model', 'llama3')
        return OllamaAdapter(
            model_name=model,
            temperature=temperature,
            ollama_base_url=cfg.get('OLLAMA_BASE_URL', cfg.get('base_url', 'http://localhost:11434/v1'))
        )
    elif provider == 'transformers':
        model = cfg.get('model', 'meta-llama/Llama-2-7b-chat-hf')
        return TransformersAdapter(
            model_name=model,
            temperature=temperature,
            device_map=cfg.get('device_map', 'auto'),
            max_new_tokens=cfg.get('max_new_tokens', 512)
        )
    else:  # 默认openai
        model = cfg.get('model', 'gpt-4')
        return OpenAIAdapter(
            api_key=cfg.get('OPENAI_API_KEY'),
            api_base=cfg.get('OPENAI_API_BASE'),
            model=model,
            temperature=temperature
        )