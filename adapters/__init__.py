# adapters/__init__.py
from .base import BaseAdapter
from .openai_adapter import OpenAIAdapter
from .azure_adapter import AzureAdapter
from .ark_adapter import ArkAdapter

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
    else:  # 默认openai
        model = cfg.get('model', 'gpt-4')
        return OpenAIAdapter(
            api_key=cfg.get('OPENAI_API_KEY'),
            api_base=cfg.get('OPENAI_API_BASE'),
            model=model,
            temperature=temperature
        )
