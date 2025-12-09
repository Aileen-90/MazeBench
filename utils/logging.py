# utils/logging.py
import logging
from pathlib import Path

def setup_logging(log_level=logging.INFO, log_file=None):
    """
    设置日志配置
    """
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
        ]
    )

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(file_handler)

def get_logger(name: str):
    """
    获取指定名称的logger
    """
    return logging.getLogger(name)
