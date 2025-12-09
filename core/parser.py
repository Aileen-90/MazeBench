# core/parser.py
import re
from typing import List, Tuple, Optional

class OutputParser:
    """模型输出解析器"""

    def parse_with_fallback(self, text: str, adapter=None, prompt: str = None) -> object:
        """
        解析模型输出，提取路径信息
        支持多种格式的fallback解析
        """
        # 尝试多种解析策略
        path = self._parse_coordinate_list(text)
        if path:
            return ParsedResult(path=path, raw_text=text)

        path = self._parse_step_by_step(text)
        if path:
            return ParsedResult(path=path, raw_text=text)

        path = self._parse_simple_list(text)
        if path:
            return ParsedResult(path=path, raw_text=text)

        # 如果都失败，返回空路径
        return ParsedResult(path=[], raw_text=text)

    def _parse_coordinate_list(self, text: str) -> Optional[List[Tuple[int, int]]]:
        """解析标准坐标列表格式：[(0,0),(0,1),...]"""
        try:
            # 使用正则表达式提取坐标列表
            pattern = r'\[\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*(?:,\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*)*\]'
            match = re.search(pattern, text)
            if not match:
                return None

            # 提取所有坐标对
            coords = re.findall(r'\(\s*(\d+)\s*,\s*(\d+)\s*\)', text)
            if len(coords) < 2:  # 至少需要起点和终点
                return None

            return [(int(y), int(x)) for y, x in coords]  # 注意：转换为(y,x)格式
        except:
            return None

    def _parse_step_by_step(self, text: str) -> Optional[List[Tuple[int, int]]]:
        """解析逐步推理格式"""
        try:
            # 查找包含坐标的句子
            coord_pattern = r'\(\s*(\d+)\s*,\s*(\d+)\s*\)'
            coords = re.findall(coord_pattern, text)

            if len(coords) < 2:
                return None

            # 去重并保持顺序
            seen = set()
            path = []
            for y, x in coords:
                coord = (int(y), int(x))
                if coord not in seen:
                    path.append(coord)
                    seen.add(coord)

            return path
        except:
            return None

    def _parse_simple_list(self, text: str) -> Optional[List[Tuple[int, int]]]:
        """解析简单数字列表格式"""
        try:
            # 尝试解析形如 [0,0, 0,1, 1,1] 的格式
            numbers = re.findall(r'\d+', text)
            if len(numbers) % 2 != 0 or len(numbers) < 4:  # 至少2个坐标点
                return None

            path = []
            for i in range(0, len(numbers), 2):
                y, x = int(numbers[i]), int(numbers[i+1])
                path.append((y, x))

            return path
        except:
            return None

class ParsedResult:
    """解析结果封装"""
    def __init__(self, path: List[Tuple[int, int]], raw_text: str):
        self.path = path
        self.raw_text = raw_text
