#!/usr/bin/env python3
"""
测试路径坐标解析功能
"""

import re

def parse_path_coordinates(response: str, current_position=None) -> list:
    """
    解析AI响应中的路径坐标，坐标格式为 (y, x)

    支持格式：
    - (1,2),(3,4),(5,6)
    - (1, 2), (3, 4), (5, 6)

    如果第一个坐标是当前自身坐标，会自动跳过
    """
    response = response.strip().lower()

    # 使用正则表达式查找所有坐标对
    coord_pattern = r'\(\s*(\d+)\s*,\s*(\d+)\s*\)'
    matches = re.findall(coord_pattern, response)

    path = []
    for y_str, x_str in matches:
        try:
            y, x = int(y_str), int(x_str)
            path.append((y, x))
        except ValueError:
            continue

    # 如果提供了当前坐标且路径第一个坐标是自身坐标，则跳过第一个坐标
    if current_position and path and path[0] == current_position:
        path = path[1:]

    return path

def test_parse_path_coordinates():
    """测试路径坐标解析函数"""
    test_cases = [
        ("(1,2),(3,4),(5,6)", [(1,2), (3,4), (5,6)]),
        ("(1, 2), (3, 4), (5, 6)", [(1,2), (3,4), (5,6)]),
        ("1,2;3,4;5,6", []),  # 不支持这种格式
        ("up", []),  # 单个动作
        ("down", []),
        ("(0,0),(0,1),(1,1)", [(0,0), (0,1), (1,1)]),
        ("我想去(1,2)然后(1,3)", [(1,2), (1,3)]),  # 带文本的坐标
    ]

    print("测试路径坐标解析:")
    for i, (input_str, expected) in enumerate(test_cases):
        result = parse_path_coordinates(input_str)
        status = "✓" if result == expected else "✗"
        print(f"测试 {i+1}: {status} 输入: '{input_str}' -> 期望: {expected}, 结果: {result}")

if __name__ == "__main__":
    test_parse_path_coordinates()