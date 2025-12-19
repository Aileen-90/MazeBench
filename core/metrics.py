# core/metrics.py
from typing import Dict

class Metrics:
    """评测指标计算器"""

    def __init__(self, size: int = 10):
        self.size = size

    def score(self, validation_result: Dict) -> Dict:
        """
        根据验证结果计算各项评分
        返回包含各项分数的字典
        """
        if not validation_result.get('ok', False):
            return {
                'total': 0,
                'S': 0,  # Success - 是否成功到达终点
                'Q': 0,  # Quality - 路径质量
                'O': 0,  # Optimality - 最优性
                'A': 0   # Accuracy - 准确性
            }

        path_length = validation_result.get('path_length', 0)
        optimal_length = validation_result.get('optimal_length', 0)
        efficiency = validation_result.get('efficiency', 0)

        # S: Success Score - 成功到达终点得满分
        S = 100

        # Q: Quality Score - 路径是否有效且连通
        Q = 100

        # O: Optimality Score - 路径长度越接近最优越好
        if optimal_length > 0:
            O = min(100, max(0, 100 * (optimal_length / path_length)))
        else:
            O = 100  # 如果没有最优路径信息，给满分
        
        # P: Path Similarity Score - 路径相似度评分（新增）
        path_similarity = validation_result.get('path_similarity', 0)
        P = min(100, max(0, 100 * path_similarity))

        # A: Accuracy Score - 综合准确性评分
        A = (S + Q + O + P) / 4  # 现在包含路径相似度

        # Total Score - 加权总分（调整权重以包含路径相似度）
        total = 0.3 * S + 0.25 * Q + 0.25 * O + 0.2 * P

        return {
            'total': round(total, 2),
            'S': S,
            'Q': Q,
            'O': round(O, 2),
            'P': round(P, 2),  # 新增：路径相似度评分
            'A': round(A, 2)
        }