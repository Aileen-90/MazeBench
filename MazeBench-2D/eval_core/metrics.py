from typing import Dict

class Metrics:
    def __init__(self, size: int):
        self.size = size
        # Reasonable dynamic weights: larger mazes emphasize optimality and overlap; success remains primary; A small
        if size <= 10:
            self.w = {'S': 0.50, 'Q': 0.25, 'O': 0.23, 'A': 0.02}
        elif size <= 20:
            self.w = {'S': 0.45, 'Q': 0.30, 'O': 0.23, 'A': 0.02}
        else:
            self.w = {'S': 0.40, 'Q': 0.35, 'O': 0.23, 'A': 0.02}

    def score(self, result: Dict) -> Dict:
        # S: Success (1 if ok else 0)
        S = 1.0 if result.get('ok') else 0.0
        # Q: Optimality (1 if optimal else 0)
        Q = 1.0 if result.get('ok') and result.get('optimal') else 0.0
        # O: Overlap (F1 over path nodes), computed regardless of ok
        O = float(result.get('overlap') or 0.0)
        # A: Anti-cheat adherence (reduced weight)
        A = 1.0 if result.get('anti_cheat_pass', True) else 0.0
        total = self.w['S']*S + self.w['Q']*Q + self.w['O']*O + self.w['A']*A
        return {
            'S': S, 'Q': Q, 'O': O, 'A': A, 'total': round(total*100, 1),
            'W_S': self.w['S'], 'W_Q': self.w['Q'], 'W_O': self.w['O'], 'W_A': self.w['A']
        }
