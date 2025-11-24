class Metrics:
    def __init__(self, size: int = 10):
        # Reasonable dynamic weights: more emphasis on Q and O as size grows; A small
        if size <= 10:
            self.wS, self.wQ, self.wO, self.wA = 0.50, 0.25, 0.23, 0.02
        elif size <= 20:
            self.wS, self.wQ, self.wO, self.wA = 0.45, 0.30, 0.23, 0.02
        else:
            self.wS, self.wQ, self.wO, self.wA = 0.40, 0.35, 0.23, 0.02

    def score(self, result: dict) -> dict:
        S = 1 if result.get('ok') else 0
        Q = 1 if result.get('ok') and result.get('optimal') else 0
        O = float(result.get('overlap') or 0.0)
        A = 1 if result.get('anti_cheat_pass', True) else 0
        total = round(100 * (self.wS*S + self.wQ*Q + self.wO*O + self.wA*A), 2)
        return {
            'S': S, 'Q': Q, 'O': O, 'A': A, 'total': total,
            'W_S': self.wS, 'W_Q': self.wQ, 'W_O': self.wO, 'W_A': self.wA
        }
