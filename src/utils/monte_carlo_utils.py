"""
equity_utils.py
---------------
Monte-Carlo equity estimator (Heads-Up) using game's HandEvaluator.
用法:
    from equity_utils import mc_equity
    p = mc_equity(hole, board, n=600)   # 0–1
"""
import random
from itertools import combinations
from game.engine.card import Card
from utils.hand_evaluator import HandEvaluator

# 52 張牌建表
ALL = [Card.from_str(s + r) for s in "CDHS" for r in "23456789TJQKA"]

def mc_equity(hole, board, n=40000):
    """
    n     : 抽樣次數
    """
    if isinstance(hole[0], str):
        hole = [Card.from_str(c) for c in hole]
    if board and isinstance(board[0], str):
        board = [Card.from_str(c) for c in board]
    
    known = hole + board
    deck  = [c for c in ALL if c not in known]

    results = []
    eval_hand = HandEvaluator.eval_hand

    for _ in range(n):
        opp_hole = random.sample(deck, 2)
        rest     = [c for c in deck if c not in opp_hole]
        future_board = board + random.sample(rest, 5 - len(board))

        self_score = eval_hand(hole, future_board)
        opp_score  = eval_hand(opp_hole, future_board)

        if self_score > opp_score:
            results.append(1)
        elif self_score == opp_score:
            results.append(0.2)
        else:
            results.append(0)

    p = sum(results) / n
    variance = sum((x - p) ** 2 for x in results) / n
    stddev = variance ** 0.5
    if stddev >= 0.6:
        return p * 0.75
    elif stddev >= 0.5:
        return p * 0.85
    elif stddev >= 0.4:
        return p * 0.9
    else:
        return p