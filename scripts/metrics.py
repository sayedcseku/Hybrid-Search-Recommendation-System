import math
from typing import List, Tuple

def recall_at_k(retrieved: List[str], positive: str, k: int) -> float:
    return 1.0 if positive in retrieved[:k] else 0.0

def ndcg_at_k(retrieved: List[str], positive: str, k: int) -> float:
    """
    Binary relevance NDCG for a single positive item.
    If positive is at rank r (1-indexed), DCG = 1/log2(r+1), IDCG = 1.0
    """
    top = retrieved[:k]
    if positive not in top:
        return 0.0
    r = top.index(positive) + 1
    return 1.0 / math.log2(r + 1)
