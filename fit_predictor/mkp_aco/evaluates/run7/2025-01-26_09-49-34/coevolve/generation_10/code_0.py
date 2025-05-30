import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    n, m = prize.shape[0], weight.shape[1]
    normalized_prize = prize / np.sum(prize)
    normalized_weight = weight / np.sum(weight, axis=1, keepdims=True)
    fairness_factor = normalized_prize * (1 - normalized_weight)
    heuristics = np.sum(fairness_factor, axis=1)
    return heuristics
