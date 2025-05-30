import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    n, m = prize.shape[0], weight.shape[1]
    item_values = prize / np.sum(prize)
    total_weight = np.sum(weight, axis=1)
    heuristics = np.dot(item_values, total_weight)
    return heuristics
