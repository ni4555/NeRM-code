import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    m = weight.shape[1]
    normalized_weight = weight / weight.sum(axis=1, keepdims=True)
    utility = prize / normalized_weight.sum(axis=1, keepdims=True)
    max_utility = utility.max(axis=1)
    return max_utility
