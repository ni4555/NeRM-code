import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    n = prize.shape[0]
    m = weight.shape[1]
    
    # Initialize heuristics array
    heuristics = np.zeros(n)
    
    # Calculate weighted ratio for each item
    for i in range(n):
        total_weight = np.sum(weight[i])
        weighted_ratio = np.sum(prize[i] * weight[i]) / total_weight
        heuristics[i] = weighted_ratio
    
    # Dynamic item sorting based on weighted ratio
    sorted_indices = np.argsort(heuristics)[::-1]
    heuristics = heuristics[sorted_indices]
    
    return heuristics