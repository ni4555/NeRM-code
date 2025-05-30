import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the multi-dimensional weighted ratio metric for each item
    weighted_ratio = np.sum(prize * weight, axis=1)
    
    # Normalize the weighted ratio to ensure the sum of all heuristics is 1
    heuristic_sum = np.sum(weighted_ratio)
    heuristics = weighted_ratio / heuristic_sum
    
    return heuristics