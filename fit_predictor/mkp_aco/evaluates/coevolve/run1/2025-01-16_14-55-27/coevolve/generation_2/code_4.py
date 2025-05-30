import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the weighted ratio for each item
    weighted_ratio = prize / weight.sum(axis=1)
    
    # Dynamic sorting based on the weighted ratio
    sorted_indices = np.argsort(weighted_ratio)[::-1]
    
    # Calculate the heuristic score based on the sorted order
    heuristics = np.zeros_like(prize)
    cumulative_weight = 0
    for i in sorted_indices:
        cumulative_weight += weight[i][0]
        if cumulative_weight <= 1:
            heuristics[i] = 1
    
    return heuristics