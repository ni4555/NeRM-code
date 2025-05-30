import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    n = prize.shape[0]
    m = weight.shape[1]
    
    # Calculate the weighted value for each item
    weighted_value = np.dot(prize, weight)
    
    # Calculate the total weight for each item
    total_weight = np.sum(weight, axis=1)
    
    # Initialize the heuristic array with the weighted value
    heuristics = weighted_value.copy()
    
    # Adjust the heuristic based on adherence to multi-dimensional constraints
    for i in range(n):
        if total_weight[i] > 1:
            heuristics[i] *= (1 - (total_weight[i] - 1) / 1)
    
    # Normalize the heuristic values
    heuristics /= np.sum(heuristics)
    
    return heuristics