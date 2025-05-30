import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    n, m = prize.shape[0], weight.shape[1]
    # Initialize the heuristic array with zeros
    heuristics = np.zeros(n)
    
    # Calculate the total weight for each item
    total_weight = np.sum(weight, axis=1)
    
    # Calculate the heuristic value for each item
    for i in range(n):
        # Calculate the weighted sum of the prize for each dimension
        weighted_prize = np.sum(prize[i] * weight[i])
        # Normalize the weighted prize by the total weight to account for the constraint
        normalized_prize = weighted_prize / total_weight[i]
        # Calculate the heuristic value as the normalized prize
        heuristics[i] = normalized_prize
    
    return heuristics