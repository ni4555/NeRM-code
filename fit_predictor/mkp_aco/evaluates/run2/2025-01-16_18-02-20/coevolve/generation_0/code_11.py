import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    n = prize.shape[0]
    m = weight.shape[1]
    
    # Initialize a matrix to store the heuristic values
    heuristics = np.zeros((n,), dtype=float)
    
    # Iterate through each item to compute its heuristic
    for i in range(n):
        # Compute the sum of weights for item i across all dimensions
        item_weight = weight[i].sum()
        # Normalize by the dimension of weights (since each dimension's max constraint is 1)
        normalized_weight = item_weight / m
        # The heuristic is the prize divided by the normalized weight
        heuristics[i] = prize[i] / normalized_weight
    
    return heuristics