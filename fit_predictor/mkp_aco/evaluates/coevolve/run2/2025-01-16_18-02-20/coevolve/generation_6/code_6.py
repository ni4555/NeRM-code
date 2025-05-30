import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    n, m = prize.shape[0], weight.shape[1]
    # Initialize the heuristic values with zeros
    heuristics = np.zeros(n)
    
    # Normalize weights within each dimension
    for j in range(m):
        weight[:, j] = weight[:, j] / np.sum(weight[:, j])
    
    # Compute the value-to-weight ratio for each item
    value_to_weight_ratio = prize / weight.sum(axis=1)
    
    # Use a stochastic sampling algorithm to determine the heuristics
    for i in range(n):
        # Sample based on the value-to-weight ratio and normalize
        sample_prob = value_to_weight_ratio[i] / np.sum(value_to_weight_ratio)
        heuristics[i] = sample_prob
    
    return heuristics