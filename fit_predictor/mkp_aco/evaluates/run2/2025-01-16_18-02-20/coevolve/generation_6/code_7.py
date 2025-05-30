import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the value-to-weight ratio for each item
    value_to_weight_ratio = prize / weight.sum(axis=1, keepdims=True)
    
    # Normalize the value-to-weight ratio to get a per-item heuristic
    normalized_ratio = value_to_weight_ratio / value_to_weight_ratio.sum(axis=0, keepdims=True)
    
    # Adjust sampling parameters based on the evolving knapsack capacities
    # This is a placeholder for the adaptive stochastic sampling algorithm
    # For simplicity, we'll just use the normalized ratio as the heuristic
    heuristics = normalized_ratio
    
    return heuristics