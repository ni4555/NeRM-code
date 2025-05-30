import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Initialize the value/weight ratio for each item
    value_weight_ratio = prize / weight.sum(axis=1)
    
    # Calculate the total value for each possible number of items to include
    cumulative_value = np.cumsum(prize * value_weight_ratio)
    
    # Compute the heuristic as the ratio of the total value to the total weight
    total_weight = np.sum(weight, axis=1)
    heuristics = cumulative_value / total_weight
    
    return heuristics