import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the value-to-weight ratio for each item
    value_to_weight_ratio = prize / weight.sum(axis=1)
    
    # Sort items based on their value-to-weight ratio in descending order
    sorted_indices = np.argsort(value_to_weight_ratio)[::-1]
    
    # Initialize the heuristics array with zeros
    heuristics = np.zeros(prize.shape[0])
    
    # Set the heuristics for the top items to 1 (most promising)
    for i in sorted_indices:
        heuristics[i] = 1
    
    return heuristics