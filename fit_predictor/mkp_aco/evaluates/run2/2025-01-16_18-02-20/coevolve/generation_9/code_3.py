import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Normalize the prize by dividing by the maximum prize to get the value per unit weight
    value_to_weight_ratio = prize / weight.sum(axis=1, keepdims=True)
    
    # Sort the ratios in descending order to prioritize items with higher value-to-weight ratio
    sorted_indices = np.argsort(-value_to_weight_ratio, axis=1)
    
    # Initialize an array to store the heuristics (prominence of each item)
    heuristics = np.zeros_like(prize, dtype=float)
    
    # Update the heuristics for each item based on its sorted position
    for i, sorted_index in enumerate(sorted_indices):
        heuristics[i][sorted_index] = 1.0
    
    return heuristics