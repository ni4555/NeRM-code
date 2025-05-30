import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the weighted ratio for each item
    weighted_ratio = prize / weight.sum(axis=1)
    
    # Sort items based on the weighted ratio in descending order
    sorted_indices = np.argsort(weighted_ratio)[::-1]
    
    # Initialize the heuristics array with zeros
    heuristics = np.zeros_like(prize)
    
    # Calculate the cumulative sum of weighted ratio
    cumulative_weighted_ratio = np.cumsum(weighted_ratio[sorted_indices])
    
    # Calculate the heuristic for each item
    for i, index in enumerate(sorted_indices):
        heuristics[index] = cumulative_weighted_ratio[i] / cumulative_weighted_ratio[-1]
    
    return heuristics