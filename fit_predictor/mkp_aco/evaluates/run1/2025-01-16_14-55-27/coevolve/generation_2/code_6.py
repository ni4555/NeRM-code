import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the weighted ratio for each item
    weighted_ratio = prize / weight.sum(axis=1)
    
    # Sort items by their weighted ratio in descending order
    sorted_indices = np.argsort(weighted_ratio)[::-1]
    
    # Initialize the heuristic values array
    heuristics = np.zeros_like(prize)
    
    # Assign heuristic values based on the sorted order
    for i, index in enumerate(sorted_indices):
        heuristics[index] = 1 / (i + 1)  # Example heuristic function: inverse of rank
    
    return heuristics