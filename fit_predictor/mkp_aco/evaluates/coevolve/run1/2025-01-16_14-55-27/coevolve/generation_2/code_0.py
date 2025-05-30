import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the weighted ratio for each item
    weighted_ratio = prize / weight.sum(axis=1, keepdims=True)
    
    # Sort items based on the weighted ratio in descending order
    sorted_indices = np.argsort(weighted_ratio, axis=1)[::-1]
    
    # Initialize the heuristics array with zeros
    heuristics = np.zeros_like(prize)
    
    # Assign heuristics based on the sorted order
    for i, sorted_index in enumerate(sorted_indices):
        heuristics[sorted_index] = 1
    
    return heuristics