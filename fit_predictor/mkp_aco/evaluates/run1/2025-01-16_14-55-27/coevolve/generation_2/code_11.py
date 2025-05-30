import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the weighted ratio for each item
    weighted_ratio = prize / weight.sum(axis=1)
    
    # Dynamic sorting by weighted ratio
    sorted_indices = np.argsort(weighted_ratio)[::-1]
    
    # Initialize heuristics array
    heuristics = np.zeros_like(prize)
    
    # Assign heuristic values based on sorted order
    for i, idx in enumerate(sorted_indices):
        heuristics[idx] = 1.0 / (i + 1)  # Using inverse order as heuristic value
    
    return heuristics