import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Initialize the heuristics array with zeros
    heuristics = np.zeros_like(prize)
    
    # Calculate the weighted ratio for each item
    weighted_ratio = prize / weight.sum(axis=1)
    
    # Calculate the heuristic as the weighted ratio divided by the sum of weights for each item
    heuristics = weighted_ratio / weight
    
    # Sort the heuristics in descending order and return the sorted heuristics
    return np.argsort(-heuristics)