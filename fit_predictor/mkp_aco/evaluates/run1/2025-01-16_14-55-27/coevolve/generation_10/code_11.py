import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Initialize the heuristic array with zeros
    heuristics = np.zeros_like(prize)
    
    # Calculate the value-to-weight ratio for each item
    value_to_weight_ratio = prize / weight.sum(axis=1, keepdims=True)
    
    # Calculate the heuristic for each item based on its value-to-weight ratio
    heuristics = value_to_weight_ratio.sum(axis=1)
    
    return heuristics