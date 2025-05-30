import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Initialize an array to hold the heuristics for each item
    heuristics = np.zeros_like(prize)
    
    # Calculate the value/weight ratio for each item in each dimension
    value_weight_ratio = prize / weight
    
    # Sum the value/weight ratios across all dimensions for each item
    heuristics = np.sum(value_weight_ratio, axis=1)
    
    # Normalize the heuristics by dividing by the maximum heuristic value
    heuristics /= np.max(heuristics)
    
    return heuristics