import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the value-to-weight ratio for each item
    value_to_weight_ratio = prize / weight
    
    # Normalize the ratios to make them suitable for comparison
    normalized_ratio = value_to_weight_ratio / np.sum(value_to_weight_ratio)
    
    # Create a heuristic score for each item based on the normalized ratio
    heuristics = normalized_ratio * (1 + np.random.rand(len(value_to_weight_ratio)))
    
    return heuristics