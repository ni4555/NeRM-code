import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the value-to-weight ratio for each item
    value_to_weight_ratio = prize / weight.sum(axis=1)
    
    # Normalize the value-to-weight ratio to handle different scales
    max_ratio = np.max(value_to_weight_ratio)
    min_ratio = np.min(value_to_weight_ratio)
    normalized_ratio = 2 * ((value_to_weight_ratio - min_ratio) / (max_ratio - min_ratio)) - 1
    
    # Adjust the normalized ratio based on the current weight usage (heuristic)
    # For simplicity, we use a linear adjustment here, but it can be replaced with more complex logic
    adjusted_ratio = normalized_ratio * (1 - weight.sum(axis=1) / weight.shape[1])
    
    # Return the heuristics array
    return adjusted_ratio