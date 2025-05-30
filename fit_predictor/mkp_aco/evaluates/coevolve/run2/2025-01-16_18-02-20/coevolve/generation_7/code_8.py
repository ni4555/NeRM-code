import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the value-to-weight ratio for each item
    value_to_weight = prize / weight
    
    # Normalize the value-to-weight ratio to create a heuristic
    max_ratio = np.max(value_to_weight)
    min_ratio = np.min(value_to_weight)
    normalized_ratio = (value_to_weight - min_ratio) / (max_ratio - min_ratio)
    
    # Return the normalized heuristic values
    return normalized_ratio