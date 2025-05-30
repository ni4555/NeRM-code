import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the value-to-weight ratio for each item
    value_to_weight_ratio = prize / weight.sum(axis=1)
    
    # Normalize the value-to-weight ratio to ensure all values are non-negative
    value_to_weight_ratio = np.abs(value_to_weight_ratio)
    
    # The heuristic is the value-to-weight ratio itself
    heuristics = value_to_weight_ratio
    return heuristics