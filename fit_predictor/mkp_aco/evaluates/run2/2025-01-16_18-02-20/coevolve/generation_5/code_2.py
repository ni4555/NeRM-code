import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the value-to-weight ratio for each item
    value_to_weight_ratio = prize / weight
    
    # Normalize the value-to-weight ratios to ensure they are positive and sum to 1
    normalized_vtw = value_to_weight_ratio / np.sum(value_to_weight_ratio)
    
    # The heuristic for each item is its normalized value-to-weight ratio
    heuristics = normalized_vtw
    
    return heuristics