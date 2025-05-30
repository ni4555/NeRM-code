import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Assuming prize is a 1-D array of length n and weight is an n x m array
    # Calculate value-to-weight ratio for each item
    value_to_weight_ratio = prize / weight.sum(axis=1)
    
    # Calculate normalized value-to-weight ratio
    max_ratio = np.max(value_to_weight_ratio)
    normalized_ratio = value_to_weight_ratio / max_ratio
    
    # Assuming each dimension's constraint is fixed to 1, the heuristic is simply the normalized ratio
    # because it is already prioritized by the value-to-weight ratio
    heuristics = normalized_ratio
    
    return heuristics