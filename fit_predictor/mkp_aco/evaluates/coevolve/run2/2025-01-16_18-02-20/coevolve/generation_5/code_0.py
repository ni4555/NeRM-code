import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the value-to-weight ratio for each item
    value_to_weight_ratio = prize / weight.sum(axis=1)
    
    # Normalize the ratios to get a rank between 0 and 1
    max_ratio = np.max(value_to_weight_ratio)
    normalized_ratio = value_to_weight_ratio / max_ratio
    
    # Calculate a heuristic based on the normalized ratio and the prize
    # This heuristic function is arbitrary and for the sake of this problem, 
    # we can just use the normalized ratio as it is.
    heuristics = normalized_ratio * prize
    
    return heuristics