import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate value-to-weight ratio for each item
    value_to_weight_ratio = prize / weight.sum(axis=1)
    
    # Calculate weighted ratio by multiplying value-to-weight ratio with total prize
    weighted_ratio = value_to_weight_ratio * prize
    
    # Normalize weighted ratio to get heuristics
    max_weighted_ratio = np.max(weighted_ratio)
    heuristics = weighted_ratio / max_weighted_ratio
    
    return heuristics