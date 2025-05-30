import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Normalize weights by summing each row to get the ratio for each item
    weight_ratio = weight / weight.sum(axis=1, keepdims=True)
    
    # Calculate weighted prize for each item
    weighted_prize = prize * weight_ratio
    
    # Use a simple heuristic: the higher the weighted prize, the more promising the item
    heuristics = weighted_prize.sum(axis=1)
    
    return heuristics