import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the value/weight ratio for each item
    value_weight_ratio = prize / weight.sum(axis=1)
    
    # Normalize the ratios to ensure they sum to 1 across all items
    normalized_ratio = value_weight_ratio / value_weight_ratio.sum()
    
    # Scale the normalized ratios to get a promising score for each item
    heuristics = normalized_ratio * (1 / (1 + weight.sum(axis=1)))
    
    return heuristics