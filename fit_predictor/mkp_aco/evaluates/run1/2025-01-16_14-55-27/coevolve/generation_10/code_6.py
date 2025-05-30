import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the value-to-weight ratio for each item
    value_to_weight_ratio = prize / weight.sum(axis=1)
    
    # Normalize the value-to-weight ratios to ensure consistency
    normalized_ratios = (value_to_weight_ratio - np.min(value_to_weight_ratio)) / (np.max(value_to_weight_ratio) - np.min(value_to_weight_ratio))
    
    # The heuristic value is simply the normalized ratio
    heuristics = normalized_ratios
    return heuristics