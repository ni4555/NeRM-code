import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the value-to-weight ratio for each item
    value_to_weight_ratio = prize / weight.sum(axis=1)
    
    # Normalize the ratios to ensure they are on a consistent scale
    normalized_ratio = (value_to_weight_ratio - np.min(value_to_weight_ratio)) / (np.max(value_to_weight_ratio) - np.min(value_to_weight_ratio))
    
    # Use the normalized ratio as the heuristic value
    heuristics = normalized_ratio
    
    return heuristics