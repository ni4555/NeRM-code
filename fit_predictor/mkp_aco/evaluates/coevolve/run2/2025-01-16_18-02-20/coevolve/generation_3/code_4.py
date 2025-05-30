import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the value-to-weight ratio for each item
    value_to_weight_ratio = prize / weight
    
    # Normalize the value-to-weight ratio by the sum of ratios to get a probability
    normalized_ratio = value_to_weight_ratio / np.sum(value_to_weight_ratio)
    
    # Use the normalized ratio as the heuristics
    heuristics = normalized_ratio
    
    return heuristics