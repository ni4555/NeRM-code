import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the value-to-weight ratio for each item
    value_to_weight = prize / weight
    
    # Normalize the value-to-weight ratios to ensure they sum to 1
    normalized_ratio = value_to_weight / np.sum(value_to_weight)
    
    # Use the normalized ratio as the heuristic value for each item
    heuristics = normalized_ratio
    
    return heuristics