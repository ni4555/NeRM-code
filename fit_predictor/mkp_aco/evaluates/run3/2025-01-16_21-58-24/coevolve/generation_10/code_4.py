import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the value-to-weight ratio for each item
    value_to_weight_ratio = prize / weight.sum(axis=1)
    
    # Normalize the ratios to ensure they sum to 1
    normalized_ratio = value_to_weight_ratio / value_to_weight_ratio.sum()
    
    # Return the normalized value-to-weight ratios as heuristics
    return normalized_ratio