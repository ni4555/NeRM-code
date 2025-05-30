import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the value-to-weight ratio for each item
    value_to_weight_ratio = prize / weight.sum(axis=1)
    
    # Normalize the value-to-weight ratio by the maximum ratio
    normalized_ratio = value_to_weight_ratio / value_to_weight_ratio.max()
    
    # Return the normalized value-to-weight ratios as the heuristics
    return normalized_ratio