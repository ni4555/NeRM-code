import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the value-to-weight ratio for each item
    value_to_weight_ratio = prize / weight
    
    # Normalize the ratios to sum to 1
    total_ratio = np.sum(value_to_weight_ratio)
    normalized_ratio = value_to_weight_ratio / total_ratio
    
    # Return the normalized value-to-weight ratios as heuristics
    return normalized_ratio