import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate value-to-weight ratio for each item
    value_to_weight_ratio = prize / weight
    
    # Normalize the value-to-weight ratio to ensure non-negative values
    normalized_ratio = np.maximum(0, value_to_weight_ratio)
    
    # Calculate the sum of normalized ratios to normalize the values to sum to 1
    total_normalized_ratio = np.sum(normalized_ratio)
    
    # If the sum is zero, all items have zero weight, which should not happen in this scenario
    if total_normalized_ratio == 0:
        raise ValueError("All items have zero weight, which is not possible with constraints fixed to 1.")
    
    # Normalize the ratios so that their sum is 1
    normalized_ratio /= total_normalized_ratio
    
    # The normalized ratio now serves as the heuristic for each item
    heuristics = normalized_ratio
    
    return heuristics