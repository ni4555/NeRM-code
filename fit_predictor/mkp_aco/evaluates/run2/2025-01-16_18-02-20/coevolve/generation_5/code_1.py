import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate value-to-weight ratio for each item
    value_to_weight_ratio = prize / weight
    
    # Normalize the value-to-weight ratio by the sum of all ratios to ensure that the sum of heuristics is 1
    normalized_ratio = value_to_weight_ratio / np.sum(value_to_weight_ratio)
    
    # The normalized ratio serves as the heuristic for each item
    return normalized_ratio