import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate value-to-weight ratio for each item
    value_to_weight_ratio = prize / weight.sum(axis=1)
    
    # Normalize the ratios to get a score between 0 and 1
    normalized_ratio = value_to_weight_ratio / value_to_weight_ratio.max()
    
    # Calculate the heuristics based on normalized ratio
    heuristics = normalized_ratio
    
    return heuristics