import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the value-to-weight ratio for each item
    value_to_weight_ratio = prize / weight.sum(axis=1)
    
    # Calculate the total weight for each item
    total_weight = weight.sum(axis=1)
    
    # Normalize the value-to-weight ratio based on the total weight
    normalized_ratio = value_to_weight_ratio / total_weight
    
    # Calculate the heuristic as the normalized ratio
    heuristics = normalized_ratio
    
    return heuristics