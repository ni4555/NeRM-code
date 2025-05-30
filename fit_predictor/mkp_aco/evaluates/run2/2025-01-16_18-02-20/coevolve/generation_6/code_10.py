import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the value-to-weight ratio for each item
    value_to_weight_ratio = prize / weight.sum(axis=1)
    
    # Normalize the value-to-weight ratio to get a promising score for each item
    normalized_scores = value_to_weight_ratio / value_to_weight_ratio.sum()
    
    # Return the normalized scores as the heuristics
    return normalized_scores