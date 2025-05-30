import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the value-to-weight ratio for each item
    value_to_weight_ratio = prize / weight
    
    # Normalize the ratio to get a heuristic score
    heuristic = value_to_weight_ratio / np.sum(value_to_weight_ratio)
    
    return heuristic