import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the value-to-weight ratio for each item
    value_to_weight_ratio = prize / weight.sum(axis=1)
    
    # Normalize the value-to-weight ratio to get a heuristic score
    max_ratio = np.max(value_to_weight_ratio)
    heuristics = value_to_weight_ratio / max_ratio
    
    return heuristics