import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the value-to-weight ratio for each item
    value_to_weight_ratio = prize / weight.sum(axis=1)
    
    # Return the value-to-weight ratio as the heuristic
    return value_to_weight_ratio