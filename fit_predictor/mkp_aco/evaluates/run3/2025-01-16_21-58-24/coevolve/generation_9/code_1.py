import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the value-to-weight ratio for each item
    value_to_weight_ratio = prize / np.sum(weight, axis=1)
    
    # The heuristic value is the negative of the value-to-weight ratio to maximize the selection
    heuristics = -value_to_weight_ratio
    
    return heuristics