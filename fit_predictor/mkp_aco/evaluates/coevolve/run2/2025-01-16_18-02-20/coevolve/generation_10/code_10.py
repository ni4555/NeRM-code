import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate value-to-weight ratio for each item
    value_to_weight = prize / weight
    
    # Normalize the value-to-weight ratios to get the heuristics
    heuristics = value_to_weight / np.sum(value_to_weight)
    
    return heuristics