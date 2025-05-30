import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the value-to-weight ratio for each item
    value_to_weight_ratio = prize / weight
    
    # Prioritize items based on the value-to-weight ratio
    heuristics = np.argsort(value_to_weight_ratio)[::-1]
    
    return heuristics