import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the value/weight ratio for each item
    value_weight_ratio = prize / weight.sum(axis=1)
    
    # Normalize the ratio to get a heuristic value
    heuristics = value_weight_ratio / value_weight_ratio.sum()
    
    return heuristics