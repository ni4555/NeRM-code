import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the value/weight ratio for each item
    value_weight_ratio = prize / weight
    
    # Calculate the heuristic for each item based on the value/weight ratio
    heuristics = value_weight_ratio
    
    return heuristics