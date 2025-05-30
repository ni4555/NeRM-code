import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the value/weight ratio for each item
    value_weight_ratio = prize / weight
    
    # Normalize the ratios to get a heuristic value for each item
    max_ratio = np.max(value_weight_ratio)
    heuristics = value_weight_ratio / max_ratio
    
    return heuristics