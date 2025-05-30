import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the value-to-weight ratio for each item
    value_to_weight_ratio = prize / weight
    
    # Normalize the ratios to sum to 1 to create a probability distribution
    total_ratio = np.sum(value_to_weight_ratio)
    heuristics = value_to_weight_ratio / total_ratio
    
    return heuristics