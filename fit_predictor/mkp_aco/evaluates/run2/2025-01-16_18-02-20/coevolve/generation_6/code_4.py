import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the value-to-weight ratio for each item
    value_to_weight_ratio = prize / weight.sum(axis=1, keepdims=True)
    
    # Normalize the value-to-weight ratio to get a probability for each item
    normalized_ratio = value_to_weight_ratio / value_to_weight_ratio.sum()
    
    # Use the probability to create a heuristics array
    heuristics = normalized_ratio * (prize / weight.sum(axis=1, keepdims=True))
    
    return heuristics