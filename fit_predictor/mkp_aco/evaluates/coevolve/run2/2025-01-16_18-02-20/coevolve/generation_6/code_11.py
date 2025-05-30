import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the value-to-weight ratio for each item
    value_to_weight_ratio = prize / weight.sum(axis=1)
    
    # Normalize the ratios to create a probability distribution
    normalized_ratio = value_to_weight_ratio / value_to_weight_ratio.sum()
    
    # Use a stochastic sampling algorithm to determine the heuristics
    heuristics = np.random.choice([0, 1], p=normalized_ratio, size=prize.shape)
    
    return heuristics