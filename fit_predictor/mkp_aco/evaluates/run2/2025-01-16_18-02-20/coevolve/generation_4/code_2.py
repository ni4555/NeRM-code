import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the value-to-weight ratio for each item
    value_to_weight_ratio = prize / weight
    
    # Normalize the ratios to sum to 1 for stochastic sampling
    normalized_ratio = value_to_weight_ratio / np.sum(value_to_weight_ratio)
    
    # Sample from the normalized ratios to determine the heuristic values
    heuristics = np.random.choice(normalized_ratio, size=len(prize))
    
    return heuristics