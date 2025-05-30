import numpy as np
import numpy as np
import random

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the value-to-weight ratio for each item
    value_to_weight_ratio = prize / weight
    
    # Normalize the ratio to ensure all values are between 0 and 1
    max_ratio = np.max(value_to_weight_ratio)
    normalized_ratio = value_to_weight_ratio / max_ratio
    
    # Apply stochastic sampling to select high-value items
    stochastic_sample = np.random.rand(len(normalized_ratio))
    normalized_sample = stochastic_sample / np.sum(stochastic_sample)
    
    # Adjust the heuristic based on the normalized sampling
    heuristics = normalized_ratio * normalized_sample
    
    return heuristics