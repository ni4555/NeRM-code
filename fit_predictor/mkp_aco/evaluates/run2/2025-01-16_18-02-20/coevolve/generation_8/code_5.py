import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    n = prize.shape[0]
    m = weight.shape[1]
    
    # Calculate the raw value-to-weight ratios
    value_to_weight_ratio = prize / weight
    
    # Normalize the ratios to ensure they are on the same scale
    max_ratio = np.max(value_to_weight_ratio)
    min_ratio = np.min(value_to_weight_ratio)
    normalized_ratio = (value_to_weight_ratio - min_ratio) / (max_ratio - min_ratio)
    
    # Apply a probabilistic factor to the normalized ratios
    # Here we use a simple exponential decay function, but other distributions could be used
    probabilistic_factor = np.exp(normalized_ratio)
    
    # Normalize the probabilistic factor to sum to 1
    probabilistic_factor /= np.sum(probabilistic_factor)
    
    # Return the heuristics as the probabilistic factor
    return probabilistic_factor