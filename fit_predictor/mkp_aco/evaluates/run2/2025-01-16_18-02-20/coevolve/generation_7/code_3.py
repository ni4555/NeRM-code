import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the value-to-weight ratio for each item
    value_to_weight_ratio = prize / weight
    
    # Normalize the value-to-weight ratio to ensure non-negative values
    min_ratio = np.min(value_to_weight_ratio)
    normalized_ratio = value_to_weight_ratio - min_ratio
    
    # Scale the normalized ratios to ensure they can be interpreted as probabilities
    max_ratio = np.max(normalized_ratio)
    scaled_ratio = normalized_ratio / max_ratio
    
    # Convert the scaled ratios to heuristics by applying a sigmoid function
    heuristics = 1 / (1 + np.exp(-scaled_ratio))
    
    return heuristics