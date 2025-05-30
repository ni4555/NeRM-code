import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Normalize weights for each item to ensure they contribute equally to the heuristic
    normalized_weight = weight / np.sum(weight, axis=1, keepdims=True)
    
    # Calculate value-to-weight ratio for each item
    value_to_weight_ratio = prize / normalized_weight
    
    # Normalize value-to-weight ratios to prevent the dominance of high-value items
    max_ratio = np.max(value_to_weight_ratio)
    min_ratio = np.min(value_to_weight_ratio)
    normalized_ratio = (value_to_weight_ratio - min_ratio) / (max_ratio - min_ratio)
    
    # Calculate heuristics based on normalized value-to-weight ratios
    heuristics = normalized_ratio
    
    return heuristics