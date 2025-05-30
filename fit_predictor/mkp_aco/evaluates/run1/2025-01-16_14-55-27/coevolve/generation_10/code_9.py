import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Normalize the prize to avoid dominance by larger numbers
    normalized_prize = prize / np.sum(prize)
    
    # Calculate the value-to-weight ratio for each item
    value_to_weight_ratio = normalized_prize / np.sum(weight, axis=1)
    
    # Apply a normalization technique to stabilize the heuristic process
    max_ratio = np.max(value_to_weight_ratio)
    min_ratio = np.min(value_to_weight_ratio)
    normalized_ratio = (value_to_weight_ratio - min_ratio) / (max_ratio - min_ratio)
    
    # Return the normalized value-to-weight ratio as the heuristic
    return normalized_ratio