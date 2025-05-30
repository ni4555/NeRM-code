import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the value-to-weight ratio for each item
    value_to_weight = prize / weight
    
    # Normalize the value-to-weight ratio
    max_ratio = np.max(value_to_weight)
    normalized_ratio = value_to_weight / max_ratio
    
    return normalized_ratio