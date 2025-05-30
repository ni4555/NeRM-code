import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the value-to-weight ratio for each item
    value_to_weight = prize / weight
    
    # Normalize the value-to-weight ratio to ensure a proper ranking
    normalized_vtw = value_to_weight / np.sum(value_to_weight)
    
    # Return the normalized value-to-weight ratio as the heuristic
    return normalized_vtw