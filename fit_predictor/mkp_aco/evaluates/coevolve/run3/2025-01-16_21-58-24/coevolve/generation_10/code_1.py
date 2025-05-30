import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the value-to-weight ratio for each item
    value_to_weight = prize / weight.sum(axis=1)
    
    # Normalize the value-to-weight ratio to ensure they are comparable
    normalized_vtw = value_to_weight / value_to_weight.max()
    
    # The heuristics array will be the normalized value-to-weight ratios
    heuristics = normalized_vtw
    return heuristics