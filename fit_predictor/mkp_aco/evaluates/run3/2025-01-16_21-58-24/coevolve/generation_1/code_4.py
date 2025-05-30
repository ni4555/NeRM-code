import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the total weight for each item
    total_weight = np.sum(weight, axis=1)
    
    # Normalize the total weight to the maximum possible value
    max_weight = np.max(total_weight)
    normalized_weight = total_weight / max_weight
    
    # Calculate the heuristic value as the ratio of prize to normalized weight
    heuristics = prize / normalized_weight
    
    return heuristics