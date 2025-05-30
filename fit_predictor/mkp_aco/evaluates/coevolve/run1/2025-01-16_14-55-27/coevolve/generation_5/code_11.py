import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the ratio of prize to weight for each item
    ratio = prize / weight
    
    # Normalize the ratio values
    normalized_ratio = ratio / np.sum(ratio)
    
    # Return the normalized ratio values as the heuristics
    return normalized_ratio