import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the ratio of prize to weight for each item
    ratio = prize / weight
    
    # Normalize the ratios to get the heuristics
    heuristics = ratio / np.sum(ratio)
    
    return heuristics