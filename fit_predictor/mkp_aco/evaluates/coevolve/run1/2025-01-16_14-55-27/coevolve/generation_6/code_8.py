import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Initialize a heuristic array with zeros
    n = prize.size
    heuristics = np.zeros(n)
    
    # Calculate the weighted ratio for each item
    ratio = prize / weight.sum(axis=1)
    
    # Normalize the ratios by summing them and dividing by the sum
    normalized_ratio = ratio / ratio.sum()
    
    # Calculate the heuristic for each item by its weighted ratio
    heuristics = normalized_ratio
    
    return heuristics