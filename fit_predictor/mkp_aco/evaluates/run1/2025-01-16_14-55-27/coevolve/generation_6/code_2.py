import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Ensure that weight is a 2D array with shape (n, m) where m is the dimension of weights
    if weight.ndim != 2 or weight.shape[1] != 1:
        raise ValueError("weight should be a 2D array with each row representing the weight of an item")
    
    # Calculate the weighted ratio for each item
    weighted_ratio = prize / weight
    
    # Apply dynamic item sorting based on the weighted ratio
    sorted_indices = np.argsort(weighted_ratio)[::-1]
    
    # Apply weighted ratio analysis to determine the heuristics
    heuristics = np.zeros(prize.shape)
    for i in sorted_indices:
        # Heuristic value is the ratio of the prize to the weight
        heuristics[i] = weighted_ratio[i]
    
    return heuristics