import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the multi-dimensional weighted ratio for each item
    weighted_ratio = np.sum(weight, axis=1) / np.sum(weight, axis=1) ** 2
    
    # Apply the cumulative sum analysis to the prize array
    cumulative_prize = np.cumsum(prize)
    
    # Combine the weighted ratio and cumulative prize to get the heuristic value
    heuristics = weighted_ratio * cumulative_prize
    
    return heuristics