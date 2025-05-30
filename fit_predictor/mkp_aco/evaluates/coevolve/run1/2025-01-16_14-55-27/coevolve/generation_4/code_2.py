import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    n = len(prize)
    m = weight.shape[1]
    
    # Calculate the weighted ratio for each item
    weighted_ratio = prize / weight.sum(axis=1)
    
    # Normalize the cumulative prize
    cumulative_prize = np.cumsum(prize)
    cumulative_prize /= cumulative_prize[-1]
    
    # Combine weighted ratio analysis and cumulative prize normalization
    heuristics = weighted_ratio * cumulative_prize
    
    # Scale the heuristics to ensure they sum to 1
    heuristics /= heuristics.sum()
    
    return heuristics