import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    n = prize.shape[0]
    m = weight.shape[1]
    
    # Calculate the cumulative prize for each item
    cumulative_prize = np.cumsum(prize)
    
    # Calculate the weighted ratio for each item
    weighted_ratio = cumulative_prize / weight.sum(axis=1)
    
    # Calculate the cumulative normalized prize for each item
    cumulative_normalized_prize = cumulative_prize / cumulative_prize.sum()
    
    # Combine the weighted ratio and cumulative normalized prize to create a heuristic value
    heuristic_values = weighted_ratio * cumulative_normalized_prize
    
    return heuristic_values