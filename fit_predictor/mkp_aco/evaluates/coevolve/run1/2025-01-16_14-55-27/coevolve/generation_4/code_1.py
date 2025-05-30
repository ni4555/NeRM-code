import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the weighted ratio for each item
    weighted_ratio = prize / weight.sum(axis=1)
    
    # Normalize the prize for each item by the cumulative prize sum
    cumulative_prize = np.cumsum(prize)
    normalized_prize = prize / cumulative_prize
    
    # Combine the weighted ratio and normalized prize to form a heuristic
    heuristics = weighted_ratio * normalized_prize
    
    return heuristics