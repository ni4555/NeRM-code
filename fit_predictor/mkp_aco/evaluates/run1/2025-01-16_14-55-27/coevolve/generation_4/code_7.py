import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the weighted ratio for each item
    weighted_ratio = prize / np.sum(weight, axis=1)
    
    # Normalize the cumulative prize for each dimension
    cumulative_prize = np.cumsum(prize)
    normalized_cumulative_prize = cumulative_prize / np.sum(cumulative_prize)
    
    # Combine weighted ratio and normalized cumulative prize
    heuristics = weighted_ratio * normalized_cumulative_prize
    
    return heuristics