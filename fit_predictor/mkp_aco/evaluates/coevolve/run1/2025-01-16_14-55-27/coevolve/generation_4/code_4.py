import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate weighted ratio for each item
    weighted_ratio = prize / np.sum(weight, axis=1)
    
    # Calculate cumulative prize normalization
    cumulative_prize = np.cumsum(prize)
    cumulative_prize_ratio = cumulative_prize / np.sum(cumulative_prize)
    
    # Combine weighted ratio and cumulative prize normalization
    heuristics = weighted_ratio * cumulative_prize_ratio
    
    return heuristics