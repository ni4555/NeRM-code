import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate weighted ratio for each item
    weighted_ratio = prize / np.sum(weight, axis=1)
    
    # Calculate cumulative prize for each item
    cumulative_prize = np.cumsum(prize)
    
    # Normalize cumulative prize
    normalized_cumulative_prize = cumulative_prize / cumulative_prize[-1]
    
    # Combine weighted ratio analysis and cumulative prize normalization
    heuristics = weighted_ratio * normalized_cumulative_prize
    
    return heuristics