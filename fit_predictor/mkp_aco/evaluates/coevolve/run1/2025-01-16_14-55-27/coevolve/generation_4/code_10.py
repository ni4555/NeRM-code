import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    n = prize.shape[0]
    m = weight.shape[1]
    
    # Calculate weighted ratio analysis
    weighted_ratio = prize / weight.sum(axis=1)
    
    # Calculate cumulative prize for each item
    cumulative_prize = np.cumsum(prize)
    
    # Normalize cumulative prize to account for different total prizes
    normalized_cumulative_prize = cumulative_prize / cumulative_prize.sum()
    
    # Combine weighted ratio and normalized cumulative prize
    heuristics = weighted_ratio * normalized_cumulative_prize
    
    return heuristics