import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    n = prize.shape[0]
    m = weight.shape[1]
    
    # Calculate the weighted ratio for each item
    weighted_ratio = prize / np.sum(weight, axis=1)
    
    # Calculate the cumulative prize normalization
    cumulative_prize = np.cumsum(prize)
    cumulative_prize_ratio = cumulative_prize / np.sum(cumulative_prize)
    
    # Combine the heuristic values
    heuristics = weighted_ratio * cumulative_prize_ratio
    
    return heuristics