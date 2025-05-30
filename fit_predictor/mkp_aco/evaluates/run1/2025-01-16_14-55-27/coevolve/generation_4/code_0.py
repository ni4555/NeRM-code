import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the weighted ratio for each item
    weighted_ratio = prize / weight.sum(axis=1)
    
    # Normalize the cumulative prize for each item
    cumulative_prize = np.cumsum(prize)
    normalized_prize = cumulative_prize / cumulative_prize[-1]
    
    # Combine the weighted ratio and normalized prize using a heuristic
    heuristics = weighted_ratio * normalized_prize
    
    # Return the heuristics array
    return heuristics