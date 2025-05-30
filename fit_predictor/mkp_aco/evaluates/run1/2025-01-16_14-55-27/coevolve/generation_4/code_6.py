import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate weighted ratio
    weighted_ratio = prize / weight.sum(axis=1, keepdims=True)
    
    # Calculate cumulative prize normalization
    cumulative_prize = np.cumsum(prize)
    cumulative_prize /= cumulative_prize[-1]  # Normalize to the last value for consistency
    
    # Combine the two heuristics with a simple weighted sum
    heuristics = weighted_ratio * 0.6 + cumulative_prize * 0.4  # Adjust weights as needed
    
    return heuristics