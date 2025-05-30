import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Normalize the prize by dividing by the sum of weights in each dimension
    normalized_prize = prize / weight.sum(axis=1, keepdims=True)
    
    # Calculate the weighted sum of normalized prizes
    weighted_normalized_prize = (normalized_prize * weight).sum(axis=1)
    
    # Calculate the heuristics as the difference between the total weighted normalized prize and the total prize
    heuristics = weighted_normalized_prize - prize.sum()
    
    return heuristics