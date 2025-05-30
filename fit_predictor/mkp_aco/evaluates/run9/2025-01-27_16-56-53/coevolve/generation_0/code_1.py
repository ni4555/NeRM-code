import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    n, m = prize.shape[0], weight.shape[1]
    heuristics = np.zeros(n)
    
    # Calculate the maximum prize for each item
    max_prize = np.max(prize, axis=1)
    
    # Normalize the prize for each item by its max prize
    normalized_prize = prize / max_prize[:, np.newaxis]
    
    # Calculate the heuristic as the normalized prize minus the sum of weights
    heuristics = normalized_prize - np.sum(weight, axis=1)
    
    return heuristics
