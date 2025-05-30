import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Ensure that weight has shape (n, m) where m is the dimension of weights and each dimension is fixed to 1
    assert weight.shape[1] == 1, "Each item must have a single weight dimension with a fixed value of 1"
    
    # Calculate the total prize for each item
    total_prize = np.sum(prize, axis=1)
    
    # Normalize the prize values by the sum of all prizes to get the relative importance of each item
    normalized_prize = total_prize / np.sum(total_prize)
    
    # Since the weight is fixed to 1 for each item, the heuristic value can be the normalized prize value
    heuristics = normalized_prize
    
    return heuristics