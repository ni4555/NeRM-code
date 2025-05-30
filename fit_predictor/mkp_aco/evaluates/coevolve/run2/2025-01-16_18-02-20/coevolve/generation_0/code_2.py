import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Assuming the heuristics are based on the ratio of prize to weight
    # We normalize the prize by dividing it by the sum of all weights to avoid overflow issues
    normalized_prize = prize / np.sum(weight)
    
    # Calculate the heuristics as the normalized prize minus the average normalized prize
    heuristics = normalized_prize - np.mean(normalized_prize)
    
    # Since the weight constraint is fixed to 1 for each dimension, we can ignore the weight array
    # and just focus on the prize-based heuristics
    
    return heuristics