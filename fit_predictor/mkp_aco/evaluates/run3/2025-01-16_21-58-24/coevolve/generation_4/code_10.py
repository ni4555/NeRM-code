import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Assuming that the weight has shape (n, m) where m=1 as per the problem constraints
    # Calculate the total weight of each item
    total_weight = weight.sum(axis=1)
    
    # Calculate the total value of each item
    total_value = prize.sum(axis=1)
    
    # Calculate the heuristic score for each item
    # The heuristic could be based on the ratio of value to weight
    heuristic_score = total_value / total_weight
    
    # Return the computed heuristic scores
    return heuristic_score