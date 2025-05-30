import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the total weight for each item
    total_weight = np.sum(weight, axis=1)
    
    # Calculate the total value for each item
    total_value = np.sum(prize, axis=1)
    
    # Calculate the heuristic score as the ratio of value to weight
    # We normalize by the maximum value to ensure a scale that is comparable across items
    max_value = np.max(total_value)
    heuristic_scores = total_value / max_value
    
    # Return the normalized heuristic scores
    return heuristic_scores