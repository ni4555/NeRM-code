import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Normalize the prize based on the sum of weights for each item
    normalized_prize = prize / np.sum(weight, axis=1)
    
    # Calculate the value per unit weight for each item
    value_per_weight = normalized_prize / np.sum(weight, axis=1)
    
    # Compute the heuristics score for each item
    heuristics = value_per_weight / np.sum(value_per_weight)
    
    return heuristics