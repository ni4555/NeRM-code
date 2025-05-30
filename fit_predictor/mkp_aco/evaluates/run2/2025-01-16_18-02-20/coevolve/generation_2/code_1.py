import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Normalize the prize values
    normalized_prize = prize / np.sum(prize)
    
    # Calculate the normalized weights for each item
    normalized_weight = np.sum(weight, axis=1) / np.sum(weight)
    
    # Calculate the heuristic value for each item as the product of normalized prize and normalized weight
    heuristics = normalized_prize * normalized_weight
    
    # Return the heuristic values as an array
    return heuristics