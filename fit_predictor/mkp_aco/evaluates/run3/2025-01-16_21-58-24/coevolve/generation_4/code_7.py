import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the normalized prize value for each item
    normalized_prize = prize / np.sum(prize)
    
    # Calculate the normalized weight for each item in each dimension
    normalized_weight = weight / np.sum(weight, axis=1, keepdims=True)
    
    # Construct the fitness metric by combining normalized prize and normalized weight
    # The fitness metric is the ratio of normalized prize to the sum of normalized weights for each item
    fitness = normalized_prize / np.sum(normalized_weight, axis=1)
    
    # Return the heuristics array
    return fitness