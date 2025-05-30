import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the normalized profit for each item
    normalized_profit = prize / np.sum(prize)
    
    # Calculate the normalized weight for each item in each dimension
    normalized_weight = weight / np.sum(weight, axis=1)[:, np.newaxis]
    
    # Calculate the heuristic value for each item
    heuristic = normalized_profit * np.prod(normalized_weight, axis=1)
    
    return heuristic