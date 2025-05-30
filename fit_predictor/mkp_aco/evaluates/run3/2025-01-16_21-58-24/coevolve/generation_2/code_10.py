import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Normalize the weights to have a maximum of 1 in each dimension
    max_weight = np.max(weight, axis=1, keepdims=True)
    normalized_weight = weight / max_weight
    
    # Calculate the potential value for each item
    potential_value = np.sum(prize * normalized_weight, axis=1)
    
    # Heuristic: The higher the potential value, the more promising the item is
    heuristics = potential_value / np.sum(potential_value)
    
    return heuristics