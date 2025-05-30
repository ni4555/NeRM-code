import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Normalize the weights to have a maximum of 1 per dimension
    normalized_weight = weight / np.sum(weight, axis=1, keepdims=True)
    
    # Calculate the potential value for each item based on the prize and normalized weight
    potential_value = prize * normalized_weight
    
    # Since the dimension constraint is fixed to 1, we can simply sum across dimensions
    heuristics = np.sum(potential_value, axis=1)
    
    return heuristics