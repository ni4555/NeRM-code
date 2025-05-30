import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the heuristic value for each item
    # The heuristic is a weighted combination of the prize and the inverse of the weight
    # since a lower weight is preferable. The constraints are fixed to 1, so we do not
    # need to enforce them explicitly here.
    
    # Normalize the weights to account for the fixed constraint of 1
    normalized_weight = weight / np.sum(weight, axis=1, keepdims=True)
    
    # Calculate the heuristic based on the prize and normalized weight
    heuristic_values = prize / normalized_weight
    
    # Return the computed heuristic values
    return heuristic_values