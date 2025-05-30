import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Assuming prize and weight are numpy arrays and the constraint is that each dimension is fixed to 1
    # Calculate the heuristic value for each item based on the prize and normalized weight
    normalized_weight = weight / np.sum(weight, axis=1, keepdims=True)
    heuristic_values = prize * np.prod(normalized_weight, axis=1)
    
    return heuristic_values