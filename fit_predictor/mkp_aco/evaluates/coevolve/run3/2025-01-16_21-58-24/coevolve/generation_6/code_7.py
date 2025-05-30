import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the heuristic value for each item
    # Since the constraint of each dimension is fixed to 1, we can calculate the total weight of each item
    total_weight = np.sum(weight, axis=1)
    
    # Calculate the heuristic as the profit divided by the total weight for each item
    # We add a small epsilon to avoid division by zero
    epsilon = 1e-6
    heuristic_values = prize / (total_weight + epsilon)
    
    # Return the computed heuristic values
    return heuristic_values