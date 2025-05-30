import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the total weight for each item by summing across all dimensions
    total_weight = np.sum(weight, axis=1)
    
    # Avoid division by zero by adding a small epsilon to the total weight
    epsilon = 1e-10
    total_weight = np.maximum(total_weight, epsilon)
    
    # Compute the heuristic as the prize divided by the total weight
    # Normalize the heuristic to the range [0, 1]
    heuristics = prize / total_weight
    
    return heuristics