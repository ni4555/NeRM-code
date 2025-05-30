import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the total weight for each item by summing the weights across all dimensions
    total_weight = weight.sum(axis=1)
    
    # Compute the heuristic value for each item, which is the prize divided by the total weight.
    # If the total weight is zero (which should not be the case given the problem description), we set the heuristic to zero.
    heuristics = np.where(total_weight > 0, prize / total_weight, 0.0)
    
    return heuristics