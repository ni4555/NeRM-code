import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Assuming the prize and weight arrays are of shape (n,) and (n, m) respectively
    # where n is the number of items and m is the number of dimensions
    
    # Calculate the total weight of each item across all dimensions
    total_weight = np.sum(weight, axis=1)
    
    # Calculate the heuristic value for each item
    # Here, we use a simple heuristic that is the ratio of prize to total weight
    # This heuristic is a normalized value and assumes that the constraint of each dimension is 1
    heuristics = prize / total_weight
    
    return heuristics