import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Assuming the heuristic is based on the ratio of prize to weight
    # and each item has the same weight in each dimension, we can
    # simply calculate the ratio for each item across all dimensions.
    
    # Calculate the total weight for each item as it's the same across all dimensions
    total_weight = weight.sum(axis=1)
    
    # Calculate the heuristic based on the prize to weight ratio
    heuristics = prize / total_weight
    
    return heuristics