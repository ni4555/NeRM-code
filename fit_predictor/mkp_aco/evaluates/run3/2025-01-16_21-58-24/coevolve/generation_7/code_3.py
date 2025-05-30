import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Initialize an array to hold the promise values for each item
    heuristics = np.zeros_like(prize)
    
    # Calculate the prize-to-weight ratio for each item in each dimension
    # Since the constraint is fixed to 1 for each dimension, we can sum the weights across dimensions
    total_weight_per_item = np.sum(weight, axis=1)
    
    # Calculate the promise as the ratio of prize to total weight per item
    # We normalize the prize by dividing by the total weight to get a per-item prize-to-weight ratio
    heuristics = prize / total_weight_per_item
    
    # Return the array of promises
    return heuristics