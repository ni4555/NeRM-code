import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Assuming each item's weight is in the same dimension as the prize vector
    # We use a simple heuristic that multiplies the prize by a normalized weight
    # The normalization is done by dividing each item's weight by the sum of all weights
    # to ensure that the total weight does not exceed the knapsack capacity
    
    # Calculate the sum of weights to normalize
    weight_sum = np.sum(weight, axis=1)
    
    # Avoid division by zero
    weight_sum[weight_sum == 0] = 1
    
    # Normalize weights
    normalized_weight = weight / weight_sum[:, np.newaxis]
    
    # Calculate heuristic values
    heuristics = normalized_weight * prize
    
    return heuristics