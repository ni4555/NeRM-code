import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # This is a simple example of a heuristic that calculates the value-to-weight ratio
    # for each item, assuming the weight matrix has only ones, which is a special case
    # for a single dimension constraint of 1.
    
    # Compute the sum of the prizes and the number of dimensions for each item
    prize_sum = np.sum(prize, axis=1)
    num_dimensions = weight.shape[1]
    
    # Calculate the value-to-weight ratio for each item
    heuristic_values = prize_sum / (weight.sum(axis=1) * num_dimensions)
    
    return heuristic_values