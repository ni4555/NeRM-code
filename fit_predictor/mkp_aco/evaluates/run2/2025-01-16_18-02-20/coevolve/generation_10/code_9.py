import numpy as np
import numpy as np
import random

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    n = prize.shape[0]
    m = weight.shape[1]
    
    # Calculate value-to-weight ratio for each item
    value_to_weight_ratio = prize / weight
    
    # Calculate the heuristic as the negative ratio, so higher ratio items have higher heuristic value
    heuristic = -value_to_weight_ratio.sum(axis=1)
    
    return heuristic

# Example usage:
# Assuming prize and weight arrays are provided for n items and m dimensions
# where each item has a weight of 1 in each dimension.
prize_example = np.array([10, 20, 30, 40])
weight_example = np.array([[1], [1], [1], [1]])
heuristic_example = heuristics_v2(prize_example, weight_example)
print(heuristic_example)