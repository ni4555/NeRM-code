import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Assuming prize and weight have shape (n,) and (n, m) respectively.
    # Each dimension weight has a constraint of 1.
    
    # Calculate the total weight capacity of the knapsack.
    knapsack_capacity = np.sum(weight, axis=1)
    
    # Calculate the heuristic value for each item.
    # The heuristic is a function of the ratio of the prize to the weight.
    # Since the weight constraint is 1 for each dimension, we use the sum of weights as the total weight.
    heuristic = prize / knapsack_capacity
    
    # Normalize the heuristic values to ensure they sum to 1.
    heuristic /= np.sum(heuristic)
    
    return heuristic