import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # This function calculates a heuristic value for each item based on the given prize and weight.
    # Given the constraints and the problem description, we will assume a simple heuristic:
    # The heuristic for each item is the normalized prize-to-weight ratio for each dimension.
    # Since the weight constraint for each dimension is fixed to 1, the weight array is expected to be of shape (n, m),
    # with each weight in the weight array already being normalized to 1.
    
    # Initialize an empty array to store the heuristics
    heuristics = np.zeros_like(prize)
    
    # Loop through each item to calculate the heuristic value
    for i in range(prize.shape[0]):
        # Calculate the prize-to-weight ratio for each dimension
        prize_to_weight_ratio = prize[i] / weight[i].sum()
        # Normalize the ratio by subtracting the mean ratio to ensure non-negative values
        normalized_ratio = prize_to_weight_ratio - np.mean(prize / weight.sum(axis=1))
        # Assign the heuristic value to the current item
        heuristics[i] = normalized_ratio
    
    return heuristics