import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the weighted ratio for each item
    weighted_ratio = prize / np.sum(weight, axis=1)
    
    # Sort items based on the weighted ratio in descending order
    sorted_indices = np.argsort(weighted_ratio)[::-1]
    
    # Initialize an empty array to store the heuristic scores
    heuristics = np.zeros_like(weighted_ratio)
    
    # Use a simple greedy approach to assign heuristic scores
    for i, index in enumerate(sorted_indices):
        # Check if adding the current item would exceed the weight constraints
        # For simplicity, we assume the weight constraints are fixed to 1 for each dimension
        if np.all(weight[index] <= 1):
            heuristics[index] = weighted_ratio[index]
        else:
            heuristics[index] = 0
    
    return heuristics

# Example usage:
# prize = np.array([60, 100, 120, 80])
# weight = np.array([[1, 1], [2, 1], [1, 2], [2, 2]])
# heuristics = heuristics_v2(prize, weight)
# print(heuristics)