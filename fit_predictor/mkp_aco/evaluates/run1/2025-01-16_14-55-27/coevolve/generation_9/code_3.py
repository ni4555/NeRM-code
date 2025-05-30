import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    n = prize.shape[0]
    m = weight.shape[1]
    
    # Calculate the weighted ratio for each item
    weighted_ratio = prize / m
    
    # Create an array to hold the heuristic values
    heuristic = np.zeros(n)
    
    # Sort the items by weighted ratio in descending order
    sorted_indices = np.argsort(weighted_ratio)[::-1]
    
    # Initialize the total weight accumulated so far
    total_weight = 0
    
    # Iterate over the sorted indices to calculate the heuristic
    for i, index in enumerate(sorted_indices):
        # Check if the item can be added without exceeding the weight limit
        if total_weight + weight[index] <= 1:  # Assuming the knapsack's capacity is 1 for each dimension
            total_weight += weight[index]
            heuristic[index] = 1  # Set the heuristic value to 1 for the selected item
    
    return heuristic