import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the total weight capacity for each knapsack
    total_capacity = np.sum(weight, axis=1)
    
    # Calculate the total prize for each item
    item_prize_weight_ratio = prize / weight
    
    # Initialize heuristics array with zeros
    heuristics = np.zeros_like(prize)
    
    # Sort items based on the prize-to-weight ratio in descending order
    sorted_indices = np.argsort(item_prize_weight_ratio)[::-1]
    
    # Iterate over each knapsack
    for i in range(weight.shape[0]):
        # Calculate the remaining capacity of the knapsack
        remaining_capacity = total_capacity[i] - np.sum(weight[:i], axis=1)
        
        # Iterate over the sorted items
        for j in sorted_indices:
            # Check if the item can be added to the current knapsack
            if weight[j, i] <= remaining_capacity:
                # Update the heuristics value
                heuristics[j] = 1
                # Update the remaining capacity of the knapsack
                remaining_capacity -= weight[j, i]
                break
    
    return heuristics
