import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the value-to-weight ratio for each item
    value_to_weight_ratio = prize / weight
    
    # Calculate the average value-to-weight ratio to normalize the values
    average_ratio = np.mean(value_to_weight_ratio)
    
    # Normalize the value-to-weight ratios by dividing by the average ratio
    normalized_ratios = value_to_weight_ratio / average_ratio
    
    # Calculate the heuristic scores based on the normalized ratios
    # Higher heuristic scores correspond to higher priority
    heuristics = -normalized_ratios  # Negative because we want higher values to have higher priority
    
    return heuristics

# Example usage:
# n = number of items
# m = number of knapsacks
# prize = array of shape (n,) with the prize value for each item
# weight = array of shape (n, m) with the weight of each item for each knapsack
# Example input:
# prize = np.array([60, 100, 120, 130])
# weight = np.array([[10, 20], [30, 50], [40, 70], [50, 90]])
# Example output:
# heuristics = heuristics_v2(prize, weight)
# print(heuristics)