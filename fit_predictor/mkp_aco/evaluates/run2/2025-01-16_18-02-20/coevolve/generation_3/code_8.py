import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate value-to-weight ratio for each item
    value_to_weight_ratio = prize / weight
    
    # Sort items based on value-to-weight ratio in descending order
    sorted_indices = np.argsort(value_to_weight_ratio)[::-1]
    
    # Initialize heuristics array with 0s
    heuristics = np.zeros_like(prize)
    
    # Calculate heuristics for each item
    for i, index in enumerate(sorted_indices):
        # Calculate cumulative weight
        cumulative_weight = np.sum(weight[sorted_indices[:i+1]])
        
        # If cumulative weight is within the weight limit, set heuristic to 1
        if cumulative_weight <= 1:
            heuristics[index] = 1
        else:
            # Otherwise, calculate the heuristic based on the remaining weight
            remaining_weight = 1 - cumulative_weight
            heuristics[index] = (prize[index] / remaining_weight) / value_to_weight_ratio[index]
    
    return heuristics