import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the total weight capacity for each knapsack
    total_capacity = np.sum(weight, axis=1)
    
    # Normalize prize values
    normalized_prize = prize / np.sum(prize)
    
    # Initialize a heuristic array with zeros
    heuristics = np.zeros_like(prize)
    
    # Calculate heuristic values based on normalized prize and total capacity
    for i in range(weight.shape[0]):
        # Calculate the total capacity that can be allocated to item i
        allocatable_capacity = np.sum(total_capacity) - np.sum(weight[:i])
        
        # Update heuristic for item i
        heuristics[i] = normalized_prize[i] * allocatable_capacity / weight[i]
    
    # Adjust heuristics to account for weight distribution across knapsacks
    # by using a simple adaptive weight distribution strategy
    for i in range(weight.shape[0]):
        for j in range(weight.shape[0]):
            if i != j:
                # Calculate the weight distribution factor
                weight_dist_factor = weight[j] / total_capacity[j]
                
                # Update heuristic for item i based on the weight distribution
                heuristics[i] += normalized_prize[i] * weight_dist_factor
    
    # Normalize heuristics to ensure they sum up to 1
    heuristics /= np.sum(heuristics)
    
    return heuristics
