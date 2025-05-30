import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    n = prize.shape[0]
    m = weight.shape[1]
    
    # Calculate the total weight of each item
    total_weight = np.sum(weight, axis=1)
    
    # Initialize the heuristics array
    heuristics = np.zeros(n)
    
    # Loop through each item and calculate its heuristic
    for i in range(n):
        # Calculate the ratio of prize to total weight
        # Use max to avoid division by zero
        ratio = np.max([prize[i], total_weight[i]])
        
        # Normalize the ratio by the number of dimensions to consider it as a weighted heuristic
        heuristics[i] = ratio / m
    
    # Normalize heuristics to be in the range of [0, 1]
    heuristics /= np.sum(heuristics)
    
    return heuristics