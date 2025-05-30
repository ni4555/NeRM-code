import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Initialize the heuristic array with zeros
    n = prize.shape[0]
    heuristics = np.zeros(n)
    
    # Calculate the fitness for each item
    for i in range(n):
        # Calculate the total value of the item
        total_value = np.sum(prize[i])
        # Calculate the total weight of the item across all dimensions
        total_weight = np.sum(weight[i])
        # Normalize the total value by the total weight to get the density
        density = total_value / total_weight
        # The heuristic is the density of the item
        heuristics[i] = density
    
    # Normalize the heuristic scores to sum to 1
    heuristics /= np.sum(heuristics)
    
    return heuristics