import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the weighted ratio for each item
    weighted_ratio = prize / weight.sum(axis=1)
    
    # Sort items based on the weighted ratio in descending order
    sorted_indices = np.argsort(weighted_ratio)[::-1]
    
    # Initialize the heuristics array with zeros
    heuristics = np.zeros_like(prize)
    
    # Calculate the cumulative prize for sorted items
    cumulative_prize = np.zeros_like(prize)
    cumulative_prize[sorted_indices[0]] = prize[sorted_indices[0]]
    
    for i in range(1, len(sorted_indices)):
        cumulative_prize[sorted_indices[i]] = cumulative_prize[sorted_indices[i-1]] + prize[sorted_indices[i]]
    
    # Calculate the heuristics based on the cumulative prize
    for i in range(len(sorted_indices)):
        heuristics[sorted_indices[i]] = cumulative_prize[sorted_indices[i]] / (cumulative_prize[-1] if cumulative_prize[-1] != 0 else 1)
    
    return heuristics