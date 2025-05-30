import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the total value of each item
    total_value = np.sum(prize, axis=1)
    
    # Calculate the weighted ratio for each dimension
    weighted_ratio = total_value / (np.sum(weight, axis=1, keepdims=True))
    
    # Normalize the weighted ratio to ensure that all dimensions are on the same scale
    min_weighted_ratio = np.min(weighted_ratio)
    max_weighted_ratio = np.max(weighted_ratio)
    normalized_weighted_ratio = (weighted_ratio - min_weighted_ratio) / (max_weighted_ratio - min_weighted_ratio)
    
    # Combine the normalized weighted ratio with the total value to create a heuristic score
    heuristic_score = normalized_weighted_ratio * total_value
    
    # Sort the items based on the heuristic score in descending order
    sorted_indices = np.argsort(heuristic_score)[::-1]
    
    # Create an array to store the heuristics
    heuristics = np.zeros_like(prize)
    
    # Calculate the heuristics for each item
    for i in sorted_indices:
        # Initialize the heuristic with the total value
        heuristics[i] = prize[i]
        
        # Subtract the minimum weighted ratio to ensure that all heuristics are non-negative
        heuristics[i] -= min_weighted_ratio
    
    return heuristics