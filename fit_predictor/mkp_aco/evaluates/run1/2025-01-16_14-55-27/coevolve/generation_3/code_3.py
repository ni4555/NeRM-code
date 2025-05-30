import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Initialize the heuristics array with zeros
    heuristics = np.zeros_like(prize)
    
    # Calculate the multi-dimensional weighted ratio metric
    weighted_sum = np.sum(weight, axis=1)
    # Avoid division by zero by adding a small epsilon
    epsilon = 1e-10
    weighted_ratio = prize / (weighted_sum + epsilon)
    
    # Calculate the cumulative sum for each item
    cumulative_sum = np.cumsum(weighted_ratio)
    
    # Apply a dynamic sorting mechanism based on the cumulative sum
    sorted_indices = np.argsort(-cumulative_sum)
    
    # Update the heuristics based on the sorted indices
    for index in sorted_indices:
        heuristics[index] = cumulative_sum[index]
    
    return heuristics