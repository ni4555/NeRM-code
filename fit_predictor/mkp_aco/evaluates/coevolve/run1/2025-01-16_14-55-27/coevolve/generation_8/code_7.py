import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Normalize weights
    weight_normalized = weight / np.sum(weight, axis=1, keepdims=True)
    
    # Calculate weighted ratio for each item
    weighted_ratio = prize * weight_normalized
    
    # Sort items based on a dynamic multi-criteria mechanism
    # The sorting criteria is a combination of weighted ratio and inverse of the sum of weights
    # which ensures that items with higher weights get less priority
    criteria = weighted_ratio / (np.sum(weight, axis=1) + 1e-8)  # Adding a small constant to avoid division by zero
    sorted_indices = np.argsort(criteria)[::-1]  # Descending order
    
    # Calculate heuristics based on sorted order
    heuristics = np.zeros_like(prize)
    heuristics[sorted_indices] = np.arange(len(prize))  # Assign heuristics based on sorted order
    
    return heuristics