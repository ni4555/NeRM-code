import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the multi-dimensional weighted ratio for each item
    weighted_ratio = np.prod(weight, axis=1) / np.sum(weight, axis=1)
    
    # Use the prize to adjust the weighted ratio, as it represents the item's desirability
    adjusted_ratio = weighted_ratio * prize
    
    # Sort items based on the adjusted ratio in descending order
    sorted_indices = np.argsort(-adjusted_ratio)
    
    # Initialize heuristics array with zeros
    heuristics = np.zeros_like(prize)
    
    # Assign a higher heuristic value to more promising items
    heuristics[sorted_indices] = adjusted_ratio[sorted_indices]
    
    return heuristics