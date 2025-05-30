import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Ensure weight is a 2D array where each row represents an item and each column represents a dimension
    assert weight.ndim == 2 and weight.shape[1] == 1, "Weight array should be of shape (n, 1)"
    
    # Calculate the weighted ratio for each item
    weighted_ratio = prize / weight
    
    # Normalize the weighted ratio to ensure values are comparable
    # We use a simple normalization method that scales the values to a 0-1 range
    min_ratio = np.min(weighted_ratio)
    max_ratio = np.max(weighted_ratio)
    normalized_ratio = (weighted_ratio - min_ratio) / (max_ratio - min_ratio)
    
    # Create the heuristic array based on the normalized weighted ratio
    heuristics = normalized_ratio
    
    return heuristics

# Example usage:
prize = np.array([60, 100, 120, 70])
weight = np.array([1, 1, 1, 1])
print(heuristics_v2(prize, weight))