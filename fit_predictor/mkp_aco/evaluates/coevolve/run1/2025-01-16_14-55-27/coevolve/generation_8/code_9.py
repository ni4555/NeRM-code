import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    n = prize.shape[0]
    m = weight.shape[1]
    
    # Calculate the weighted ratio for each item
    weighted_ratio = prize / weight.sum(axis=1)
    
    # Calculate the sum of weights for each item
    weight_sum = weight.sum(axis=1)
    
    # Combine weighted ratio and weight sum for dynamic sorting
    combined_heuristics = weighted_ratio * weight_sum
    
    # Sort the items based on the combined heuristics (e.g., descending order)
    sorted_indices = np.argsort(-combined_heuristics)
    
    # Normalize the sorted heuristics to the range [0, 1]
    max_heuristic = combined_heuristics[sorted_indices[0]]
    normalized_heuristics = combined_heuristics / max_heuristic
    
    return normalized_heuristics[sorted_indices]

# Example usage:
# n = 4
# m = 2
# prize = np.array([10, 20, 30, 40])
# weight = np.array([[1, 2], [1, 3], [1, 4], [1, 5]])
# print(heuristics_v2(prize, weight))