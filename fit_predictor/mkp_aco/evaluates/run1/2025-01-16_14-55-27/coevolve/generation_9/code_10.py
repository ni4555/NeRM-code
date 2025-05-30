import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate weighted ratio for each item
    weighted_ratio = prize / np.sum(weight, axis=1)
    
    # Sort items based on weighted ratio
    sorted_indices = np.argsort(weighted_ratio)[::-1]
    
    # Initialize heuristic array with the sorted indices
    heuristics = np.zeros_like(weighted_ratio, dtype=float)
    heuristics[sorted_indices] = 1.0
    
    # Apply greedy algorithm
    remaining_capacity = np.ones_like(weight, dtype=float)  # Remaining capacity for each dimension
    
    for i in sorted_indices:
        if np.all(remaining_capacity <= weight[i]):
            # If the item fits within all remaining capacities, add it to the solution
            heuristics[i] = 1.0
        else:
            # Otherwise, calculate how much of the item can be added without exceeding capacity
            for dim in range(weight.shape[1]):
                remaining_capacity[dim] = min(remaining_capacity[dim], weight[i, dim])
            heuristics[i] = np.prod(remaining_capacity / weight[i])
    
    return heuristics

# Example usage:
prize = np.array([60, 100, 120, 130])
weight = np.array([[1, 1], [1, 2], [1, 3], [1, 4]])
print(heuristics_v2(prize, weight))