import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Step 1: Calculate weighted ratio (prize/total weight for each item)
    weighted_ratio = prize / np.sum(weight, axis=1)
    
    # Step 2: Sort items based on weighted ratio in descending order
    sorted_indices = np.argsort(weighted_ratio)[::-1]
    
    # Step 3: Normalize the sorted items
    # Calculate the maximum weighted ratio
    max_ratio = np.max(weighted_ratio)
    # Normalize by dividing by the max ratio
    normalized_ratios = weighted_ratio / max_ratio
    
    # Step 4: Return the sorted and normalized heuristic values
    return normalized_ratios[sorted_indices]

# Example usage:
# Assuming we have 3 items with prizes and weights
prize = np.array([50, 60, 40])
weight = np.array([[1, 2], [1, 3], [1, 1]])

# Call the function
heuristic_values = heuristics_v2(prize, weight)
print(heuristic_values)