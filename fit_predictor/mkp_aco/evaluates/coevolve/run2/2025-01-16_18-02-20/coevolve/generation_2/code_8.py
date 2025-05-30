import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the total normalized value for each item
    normalized_value = prize / np.min(weight, axis=1)
    
    # Return the normalized value as the heuristic
    return normalized_value

# Example usage:
# n = 4, m = 2
# prize = np.array([60, 100, 120, 70])
# weight = np.array([[1, 2], [1, 1], [2, 2], [1, 1]])
# heuristics = heuristics_v2(prize, weight)
# print(heuristics)