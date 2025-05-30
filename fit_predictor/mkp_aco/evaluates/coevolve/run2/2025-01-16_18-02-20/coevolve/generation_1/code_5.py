import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Since the weight for each dimension is fixed to 1, we can simply use the prize array.
    # Each element in the prize array is the heuristic value for the corresponding item.
    return prize

# Example usage:
# n = 5, m = 1 (1-dimensional weight, so prize and weight are effectively the same)
prize = np.array([10, 20, 30, 40, 50])
weight = np.array([[1], [1], [1], [1], [1]])

# Get heuristics
heuristics = heuristics_v2(prize, weight)
print(heuristics)