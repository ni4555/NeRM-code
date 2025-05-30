import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Assuming weight is a 2D array of shape (n, m) with each dimension's constraint as 1
    # Since the constraint of each dimension is fixed to 1, we can ignore the weight array
    # and just use the prize array to calculate the heuristic.
    # The heuristic is calculated as the prize-to-weight ratio, which is the prize since weight is fixed at 1.
    return prize

# Example usage:
# Let's say we have 5 items with prize values [10, 20, 30, 40, 50]
prize = np.array([10, 20, 30, 40, 50])
# Since each dimension's constraint is 1, the weight for each item will be [1, 1, 1, 1, 1]
weight = np.array([[1], [1], [1], [1], [1]])

# We calculate the heuristics for each item
heuristics = heuristics_v2(prize, weight)
print(heuristics)