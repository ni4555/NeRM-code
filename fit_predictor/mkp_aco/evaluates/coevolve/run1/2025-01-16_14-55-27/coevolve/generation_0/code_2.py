import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Since the weight constraint for each dimension is fixed to 1,
    # we can directly use the prize values as the heuristic scores
    # because the "prominence" of an item is directly proportional to its prize value.
    # Here we assume that the higher the prize, the more promising it is to include the item.
    return prize

# Example usage:
# n = 4 (number of items)
# m = 1 (number of dimensions per item, fixed to 1)
# prize = np.array([10, 20, 30, 40])
# weight = np.array([[1], [1], [1], [1]])
# The function should return the prize array as the heuristic scores.
# heuristics_v2(prize, weight) -> np.array([10, 20, 30, 40])