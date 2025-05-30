import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the heuristic for each item
    # In this simple case, we use the prize divided by the total weight (which is equal to the number of dimensions)
    # This assumes that the weight array has shape (n, m) and that the sum of weights for each item is equal to m
    # where m is the dimension of weights each item has
    heuristics = prize / weight.sum(axis=1)
    return heuristics