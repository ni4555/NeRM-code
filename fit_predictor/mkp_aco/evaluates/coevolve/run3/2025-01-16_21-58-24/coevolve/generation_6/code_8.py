import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the heuristic as the prize divided by the weight
    # Since the weight is a 2D array and the constraint for each dimension is 1,
    # we need to sum the weights along the dimension to get the total weight for each item.
    total_weight = np.sum(weight, axis=1)
    # Avoid division by zero for items with zero weight
    total_weight[total_weight == 0] = 1
    # Calculate the heuristic value for each item
    heuristics = prize / total_weight
    return heuristics