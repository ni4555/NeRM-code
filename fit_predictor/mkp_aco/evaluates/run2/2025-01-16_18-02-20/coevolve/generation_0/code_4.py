import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Assuming the heuristics are based on the ratio of prize to weight for each item
    # since the weight constraint for each dimension is 1, the total weight of an item is the sum of its weights in all dimensions.
    total_weight = weight.sum(axis=1)
    # Avoid division by zero for items with zero weight
    total_weight[total_weight == 0] = 1
    # Calculate the heuristics as the ratio of prize to total weight
    heuristics = prize / total_weight
    return heuristics