import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Since the constraint of each dimension is fixed to 1, we can just sum the weights for each item.
    total_weight = weight.sum(axis=1)
    # The heuristic for each item is simply the prize of the item.
    heuristics = prize / total_weight
    return heuristics