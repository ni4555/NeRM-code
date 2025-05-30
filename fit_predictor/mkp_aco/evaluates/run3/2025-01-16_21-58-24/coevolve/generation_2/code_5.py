import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the heuristic value for each item as the ratio of prize to weight
    # Since the constraint of each dimension is fixed to 1, we sum the weights across dimensions
    total_weight = np.sum(weight, axis=1)
    heuristics = prize / total_weight
    return heuristics