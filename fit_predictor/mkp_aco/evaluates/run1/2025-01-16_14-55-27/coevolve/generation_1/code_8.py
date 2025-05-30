import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Assuming that the heuristic is based on the ratio of prize to weight
    # This is a simple heuristic where we calculate the value per unit weight for each item
    # and then normalize by the sum of all values to ensure the heuristics sum to 1.
    value_per_weight = prize / weight.sum(axis=1)
    normalized_heuristics = value_per_weight / value_per_weight.sum()
    return normalized_heuristics