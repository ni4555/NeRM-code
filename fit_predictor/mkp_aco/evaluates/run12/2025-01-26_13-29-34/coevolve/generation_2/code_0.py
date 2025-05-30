import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    n, m = prize.shape[0], weight.shape[1]
    heuristic_values = np.zeros(n)
    for i in range(n):
        # Calculate the sum of weights across all dimensions
        weight_sum = np.sum(weight[i])
        # Calculate the normalized weight
        normalized_weight = weight_sum if weight_sum != 0 else 1
        # Compute the heuristic value as the ratio of prize to normalized weight
        heuristic_values[i] = prize[i] / normalized_weight
    return heuristic_values
