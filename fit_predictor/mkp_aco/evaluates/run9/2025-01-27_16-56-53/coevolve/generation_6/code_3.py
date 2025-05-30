import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    n, m = prize.shape[0], weight.shape[1]
    # Initialize heuristic values to zero
    heuristics = np.zeros(n)
    # Normalize weights for each item across all dimensions
    normalized_weight = weight / np.sum(weight, axis=1)[:, np.newaxis]
    # Calculate the heuristic based on the normalized weights and prize values
    for i in range(n):
        heuristics[i] = np.dot(normalized_weight[i], prize)
    # Apply a threshold to balance exploration and exploitation
    threshold = np.percentile(heuristics, 90)
    heuristics[heuristics < threshold] = 0
    heuristics[heuristics >= threshold] = 1
    return heuristics
