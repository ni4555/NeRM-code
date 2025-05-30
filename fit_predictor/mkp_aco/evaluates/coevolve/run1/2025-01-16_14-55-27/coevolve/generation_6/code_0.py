import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Initialize the heuristics array with the ratio of prize to weight
    heuristics = prize / weight

    # Sort the heuristics based on the weighted ratio in descending order
    # The constraint of each dimension is fixed to 1, so we can use the first dimension
    # for sorting purposes, assuming all dimensions contribute equally to the weight
    sorted_indices = np.argsort(-heuristics[:, 0])

    # Apply the sorted indices to the heuristics array
    heuristics_sorted = np.zeros_like(heuristics)
    heuristics_sorted[sorted_indices] = heuristics[sorted_indices]

    return heuristics_sorted