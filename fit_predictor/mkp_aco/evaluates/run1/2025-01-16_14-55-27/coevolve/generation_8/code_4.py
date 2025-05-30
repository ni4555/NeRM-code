import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the weighted ratio for each item
    weighted_ratio = prize / weight.sum(axis=1)

    # Sort items by weighted ratio in descending order
    sorted_indices = np.argsort(weighted_ratio)[::-1]

    # Apply dynamic multi-criteria sorting based on constraints
    constraints = np.ones((weight.shape[0], 1))
    for i in range(weight.shape[1]):
        # Normalize the weight by the sum of weights to ensure each dimension is considered
        normalized_weight = weight[:, i] / weight.sum(axis=1)
        # Sort items by normalized weight in each dimension, descending order
        sorted_indices_by_dim = np.argsort(normalized_weight)[::-1]
        # Update sorted indices based on the sorted indices by dimension
        sorted_indices = np.intersect1d(sorted_indices, sorted_indices_by_dim)

    # Compute the heuristic value for each item
    heuristics = np.zeros(prize.shape)
    heuristics[sorted_indices] = weighted_ratio[sorted_indices]

    return heuristics