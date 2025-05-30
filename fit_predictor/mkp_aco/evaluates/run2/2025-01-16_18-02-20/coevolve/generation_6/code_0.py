import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Initialize a numpy array with the same shape as `prize`
    heuristics = np.zeros_like(prize)

    # Normalize the weights for each item to the range [0, 1]
    weight_normalized = weight / np.sum(weight, axis=1, keepdims=True)

    # Calculate the value-to-weight ratio for each item
    value_to_weight_ratio = prize / weight_normalized

    # Compute the heuristics as the sum of the value-to-weight ratio for each item
    heuristics = np.sum(value_to_weight_ratio, axis=1)

    return heuristics