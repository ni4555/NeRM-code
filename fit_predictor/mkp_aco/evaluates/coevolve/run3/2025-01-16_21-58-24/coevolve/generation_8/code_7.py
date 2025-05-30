import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the heuristic value for each item
    # Here we use a simple heuristic based on the ratio of prize to weight sum
    # and normalize it to the range [0, 1].
    # This is a naive heuristic that might not be optimal for the MKP but serves as an example.
    weight_sum = np.sum(weight, axis=1)
    heuristic_values = prize / (weight_sum + 1e-10)  # Adding a small constant to avoid division by zero
    normalized_heuristic = heuristic_values / np.sum(heuristic_values)
    return normalized_heuristic