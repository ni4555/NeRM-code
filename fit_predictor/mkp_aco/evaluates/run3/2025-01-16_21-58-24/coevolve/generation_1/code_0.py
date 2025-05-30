import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Example implementation using a simple heuristic (ratio of prize to total weight for each item)
    # Adjust this heuristic as necessary to improve performance
    n, m = weight.shape
    heuristic_values = prize / (weight.sum(axis=1, keepdims=True) + 1e-6)  # Adding a small constant to avoid division by zero
    return heuristic_values