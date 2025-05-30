import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Assuming the heuristic function uses a simple ratio of prize to weight sum
    # for each item in each dimension, since the constraint is fixed to 1 for each dimension.
    # This is a naive heuristic for the MKP.
    heuristic_values = prize / (weight.sum(axis=1) + 1e-8)  # Adding a small constant to avoid division by zero
    return heuristic_values