import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the heuristic for each item based on some heuristic function
    # Here we use a simple heuristic based on the ratio of prize to weight sum across all dimensions
    heuristics = prize / weight.sum(axis=1)
    return heuristics