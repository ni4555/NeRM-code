import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the heuristic for each item based on the prize and weight
    # For simplicity, we'll use a heuristic that is the ratio of prize to weight
    # across all dimensions, which should be 1 given the constraint that weight has dimension m and is fixed to 1 for each item.
    heuristics = prize / weight
    return heuristics