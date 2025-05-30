import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    n = prize.shape[0]
    m = weight.shape[1]
    heuristics = np.zeros(n)

    # Calculate weighted ratio for each item
    weighted_ratio = prize / (weight.sum(axis=1))

    # Calculate heuristic value based on weighted ratio and dynamic item sorting
    for i in range(n):
        # Apply some heuristic logic to determine the heuristic value
        # For example, using weighted ratio and a simple sort
        heuristics[i] = weighted_ratio[i]  # This is a placeholder; actual heuristic logic would go here

    return heuristics