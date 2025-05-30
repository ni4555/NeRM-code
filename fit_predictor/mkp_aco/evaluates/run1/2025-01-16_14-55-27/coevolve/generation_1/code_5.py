import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Initialize an array to hold the heuristic values
    heuristics = np.zeros_like(prize)

    # Compute the heuristic for each item
    for i in range(prize.shape[0]):
        # Calculate the heuristic value as the ratio of prize to weight (since weight constraint is fixed to 1)
        heuristics[i] = prize[i] / weight[i].sum()

    return heuristics