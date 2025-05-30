import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Initialize an array of zeros with shape (n,)
    heuristics = np.zeros_like(prize, dtype=float)

    # Iterate over each item
    for i in range(prize.shape[0]):
        # Calculate the ratio of the prize to the total weight across all dimensions
        prize_ratio = prize[i] / weight[i].sum()
        # Update the heuristic value for the current item
        heuristics[i] = prize_ratio

    return heuristics