import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Initialize the heuristics array with zeros
    heuristics = np.zeros_like(prize)

    # Iterate over each item to calculate its heuristics value
    for i in range(len(prize)):
        # Calculate the weighted ratio for the current item
        weighted_ratio = prize[i] / np.sum(weight[i])

        # Update the heuristics value based on the weighted ratio
        heuristics[i] = weighted_ratio

    return heuristics