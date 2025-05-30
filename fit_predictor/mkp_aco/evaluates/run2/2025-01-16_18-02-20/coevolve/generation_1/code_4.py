import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Initialize the heuristic values for each item to be equal
    n = prize.shape[0]
    heuristics = np.ones(n)

    # Dynamic weight adjustment and iterative item selection
    for i in range(n):
        # Adjust the weight to emphasize the most promising items
        adjusted_weight = weight[i] * heuristics[i]

        # Update the heuristic based on prize and adjusted weight
        heuristics[i] = prize[i] / adjusted_weight

    # Normalize the heuristics to sum to 1
    heuristics /= heuristics.sum()

    return heuristics