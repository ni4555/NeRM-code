import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    n = prize.shape[0]
    m = weight.shape[1]
    # Initialize heuristics to the lowest possible float
    heuristics = np.zeros(n)
    # Apply a simple greedy heuristic considering each weight dimension
    for item in range(n):
        heuristics[item] = prize[item] / weight[item].sum()
    return heuristics
