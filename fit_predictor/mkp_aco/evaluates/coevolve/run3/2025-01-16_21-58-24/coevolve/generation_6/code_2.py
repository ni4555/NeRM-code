import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Assuming that all weights are normalized to 1 for the given constraint
    # The heuristic could be based on the ratio of prize to weight for each item
    # This is a simple heuristic based on the greedy approach
    heuristic = prize / weight.sum(axis=1)
    return heuristic