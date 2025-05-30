import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Assuming that the heuristics are calculated based on the sum of each item's prize values
    # while considering the weight constraint of 1 in each dimension.
    # This is a simplistic heuristic for demonstration purposes.
    heuristics = np.sum(prize * weight, axis=1)
    return heuristics