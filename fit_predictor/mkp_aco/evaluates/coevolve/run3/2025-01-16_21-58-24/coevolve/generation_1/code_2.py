import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Assuming that the weight constraint is 1 for each dimension,
    # we can use the prize-to-weight ratio as a heuristic.
    # This heuristic is simple and can be used to rank items based on their value.
    return prize / weight.sum(axis=1)