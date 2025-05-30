import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Since each dimension weight is fixed to 1 and m is the dimension of weights,
    # we can calculate the heuristic based on the ratio of prize to weight in each dimension.
    # We use the maximum ratio as the heuristic score because we want to maximize the prize.
    heuristics = np.max(prize / weight, axis=1)
    return heuristics