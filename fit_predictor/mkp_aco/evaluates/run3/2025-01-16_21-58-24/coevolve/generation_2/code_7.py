import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Assuming prize and weight are of shape (n,) and (n, m) respectively,
    # and m=1 according to the problem description.
    n = prize.shape[0]
    m = weight.shape[1]
    
    # Compute the heuristic values as the inverse of the sum of weights.
    # The rationale is that a lower sum of weights for an item makes it more
    # promising, so we use its inverse as the heuristic.
    # We use 1 / (1 + sum(weight)) to ensure the heuristic values are non-negative.
    heuristics = 1 / (1 + np.sum(weight, axis=1))
    
    return heuristics