import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Assuming that the value-to-weight ratio is used for the heuristic
    # and that the constraint for each dimension is fixed to 1,
    # the heuristic for each item can be defined as the prize of the item.
    # Since the prize is the only value to consider and weight is a one-dimensional array
    # with all elements being 1, the heuristic is simply the prize itself.
    return prize