import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Assuming prize and weight are of shape (n,) and (n, m) respectively, and each dimension weight[i][j] is 1.
    # We calculate the "promise" of each item as the ratio of its prize to the sum of weights across all dimensions.
    # This assumes that all items are equally "promising" in each dimension, which is a simplification.
    total_weight = weight.sum(axis=1)
    heuristics = prize / total_weight
    return heuristics