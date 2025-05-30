import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Assuming the heuristics are calculated as the ratio of the prize to the sum of weights
    # for each item, normalized by the maximum ratio found.
    total_weight = np.sum(weight, axis=1)
    max_ratio = np.max(prize / total_weight)
    return prize / (total_weight * max_ratio)