import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    n, m = weight.shape
    total_weight = np.sum(weight, axis=1)
    heuristic_values = np.max(prize / total_weight[:, np.newaxis], axis=1)
    return heuristic_values
