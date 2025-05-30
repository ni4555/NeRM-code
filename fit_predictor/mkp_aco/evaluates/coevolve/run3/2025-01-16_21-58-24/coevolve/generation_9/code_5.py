import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    value_to_weight_ratio = prize / weight.sum(axis=1)
    return value_to_weight_ratio