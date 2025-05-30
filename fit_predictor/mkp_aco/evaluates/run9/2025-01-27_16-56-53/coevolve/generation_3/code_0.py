import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    n, m = prize.shape[0], weight.shape[1]
    heuristic = np.dot(prize, np.sum(weight, axis=1)) / np.sum(weight**2, axis=1)
    return heuristic
