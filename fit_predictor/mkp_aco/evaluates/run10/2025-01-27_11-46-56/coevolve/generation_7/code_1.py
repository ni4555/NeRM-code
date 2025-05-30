import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    m = weight.shape[1]
    utility = prize / np.sum(weight, axis=1)
    max_utility = np.max(utility)
    normalized_utility = utility / max_utility
    heuristics = np.argmax(normalized_utility, axis=1)
    return heuristics
