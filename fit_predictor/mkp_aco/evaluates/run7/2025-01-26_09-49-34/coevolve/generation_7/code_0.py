import numpy as np
import numpy as np
import random

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    n, m = prize.shape[0], weight.shape[1]
    max_contribution = np.sum(prize * weight, axis=1)
    probability = max_contribution / np.sum(max_contribution)
    return np.random.choice(np.arange(n), p=probability, size=m)
