import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    m = weight.shape[1]
    max_rew = np.sum(prize * np.ones(m), axis=1)
    for i in range(m):
        weight[:, i] /= weight[:, i].max()
    heuristic = np.dot(prize, weight) / max_rew
    return heuristic
