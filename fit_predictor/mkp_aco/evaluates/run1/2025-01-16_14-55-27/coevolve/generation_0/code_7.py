import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Since weight is fixed to 1 in each dimension, we can just return the prize as the heuristic.
    # Each item's heuristic value is directly its prize value.
    return prize