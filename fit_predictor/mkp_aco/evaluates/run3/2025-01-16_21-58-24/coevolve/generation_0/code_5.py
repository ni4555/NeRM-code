import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Assuming the heuristic is based on the ratio of prize to weight for each item
    # Since the weight dimension is fixed to 1, we only need to consider the prize for each item
    return prize / weight