import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Assuming the heuristic is a simple ratio of prize to weight sum across dimensions
    heuristic_values = prize / np.sum(weight, axis=1)
    return heuristic_values