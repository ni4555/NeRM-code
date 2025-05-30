import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Assuming that the heuristic is based on the prize to weight ratio
    # Since each dimension weight is fixed to 1, we can calculate the heuristic as prize per unit weight
    heuristics = prize / weight.sum(axis=1)
    return heuristics