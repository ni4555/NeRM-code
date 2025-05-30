import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Assuming that all weights are 1 for each dimension (as per the problem description)
    # Calculate the heuristic based on prize value
    heuristics = prize / np.sum(weight, axis=1)
    return heuristics