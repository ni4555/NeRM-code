import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming a simple heuristic: the heuristic value is the inverse of the distance
    # This is a naive approach and might not be suitable for a high-precision requirement
    # or a complex distance matrix. A more sophisticated heuristic function would be
    # needed for the algorithm to surpass the given fitness threshold.
    return 1.0 / (distance_matrix + 1e-8)  # Adding a small constant to avoid division by zero