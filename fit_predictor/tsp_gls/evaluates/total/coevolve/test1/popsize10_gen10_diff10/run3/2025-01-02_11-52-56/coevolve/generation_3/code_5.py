import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Invert the distance matrix to use the inverse as the heuristic
    # Assuming that a lower distance is better, the heuristic will be the inverse of the distance.
    return 1.0 / distance_matrix