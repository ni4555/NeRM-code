import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming that a larger heuristic value indicates a "worse" edge to include.
    # We'll use the inverse of the distance as the heuristic to reflect this.
    return 1.0 / (distance_matrix + 1e-10)  # Adding a small constant to avoid division by zero.