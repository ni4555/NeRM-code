import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming the heuristic here is to return the inverse of the distance matrix
    # since smaller distances are better to include in the path.
    return 1.0 / (distance_matrix + 1e-10)  # Adding a small constant to avoid division by zero