import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming that the heuristic is based on the inverse of the distance
    # The smaller the distance, the better the heuristic value
    return 1 / (distance_matrix + 1e-8)  # Adding a small constant to avoid division by zero