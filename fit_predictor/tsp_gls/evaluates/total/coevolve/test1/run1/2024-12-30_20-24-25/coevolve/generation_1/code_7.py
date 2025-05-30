import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming the heuristic is the reciprocal of the distance
    # where distance is non-zero to avoid division by zero errors
    return np.reciprocal(distance_matrix[distance_matrix > 0])