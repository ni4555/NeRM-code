import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Calculate Manhattan distance for each edge in the matrix
    Manhattan_distances = np.abs(distance_matrix - distance_matrix.T)
    return Manhattan_distances