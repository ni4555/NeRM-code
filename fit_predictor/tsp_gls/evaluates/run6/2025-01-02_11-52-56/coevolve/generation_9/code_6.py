import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Ensure the distance matrix is square and symmetric
    if not np.array_equal(distance_matrix, distance_matrix.T):
        raise ValueError("Distance matrix must be symmetric.")
    
    # Compute the heuristic by taking the inverse of the distance matrix
    # This is a simple heuristic assuming that shorter distances are better
    heuristic_matrix = 1.0 / distance_matrix
    
    # Replace any infinities or NaNs with a large number
    heuristic_matrix[np.isinf(heuristic_matrix)] = np.finfo(float).max
    heuristic_matrix[np.isnan(heuristic_matrix)] = np.finfo(float).max
    
    return heuristic_matrix