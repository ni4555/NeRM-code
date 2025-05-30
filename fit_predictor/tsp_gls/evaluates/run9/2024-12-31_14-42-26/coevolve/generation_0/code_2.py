import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Ensure the distance matrix is a square matrix
    if not np.array_equal(distance_matrix, distance_matrix.T):
        raise ValueError("Distance matrix must be symmetric.")
    
    # Calculate the heuristic matrix by taking the reciprocal of the distance matrix
    # where the diagonal elements are set to 0 because a node to itself has no "badness"
    heuristic_matrix = np.reciprocal(distance_matrix)
    np.fill_diagonal(heuristic_matrix, 0)
    
    return heuristic_matrix