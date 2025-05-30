import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming that the distance_matrix is symmetric, the diagonal elements are zero,
    # and all elements are positive.
    assert distance_matrix.ndim == 2, "distance_matrix must be a 2D array"
    assert np.all(distance_matrix >= 0), "distance_matrix contains negative values"
    assert np.allclose(distance_matrix, distance_matrix.T), "distance_matrix must be symmetric"
    
    # Calculate the heuristic values based on the edge weights.
    # A simple heuristic could be the edge weight itself, since we want to minimize them.
    # However, for demonstration purposes, we will use the reciprocal of the distance as
    # the heuristic value, which is common in many metaheuristics for the TSP.
    heuristics_matrix = 1 / (distance_matrix + 1e-8)  # Adding a small value to avoid division by zero
    
    # Return the same shape matrix with the heuristic values.
    return heuristics_matrix