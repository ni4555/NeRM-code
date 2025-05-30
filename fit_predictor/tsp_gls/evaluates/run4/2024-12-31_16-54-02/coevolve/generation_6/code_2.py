import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # This function assumes that the distance_matrix is symmetric and the diagonal is filled with zeros.
    # It returns a heuristic matrix where each element is a prior indicator of how bad it is to include
    # that edge in a solution. The heuristic is based on the inverse of the distance (i.e., shorter distances
    # are better as they imply a less costly edge to include in the solution).
    
    # Calculate the inverse of the distance matrix
    inv_distance_matrix = 1.0 / (distance_matrix + 1e-8)  # Adding a small constant to avoid division by zero
    
    # Since the matrix is symmetric, we can use the lower triangle and then transpose it
    lower_triangle = np.tril(inv_distance_matrix)
    upper_triangle = np.triu(inv_distance_matrix)
    
    # Combine the lower and upper triangles to form the heuristic matrix
    heuristic_matrix = lower_triangle + upper_triangle
    
    return heuristic_matrix