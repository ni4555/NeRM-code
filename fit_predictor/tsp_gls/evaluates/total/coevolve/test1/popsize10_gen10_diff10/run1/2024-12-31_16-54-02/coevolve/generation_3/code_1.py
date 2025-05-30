import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming the distance matrix is symmetric and the diagonal is filled with zeros
    # The heuristic will be based on the negative of the distance matrix
    # since we want to minimize the total path length, which corresponds to minimizing the sum of edge weights
    
    # Generate a matrix with the negative distances
    negative_distance_matrix = -distance_matrix
    
    # Return the negative distance matrix as the heuristic
    return negative_distance_matrix