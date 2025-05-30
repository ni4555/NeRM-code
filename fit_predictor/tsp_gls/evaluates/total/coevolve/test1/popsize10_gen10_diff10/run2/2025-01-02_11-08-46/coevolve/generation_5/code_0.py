import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # This function assumes that the distance matrix is symmetric and the diagonal elements are 0.
    # It also assumes that the distance matrix is square (n x n).
    
    # Calculate the sum of all distances in the matrix to use for normalization
    total_distance = np.sum(distance_matrix)
    
    # Perform distance-weighted normalization
    normalized_distance_matrix = distance_matrix / total_distance
    
    # Create a matrix where each element is the inverse of the normalized distance
    # (i.e., how "bad" it is to include that edge in the solution)
    heuristics_matrix = 1 / (normalized_distance_matrix + 1e-10)  # Adding a small constant to avoid division by zero
    
    return heuristics_matrix