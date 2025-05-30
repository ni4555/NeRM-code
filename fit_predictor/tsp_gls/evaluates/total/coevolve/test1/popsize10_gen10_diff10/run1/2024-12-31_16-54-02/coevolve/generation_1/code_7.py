import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize a matrix with the same shape as distance_matrix to store heuristics
    heuristics = np.zeros_like(distance_matrix, dtype=np.float64)
    
    # Compute heuristics for each edge
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            if i != j:
                # Example heuristic: the sum of the minimum distances from node i to all other nodes
                # and from node j to all other nodes, minus the distance between i and j
                min_distances_from_i = np.min(distance_matrix[i, :])
                min_distances_from_j = np.min(distance_matrix[j, :])
                heuristics[i, j] = min_distances_from_i + min_distances_from_j - distance_matrix[i, j]
    
    return heuristics