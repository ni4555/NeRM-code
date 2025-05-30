import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize the heuristics array with the same shape as the distance matrix
    heuristics = np.zeros_like(distance_matrix, dtype=float)
    
    # Compute the minimum pairwise distances for each edge
    for i in range(distance_matrix.shape[0]):
        for j in range(i + 1, distance_matrix.shape[1]):
            min_distance = np.min(distance_matrix[i, :]) + np.min(distance_matrix[:, j])
            heuristics[i, j] = min_distance - distance_matrix[i, j]
            heuristics[j, i] = heuristics[i, j]  # Since the matrix is symmetric
    
    return heuristics