import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize the heuristics matrix with the same shape as the distance matrix
    heuristics = np.zeros_like(distance_matrix, dtype=float)
    
    # Calculate a simple distance-based heuristic for initial path estimation
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            heuristics[i, j] = distance_matrix[i, j] / (np.sum(distance_matrix[i]) + 1e-8)
    
    # Apply a symmetric distance matrix for further exploration
    heuristics = heuristics + heuristics.T - np.diag(heuristics.diagonal())
    
    return heuristics