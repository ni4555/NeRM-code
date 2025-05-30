import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize the heuristics array with zeros, same shape as distance_matrix
    heuristics = np.zeros_like(distance_matrix, dtype=np.float64)
    
    # Fill the diagonal with 0s because the distance from a node to itself is 0
    np.fill_diagonal(heuristics, 0)
    
    # Compute the heuristics as the maximum distance to any other node
    for i in range(len(distance_matrix)):
        for j in range(len(distance_matrix)):
            if i != j:
                heuristics[i][j] = np.max(distance_matrix[i])
    
    return heuristics