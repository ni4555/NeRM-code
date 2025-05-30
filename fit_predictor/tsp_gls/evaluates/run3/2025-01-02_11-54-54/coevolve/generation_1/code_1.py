import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize the heuristics array with zeros
    heuristics = np.zeros_like(distance_matrix)
    
    # Compute the heuristic for each edge
    for i in range(len(distance_matrix)):
        for j in range(i+1, len(distance_matrix)):  # since the matrix is symmetric
            # The heuristic is based on the sum of the distances of the two nodes
            heuristics[i, j] = heuristics[j, i] = distance_matrix[i, j] + distance_matrix[i, 0] + distance_matrix[0, j]
    
    return heuristics