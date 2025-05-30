import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize a matrix of the same shape as the distance matrix with zeros
    heuristics_matrix = np.zeros_like(distance_matrix)
    
    # Compute the heuristics for each edge
    for i in range(len(distance_matrix)):
        for j in range(i+1, len(distance_matrix)):
            # Example heuristic: the heuristic value is the negative of the distance
            heuristics_matrix[i][j] = -distance_matrix[i][j]
            heuristics_matrix[j][i] = heuristics_matrix[i][j]  # The matrix is symmetric
    
    return heuristics_matrix