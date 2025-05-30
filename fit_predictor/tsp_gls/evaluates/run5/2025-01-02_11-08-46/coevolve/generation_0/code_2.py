import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize a matrix of zeros with the same shape as the distance matrix
    heuristics_matrix = np.zeros_like(distance_matrix)
    
    # Calculate the heuristics for each edge
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            if i != j:
                # A simple heuristic could be the negative of the distance
                heuristics_matrix[i][j] = -distance_matrix[i][j]
            else:
                # No heuristic for the diagonal elements (self-loops)
                heuristics_matrix[i][j] = 0
    
    return heuristics_matrix