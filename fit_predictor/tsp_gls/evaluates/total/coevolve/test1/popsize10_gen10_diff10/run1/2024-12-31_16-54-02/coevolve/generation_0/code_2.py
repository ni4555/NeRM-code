import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize the heuristics array with zeros
    heuristics = np.zeros_like(distance_matrix)
    
    # Iterate over the rows and columns of the distance matrix
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            if i != j:
                # For each edge (i, j), calculate the heuristic
                # A simple heuristic could be the distance itself
                heuristics[i][j] = distance_matrix[i][j]
    
    return heuristics