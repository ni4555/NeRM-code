import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize the heuristic array with zeros
    heuristics = np.zeros_like(distance_matrix)
    
    # Calculate the heuristic for each edge
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            # The heuristic is the inverse of the distance
            heuristics[i, j] = 1 / distance_matrix[i, j]
    
    return heuristics