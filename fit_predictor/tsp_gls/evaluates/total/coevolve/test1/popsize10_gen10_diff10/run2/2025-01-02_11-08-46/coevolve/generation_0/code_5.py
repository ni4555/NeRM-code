import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize the heuristic array with zeros
    heuristics = np.zeros_like(distance_matrix)
    
    # Compute the heuristics for each edge based on the distance matrix
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            # The heuristic is the negative of the distance, as we are looking for the shortest path
            heuristics[i, j] = -distance_matrix[i, j]
    
    return heuristics