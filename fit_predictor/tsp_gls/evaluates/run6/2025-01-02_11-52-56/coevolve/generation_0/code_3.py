import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize the heuristics array with zeros
    heuristics = np.zeros_like(distance_matrix)
    
    # Calculate the heuristics for each edge
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            if i != j:
                # Calculate the heuristic as the negative of the distance (assuming we want to minimize)
                heuristics[i][j] = -distance_matrix[i][j]
            else:
                # The heuristic for an edge connecting a node to itself is considered infinite (or very large)
                heuristics[i][j] = np.inf
    
    return heuristics