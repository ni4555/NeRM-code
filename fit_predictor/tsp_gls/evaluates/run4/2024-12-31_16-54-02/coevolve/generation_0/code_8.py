import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming that distance_matrix is symmetric, meaning the distance from i to j is the same as from j to i.
    # Initialize the output matrix with zeros.
    heuristics_matrix = np.zeros_like(distance_matrix)
    
    # Calculate the heuristic for each edge.
    for i in range(len(distance_matrix)):
        for j in range(len(distance_matrix)):
            if i != j:
                # The heuristic can be the inverse of the distance or some other function.
                # For simplicity, we'll use the inverse of the distance (larger values are worse).
                heuristics_matrix[i][j] = 1 / distance_matrix[i][j]
            else:
                # No heuristic for the self-loop.
                heuristics_matrix[i][j] = 0
    
    return heuristics_matrix