import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize a matrix with zeros, same shape as the input distance matrix
    heuristics_matrix = np.zeros_like(distance_matrix)
    
    # Loop over each pair of nodes
    for i in range(len(distance_matrix)):
        for j in range(i + 1, len(distance_matrix)):
            # Calculate the heuristic value as the negative of the distance (since we are looking for the shortest path)
            heuristics_matrix[i][j] = -distance_matrix[i][j]
            heuristics_matrix[j][i] = heuristics_matrix[i][j]  # Symmetry property of the distance matrix
    
    return heuristics_matrix