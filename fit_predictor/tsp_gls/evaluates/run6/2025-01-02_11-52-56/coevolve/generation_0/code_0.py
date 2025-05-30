import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize the heuristics array with zeros
    heuristics = np.zeros_like(distance_matrix)
    
    # Iterate over all unique pairs of nodes (i, j) where i < j
    for i in range(distance_matrix.shape[0]):
        for j in range(i + 1, distance_matrix.shape[1]):
            # Calculate the heuristic value as the distance between the two nodes
            heuristics[i, j] = distance_matrix[i, j]
            heuristics[j, i] = distance_matrix[i, j]  # Symmetry of the matrix
    
    return heuristics