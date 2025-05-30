import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize the same shape matrix with zeros
    heuristics_matrix = np.zeros_like(distance_matrix)
    
    # Compute Manhattan distance between each pair of nodes
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            if i != j:  # Skip the diagonal
                # Assuming that the matrix is symmetric
                # Calculate Manhattan distance
                manhattan_distance = np.abs(i - j)
                heuristics_matrix[i][j] = manhattan_distance
    
    return heuristics_matrix