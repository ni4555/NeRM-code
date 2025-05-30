import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming the distance_matrix is symmetric and square
    size = distance_matrix.shape[0]
    
    # Create a matrix of the same shape as distance_matrix to store heuristics
    heuristics_matrix = np.zeros_like(distance_matrix)
    
    # Calculate the heuristic for each edge
    for i in range(size):
        for j in range(i+1, size):  # Only calculate for the upper triangle to avoid redundancy
            # The heuristic is the inverse of the distance (assuming distance > 0)
            heuristics_matrix[i, j] = 1 / distance_matrix[i, j]
            heuristics_matrix[j, i] = heuristics_matrix[i, j]  # Since the matrix is symmetric
    
    return heuristics_matrix