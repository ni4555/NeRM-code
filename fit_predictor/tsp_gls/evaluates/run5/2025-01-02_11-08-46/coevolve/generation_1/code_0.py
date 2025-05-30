import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Ensure the distance matrix is square
    assert distance_matrix.shape[0] == distance_matrix.shape[1], "Distance matrix must be square."
    
    # Initialize the heuristic matrix with zeros
    heuristic_matrix = np.zeros_like(distance_matrix)
    
    # Compute Manhattan distances as the heuristic
    for i in range(len(distance_matrix)):
        for j in range(i + 1, len(distance_matrix)):  # avoid symmetry to save computation
            # Manhattan distance is the sum of the absolute differences of their Cartesian coordinates
            heuristic = np.abs(distance_matrix[i, 0] - distance_matrix[j, 0]) + np.abs(distance_matrix[i, 1] - distance_matrix[j, 1])
            heuristic_matrix[i, j] = heuristic
            heuristic_matrix[j, i] = heuristic  # due to symmetry
    
    return heuristic_matrix