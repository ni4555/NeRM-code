import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize the heuristic matrix with the same shape as the distance matrix
    heuristic_matrix = np.zeros_like(distance_matrix, dtype=np.float64)
    
    # Compute Manhattan distance for each edge
    for i in range(len(distance_matrix)):
        for j in range(i + 1, len(distance_matrix)):  # No need to compute for i < j
            # Compute Manhattan distance
            heuristic = np.abs(i - j)  # This is a simplified Manhattan distance for demonstration purposes
            # Assign the heuristic value to both directions (i, j) and (j, i)
            heuristic_matrix[i, j] = heuristic
            heuristic_matrix[j, i] = heuristic
    
    return heuristic_matrix