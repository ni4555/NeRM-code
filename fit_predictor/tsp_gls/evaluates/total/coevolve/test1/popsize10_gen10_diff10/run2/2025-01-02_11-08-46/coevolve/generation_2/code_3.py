import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming the distance matrix is symmetric, we only need to compute half of it
    n = distance_matrix.shape[0]
    heuristic_matrix = np.zeros_like(distance_matrix)

    # Calculate edge-based heuristics
    for i in range(n):
        for j in range(i + 1, n):
            # Example heuristic: the heuristic is the average distance to all other nodes
            # minus the distance to the current node
            heuristic = np.mean(distance_matrix[i, :]) - distance_matrix[i, j]
            heuristic_matrix[i, j] = heuristic
            heuristic_matrix[j, i] = heuristic

    return heuristic_matrix