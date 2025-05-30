import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize an array of the same shape as distance_matrix with zeros
    heuristics = np.zeros_like(distance_matrix, dtype=float)
    
    # Calculate the heuristic value for each edge
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            # The heuristic value for an edge can be a simple inverse of the distance
            heuristics[i, j] = 1.0 / (distance_matrix[i, j] + 1e-8)  # Adding a small value to avoid division by zero
    
    return heuristics