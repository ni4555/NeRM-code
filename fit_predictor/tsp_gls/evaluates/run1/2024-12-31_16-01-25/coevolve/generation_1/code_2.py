import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize the heuristics array with zeros, same shape as the distance matrix
    heuristics = np.zeros_like(distance_matrix)
    
    # Compute the heuristic values
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            if i != j:
                # Heuristic value could be based on the inverse of the distance
                heuristics[i, j] = 1 / (distance_matrix[i, j] + 1e-6)  # Adding a small constant to avoid division by zero
    
    return heuristics