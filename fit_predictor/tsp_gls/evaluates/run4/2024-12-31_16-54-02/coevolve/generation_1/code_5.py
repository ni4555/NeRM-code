import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize the heuristics array with zeros
    heuristics = np.zeros_like(distance_matrix, dtype=np.float64)
    
    # Compute the heuristic for each edge
    for i in range(distance_matrix.shape[0]):
        for j in range(i + 1, distance_matrix.shape[1]):
            # A simple heuristic could be the inverse of the distance
            # You can modify this heuristic to be more sophisticated
            heuristics[i, j] = 1.0 / distance_matrix[i, j]
            heuristics[j, i] = heuristics[i, j]  # Symmetry
    
    return heuristics