import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize the heuristics array with zeros
    heuristics = np.zeros_like(distance_matrix)
    
    # Compute the heuristics for each edge based on some heuristic
    # For example, we can use the maximum distance to any other node as a heuristic
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            if i != j:
                heuristics[i, j] = np.max(distance_matrix[i, :]) + np.max(distance_matrix[:, j])
    
    return heuristics