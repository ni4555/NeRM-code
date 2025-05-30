import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize an array of zeros with the same shape as distance_matrix
    heuristics = np.zeros_like(distance_matrix)
    
    # For each pair of nodes (i, j) where i is not equal to j, calculate the heuristic
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            if i != j:
                # Assuming the heuristic is the distance from node i to node j
                heuristics[i, j] = distance_matrix[i, j]
    
    return heuristics