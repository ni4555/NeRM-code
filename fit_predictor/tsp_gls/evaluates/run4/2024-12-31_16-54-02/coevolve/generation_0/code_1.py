import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize an array of the same shape as the distance matrix with zeros
    heuristics = np.zeros_like(distance_matrix, dtype=float)
    
    # Calculate the heuristics for each edge
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            if i != j:
                # The heuristic is the negative of the distance, as smaller values are better
                heuristics[i, j] = -distance_matrix[i, j]
    
    return heuristics