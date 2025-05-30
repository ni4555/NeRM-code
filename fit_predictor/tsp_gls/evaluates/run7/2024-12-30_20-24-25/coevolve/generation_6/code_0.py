import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize a matrix with the same shape as the distance matrix to store the heuristics
    heuristics_matrix = np.zeros_like(distance_matrix, dtype=float)
    
    # Compute Manhattan distance for each edge
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            if i != j:
                heuristics_matrix[i][j] = abs(i - j)  # Manhattan distance is the sum of absolute differences
    
    return heuristics_matrix