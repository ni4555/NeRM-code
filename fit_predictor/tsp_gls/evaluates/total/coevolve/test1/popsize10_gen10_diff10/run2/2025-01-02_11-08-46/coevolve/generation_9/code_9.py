import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize a matrix with zeros of the same shape as the input
    heuristics_matrix = np.zeros_like(distance_matrix)
    
    # Compute the heuristic for each edge
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            # Apply the innovative heuristic here
            # Placeholder for the actual heuristic logic
            heuristics_matrix[i][j] = np.abs(distance_matrix[i][j] - np.mean(distance_matrix))
    
    return heuristics_matrix