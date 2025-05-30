import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize a matrix of the same shape as the distance matrix with zeros
    heuristics_matrix = np.zeros_like(distance_matrix)
    
    # Iterate over the distance matrix to calculate the heuristics
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            # Calculate the heuristic value for the edge from i to j
            # This is a simple example, where the heuristic is the inverse of the distance
            heuristics_matrix[i, j] = 1 / (distance_matrix[i, j] + 1e-10)  # Adding a small constant to avoid division by zero
    
    return heuristics_matrix