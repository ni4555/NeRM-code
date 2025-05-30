import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Calculate the maximum distance in the matrix
    max_distance = np.max(distance_matrix)
    
    # Generate a matrix of the same shape as the input with values from 0 to max_distance
    heuristics_matrix = np.arange(max_distance + 1)
    
    # For each edge in the distance matrix, subtract the distance from the corresponding
    # value in the heuristics matrix to indicate how bad it is to include that edge
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            if i != j:
                heuristics_matrix[i, j] -= distance_matrix[i, j]
    
    return heuristics_matrix