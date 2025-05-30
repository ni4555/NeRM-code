import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming the distance matrix is symmetric and the diagonal is filled with zeros
    # Calculate the Manhattan distance from each city to the origin (first city)
    n = distance_matrix.shape[0]
    heuristic_matrix = np.zeros_like(distance_matrix)
    
    for i in range(n):
        for j in range(n):
            if i != j:
                # Calculate the heuristic based on the Manhattan distance
                heuristic_matrix[i, j] = np.abs(distance_matrix[i, 0] - distance_matrix[j, 0]) + \
                                         np.abs(distance_matrix[i, 1] - distance_matrix[j, 1])
    
    return heuristic_matrix