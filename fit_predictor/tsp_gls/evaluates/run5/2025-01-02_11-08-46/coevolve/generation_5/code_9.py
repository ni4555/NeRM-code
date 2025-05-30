import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize the heuristic matrix with the same shape as the distance matrix
    heuristic_matrix = np.zeros_like(distance_matrix)
    
    # Calculate the sum of all distances in the distance matrix
    total_distance = np.sum(distance_matrix)
    
    # Calculate the heuristic for each edge
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            if i != j:
                # Subtract the distance of the edge from the total distance
                # to get the heuristic value
                heuristic_matrix[i][j] = total_distance - distance_matrix[i][j]
    
    return heuristic_matrix