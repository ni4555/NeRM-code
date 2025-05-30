import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize the heuristic matrix with zeros
    heuristic_matrix = np.zeros_like(distance_matrix, dtype=np.float64)
    
    # Calculate the heuristic for each edge based on the sum of the minimum distances to the start and end nodes
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            if i != j:
                # Add the distance to the nearest neighbor
                heuristic_matrix[i, j] = np.min(distance_matrix[i, :]) + np.min(distance_matrix[:, j])
            else:
                # Distance to itself is zero
                heuristic_matrix[i, j] = 0
    
    return heuristic_matrix