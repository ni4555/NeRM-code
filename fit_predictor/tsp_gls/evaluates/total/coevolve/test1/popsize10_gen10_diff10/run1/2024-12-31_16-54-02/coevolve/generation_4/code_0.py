import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize the heuristic matrix with zeros
    heuristic_matrix = np.zeros_like(distance_matrix)
    
    # Compute the heuristics based on the edge distance
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            if i != j:
                # Heuristic value is calculated based on the edge distance
                # This is a simple example where we assume the heuristic is the negative distance
                heuristic_matrix[i, j] = -distance_matrix[i, j]
    
    return heuristic_matrix