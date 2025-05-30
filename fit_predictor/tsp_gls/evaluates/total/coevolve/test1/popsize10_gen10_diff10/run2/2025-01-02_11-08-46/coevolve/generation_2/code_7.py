import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming distance_matrix is symmetric and the diagonal is filled with 0s
    # Create a heuristic matrix initialized with zeros
    heuristic_matrix = np.zeros_like(distance_matrix)
    
    # Calculate the minimum distance to each city from every other city
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            if i != j:
                # For each edge, calculate the heuristic value
                # Here we use a simple heuristic that is the minimum distance to any other city
                # This is a simplistic approach and can be replaced with more complex heuristics
                heuristic_matrix[i][j] = np.min(distance_matrix[i]) + np.min(distance_matrix[j])
    
    return heuristic_matrix