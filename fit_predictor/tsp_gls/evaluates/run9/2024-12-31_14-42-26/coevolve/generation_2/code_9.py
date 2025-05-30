import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize the heuristic matrix with zeros
    heuristic_matrix = np.zeros_like(distance_matrix)
    
    # Calculate the heuristic values
    for i in range(len(distance_matrix)):
        for j in range(i + 1, len(distance_matrix)):
            # Implementing a simple heuristic: the heuristic value is the negative of the distance
            heuristic_matrix[i, j] = -distance_matrix[i, j]
            heuristic_matrix[j, i] = -distance_matrix[j, i]
    
    return heuristic_matrix