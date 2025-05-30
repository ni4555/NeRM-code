import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize the heuristic matrix with zeros
    heuristic_matrix = np.zeros_like(distance_matrix)
    
    # Calculate the heuristic values for each edge
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            if i != j:
                # For simplicity, let's use the reciprocal of the distance as the heuristic
                # You can replace this with more sophisticated heuristics
                heuristic_matrix[i][j] = 1 / distance_matrix[i][j]
    
    return heuristic_matrix