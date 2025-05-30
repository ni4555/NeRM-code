import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize the heuristic matrix with zeros
    heuristic_matrix = np.zeros_like(distance_matrix)
    
    # Calculate the heuristic values based on some heuristic algorithm (e.g., nearest neighbor)
    # For demonstration, we'll use a simple heuristic where we set the heuristic as the
    # distance to the nearest node in the matrix.
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            if i != j:  # No distance to itself
                min_distance = np.min(distance_matrix[i, :])
                heuristic_matrix[i, j] = min_distance
    
    return heuristic_matrix