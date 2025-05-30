import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize a new matrix of the same shape as the distance matrix
    heuristic_matrix = np.zeros_like(distance_matrix)
    
    # Assuming the heuristic is calculated based on some heuristic function
    # For demonstration, we'll use a simple heuristic that assumes the distance
    # between any two cities is the reciprocal of their distance. This is a common
    # heuristic approach in the TSP, but it can be replaced with any other heuristic.
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            # We avoid the diagonal as it represents the distance from a city to itself
            if i != j:
                # The heuristic value is the reciprocal of the distance, with a small epsilon
                # to avoid division by zero
                heuristic_value = 1.0 / (distance_matrix[i][j] + 1e-10)
                heuristic_matrix[i][j] = heuristic_value
    
    return heuristic_matrix