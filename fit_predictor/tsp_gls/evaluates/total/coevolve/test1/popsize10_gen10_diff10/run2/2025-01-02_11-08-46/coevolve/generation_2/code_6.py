import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize the heuristics array with zeros
    heuristics = np.zeros_like(distance_matrix)
    
    # Calculate the heuristics based on the distance matrix
    # This is a placeholder for the actual heuristic calculation logic
    # which would involve the integration of various strategies as described.
    # For demonstration purposes, we'll use a simple heuristic based on the minimum distance
    # to any other node (excluding the current node itself).
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            if i != j:
                heuristics[i, j] = np.min(distance_matrix[i, :]) + np.min(distance_matrix[:, j])
            else:
                heuristics[i, j] = float('inf')  # No self-loops
    
    return heuristics