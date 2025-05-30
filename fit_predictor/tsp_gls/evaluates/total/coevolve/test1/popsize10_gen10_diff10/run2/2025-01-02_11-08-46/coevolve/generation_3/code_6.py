import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize the heuristic matrix with zeros
    heuristic_matrix = np.zeros_like(distance_matrix)
    
    # Normalize distances based on the maximum distance in the matrix
    max_distance = np.max(distance_matrix)
    normalized_distance_matrix = distance_matrix / max_distance
    
    # Apply the minimum sum heuristic
    # This is a simple example where we use the sum of the row (from a given city to all others)
    # as a heuristic value for the edge from the first city to the last city in the row.
    for i in range(distance_matrix.shape[0]):
        heuristic_matrix[i, -1] = np.sum(normalized_distance_matrix[i, :-1])
    
    return heuristic_matrix