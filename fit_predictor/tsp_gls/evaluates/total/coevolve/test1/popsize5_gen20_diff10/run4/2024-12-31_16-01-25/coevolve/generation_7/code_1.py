import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Calculate the diagonal of the distance matrix
    # Diagonal elements are the distances to the starting city
    start_distances = np.diag(distance_matrix)
    
    # Calculate the minimum distance from each city to all other cities
    min_distances = np.min(distance_matrix, axis=1)
    
    # Create a new matrix of the same shape as the distance matrix
    heuristics = np.zeros_like(distance_matrix, dtype=np.float64)
    
    # Fill the matrix with the calculated heuristic values
    # The heuristic for an edge (i, j) is the difference between the sum of distances
    # from i to all other cities and from j to all other cities
    # minus the distance between i and j itself.
    for i in range(len(distance_matrix)):
        for j in range(len(distance_matrix)):
            if i != j:
                heuristics[i, j] = (start_distances[i] + start_distances[j] - distance_matrix[i, j])
    
    return heuristics