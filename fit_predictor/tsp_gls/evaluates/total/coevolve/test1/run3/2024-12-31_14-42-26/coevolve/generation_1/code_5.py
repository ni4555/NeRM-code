import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # This function will calculate the heuristics for each edge in the distance matrix
    # For simplicity, let's assume a naive heuristic: the sum of the distances to the nearest neighbor
    
    # Initialize an array of the same shape with the same type, filled with zeros
    heuristics = np.zeros_like(distance_matrix, dtype=distance_matrix.dtype)
    
    # Iterate over each edge in the distance matrix
    for i in range(len(distance_matrix)):
        for j in range(i+1, len(distance_matrix)):  # Skip the diagonal, since it's an edge to itself
            # Find the nearest neighbor for node i and node j
            nearest_i = np.argmin(distance_matrix[i])
            nearest_j = np.argmin(distance_matrix[j])
            
            # Calculate the heuristic as the sum of distances to the nearest neighbors
            heuristics[i, j] = distance_matrix[i, nearest_i] + distance_matrix[j, nearest_j]
            heuristics[j, i] = distance_matrix[j, nearest_j] + distance_matrix[i, nearest_i]  # Symmetry
    
    return heuristics