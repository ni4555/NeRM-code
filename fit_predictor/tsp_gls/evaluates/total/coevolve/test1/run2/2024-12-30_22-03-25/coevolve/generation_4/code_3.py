import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Calculate the sum of the longest edges for each node pair
    max_edge_sums = np.max(distance_matrix, axis=0) + np.max(distance_matrix, axis=1)
    
    # Calculate the minimum distance from each node to all others
    min_distances = np.min(distance_matrix, axis=1)
    
    # Combine both criteria to form the heuristic values
    # The heuristic value for each edge is the sum of the maximum edge sums and the minimum distances
    heuristic_values = max_edge_sums + min_distances
    
    # Create an array with the same shape as the distance matrix
    # Filling with 1.0 assumes that the higher the heuristic value, the less favorable the edge
    heuristics = np.full(distance_matrix.shape, fill_value=1.0)
    
    # For each edge, assign the heuristic value to the corresponding indices
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            if distance_matrix[i][j] != 0:  # Avoid assigning heuristics to zero diagonal elements
                heuristics[i][j] = heuristic_values[i] + heuristic_values[j]
    
    return heuristics