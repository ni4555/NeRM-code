import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming distance_matrix is symmetric and the diagonal elements are zeros
    num_nodes = distance_matrix.shape[0]
    
    # Initialize the heuristic matrix with zeros
    heuristic_matrix = np.zeros_like(distance_matrix)
    
    # Calculate the heuristic values for each edge
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            # A simple heuristic that considers the distance divided by the sum of distances
            # from i to all other nodes and from j to all other nodes
            sum_distances_from_i = np.sum(distance_matrix[i, :])
            sum_distances_from_j = np.sum(distance_matrix[j, :])
            heuristic_value = distance_matrix[i, j] / (sum_distances_from_i + sum_distances_from_j)
            heuristic_matrix[i, j] = heuristic_value
            heuristic_matrix[j, i] = heuristic_value  # Since the matrix is symmetric
    
    return heuristic_matrix