import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming the distance_matrix is symmetric and contains 0s on the diagonal
    # We will compute the heuristic for each edge based on some heuristic
    # For simplicity, let's use the sum of the distances from the start node to all other nodes
    # as a simple heuristic for the edge cost.
    
    # Compute the heuristic for each edge
    num_nodes = distance_matrix.shape[0]
    heuristic_matrix = np.zeros_like(distance_matrix)
    
    # The heuristic for an edge (i, j) can be the sum of distances from node i to all other nodes
    # and node j to all other nodes minus the distance between i and j itself
    # This heuristic is based on the assumption that the best path from i to j does not pass through j
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                heuristic_matrix[i, j] = np.sum(distance_matrix[i, :]) + np.sum(distance_matrix[j, :]) - distance_matrix[i, j]
    
    return heuristic_matrix