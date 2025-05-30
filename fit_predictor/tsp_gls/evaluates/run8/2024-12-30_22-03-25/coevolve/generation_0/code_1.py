import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Number of nodes in the distance matrix
    num_nodes = distance_matrix.shape[0]
    
    # Initialize the heuristics array with zeros
    heuristics = np.zeros_like(distance_matrix)
    
    # Compute the heuristic for each edge
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                # The heuristic is the sum of the distances from node i to all other nodes
                # and node j to all other nodes minus the distance from node i to node j
                # (since that's the distance we're avoiding).
                heuristics[i, j] = np.sum(distance_matrix[i, :]) + np.sum(distance_matrix[j, :]) - distance_matrix[i, j]
    
    return heuristics