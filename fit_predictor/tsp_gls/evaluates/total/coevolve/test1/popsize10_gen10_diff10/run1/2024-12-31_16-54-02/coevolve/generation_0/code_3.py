import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Calculate the heuristics for each edge in the distance matrix
    # by considering the shortest path between any two nodes without
    # returning to the starting node.
    
    # Initialize an array to hold the heuristics for each edge
    num_nodes = distance_matrix.shape[0]
    heuristics = np.zeros_like(distance_matrix, dtype=np.float64)
    
    # Iterate over all pairs of nodes
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            # Find the shortest path from node i to node j, excluding
            # the edge between i and j itself.
            min_path_length = np.min(distance_matrix[i] * (distance_matrix[i] > 0) + 
                                     distance_matrix[j] * (distance_matrix[j] > 0) +
                                     distance_matrix[i, j])
            heuristics[i, j] = heuristics[j, i] = min_path_length
    
    return heuristics