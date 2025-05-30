import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming the distance matrix is symmetric and the diagonal elements are 0
    # Calculate the pairwise distances using a simple heuristic, such as the sum of distances
    # minus the minimum distance found in the neighborhood of each node.
    n_nodes = distance_matrix.shape[0]
    heuristics_matrix = np.zeros_like(distance_matrix)
    
    for i in range(n_nodes):
        # For each node, find the minimum distance to any other node
        min_distance = np.min(distance_matrix[i, :])
        
        # Calculate the heuristic for the current node
        for j in range(n_nodes):
            if i != j:
                heuristics_matrix[i, j] = distance_matrix[i, j] - min_distance
    
    return heuristics_matrix