import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    num_nodes = distance_matrix.shape[0]
    heuristics_matrix = np.full_like(distance_matrix, np.inf)
    
    for i in range(num_nodes):
        # For each node, compute the minimum distance to all other nodes
        min_distances = np.min(distance_matrix[i], axis=0)
        
        # For each edge, compute the heuristic as the ratio of the edge distance
        # to the minimum distance from the current node to any other node
        for j in range(num_nodes):
            if i != j:
                heuristics_matrix[i][j] = distance_matrix[i][j] / min_distances[j]
    
    return heuristics_matrix