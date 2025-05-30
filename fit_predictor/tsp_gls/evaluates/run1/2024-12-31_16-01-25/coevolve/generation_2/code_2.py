import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Calculate the number of nodes
    num_nodes = distance_matrix.shape[0]
    
    # Initialize the heuristic matrix with zeros
    heuristics = np.zeros_like(distance_matrix)
    
    # Calculate the Manhattan distance for each edge to the nearest vertex
    for i in range(num_nodes):
        for j in range(num_nodes):
            # Avoid considering the same node twice or the edge that leads back to the same node
            if i != j:
                # Compute Manhattan distance to all other nodes and take the minimum
                heuristics[i, j] = np.min(distance_matrix[i] + distance_matrix[j])
                
    return heuristics