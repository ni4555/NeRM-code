import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize the heuristic matrix with zeros
    heuristic_matrix = np.zeros_like(distance_matrix)
    
    # Calculate the minimum spanning tree (MST) using Prim's algorithm
    # and normalize distances based on the MST
    mst = np.zeros_like(distance_matrix, dtype=bool)
    num_nodes = distance_matrix.shape[0]
    visited = np.zeros(num_nodes, dtype=bool)
    
    # Start from an arbitrary node (node 0)
    visited[0] = True
    for _ in range(num_nodes - 1):
        min_distance = np.inf
        next_node = -1
        for i in range(num_nodes):
            if not visited[i]:
                for j in range(num_nodes):
                    if not mst[i, j] and distance_matrix[i, j] < min_distance:
                        min_distance = distance_matrix[i, j]
                        next_node = j
        mst[next_node, range(num_nodes)] = True
        mst[range(num_nodes), next_node] = True
        visited[next_node] = True
    
    # Normalize distances based on the MST
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j and not mst[i, j]:
                heuristic_matrix[i, j] = 1 - (distance_matrix[i, j] / min_distance)
    
    return heuristic_matrix