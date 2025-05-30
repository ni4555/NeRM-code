import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Normalize the distance matrix by the maximum distance in each row
    max_distances = np.max(distance_matrix, axis=1)
    normalized_matrix = distance_matrix / max_distances[:, np.newaxis]
    
    # Construct a dynamic minimum spanning tree (MST) using Prim's algorithm
    # Initialize an array to store the minimum edge weights for each vertex
    min_edge_weights = np.inf
    # Start with the first vertex
    min_edge_weights[0] = 0
    # Create an array to keep track of which vertices are included in the MST
    in_mst = np.zeros(distance_matrix.shape[0], dtype=bool)
    in_mst[0] = True
    
    for _ in range(distance_matrix.shape[0] - 1):
        # Find the vertex with the minimum edge weight not in the MST
        current_min = np.inf
        current_vertex = -1
        for i in range(distance_matrix.shape[0]):
            if not in_mst[i] and min_edge_weights[i] < current_min:
                current_min = min_edge_weights[i]
                current_vertex = i
        # Update the MST
        in_mst[current_vertex] = True
        min_edge_weights[current_vertex] = current_min
    
    # Use the MST to guide the heuristic
    # The heuristic is the inverse of the MST edge weights
    heuristic_matrix = 1 / min_edge_weights
    
    return heuristic_matrix