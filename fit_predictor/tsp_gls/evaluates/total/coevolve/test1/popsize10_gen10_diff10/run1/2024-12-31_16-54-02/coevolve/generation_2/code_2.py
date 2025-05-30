import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize an array with the same shape as distance_matrix to store the heuristics
    heuristics = np.zeros_like(distance_matrix)
    
    # For each pair of nodes (i, j), compute the shortest path length
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            if i != j:  # We do not need to compute the shortest path from a node to itself
                # The shortest path is the minimum distance between node i and node j
                shortest_path = np.min(distance_matrix[i, :]) + np.min(distance_matrix[j, :])
                # Calculate the "badness" of the edge (i, j)
                heuristics[i, j] = distance_matrix[i, j] - shortest_path
    
    return heuristics