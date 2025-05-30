import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize the heuristics array with zeros
    heuristics = np.zeros_like(distance_matrix)
    
    # Compute the heuristics using the shortest path algorithm
    for i in range(len(distance_matrix)):
        for j in range(i + 1, len(distance_matrix)):
            # Sum the distances between the current node and all other nodes
            # to get the total distance excluding the direct edge
            sum_distances = np.sum(distance_matrix[i]) + np.sum(distance_matrix[j])
            # Subtract the distance between the two nodes to get the
            # heuristic value for this edge
            heuristics[i, j] = sum_distances - distance_matrix[i, j]
            heuristics[j, i] = heuristics[i, j]  # Symmetric matrix
    
    return heuristics