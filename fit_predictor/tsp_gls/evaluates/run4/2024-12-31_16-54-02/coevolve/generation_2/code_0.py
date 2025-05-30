import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # This function computes the heuristic values for each edge in the distance matrix.
    # We assume that the distance matrix is symmetric and that the diagonal elements are 0.
    # The heuristic for each edge is calculated as the sum of the distances from the
    # two nodes to any other node, effectively calculating a "double shortest path" heuristic.
    
    # Initialize the heuristic matrix with zeros
    heuristics_matrix = np.zeros_like(distance_matrix)
    
    # Loop through all unique pairs of nodes (i, j) where i < j to avoid duplicate edges
    for i in range(distance_matrix.shape[0]):
        for j in range(i + 1, distance_matrix.shape[1]):
            # For each edge, compute the heuristic by summing the distances to all other nodes
            heuristics_matrix[i, j] = np.sum(distance_matrix[i, :]) + np.sum(distance_matrix[j, :])
    
    return heuristics_matrix