import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming that the distance_matrix is a square matrix of shape (n, n)
    # where n is the number of nodes in the TSP problem.
    n = distance_matrix.shape[0]
    
    # The heuristic for each edge can be defined as the sum of the distances
    # from the start node to one endpoint of the edge and from the other endpoint
    # to the end node (minus the distance between the two endpoints to avoid double-counting).
    # For simplicity, we'll use the sum of the distances from the start node to one endpoint
    # and from the other endpoint to the end node as the heuristic for each edge.
    
    # Create a matrix to store the heuristic values
    heuristic_matrix = np.zeros_like(distance_matrix)
    
    # The heuristic for the first edge is simply the distance from the start node to the first node
    heuristic_matrix[0, 1:] = distance_matrix[0, 1:]
    
    # The heuristic for the edge from node i to node j is the sum of the distances
    # from node i to node j and from node j to the end node
    for i in range(1, n):
        for j in range(i+1, n):
            heuristic_matrix[i, j] = distance_matrix[i, j] + distance_matrix[j, n-1]
            heuristic_matrix[j, i] = distance_matrix[j, i] + distance_matrix[i, n-1]
    
    # The heuristic for the edge from the last node back to the start node
    # is the distance from the last node to the start node
    heuristic_matrix[n-1, 0] = distance_matrix[n-1, 0]
    
    return heuristic_matrix