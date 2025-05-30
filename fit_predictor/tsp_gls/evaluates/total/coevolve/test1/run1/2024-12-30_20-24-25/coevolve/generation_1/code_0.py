import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize a matrix of the same shape as the distance matrix with zeros
    heuristic_matrix = np.zeros_like(distance_matrix)
    
    # Assuming that the heuristic is based on some function of the distances
    # For example, a simple heuristic could be the average distance from a vertex to all other vertices
    for i in range(distance_matrix.shape[0]):
        # Sum the distances from vertex i to all other vertices
        total_distance = np.sum(distance_matrix[i])
        # Divide by the number of vertices minus one (not including the distance to itself)
        num_vertices = distance_matrix.shape[0]
        heuristic_matrix[i] = total_distance / (num_vertices - 1)
    
    return heuristic_matrix