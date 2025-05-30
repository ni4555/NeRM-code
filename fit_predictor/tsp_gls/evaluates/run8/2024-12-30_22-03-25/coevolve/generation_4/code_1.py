import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Calculate the minimum distance from each node to all others
    min_distances = np.min(distance_matrix, axis=1)
    
    # Initialize the heuristic matrix with zeros
    heuristic_matrix = np.zeros_like(distance_matrix)
    
    # Set the heuristic value for each edge to the difference between
    # the minimum distance to the destination node and the actual distance
    # between the two nodes in the distance matrix.
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            if i != j:
                heuristic_matrix[i, j] = min_distances[j] - distance_matrix[i, j]
    
    return heuristic_matrix