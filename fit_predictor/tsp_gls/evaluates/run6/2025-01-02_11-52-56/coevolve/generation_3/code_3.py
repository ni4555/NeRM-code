import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming the distance_matrix is square and symmetric
    n = distance_matrix.shape[0]
    
    # Initialize heuristics matrix with zeros
    heuristics = np.zeros_like(distance_matrix)
    
    # Calculate the average distance from each node to all other nodes
    average_distances = np.sum(distance_matrix, axis=1) / (n - 1)
    
    # Calculate the heuristics for each edge
    for i in range(n):
        for j in range(n):
            if i != j:
                # The heuristic for edge i-j is the difference between the
                # average distance of i to all other nodes and the distance
                # from i to j, plus a small constant to avoid zero values
                heuristics[i][j] = average_distances[i] - distance_matrix[i][j] + 0.0001
    
    return heuristics