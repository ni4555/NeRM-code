import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Calculate the heuristic value for each edge by computing the sum of the distances
    # between the endpoints of the edge and all other nodes, minus the distance
    # between the endpoints themselves. This assumes that the distance matrix is symmetric.
    n = distance_matrix.shape[0]
    heuristics = np.zeros_like(distance_matrix)
    
    for i in range(n):
        for j in range(n):
            if i != j:
                heuristics[i, j] = np.sum(distance_matrix[i, :]) + np.sum(distance_matrix[:, j]) - 2 * distance_matrix[i, j]
                
    return heuristics