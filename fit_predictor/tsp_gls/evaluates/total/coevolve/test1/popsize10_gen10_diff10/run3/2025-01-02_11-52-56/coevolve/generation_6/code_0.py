import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming the distance_matrix is symmetric (distance from i to j is the same as from j to i)
    # We will use a simple heuristic that considers the average distance from each node to all others
    # to estimate the "badness" of an edge. The lower the value, the better the edge.
    num_nodes = distance_matrix.shape[0]
    heuristic_matrix = np.zeros_like(distance_matrix)
    
    for i in range(num_nodes):
        row_sum = np.sum(distance_matrix[i])
        heuristic_matrix[i] = distance_matrix[i] / row_sum
    
    return heuristic_matrix