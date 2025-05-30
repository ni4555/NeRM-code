import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Calculate the diagonal of the distance matrix, which represents the distance from a node to itself
    identity = np.eye(distance_matrix.shape[0])
    
    # Calculate the sum of the distances for each edge (excluding the diagonal)
    edge_sums = distance_matrix - identity
    
    # Use the edge sums to create a heuristic value for each edge
    # The heuristic function is a simple mean of the edge sums, normalized by the number of nodes minus 1
    heuristics = (edge_sums.sum(axis=1) / (distance_matrix.shape[0] - 1)).reshape(-1, 1)
    
    return heuristics