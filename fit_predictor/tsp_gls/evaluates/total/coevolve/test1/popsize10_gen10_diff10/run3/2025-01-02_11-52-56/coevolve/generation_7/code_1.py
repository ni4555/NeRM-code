import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Calculate the diagonal matrix of the distance matrix
    diag = np.diag(distance_matrix)
    
    # Compute the maximum distance from each node to any other node
    max_distances = np.max(distance_matrix, axis=1)
    
    # Compute the heuristic values as the sum of the maximum distance from each node to any other node
    # and the distance to the nearest node (diagonal element).
    # This heuristic assumes that including an edge will not worsen the tour by more than the
    # distance to the nearest node plus the maximum distance from that node to any other node.
    heuristics = max_distances + np.maximum(0, distance_matrix.diagonal())
    
    return heuristics