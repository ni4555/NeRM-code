import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # The following is a simple example of a distance-based heuristic that assumes
    # that we want to minimize the distance. Therefore, a high heuristic value for
    # an edge will be one that is relatively longer, indicating a bad choice.
    
    # Calculate the maximum distance from each node to any other node, which will be
    # used to penalize edges that are long in comparison.
    max_distances = np.max(distance_matrix, axis=1)
    
    # Create an empty array for the heuristics with the same shape as the distance matrix
    heuristics = np.zeros_like(distance_matrix)
    
    # For each edge (i, j) in the distance matrix, compute the heuristic value
    # by taking the ratio of the edge distance to the maximum distance from node i
    # to any other node. This ratio penalizes longer edges more heavily.
    for i in range(distance_matrix.shape[0]):
        for j in range(i+1, distance_matrix.shape[1]):
            edge_length = distance_matrix[i, j]
            max_dist_from_i = max_distances[i]
            heuristics[i, j] = heuristics[j, i] = edge_length / max_dist_from_i
            
    return heuristics