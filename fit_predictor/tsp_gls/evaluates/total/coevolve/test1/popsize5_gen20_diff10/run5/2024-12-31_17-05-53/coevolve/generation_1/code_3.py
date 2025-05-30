import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming the distance_matrix is symmetric and the diagonal elements are zeros
    # The heuristic for each edge is calculated as the sum of the edge's distance and
    # the distance from the edge's end node to the next node in the tour (if any).
    # This heuristic assumes a simple greedy approach, where the heuristic for
    # an edge (i, j) is the distance between i and j plus the distance from j to the
    # next node in the tour.
    # For simplicity, we'll consider the next node to be the node with the smallest
    # distance from j, which will be node 0 (as the first node in the tour).
    
    # Precompute the distances from each node to all other nodes except itself
    distance_to_all = np.copy(distance_matrix)
    np.fill_diagonal(distance_to_all, np.inf)
    
    # Calculate the heuristic for each edge
    heuristic_matrix = np.zeros_like(distance_matrix)
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            if distance_matrix[i, j] != 0:
                # The heuristic for the edge (i, j) is the distance (i, j) plus the distance
                # from j to the next node in the tour (node 0)
                heuristic_matrix[i, j] = distance_matrix[i, j] + distance_to_all[j, 0]
    
    return heuristic_matrix