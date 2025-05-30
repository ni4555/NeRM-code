import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize an array with the same shape as the distance matrix to store heuristics
    heuristics = np.zeros_like(distance_matrix)

    # Calculate the sum of distances in each row (from a vertex to all others)
    row_sums = np.sum(distance_matrix, axis=1)

    # Calculate the sum of distances in each column (to a vertex from all others)
    col_sums = np.sum(distance_matrix, axis=0)

    # Compute the heuristics based on the distance-based normalization and dynamic minimum spanning tree construction
    # The heuristic for each edge is the difference between the sum of distances from one vertex to all others
    # and the sum of distances to that vertex from all others, divided by the maximum possible distance
    # (which is the diameter of the graph, which we assume to be the maximum value in the distance matrix)
    max_distance = np.max(distance_matrix)
    heuristics = (row_sums - col_sums) / max_distance

    return heuristics