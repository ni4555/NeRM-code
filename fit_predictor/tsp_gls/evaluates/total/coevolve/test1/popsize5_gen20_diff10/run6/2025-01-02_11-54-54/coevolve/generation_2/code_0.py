import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Calculate the lower triangular matrix of the distance matrix
    # since the distance from node i to node j is the same as from j to i
    lower_triangular_matrix = np.tril(distance_matrix)
    
    # Calculate the sum of the distances in the lower triangular matrix
    total_distance = np.sum(lower_triangular_matrix)
    
    # The heuristic for each edge can be calculated as the distance
    # divided by the total distance, which gives a measure of the
    # relative importance of the edge in the context of the graph.
    heuristics = lower_triangular_matrix / total_distance
    
    # The resulting heuristics array will be the same shape as the input
    # distance matrix, with the diagonal elements set to zero (self-loops
    # have no impact on the heuristic).
    return heuristics