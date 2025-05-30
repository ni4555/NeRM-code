import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Calculate the heuristic by inverting the distance matrix values.
    # Note that for the TSP, the distance matrix is symmetric, so we could
    # use either the upper or lower triangle of the matrix to compute the heuristic.
    # Here, we are using the upper triangle to avoid redundancy.
    heuristics = 1.0 / np.triu(distance_matrix)
    
    # Fill the diagonal with a large number to avoid including the trivial edge
    # (the edge to the same node, which would be zero distance).
    np.fill_diagonal(heuristics, np.inf)
    
    return heuristics