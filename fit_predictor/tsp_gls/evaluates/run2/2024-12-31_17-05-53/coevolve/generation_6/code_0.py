import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Calculate the Manhattan distance for each edge
    n = distance_matrix.shape[0]
    manhattan_distances = np.abs(np.diff(distance_matrix, axis=0, append=True))
    
    # Compute the average distance for each edge
    average_distances = np.mean(manhattan_distances, axis=1)
    
    # Use the average distance as a heuristic value for each edge
    heuristics = average_distances.reshape(n, n)
    
    # To ensure we do not consider the same edge twice (self-loops),
    # we can set the diagonal to a large number or simply not use it.
    np.fill_diagonal(heuristics, np.inf)
    
    return heuristics