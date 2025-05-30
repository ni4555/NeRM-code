import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Calculate the diagonal of the distance matrix
    # to avoid considering the distance from a city to itself
    np.fill_diagonal(distance_matrix, np.inf)
    
    # Apply a distance-based heuristic such as the nearest neighbor
    # For simplicity, we use the Manhattan distance
    # This heuristic assumes that each edge is equally weighted
    # and the salesman should visit the nearest city next.
    heuristics = np.abs(np.subtract.outer(np.arange(distance_matrix.shape[0]), 
                                          np.arange(distance_matrix.shape[0])))
    heuristics = np.sum(distance_matrix * heuristics, axis=1)
    
    return heuristics