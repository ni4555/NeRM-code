import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # This function calculates the heuristics for each edge in the distance matrix.
    # The heuristic is based on the distance to the nearest city for each city.
    # It assumes that the distance matrix is symmetric and that each row and column represents a city.
    
    # Initialize the heuristics array with the same shape as the distance matrix
    # and set the diagonal elements to 0 since the distance to a city itself is zero.
    heuristics = np.full(distance_matrix.shape, np.inf)
    np.fill_diagonal(heuristics, 0)
    
    # For each city, find the minimum distance to any other city
    for i in range(distance_matrix.shape[0]):
        heuristics[i] = np.min(distance_matrix[i])
    
    return heuristics