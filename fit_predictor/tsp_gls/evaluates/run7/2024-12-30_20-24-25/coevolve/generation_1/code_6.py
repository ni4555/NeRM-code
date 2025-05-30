import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming the heuristic is the minimum distance to the nearest city
    # Initialize an array of the same shape as the distance matrix to hold the heuristics
    heuristics = np.full(distance_matrix.shape, np.inf)
    
    # Iterate over each city
    for i in range(len(distance_matrix)):
        # For each city, find the minimum distance to any other city
        min_distance = np.min(distance_matrix[i])
        # Update the heuristic array with the minimum distance found
        heuristics[i] = min_distance
    
    return heuristics