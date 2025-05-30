import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Calculate the minimum distance for each pair of cities
    min_distances = np.min(distance_matrix, axis=1)
    
    # Calculate the heuristic values by subtracting the minimum distance
    # from the total distance to each city
    heuristics = distance_matrix.sum(axis=1) - min_distances
    
    return heuristics