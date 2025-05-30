import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Calculate the maximum distance in the matrix
    max_distance = np.max(distance_matrix)
    
    # Create a matrix of ones with the same shape as the distance matrix
    heuristics_matrix = np.ones_like(distance_matrix)
    
    # For each edge, subtract the maximum distance if the current distance is less than the maximum
    # This effectively penalizes shorter edges (which should be preferred in a TSP solution)
    heuristics_matrix[distance_matrix < max_distance] = -max_distance
    
    return heuristics_matrix