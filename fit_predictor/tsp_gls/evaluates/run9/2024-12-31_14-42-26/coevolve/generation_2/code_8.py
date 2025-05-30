import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming that the distance_matrix is a symmetric matrix
    num_cities = distance_matrix.shape[0]
    heuristic_matrix = np.zeros_like(distance_matrix)
    
    # We could use the maximum distance from a city to any other city as a simple heuristic
    # This assumes that we want to minimize the longest distance first
    for i in range(num_cities):
        max_distance = np.max(distance_matrix[i])
        for j in range(num_cities):
            # If j is not i, set the heuristic value to be the maximum distance
            # If j is i, set the heuristic value to a very low number to indicate this city should not be visited
            if i != j:
                heuristic_matrix[i][j] = max_distance
            else:
                heuristic_matrix[i][j] = float('-inf')
    
    return heuristic_matrix