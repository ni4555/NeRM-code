import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize the heuristic matrix with zeros
    heuristic_matrix = np.zeros_like(distance_matrix)
    
    # Calculate the heuristic for each city
    for i in range(distance_matrix.shape[0]):
        # Find the nearest city for city i
        nearest_city = np.argmin(distance_matrix[i])
        # Calculate the heuristic value as the distance to the nearest city
        heuristic_matrix[i][nearest_city] = distance_matrix[i][nearest_city]
    
    return heuristic_matrix