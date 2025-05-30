import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming that the distance matrix is symmetric (distance[i][j] == distance[j][i])
    # and the first row and column represent the distance from/to the city itself, which are 0.
    n_cities = distance_matrix.shape[0]
    
    # Create a heuristic matrix initialized with zeros
    heuristic_matrix = np.zeros_like(distance_matrix)
    
    # Calculate the Manhattan distance for each edge
    for i in range(n_cities):
        for j in range(n_cities):
            if i != j:
                # Calculate Manhattan distance between city i and city j
                heuristic_value = np.abs(i - j)
                heuristic_matrix[i][j] = heuristic_value
    
    return heuristic_matrix