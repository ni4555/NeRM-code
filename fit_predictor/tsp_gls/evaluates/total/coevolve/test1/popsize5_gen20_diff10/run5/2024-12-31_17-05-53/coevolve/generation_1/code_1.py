import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming distance_matrix is a square matrix with positive entries
    num_cities = distance_matrix.shape[0]
    
    # Initialize a matrix of the same shape as the distance matrix with zeros
    heuristics_matrix = np.zeros_like(distance_matrix)
    
    # Iterate over all pairs of cities
    for i in range(num_cities):
        for j in range(num_cities):
            if i != j:
                # Calculate the heuristic for edge (i, j)
                # This is a simple example, where we assume that the heuristic is the inverse
                # of the distance (the shorter the distance, the better the heuristic)
                heuristics_matrix[i, j] = 1 / distance_matrix[i, j]
    
    return heuristics_matrix