import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming distance_matrix is a square matrix with shape (n, n)
    # where n is the number of cities
    n = distance_matrix.shape[0]
    
    # Calculate the average distance for each edge
    # This heuristic assumes that a higher average distance is "bad"
    heuristic_matrix = np.zeros_like(distance_matrix)
    for i in range(n):
        for j in range(i+1, n):  # Only calculate for upper triangle to avoid double counting
            # Calculate the average distance to all other points from city i and city j
            avg_distance_i = np.mean(distance_matrix[i, :])
            avg_distance_j = np.mean(distance_matrix[j, :])
            # Store the average distance as the heuristic value for the edge (i, j)
            heuristic_matrix[i, j] = heuristic_matrix[j, i] = (avg_distance_i + avg_distance_j) / 2
    
    return heuristic_matrix