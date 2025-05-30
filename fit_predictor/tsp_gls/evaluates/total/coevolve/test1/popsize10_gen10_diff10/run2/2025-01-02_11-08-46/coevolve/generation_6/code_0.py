import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Normalize the distance matrix
    normalized_distances = distance_matrix / np.sum(distance_matrix, axis=0)
    
    # Create a minimum sum heuristic by summing the normalized distances along the diagonal
    # This encourages paths that visit fewer unique cities
    min_sum_heuristic = np.sum(normalized_distances, axis=1)
    
    # Subtract from 1 to turn the heuristic into a penalty function, where lower values are better
    return 1 - min_sum_heuristic

# Example usage:
# Assuming 'dist_matrix' is a distance matrix of shape (n, n) where n is the number of cities
# dist_matrix = np.random.rand(n, n)  # Replace this with the actual distance matrix
# print(heuristics_v2(dist_matrix))