import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize the heuristics array with zeros, of the same shape as the distance matrix
    heuristics = np.zeros_like(distance_matrix)
    
    # Normalize the distance matrix by subtracting the minimum distance from each edge
    normalized_distance_matrix = distance_matrix - np.min(distance_matrix, axis=0)
    
    # Calculate the minimum sum heuristic by taking the minimum sum of distances for each starting point
    min_sum_heuristic = np.min(normalized_distance_matrix, axis=1)
    
    # Calculate the heuristics based on the normalized distances and minimum sum heuristic
    heuristics = normalized_distance_matrix / min_sum_heuristic[:, np.newaxis]
    
    return heuristics