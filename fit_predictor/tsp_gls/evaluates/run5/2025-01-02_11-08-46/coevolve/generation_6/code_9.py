import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Calculate the maximum distance in the matrix
    max_distance = np.max(distance_matrix)
    
    # Normalize each edge by the maximum distance
    normalized_distance_matrix = distance_matrix / max_distance
    
    # Calculate the minimum sum heuristic (the sum of the minimum distances from each vertex)
    min_sum_heuristic = np.sum(np.min(distance_matrix, axis=1))
    
    # Create a matrix of the heuristic values
    heuristic_matrix = np.full(distance_matrix.shape, max_distance)
    
    # Apply the distance-based normalization to the heuristic matrix
    heuristic_matrix = heuristic_matrix * normalized_distance_matrix
    
    # Subtract the minimum sum heuristic from each edge's heuristic value
    heuristic_matrix -= min_sum_heuristic
    
    # Ensure that the heuristic values are non-negative
    heuristic_matrix = np.maximum(heuristic_matrix, 0)
    
    return heuristic_matrix