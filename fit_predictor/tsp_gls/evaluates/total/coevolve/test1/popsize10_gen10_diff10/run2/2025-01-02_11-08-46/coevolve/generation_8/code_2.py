import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assume distance_matrix is a symmetric, square matrix
    # Calculate the maximum distance in the matrix to normalize distances
    max_distance = np.max(distance_matrix)
    
    # Normalize distances by the maximum distance to create a normalized distance matrix
    normalized_distance_matrix = distance_matrix / max_distance
    
    # Calculate the heuristic value for each edge by using the normalized distance
    # A lower normalized distance indicates a better (i.e., less costly) edge to include in the tour
    heuristics_values = 1 - normalized_distance_matrix
    
    # Use the minimum sum heuristic to refine the heuristic values
    # We take the minimum value for each edge considering all possible tours starting from that edge
    min_sum_heuristic = np.min(heuristics_values, axis=0)
    
    return min_sum_heuristic