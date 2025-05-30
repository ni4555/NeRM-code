import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize an array of the same shape as distance_matrix with high values
    heuristic_values = np.full(distance_matrix.shape, np.inf)
    
    # Apply distance-based normalization
    max_distance = np.max(distance_matrix)
    min_distance = np.min(distance_matrix)
    distance_matrix_normalized = (distance_matrix - min_distance) / (max_distance - min_distance)
    
    # Calculate a robust minimum sum heuristic
    # We can use the mean or median for robustness; here, we choose the median
    median_distance = np.median(distance_matrix_normalized)
    
    # Set the heuristic value for each edge to be the negative of the normalized distance
    # minus a robust measure based on the median
    heuristic_values = -distance_matrix_normalized + (distance_matrix_normalized >= median_distance)
    
    return heuristic_values