import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Calculate the distance-based normalization
    max_distance = np.max(distance_matrix)
    min_distance = np.min(distance_matrix)
    normalized_distances = (distance_matrix - min_distance) / (max_distance - min_distance)
    
    # Calculate the robust minimum sum heuristic
    min_row_sums = np.min(distance_matrix, axis=1)
    min_col_sums = np.min(distance_matrix, axis=0)
    min_sum_heuristic = np.minimum(min_row_sums, min_col_sums)
    
    # Combine the two heuristics using a weighted sum
    alpha = 0.5  # Weight for distance-based normalization
    beta = 0.5   # Weight for minimum sum heuristic
    combined_heuristic = alpha * normalized_distances + beta * min_sum_heuristic
    
    # Return the combined heuristic as an array of the same shape as the distance matrix
    return combined_heuristic