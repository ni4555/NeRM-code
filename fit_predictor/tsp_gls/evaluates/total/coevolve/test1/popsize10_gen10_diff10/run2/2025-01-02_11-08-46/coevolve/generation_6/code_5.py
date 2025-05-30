import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Normalize the distance matrix
    min_distance = np.min(distance_matrix)
    max_distance = np.max(distance_matrix)
    normalized_matrix = (distance_matrix - min_distance) / (max_distance - min_distance)
    
    # Apply a robust minimum sum heuristic
    # This can be a simple operation like taking the minimum of each row or column
    min_sum_per_row = np.min(normalized_matrix, axis=1)
    min_sum_per_col = np.min(normalized_matrix, axis=0)
    min_sum_heuristic = np.maximum(min_sum_per_row, min_sum_per_col)
    
    # The heuristic value for each edge is the negative of the minimum sum heuristic
    # This encourages the selection of edges with lower sums
    heuristic_matrix = -min_sum_heuristic
    
    return heuristic_matrix