import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming that the distance matrix is symmetric
    # Calculate the row and column minimums for each node
    min_row = np.min(distance_matrix, axis=1)
    min_col = np.min(distance_matrix, axis=0)
    
    # Compute the minimum sum heuristic for each edge
    min_sum_heuristic = np.maximum(min_row, min_col) - distance_matrix
    
    # Apply advanced distance-based normalization techniques
    # Here we use a simple normalization approach, but it can be replaced with more advanced methods
    normalized_min_sum_heuristic = min_sum_heuristic / np.max(min_sum_heuristic)
    
    return normalized_min_sum_heuristic