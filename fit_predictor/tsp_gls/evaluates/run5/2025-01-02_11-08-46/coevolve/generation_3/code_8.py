import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Incorporating distance-based normalization and a minimum sum heuristic
    # Normalize the distance matrix by dividing each element by the sum of its row
    normalized_distances = distance_matrix / np.sum(distance_matrix, axis=1, keepdims=True)
    
    # Calculate the minimum sum heuristic for each row
    min_row_sums = np.min(distance_matrix, axis=1)
    
    # The heuristic value for each edge is the normalized distance minus the min row sum
    heuristic_values = normalized_distances - min_row_sums
    
    # Ensure non-negative values for the heuristic function
    heuristic_values = np.maximum(heuristic_values, 0)
    
    return heuristic_values