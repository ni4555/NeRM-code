import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Normalize the distance matrix using the maximum distance in each row
    row_max = np.max(distance_matrix, axis=1, keepdims=True)
    normalized_matrix = distance_matrix / row_max
    
    # Calculate the minimum sum of the row max values as a base for the heuristic
    min_sum = np.sum(row_max)
    
    # Generate a new heuristic matrix where each element is the difference between
    # the minimum sum and the corresponding normalized distance
    heuristic_matrix = min_sum - normalized_matrix
    
    # Apply a robust minimum sum heuristic for precise edge selection
    # (This part is conceptual, as the actual implementation would depend on the
    # specifics of the robust minimum sum heuristic used)
    robust_heuristic_matrix = heuristic_matrix - np.min(heuristic_matrix)
    
    # The resulting robust heuristic matrix should indicate how bad it is to include each edge
    return robust_heuristic_matrix