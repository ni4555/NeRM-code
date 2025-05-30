import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Normalize the distance matrix by subtracting the minimum distance from each edge
    normalized_distance_matrix = distance_matrix - np.min(distance_matrix)
    
    # Calculate the minimum sum heuristic for each edge
    min_sum_heuristic = np.sum(normalized_distance_matrix, axis=0)
    
    # Return the combined heuristics
    return normalized_distance_matrix - min_sum_heuristic[:, np.newaxis]