import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Normalize distance matrix with respect to the minimum distance
    min_distance = np.min(distance_matrix, axis=1, keepdims=True)
    normalized_distance_matrix = distance_matrix / min_distance
    
    # Apply minimum sum heuristic for edge selection
    # The idea is to select edges with the smallest sum of heuristics
    min_sum_heuristic = np.sum(normalized_distance_matrix, axis=0)
    
    # Invert the values to create a heuristic function that encourages selection of smaller distances
    heuristics = 1 / (1 + min_sum_heuristic)
    
    return heuristics