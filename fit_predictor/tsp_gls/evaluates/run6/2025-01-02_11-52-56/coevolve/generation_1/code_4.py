import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Calculate the Manhattan distance between each pair of nodes
    row_diffs = np.abs(np.diff(distance_matrix, axis=0))
    col_diffs = np.abs(np.diff(distance_matrix, axis=1))
    
    # Sum the Manhattan distances to create an estimate of the total path length
    heuristics = row_diffs.sum(axis=1) + col_diffs.sum(axis=0)
    
    # Subtract the maximum distance in the matrix from each heuristic value to normalize
    max_distance = np.max(distance_matrix)
    normalized_heuristics = heuristics - max_distance
    
    return normalized_heuristics