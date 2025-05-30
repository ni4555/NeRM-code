import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming Manhattan distance is used for the heuristic matrix
    Manhattan_distance_matrix = np.abs(np.diff(distance_matrix, axis=0, append=False))
    Manhattan_distance_matrix = np.abs(np.diff(Manhattan_distance_matrix, axis=1, append=False))
    
    # The heuristic value for each edge is the Manhattan distance
    heuristic_matrix = Manhattan_distance_matrix.sum(axis=1)
    
    # Since the heuristic should be a measure of how "bad" it is to include an edge,
    # we can invert the heuristic to get a better measure (smaller values are better).
    # Subtracting from the maximum possible value of the Manhattan distance (sum of rows)
    max_manhattan_distance = Manhattan_distance_matrix.sum(axis=1).max()
    return max_manhattan_distance - heuristic_matrix