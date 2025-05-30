import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming Manhattan distance is used, calculate the Manhattan distance matrix
    manhattan_distance_matrix = np.abs(np.subtract.outer(distance_matrix, distance_matrix))
    
    # The heuristic value for each edge can be the sum of the row and column distances
    # This value represents the cost of visiting all cities in the row before the column
    heuristics_matrix = np.sum(manhattan_distance_matrix, axis=0)
    
    return heuristics_matrix