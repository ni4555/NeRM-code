import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Convert distance matrix to a difference matrix
    # The difference matrix represents the cost of moving horizontally or vertically between cities
    diff_matrix = np.abs(np.diff(distance_matrix, axis=0)).sum(axis=1) + np.abs(np.diff(distance_matrix, axis=1)).sum(axis=0)
    
    # The heuristic value for each edge is the sum of the two possible costs of moving between the cities
    # The heuristic for the edge between city i and city j is the maximum of the two possible Manhattan distances
    heuristic_matrix = np.maximum.accumulate(diff_matrix, axis=0) + np.maximum.accumulate(diff_matrix, axis=1) - diff_matrix
    
    return heuristic_matrix