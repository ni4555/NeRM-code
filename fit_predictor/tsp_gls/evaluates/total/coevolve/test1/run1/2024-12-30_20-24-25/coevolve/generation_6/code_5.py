import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Calculate the Manhattan distance matrix
    manhattan_distance_matrix = np.abs(np.subtract(distance_matrix, np.min(distance_matrix, axis=0)))
    manhattan_distance_matrix = np.abs(np.subtract(distance_matrix, np.min(distance_matrix, axis=1)))

    # Calculate the sum of the Manhattan distances to get the heuristic matrix
    heuristic_matrix = np.sum(manhattan_distance_matrix, axis=1)
    return heuristic_matrix