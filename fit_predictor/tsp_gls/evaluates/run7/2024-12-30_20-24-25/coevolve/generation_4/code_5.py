import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Calculate the Manhattan distance matrix
    manhattan_distance_matrix = np.abs(distance_matrix.sum(axis=0) - distance_matrix.sum(axis=1))

    # Apply a heuristic to create a prior indicator for each edge
    # This is a simple example where we consider the Manhattan distance as the heuristic
    heuristic_matrix = manhattan_distance_matrix / np.max(manhattan_distance_matrix)

    return heuristic_matrix