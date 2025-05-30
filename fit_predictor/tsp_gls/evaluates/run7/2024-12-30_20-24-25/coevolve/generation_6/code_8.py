import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming Manhattan distance is used for the heuristic matrix
    Manhattan_distance = np.abs(np.subtract(distance_matrix, np.mean(distance_matrix, axis=0)))
    # The heuristic matrix is the Manhattan distance matrix
    return Manhattan_distance