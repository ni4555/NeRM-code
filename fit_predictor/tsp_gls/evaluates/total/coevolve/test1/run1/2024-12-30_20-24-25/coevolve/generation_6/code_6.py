import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Compute the Manhattan distance matrix, which is used as the heuristic matrix
    heuristic_matrix = np.abs(np.subtract(distance_matrix, np.mean(distance_matrix, axis=0)))
    return heuristic_matrix