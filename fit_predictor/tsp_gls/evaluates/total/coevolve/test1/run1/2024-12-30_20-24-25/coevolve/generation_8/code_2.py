import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # This function assumes a heuristic based on the Manhattan distance
    # for the purpose of this example. This is just one possible heuristic
    # and does not necessarily represent the heuristic used in the described
    # state-of-the-art TSP solver.
    
    # Compute Manhattan distance heuristics for each edge
    heuristics = np.abs(np.diff(distance_matrix, axis=0)) + np.abs(np.diff(distance_matrix, axis=1))
    
    # Normalize heuristics to ensure they are non-negative and have the same shape as the distance matrix
    heuristics = heuristics.astype(np.float32)
    return heuristics