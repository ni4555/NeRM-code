import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # This function assumes that distance_matrix is symmetric
    # as per the description of the symmetric distance matrix.
    # The heuristic will return the inverse of the distance for each edge,
    # which can be interpreted as a measure of the "goodness" of including that edge
    # in the path. Lower values indicate a better edge to include.
    
    # Calculate the inverse of the distance matrix where the distance is not zero
    # (assuming that zero distance represents the same location, and should not be
    # included in the heuristic).
    heuristic_matrix = np.where(distance_matrix != 0, 1 / distance_matrix, 0)
    
    return heuristic_matrix