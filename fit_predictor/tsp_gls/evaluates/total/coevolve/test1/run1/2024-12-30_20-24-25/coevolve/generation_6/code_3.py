import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Calculate Manhattan distance for each edge
    manhattan_distances = np.abs(np.diff(distance_matrix, axis=0)).sum(axis=1)
    
    # Return the Manhattan distance as the heuristic value for each edge
    return manhattan_distances