import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Calculate the minimum pairwise distances among nodes
    min_distances = np.min(distance_matrix, axis=1)
    
    # Create a heuristic matrix based on the minimum distances
    # We assume that the heuristic is the inverse of the minimum distance
    # because lower distances are better (i.e., more "favorable" edges)
    heuristics = 1.0 / min_distances
    
    # Return the heuristics matrix, ensuring no division by zero
    heuristics[distance_matrix == 0] = 0
    return heuristics