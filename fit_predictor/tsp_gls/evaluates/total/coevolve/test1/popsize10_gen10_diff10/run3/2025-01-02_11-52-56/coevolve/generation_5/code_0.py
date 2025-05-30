import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Calculate the initial heuristic values based on the symmetric distance matrix
    heuristics = np.abs(np.triu(distance_matrix, k=1)) + np.abs(np.tril(distance_matrix, k=-1))
    
    # Apply a simple distance-based heuristic for initial path estimation
    heuristics += np.min(distance_matrix, axis=1) + np.min(distance_matrix, axis=0)
    
    return heuristics