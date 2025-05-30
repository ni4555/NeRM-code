import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Calculate Manhattan distance for each edge in the distance matrix
    manhattan_distance = np.abs(np.diff(distance_matrix, axis=0, prepend=distance_matrix[-1, :]) + 
                                 np.diff(distance_matrix, axis=1, prepend=distance_matrix[:, -1]))
    
    # Normalize the Manhattan distances by the direct distances to create a heuristic
    # We use direct distance matrix as a base for normalization to ensure that the heuristic
    # is consistent with the actual distances.
    direct_distance = np.linalg.norm(distance_matrix, axis=1)
    heuristic = manhattan_distance / direct_distance[:, np.newaxis]
    
    # Return the heuristic matrix
    return heuristic