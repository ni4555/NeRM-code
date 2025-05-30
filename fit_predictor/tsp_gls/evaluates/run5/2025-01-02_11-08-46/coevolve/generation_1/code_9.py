import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Calculate the minimum distance from each node to any other node
    min_distances = np.min(distance_matrix, axis=1)
    
    # Calculate the heuristic for each edge
    # The heuristic is the difference between the distance to the nearest node and the actual distance
    heuristics = distance_matrix - np.outer(min_distances, min_distances)
    
    # To ensure that the heuristics are non-negative, we clip the values at zero
    heuristics = np.clip(heuristics, 0, None)
    
    return heuristics