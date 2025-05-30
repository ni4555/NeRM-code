import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize the heuristic array with zeros
    heuristic_matrix = np.zeros_like(distance_matrix)
    
    # Set the diagonal to infinity as no edge has zero distance to itself
    np.fill_diagonal(heuristic_matrix, np.inf)
    
    # Calculate the minimum distance for each edge
    min_distance = np.min(distance_matrix, axis=1)
    
    # Subtract the minimum distance from all other distances to get the heuristic
    heuristic_matrix = distance_matrix - min_distance[:, np.newaxis]
    
    return heuristic_matrix