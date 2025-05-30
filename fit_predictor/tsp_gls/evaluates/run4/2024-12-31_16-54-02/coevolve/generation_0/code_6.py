import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize the heuristic matrix with zeros
    heuristic_matrix = np.zeros_like(distance_matrix)
    
    # Calculate the maximum distance in the matrix
    max_distance = np.max(distance_matrix)
    
    # Calculate the heuristic for each edge as the ratio of the distance to the maximum distance
    heuristic_matrix = distance_matrix / max_distance
    
    return heuristic_matrix