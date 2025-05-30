import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Calculate the Manhattan distance for each edge in the distance matrix
    manhattan_distances = np.abs(distance_matrix - np.mean(distance_matrix, axis=0))
    
    # Compute the average Manhattan distance for each edge
    average_distances = np.mean(manhattan_distances, axis=1)
    
    # Use the average distance as the heuristic value for each edge
    heuristics = average_distances
    
    return heuristics